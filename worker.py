#!/usr/bin/env python3

"""
一个用于分析和（可选）微调 DNN 模型性能的 Python 脚本。
...
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.nvtx as nvtx
import time
import math
import sys
import logging # 导入 logging
from torch.cuda import Stream
from typing import List, Set, Dict, Tuple, Optional, Any

# 导入重构的模块
import config
import utils
from models import (
    load_model, 
    setup_model_for_inference, 
    setup_retrain_components
)
from data_pipeline import (
    initialize_video_capture, 
    build_preprocessing_pipelines, 
    read_batch_from_video, 
    preprocess_batch
)

# ==================================================================
# 核心执行步骤 (保留在主 worker 中)
# ==================================================================

def execute_step(mode: str,
                 model: nn.Module,
                 inputs_on_device: torch.Tensor,
                 labels_on_device: Optional[torch.Tensor],
                 use_amp: bool,
                 optimizer: Optional[torch.optim.Optimizer],
                 criterion: Optional[nn.Module],
                 scaler: torch.cuda.amp.GradScaler,
                 nvtx_prefix: str) -> float:
    """
    执行单个推理或训练步骤。
    """
    loss_item = 0.0

    if mode == 'inference':
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            nvtx.range_push(f"{nvtx_prefix}_D_inference_Batch")
            outputs = model(inputs_on_device)
            nvtx.range_pop()
    
    elif mode == 'retrain':
        if optimizer is None or criterion is None or labels_on_device is None:
            logging.error("训练模式缺少 optimizer, criterion, 或 labels。")
            return 0.0
            
        with torch.cuda.amp.autocast(enabled=use_amp):
            nvtx.range_push(f"{nvtx_prefix}_D_retrain_Batch")
            
            optimizer.zero_grad()
            outputs = model(inputs_on_device)
            loss = criterion(outputs, labels_on_device)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            nvtx.range_pop()
        
        loss_item = loss.item()
    
    return loss_item


# ==================================================================
# 主函数 (Main)
# ==================================================================

def main():
    """
    脚本的主执行函数。
    """
    
    # --- 1. 设置 ---
    args = config.parse_arguments()
    
    # [新] 初始化日志系统
    utils.setup_logging(model_name=args.model_name)

    utils.log_system_info()
    
    # 现在 setup_cpu_affinity 将使用 logging
    utils.setup_cpu_affinity(args.core_bind)

    # --- 2. 配置与设备 ---
    MODE = args.mode
    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size
    PREP_ON_GPU = args.prep_on_gpu
    INFER_ON_GPU = args.infer_on_gpu
    USE_AMP = args.use_amp

    # [修改] 初始化视频源 (包含 is_stream)
    cap, fps, total_frames, is_camera, is_stream = initialize_video_capture(args.input_path)
    
    DEFAULT_FRAMES_TO_PROFILE = 900
    if is_camera or is_stream:
        logging.info(f"检测到摄像头/流输入。将使用默认帧数 {DEFAULT_FRAMES_TO_PROFILE} 进行分析。")
        TOTAL_FRAMES_TO_PROFILE = DEFAULT_FRAMES_TO_PROFILE
    else:
        logging.info(f"检测到视频文件。将处理所有 {int(total_frames)} 帧。")
        TOTAL_FRAMES_TO_PROFILE = total_frames
    NUM_STEPS_TO_PROFILE = math.ceil(TOTAL_FRAMES_TO_PROFILE / BATCH_SIZE)


    if args.max_steps is not None and args.max_steps > 0:
        if NUM_STEPS_TO_PROFILE > args.max_steps and not (is_camera or is_stream):
            logging.info(f"--- [!] 已应用 --max-steps 限制 ---")
            logging.info(f"    > 原定 Steps: {NUM_STEPS_TO_PROFILE} (基于文件总帧数)")
            logging.info(f"    > 限制 Steps: {args.max_steps}")
            NUM_STEPS_TO_PROFILE = args.max_steps
        elif (is_camera or is_stream):
            logging.info(f"--- [!] 已应用 --max-steps 限制 (覆盖默认值) ---")
            logging.info(f"    > 限制 Steps: {args.max_steps}")
            NUM_STEPS_TO_PROFILE = args.max_steps
        else:
             logging.info(f"--max-steps ({args.max_steps}) 小于或等于文件总 steps ({NUM_STEPS_TO_PROFILE})，将按文件总数运行。")

    # 设备设置与 CUDA 回退逻辑
    use_cuda = INFER_ON_GPU and torch.cuda.is_available()
    inference_device = torch.device("cuda:0" if use_cuda else "cpu")
    prep_device = torch.device("cuda:0" if PREP_ON_GPU and torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available() and (PREP_ON_GPU or INFER_ON_GPU):
        logging.warning("未检测到 CUDA! 强制所有操作在 CPU 上运行。")
        PREP_ON_GPU = False
        INFER_ON_GPU = False
        use_cuda = False
        inference_device = torch.device("cpu")
        prep_device = torch.device("cpu")


    if args.split_point and use_cuda and MODE == 'inference':
        model_input_device = cpu_device
        logging.info(f"启用跨层分割。模型输入 (Part 1) 将在: {model_input_device}")
    else:
        # 否则，模型输入在主推理设备 (GPU 或 CPU)
        model_input_device = inference_device
        if args.split_point:
            logging.warning("跨层分割仅在 use_cuda=True 和 mode=inference 时激活。")

    # --- 3. 打印配置 ---
    logging.info(f"配置 (模型: {MODEL_NAME})")
    # ... (打印不变)
    print(f"   > 推理/训练将在: {inference_device} (Part2)")
    if args.split_point and model_input_device == cpu_device:
         print(f"   > 跨层分割: 启用 (Part1: {model_input_device})")
    print(f"   > 运行模式: {MODE}")
    print(f"   > 批次大小 (Batch Size): {BATCH_SIZE}")
    print(f"   > 目标帧数: {TOTAL_FRAMES_TO_PROFILE}")
    print(f"   > 计算的 Steps: {NUM_STEPS_TO_PROFILE}")
    print(f"   > 预处理将在: {prep_device}")
    print(f"   > 预处理级别: {args.prep_level}")
    print(f"   > 推理/训练将在: {inference_device}")
    if MODE == 'inference':
        print(f"   > torch.compile() 已: {'启用' if args.enable_compile and use_cuda else '禁用'}")
    else:
        print(f"   > 学习率: {args.learning_rate}, Epochs: {args.retrain_epochs}")
    print(f"   > 使用 AMP: {'是' if USE_AMP and use_cuda else '否'}")
    print(f"------------")

    # --- 4. 加载模型、流水线和数据 ---
    model = load_model(
        MODEL_NAME, 
        device=inference_device, # 主设备 (GPU)
        split_point=args.split_point,
        cpu_device=cpu_device
    )
    
    optimizer: Optional[torch.optim.Optimizer] = None
    criterion: Optional[nn.Module] = None
    num_classes = 2  # 占位符

    if MODE == 'inference':
        if use_cuda:
            cudnn.benchmark = True
        input_shape = (BATCH_SIZE, 3, 224, 224)
        
        # --- 修改：dummy_input 的设备 ---
        # dummy_input 应该在 *模型输入* 设备上
        dummy_input = torch.randn(*input_shape, device=model_input_device) 
        
        model = setup_model_for_inference(
            model, use_cuda, dummy_input, 
            enable_compile=args.enable_compile,
            warmup_iters=args.warmup_iters # [*] 修改：传递参数
        )
    elif MODE == 'retrain':
        # (此时 args.split_point 必为 None, 已在 config.py 中处理)
        model, optimizer, criterion = setup_retrain_components(
            model, MODEL_NAME, inference_device, num_classes, args.learning_rate
        )

    cpu_pipeline, gpu_prep_ops, _normalize_op, _to_tensor_op = \
        build_preprocessing_pipelines(args.prep_level)
        
    # --- 5. CUDA 与指标工具 ---
    use_gpu_ops = use_cuda or PREP_ON_GPU
    stream = Stream() if use_gpu_ops else None
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and use_gpu_ops)
    
    # [新] 初始化指标收集器
    NVTX_PREFIX = f"M_{MODEL_NAME}_P_{utils.PID}"
    collector = utils.MetricsCollector(MODEL_NAME, utils.PID, output_dir=".")
    
    # --- 6. 主循环 ---
    logging.info(f"开始主循环 (Mode: {MODE})... 总共 {NUM_STEPS_TO_PROFILE} 个批次。")
    start_time = time.time()
    
    running_loss = 0.0
    total_steps_processed_all_epochs = 0
    num_epochs = args.retrain_epochs if MODE == 'retrain' else 1

    for epoch in range(num_epochs):
        
        if MODE == 'retrain':
            logging.info(f"--- (Epoch {epoch+1}/{num_epochs}) ---")
            
        # 重置视频捕获 (如果是文件)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) if not is_camera and not is_stream else None
        step = 0
        
        while step < NUM_STEPS_TO_PROFILE:
            nvtx.range_push(f"{NVTX_PREFIX}_step_batch_{step}")
            
            # [新] 初始化指标字典并开始计时
            step_metrics = {"step": step}
            step_start_time = time.time()

            # 6a. 读取
            nvtx.range_push(f"{NVTX_PREFIX}_A_Load_Decode_Batch (CPU)")
            t_read_start = time.time()
            frame_buffer, label_buffer = read_batch_from_video(
                cap, BATCH_SIZE, MODE, num_classes, is_camera, is_stream # [修改] 传递 is_stream
            )
            step_metrics["read_latency_ms"] = (time.time() - t_read_start) * 1000
            nvtx.range_pop()
            
            if not frame_buffer:
                logging.info("视频源结束，退出循环。")
                nvtx.range_pop() # 弹出 step_batch
                break

            # 6b. 预处理
            t_prep_start = time.time()
            # --- 关键修改：目标设备是 model_input_device ---
            inputs_on_device, labels_on_device = preprocess_batch(
                frame_buffer, label_buffer, _to_tensor_op, _normalize_op,
                cpu_pipeline, gpu_prep_ops, PREP_ON_GPU,
                prep_device, 
                model_input_device, # <-- 传递模型输入设备
                stream, NVTX_PREFIX
            )
            step_metrics["prep_latency_ms"] = (time.time() - t_prep_start) * 1000
            
            # 6c. 执行 (推理或训练)
            t_exec_start = time.time()
            lloss_item = execute_step(
                MODE, model, inputs_on_device, labels_on_device,
                USE_AMP and use_gpu_ops, optimizer, criterion, scaler, NVTX_PREFIX
            )
            step_metrics["exec_latency_ms"] = (time.time() - t_exec_start) * 1000
            step_metrics["loss"] = loss_item

            # [新] 记录总时间和指标
            step_metrics["total_step_latency_ms"] = (time.time() - step_start_time) * 1000
            collector.add_step_record(step_metrics)

            # 6d. 记录与同步
            if MODE == 'retrain':
                running_loss += loss_item
                if (step + 1) % 10 == 0:
                    avg_loss = running_loss / 10
                    logging.info(f"[Epoch {epoch+1}, Step {step+1}/{NUM_STEPS_TO_PROFILE}] 平均损失: {avg_loss:.4f}")
                    running_loss = 0.0

            if (step + 1) % 10 == 0:
                if use_gpu_ops:
                    torch.cuda.synchronize()
                if MODE == 'inference':
                    logging.info(f"Analyzed batch {step+1}/{NUM_STEPS_TO_PROFILE}")
            
            nvtx.range_pop() # 弹出 step_batch
            step += 1
            
        # --- Epoch 结束 ---
        total_steps_processed_all_epochs += step
        
        if MODE == 'retrain':
            torch.cuda.synchronize() if use_gpu_ops else None
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_frames = step * BATCH_SIZE
            epoch_fps = epoch_frames / epoch_duration if epoch_duration > 0 else 0
            
            print("="*50)
            logging.info(f"(Epoch {epoch+1}) 完成。")
            print(f"   > 耗时: {epoch_duration:.2f} 秒")
            print(f"   > 处理帧: {epoch_frames} 帧")
            print(f"   > 平均吞吐量: {epoch_fps:.2f} 帧/秒 (FPS)")
            print("="*50)

    # --- 7. 最终结果 ---
    if use_gpu_ops:
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_frames_processed = total_steps_processed_all_epochs * BATCH_SIZE
    
    # [新] 导出 CSV 指标
    collector.export_to_csv()

    utils.print_final_results(
        start_time, end_time, total_frames_processed, 
        num_epochs, MODEL_NAME, MODE
    )

    cap.release()
    logging.info("视频捕获已释放，程序即将退出。")


if __name__ == "__main__":
    main()
