#!/usr/bin/env python3

"""
一个用于分析和（可选）微调 DNN 模型性能的 Python 脚本。

该脚本被设计为从视频文件或摄像头流中读取数据，执行预处理（CPU或GPU），
然后执行推理或训练步骤（CPU或GPU）。

它包括以下高级功能：
- 命令行参数解析 (argparse)
- 通过 NVTX 标记进行深度性能分析
- CPU 核心亲和性设置 (os.sched_setaffinity)
- 可选的 torch.compile() (JIT 编译)
- 可选的自动混合精度 (AMP)
- 可配置的预处理复杂度
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as T
from torchvision import models
import torch.cuda.nvtx as nvtx
import time
import os
import argparse
import math
import sys
import cv2  # 用于视频捕获
import numpy as np
from torch.cuda import Stream  # 用于异步操作
from typing import List, Set, Dict, Tuple, Optional, Any

# --- [全局 Worker PID 标记] ---
# 在脚本开始时获取，用于日志记录和绑核
PID = os.getpid()


# ==================================================================
# 1. 辅助与设置函数
# ==================================================================

def parse_core_list(core_string: str) -> Set[int]:
    """
    解析一个 CPU 核心字符串 (例如 '0-3_5_7-8') 为一个整数集合。

    Args:
        core_string: 格式如 '1-3' 或 '1_3_5' 或 '0-3_7-10' 的字符串。

    Returns:
        一个包含所有指定 CPU 核心 ID 的集合。
    """
    core_set = set()
    parts = core_string.split('_')
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                core_set.update(range(start, end + 1))
            except Exception as e:
                print(f"!!! [PID {PID}] 无法解析核心范围 '{part}': {e}")
        else:
            try:
                core_set.add(int(part))
            except Exception as e:
                print(f"!!! [PID {PID}] 无法解析核心 ID '{part}': {e}")
    return core_set


def parse_arguments() -> argparse.Namespace:
    """
    配置并解析脚本的命令行参数。

    Returns:
        一个包含所有已解析参数的 argparse.Namespace 对象。
    """
    parser = argparse.ArgumentParser(description="运行 DNN 分析 Worker。")
    
    # --- 模型与执行参数 ---
    parser.add_argument('--model-name', type=str, default='vit_b_16',
                        help="要加载的模型 (例如 'vit_b_16', 'resnet50')")
    parser.add_argument('--mode', type=str, default='inference',
                        choices=['inference', 'retrain'],
                        help="运行模式: 'inference' (分析) 或 'retrain' (微调)")
    
    # --- 硬件与性能参数 ---
    parser.add_argument('--prep-on-gpu', action='store_true',
                        help="在 GPU 上运行预处理。")
    parser.add_argument('--infer-on-gpu', action='store_true',
                        help="在 GPU 上运行推理/训练。")
    parser.add_argument('--core-bind', type=str, default="all",
                        help="要绑定的核心 (例如 '16', '0-3', 'all')。")
    parser.add_argument('--batch-size', type=int, default=8,
                        help="用于分析的批次大小。")
    parser.add_argument('--enable-compile', action='store_true',
                        help="为推理启用 torch.compile()。")
    parser.add_argument('--use-amp', action='store_true',
                        help="使用自动混合精度 (AMP) 加速 GPU 操作。")

    # --- 数据与预处理参数 ---
    parser.add_argument('--input_path', type=str, default='video.mp4',
                        help="视频文件路径或摄像头索引 (e.g., 0 for default camera)")
    parser.add_argument('--prep-level', type=str, default='complex',
                        choices=['simple', 'medium', 'complex'],
                        help="预处理的复杂度级别。")

    # --- 训练特定参数 ---
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help="用于重训练的学习率")
    parser.add_argument('--retrain-epochs', type=int, default=3,
                        help="重训练的 Epoch 数量")

    args = parser.parse_args()

    # --- 参数验证 ---
    if args.mode == 'retrain' and args.enable_compile:
        print(f"--- [PID {PID}] 警告: 训练模式下 torch.compile() 已被强制禁用。 ---")
        args.enable_compile = False

    return args


def setup_cpu_affinity(core_bind_string: str) -> Set[int]:
    """
    根据配置字符串设置当前进程的 CPU 亲和性，并设置 Torch 线程数。

    Args:
        core_bind_string: 来自 argparse 的核心绑定字符串。

    Returns:
        绑定的核心集合 (如果 'all' 则为空集)。
    """
    core_set = set()
    if core_bind_string.lower() == "all":
        print(f"--- [PID {PID}] 绑核已跳过 (配置为 'all') ---")
        print(f"--- [PID {PID}] 允许 Torch 使用默认线程数 ---")
        return core_set

    try:
        core_set = parse_core_list(core_bind_string)
        if not core_set:
            raise ValueError("解析后的核心 set 为空。")

        os.sched_setaffinity(PID, core_set)
        num_threads_to_use = len(core_set)
        
        print(f"--- 绑核成功 ---")
        print(f"进程 {PID} 已被绑定到 CPU 核心: {core_set}")
        print(f"--- [PID {PID}] 正在设置 Torch 线程数 = {num_threads_to_use} ---")
        torch.set_num_threads(num_threads_to_use)

    except Exception as e:
        print(f"!!! [PID {PID}] 绑核失败 (Cores: {core_bind_string}) !!!")
        print(f"错误: {e}")
        print(f"--- [PID {PID}] 允许 Torch 使用默认线程数 ---")
        return set()
    
    return core_set


def setup_model_for_inference(model: nn.Module, 
                              use_cuda_infer: bool, 
                              dummy_input_for_warmup: Optional[torch.Tensor], 
                              enable_compile: bool = False, 
                              warmup_iters: int = 3) -> nn.Module:
    """
    为推理准备模型：可选编译 (torch.compile) 和预热 (warm-up)。

    Args:
        model: 要配置的 PyTorch 模型。
        use_cuda_infer: 推理是否在 CUDA 上运行。
        dummy_input_for_warmup: 用于预热的示例输入张量。
        enable_compile: 是否启用 torch.compile()。
        warmup_iters: 预热迭代次数。

    Returns:
        配置完成的 (可能已编译的) 模型。
    """
    if not use_cuda_infer:
        print(f"--- [PID {PID}] 推理在 CPU 上: 跳过 torch.compile() 和 warm-up ---")
        return model

    # 1. 可选的 torch.compile()
    if enable_compile:
        print(f"--- [PID {PID}] CUDA 推理: 正在启用 torch.compile() ---")
        try:
            model = torch.compile(model)
            print(f"--- [PID {PID}] torch.compile() 成功 ---")
        except Exception as e:
            print(f"!!! [PID {PID}] torch.compile() 失败: {e}")
            print("--- 将回退到 Eager 模式运行 ---")
    else:
        print(f"--- [PID {PID}] torch.compile() 已禁用, 在 Eager 模式下运行 ---")

    # 2. 预热 (Warm-up)
    warmup_type = "JIT 编译/缓存" if enable_compile else "Eager 模式缓存"
    print(f"--- [PID {PID}] 正在执行 {warmup_iters} 次 WARM-UP 运行 ({warmup_type})... ---")

    if dummy_input_for_warmup is None:
        print(f"!!! [PID {PID}] 警告: 缺少 WARM-UP 用的 dummy_input. 跳过 warm-up. !!!")
        return model

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input_for_warmup)
        # 确保所有预热操作完成
        torch.cuda.synchronize()

    print(f"--- [PID {PID}] WARM-UP 完成 ---")
    return model


def load_model(model_name: str, device: torch.device) -> nn.Module:
    """
    根据名称加载预训练模型并将其移动到指定设备。

    Args:
        model_name: 模型的名称 (例如 'vit_b_16')。
        device: 目标设备 (CPU 或 CUDA)。

    Returns:
        加载的 PyTorch 模型。
    """
    print(f"--- [PID {PID}] 正在加载模型: {model_name} ... ---")
    torch.hub.set_dir('models')
    
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    else:
        print(f"!!! [PID {PID}] 错误: 未知的模型名称 '{model_name}'")
        sys.exit(1)
        
    model.to(device)
    print(f"--- [PID {PID}] 模型已加载到 {device} ---")
    return model


def setup_retrain_components(model: nn.Module, 
                             model_name: str, 
                             device: torch.device, 
                             num_classes: int, 
                             learning_rate: float) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """
    为 'retrain' 模式配置模型：冻结层、替换分类头、设置优化器和损失函数。

    Args:
        model: 要修改的模型。
        model_name: 模型名称 (用于确定如何替换头)。
        device: 目标设备。
        num_classes: 新分类头的输出类别数。
        learning_rate: 优化器的学习率。

    Returns:
        一个元组 (model, optimizer, criterion)。
    """
    print(f"--- [PID {PID}] 模式: 'retrain'. 正在设置 model.train() 和解冻最后几层... ---")
    model.train()

    # 1. 冻结所有现有参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 替换分类头
    if model_name == 'vit_b_16':
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
        # 仅解冻新头的参数
        for param in model.heads.head.parameters():
            param.requires_grad = True
            
    elif model_name == 'resnet50' or model_name == 'resnet101':
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == 'efficientnet_b0':
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        for param in model.classifier[1].parameters():
            param.requires_grad = True
    
    # 3. 将新头移动到设备
    model.to(device)
    
    # 4. 设置优化器和损失函数
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss().to(device)
    
    print(f"--- [PID {PID}] 微调设置完成。优化器将更新以下参数: ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   > {name}")
    print(f"------------------")
    
    return model, optimizer, criterion


def build_preprocessing_pipelines(prep_level: str) -> Tuple[T.Compose, Dict[str, nn.Module], T.Normalize, T.ToTensor]:
    """
    根据复杂度级别构建 CPU 和 GPU 预处理操作。

    Args:
        prep_level: 'simple', 'medium', 或 'complex'。

    Returns:
        - cpu_pipeline: 用于 CPU 预处理的 T.Compose 对象。
        - gpu_prep_ops: 一个包含 GPU 预处理操作 (nn.Module) 的字典。
        - normalize_op: 归一化操作 (在 CPU 和 GPU 上都需要)。
        - to_tensor_op: 转换为张量的操作 (总是在 CPU 上首先执行)。
    """
    # 定义基础操作
    _resize_256_op = T.Resize(256)
    _resize_224_op = T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST)
    _crop_224_op = T.CenterCrop(224)
    _normalize_op = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    _random_rotation_op = T.RandomRotation(degrees=15)
    _gaussian_blur_op = T.GaussianBlur(kernel_size=3)
    _to_tensor_op = T.ToTensor()

    print(f"--- [PID {PID}] 正在构建 '{prep_level}' 预处理流水线... ---")

    cpu_prep_ops_list = []
    # GPU 操作字典，默认为无操作
    gpu_prep_ops = {
        "resize": nn.Identity(),
        "crop": nn.Identity(),
        "rotation": nn.Identity(),
        "blur": nn.Identity(),
    }

    if prep_level == 'simple':
        cpu_prep_ops_list = [_resize_224_op]
        gpu_prep_ops["resize"] = _resize_224_op

    elif prep_level == 'medium':
        cpu_prep_ops_list = [_resize_256_op, _crop_224_op]
        gpu_prep_ops["resize"] = _resize_256_op
        gpu_prep_ops["crop"] = _crop_224_op

    elif prep_level == 'complex':
        cpu_prep_ops_list = [_resize_256_op, _crop_224_op, _random_rotation_op, _gaussian_blur_op]
        gpu_prep_ops["resize"] = _resize_256_op
        gpu_prep_ops["crop"] = _crop_224_op
        gpu_prep_ops["rotation"] = _random_rotation_op
        gpu_prep_ops["blur"] = _gaussian_blur_op

    # CPU 流水线包含所有操作 + 归一化
    cpu_pipeline = T.Compose(cpu_prep_ops_list + [_normalize_op])
    
    # GPU 流水线不包含归一化 (单独应用)
    return cpu_pipeline, gpu_prep_ops, _normalize_op, _to_tensor_op


def initialize_video_capture(input_path: str) -> Tuple[cv2.VideoCapture, float, float]:
    """
    初始化视频捕获 (来自文件或摄像头)。

    Args:
        input_path: 视频文件路径或摄像头索引。

    Returns:
        一个元组 (cap, fps, total_frames)。
        如果
        是摄像头，total_frames 将是 float('inf')。
    """
    try:
        if input_path.isdigit():
            cap = cv2.VideoCapture(int(input_path))  # 摄像头索引
            is_camera = True
        else:
            cap = cv2.VideoCapture(input_path)  # 视频文件
            is_camera = False
            
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频源: {input_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = float('inf') if is_camera else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"从 {input_path} 加载视频, FPS: {fps}, 总帧数: {total_frames}")
        return cap, fps, total_frames
        
    except Exception as e:
        print(f"加载视频路径时发生错误: {e}")
        sys.exit(1)


def get_dummy_label(num_classes: int) -> torch.Tensor:
    """为训练模式生成一个虚拟标签。"""
    return torch.randint(0, num_classes, (1,))


def print_final_results(start_time: float, 
                        end_time: float, 
                        total_frames_processed: int, 
                        num_epochs: int, 
                        model_name: str, 
                        mode: str, 
                        pid: int):
    """
    计算并打印最终的性能总结。
    """
    total_time_taken = end_time - start_time
    avg_fps = total_frames_processed / total_time_taken if total_time_taken > 0 else 0

    print("="*50)
    print(f"--- [PID {pid}] (模型: {model_name}) [Mode: {mode}] 任务完成。 ---")
    print(f"总耗时 ({num_epochs} Epochs): {total_time_taken:.4f} 秒")
    print(f"总处理帧 (所有 Epochs): {total_frames_processed} 帧")
    print(f"平均吞吐量 (所有 Epochs): {avg_fps:.2f} 帧/秒 (FPS)")
    print("="*50)


# ==================================================================
# 2. 主循环逻辑函数
# ==================================================================

def read_batch_from_video(cap: cv2.VideoCapture, 
                          batch_size: int, 
                          mode: str, 
                          num_classes: int, 
                          is_camera: bool) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
    """
    从视频源 (cap) 读取一个批次的帧。

    Args:
        cap: cv2.VideoCapture 对象。
        batch_size: 需要读取的帧数。
        mode: 'retrain' (需要标签) 或 'inference'。
        num_classes: 标签类别数 (如果 mode=='retrain')。
        is_camera: 是否为摄像头 (影响 EOF 处理)。

    Returns:
        一个元组 (frame_buffer, label_buffer)。
    """
    frame_buffer = []
    label_buffer = []

    while len(frame_buffer) < batch_size and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_camera:
                time.sleep(0.001)  # 避免摄像头忙等待
                continue
            else:
                break  # 视频文件结束
                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        frame_buffer.append(frame)
        if mode == 'retrain':
            label_buffer.append(get_dummy_label(num_classes))  # 替换为真实标签源

    if len(frame_buffer)<batch_size: frame_buffer=[]
    return frame_buffer, label_buffer


def preprocess_batch(frame_buffer: List[np.ndarray],
                     label_buffer: List[torch.Tensor],
                     to_tensor_op: T.ToTensor,
                     normalize_op: T.Normalize,
                     cpu_pipeline: T.Compose,
                     gpu_prep_ops: Dict[str, nn.Module],
                     prep_on_gpu: bool,
                     prep_device: torch.device,
                     inference_device: torch.device,
                     stream: Optional[Stream],
                     nvtx_prefix: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    对一个批次的帧执行预处理和 HtoD 传输。

    Args:
        frame_buffer: 来自 read_batch_from_video 的原始帧列表。
        label_buffer: 标签列表。
        to_tensor_op: T.ToTensor() 操作。
        normalize_op: T.Normalize() 操作。
        cpu_pipeline: CPU 转换 T.Compose 对象。
        gpu_prep_ops: GPU 转换操作字典。
        prep_on_gpu: 是否在 GPU 上预处理。
        prep_device: 预处理设备。
        inference_device: 推理设备。
        stream: CUDA 流。
        nvtx_prefix: NVTX 标记前缀。

    Returns:
        一个元组 (inputs_on_device, labels_on_device)。
    """
    
    # 步骤 1: 始终在 CPU 上将 Numpy 数组转换为张量
    # [B, H, W, C] (Numpy) -> List[C, H, W] (Tensor) -> [B, C, H, W] (Tensor)
    nvtx.range_push(f"{nvtx_prefix}_A_Pre_ToTensor_Stack (CPU)")
    inputs = [ to_tensor_op(img) for img in frame_buffer ]
    inputs = torch.stack(inputs)
    nvtx.range_pop()

    # 步骤 2: 处理标签 (如果存在)
    labels_on_device = None
    if label_buffer:
        nvtx.range_push(f"{nvtx_prefix}_A_Pre_Label_Processing")
        labels = torch.cat(label_buffer)
        # 标签总是直接发送到推理设备
        labels_on_device = labels.to(inference_device, non_blocking=True) if labels is not None else None
        nvtx.range_pop()

    # 步骤 3: 执行预处理
    if prep_on_gpu:
        # 异步 HtoD -> GPU 预处理
        with torch.cuda.stream(stream):
            nvtx.range_push(f"{nvtx_prefix}_B_data_copy_HtoD (RAW Frame Batch)")
            inputs_gpu_float = inputs.to(prep_device, non_blocking=True)
            nvtx.range_pop()
            
            nvtx.range_push(f"{nvtx_prefix}_C_Preprocessing_GPU (Transforms)")
            # 应用字典中的 GPU 操作
            inputs_gpu_float = gpu_prep_ops["resize"](inputs_gpu_float)
            inputs_gpu_float = gpu_prep_ops["crop"](inputs_gpu_float)
            inputs_gpu_float = gpu_prep_ops["rotation"](inputs_gpu_float)
            inputs_gpu_float = gpu_prep_ops["blur"](inputs_gpu_float)
            # 归一化是最后一步
            inputs_gpu_final = normalize_op(inputs_gpu_float)
            
            # 如果 prep_device 和 inference_device 不同，可能需要 DtoD 复制
            inputs_on_device = inputs_gpu_final.to(inference_device, non_blocking=True)
            nvtx.range_pop()
    
    else:
        # CPU 预处理 -> 异步 HtoD
        nvtx.range_push(f"{nvtx_prefix}_A_Preprocessing_Batch (CPU)")
        # 应用组合的 CPU 流水线
        inputs = cpu_pipeline(inputs)
        nvtx.range_pop()

        nvtx.range_push(f"{nvtx_prefix}_B_data_copy_HtoD (Processed Batch)")
        inputs_on_device = inputs.to(inference_device, non_blocking=True)
        nvtx.range_pop()

    return inputs_on_device, labels_on_device


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

    Args:
        mode: 'inference' 或 'retrain'。
        (其他): ...

    Returns:
        该步骤的损失值 (如果 'inference' 则为 0.0)。
    """
    loss_item = 0.0

    if mode == 'inference':
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            nvtx.range_push(f"{nvtx_prefix}_D_inference_Batch")
            outputs = model(inputs_on_device)
            nvtx.range_pop()
    
    elif mode == 'retrain':
        # 确保优化器和损失函数已提供
        if optimizer is None or criterion is None or labels_on_device is None:
            print(f"!!! [PID {PID}] 错误: 训练模式缺少 optimizer, criterion, 或 labels。")
            return 0.0
            
        with torch.cuda.amp.autocast(enabled=use_amp):
            nvtx.range_push(f"{nvtx_prefix}_D_retrain_Batch")
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs_on_device)
            loss = criterion(outputs, labels_on_device)
            
            # 反向传播 (使用 AMP scaler)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            nvtx.range_pop()
        
        loss_item = loss.item()
    
    return loss_item


# ==================================================================
# 3. 主函数 (Main)
# ==================================================================

def main():
    """
    脚本的主执行函数。
    """
    global PID
    
    # --- 1. 设置 ---
    args = parse_arguments()
    setup_cpu_affinity(args.core_bind)

    # --- 2. 配置与设备 ---
    MODE = args.mode
    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size
    PREP_ON_GPU = args.prep_on_gpu
    INFER_ON_GPU = args.infer_on_gpu
    USE_AMP = args.use_amp

    # 初始化视频源
    cap, fps, total_frames = initialize_video_capture(args.input_path)
    is_camera = args.input_path.isdigit()
    
    # 目标帧数 (用于基准测试)
    DEFAULT_FRAMES_TO_PROFILE = 900
    if is_camera:
        print(f"--- [PID {PID}] 检测到摄像头输入。将使用默认帧数 {DEFAULT_FRAMES_TO_PROFILE} 进行分析。")
        TOTAL_FRAMES_TO_PROFILE = DEFAULT_FRAMES_TO_PROFILE
    else:
        # 如果是视频文件，则使用其实际帧数
        print(f"--- [PID {PID}] 检测到视频文件。将处理所有 {int(total_frames)} 帧。")
        TOTAL_FRAMES_TO_PROFILE = total_frames
    NUM_STEPS_TO_PROFILE = math.ceil(TOTAL_FRAMES_TO_PROFILE / BATCH_SIZE)

    # 设备设置与 CUDA 回退逻辑
    use_cuda = INFER_ON_GPU and torch.cuda.is_available()
    inference_device = torch.device("cuda:0" if use_cuda else "cpu")
    prep_device = torch.device("cuda:0" if PREP_ON_GPU and torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available() and (PREP_ON_GPU or INFER_ON_GPU):
        print(f"--- [PID {PID}] 警告: 未检测到 CUDA! 强制所有操作在 CPU 上运行。")
        PREP_ON_GPU = False
        INFER_ON_GPU = False
        use_cuda = False
        inference_device = torch.device("cpu")
        prep_device = torch.device("cpu")

    # --- 3. 打印配置 ---
    print(f"--- [PID {PID}] 配置 (模型: {MODEL_NAME}) ---")
    print(f"运行模式: {MODE}")
    print(f"批次大小 (Batch Size): {BATCH_SIZE}")
    print(f"目标帧数: {TOTAL_FRAMES_TO_PROFILE}")
    print(f"计算的 Steps: {NUM_STEPS_TO_PROFILE}")
    print(f"预处理将在: {prep_device}")
    print(f"预处理级别: {args.prep_level}")
    print(f"推理/训练将在: {inference_device}")
    if MODE == 'inference':
        print(f"torch.compile() 已: {'启用' if args.enable_compile and use_cuda else '禁用'}")
    else:
        print(f"学习率: {args.learning_rate}, Epochs: {args.retrain_epochs}")
    print(f"使用 AMP: {'是' if USE_AMP and use_cuda else '否'}")
    print(f"------------")

    # --- 4. 加载模型、流水线和数据 ---
    model = load_model(MODEL_NAME, inference_device)
    
    optimizer: Optional[torch.optim.Optimizer] = None
    criterion: Optional[nn.Module] = None
    num_classes = 2  # 占位符，根据您的任务调整

    if MODE == 'inference':
        if use_cuda:
            cudnn.benchmark = True
        input_shape = (BATCH_SIZE, 3, 224, 224)
        dummy_input = torch.randn(*input_shape, device=inference_device)
        model = setup_model_for_inference(
            model, use_cuda, dummy_input, enable_compile=args.enable_compile
        )
    elif MODE == 'retrain':
        model, optimizer, criterion = setup_retrain_components(
            model, MODEL_NAME, inference_device, num_classes, args.learning_rate
        )

    # 获取预处理操作
    cpu_pipeline, gpu_prep_ops, _normalize_op, _to_tensor_op = \
        build_preprocessing_pipelines(args.prep_level)
        
    

    # --- 5. CUDA 工具 ---
    use_gpu_ops = use_cuda or PREP_ON_GPU
    stream = Stream() if use_gpu_ops else None
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and use_gpu_ops)
    
    # --- 6. 主循环 ---
    print(f"--- [PID {PID}] 开始主循环 (Mode: {MODE})... 总共 {NUM_STEPS_TO_PROFILE} 个批次。")
    start_time = time.time()
    
    running_loss = 0.0
    total_steps_processed_all_epochs = 0
    NVTX_PREFIX = f"M_{MODEL_NAME}_P_{PID}"

    num_epochs = args.retrain_epochs if MODE == 'retrain' else 1

    for epoch in range(num_epochs):
        
        if MODE == 'retrain':
            print(f"\n--- [PID {PID}] (Epoch {epoch+1}/{num_epochs}) ---")
            epoch_start_time = time.time()
            
        # 重置视频捕获 (如果是文件)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) if not is_camera else None

        step = 0
        
        # 主循环，直到达到目标步数
        while step < NUM_STEPS_TO_PROFILE:
            
            nvtx.range_push(f"{NVTX_PREFIX}_step_batch_{step}")

            nvtx.range_push(f"{NVTX_PREFIX}_A_Load_Decode_Batch (CPU)")
            # 6a. 从视频源读取批次
            frame_buffer, label_buffer = read_batch_from_video(
                cap, BATCH_SIZE, MODE, num_classes, is_camera
            )
            nvtx.range_pop() # 结束 A_Load_Decode
            
            # 如果是视频文件末尾且缓冲区为空，则退出
            if not frame_buffer:
                print(f"--- [PID {PID}] 视频源结束，退出循环。")
                nvtx.range_pop()
                break

            # 6b. 预处理
            inputs_on_device, labels_on_device = preprocess_batch(
                frame_buffer, label_buffer, _to_tensor_op, _normalize_op,
                cpu_pipeline, gpu_prep_ops, PREP_ON_GPU,
                prep_device, inference_device, stream, NVTX_PREFIX
            )

            
            # 6c. 执行 (推理或训练)
            loss_item = execute_step(
                MODE, model, inputs_on_device, labels_on_device,
                USE_AMP and use_gpu_ops, optimizer, criterion, scaler, NVTX_PREFIX
            )

            # 6d. 记录与同步
            nvtx.range_push(f"{NVTX_PREFIX}_E_Sync_and_Log")
            if MODE == 'retrain':
                running_loss += loss_item
                if (step + 1) % 10 == 0:
                    avg_loss = running_loss / 10
                    print(f"--- [PID {PID}] [Epoch {epoch+1}, Step {step+1}/{NUM_STEPS_TO_PROFILE}] 平均损失: {avg_loss:.4f}")
                    running_loss = 0.0

            # 定期同步以防止 CUDA 队列溢出并获取更准确的日志
            if (step + 1) % 10 == 0:
                if use_gpu_ops:
                    torch.cuda.synchronize()
                
                if MODE == 'inference':
                    print(f"--- [PID {PID}] Analyzed batch {step+1}/{NUM_STEPS_TO_PROFILE}")
            nvtx.range_pop()# 结束 E_Sync_and_Log
            
            nvtx.range_pop()

            step += 1
            
        # --- Epoch 结束 ---
        total_steps_processed_all_epochs += step
        
        if MODE == 'retrain':
            torch.cuda.synchronize() if use_gpu_ops else None
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            # epoch_frames = min(step * BATCH_SIZE, total_frames if not is_camera else float('inf'))
            epoch_frames = step * BATCH_SIZE
            epoch_fps = epoch_frames / epoch_duration if epoch_duration > 0 else 0
            
            print("="*50)
            print(f"--- [PID {PID}] (Epoch {epoch+1}) 完成。 ---")
            print(f"耗时: {epoch_duration:.2f} 秒")
            print(f"处理帧: {epoch_frames} 帧")
            print(f"平均吞吐量: {epoch_fps:.2f} 帧/秒 (FPS)")
            print("="*50)

    # --- 7. 最终结果 ---
    if use_gpu_ops:
        torch.cuda.synchronize()
    end_time = time.time()
    
    # 总帧数是所有 epoch 中实际处理的步数 * BATCH_SIZE
    total_frames_processed = total_steps_processed_all_epochs * BATCH_SIZE

    print_final_results(
        start_time, end_time, total_frames_processed, 
        num_epochs, MODEL_NAME, MODE, PID
    )

    cap.release()


if __name__ == "__main__":
    main()