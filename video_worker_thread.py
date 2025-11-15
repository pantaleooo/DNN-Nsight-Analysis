#!/usr/bin/env python3

"""
一个用于分析和（可选）微调 DNN 模型性能的 Python 脚本。

该脚本被设计为 *单进程* worker, 根据 '--mode' 参数执行以下任一操作：
1. 'inference': 从视频/摄像头读取, 执行实时推理分析。
2. 'retrain': 从本地磁盘 (ImageFolder) 读取, 执行重训练分析。 (已按要求修改)

外部的 shell 脚本将负责启动此文件的多个实例, 并通过
命令行参数 (如 '--core-bind') 来进行资源分配和调度。
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder # <--- [新] 添加 ImageFolder
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
PID = os.getpid()


# ==================================================================
# 1. 辅助与设置函数 (大部分与原版相同)
# ==================================================================

def parse_core_list(core_string: str) -> Set[int]:
    """
    解析一个 CPU 核心字符串 (例如 '0-3_5_7-8') 为一个整数集合。
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
    """
    parser = argparse.ArgumentParser(description="运行 DNN 分析 Worker (单进程模式)。")
    
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
                        help="[关键] 此 worker 进程要绑定的核心 (例如 '0-3', '4-7', 'all')。")
    
    parser.add_argument('--batch-size-infer', type=int, default=8,
                        help="用于推理的批次大小。")
    parser.add_argument('--batch-size-retrain', type=int, default=32,
                        help="用于重训练的批次大小。")
                        
    parser.add_argument('--enable-compile', action='store_true',
                        help="为推理启用 torch.compile()。")
    parser.add_argument('--use-amp', action='store_true',
                        help="使用自动混合精度 (AMP) 加速 GPU 操作。")

    # --- 数据与预处理参数 ---
    parser.add_argument('--input-path', type=str, default='video.mp4',
                        help="[仅推理] 视频文件路径或摄像头索引 (e.g., 0 for default camera)")
    parser.add_argument('--prep-level', type=str, default='complex',
                        choices=['simple', 'medium', 'complex'],
                        help="[仅推理] 预处理的复杂度级别。")

    # --- 训练特定参数 ---
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help="[仅训练] 用于重训练的学习率")
    parser.add_argument('--retrain-epochs', type=int, default=3,
                        help="[仅训练] 重训练的 Epoch 数量")
    
    # --- [新] 训练数据加载参数 ---
    parser.add_argument('--retrain-data-path', type=str, default=None,
                        help="[仅训练] 本地训练数据集的路径 (例如 ImageFolder 格式)。")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="[仅训练] DataLoader 使用的 num-workers。")
    # ---

    args = parser.parse_args()

    # --- 参数验证 ---
    if args.mode == 'retrain' and args.enable_compile:
        print(f"--- [PID {PID}] 警告: 训练模式下 torch.compile() 已被强制禁用。 ---")
        args.enable_compile = False
        
    if args.mode == 'retrain' and not args.retrain_data_path:
        print(f"!!! [PID {PID}] 错误: 运行 'retrain' 模式必须提供 '--retrain-data-path' 参数。")
        sys.exit(1)

    return args


def setup_cpu_affinity(core_bind_string: str) -> Set[int]:
    """
    (原始版本) 根据配置字符串设置 *当前进程* 的 CPU 亲和性。
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

        # 绑定当前进程 (PID)
        os.sched_setaffinity(PID, core_set)
        num_threads_to_use = len(core_set)
        
        print(f"--- [PID {PID}] 进程绑核成功 ---")
        print(f"进程 {PID} 已被绑定到 CPU 核心: {core_set}")
        print(f"--- [PID {PID}] 正在设置 Torch 线程数 = {num_threads_to_use} ---")
        torch.set_num_threads(num_threads_to_use)

    except Exception as e:
        print(f"!!! [PID {PID}] 绑核失败 (Cores: {core_bind_string}) !!!")
        print(f"错误: {e}")
        print(f"--- [PID {PID}] 允许 Torch 使用默认线程数 ---")
        return set()
    
    return core_set

# --- (复制您原始的 setup_model_for_inference, load_model, setup_retrain_components) ---
def setup_model_for_inference(model: nn.Module, 
                              use_cuda_infer: bool, 
                              dummy_input_for_warmup: Optional[torch.Tensor], 
                              enable_compile: bool = False, 
                              warmup_iters: int = 3) -> nn.Module:
    """
    为推理准备模型：可选编译 (torch.compile) 和预热 (warm-up)。
    """
    if not use_cuda_infer:
        print(f"--- [PID {PID}] 推理在 CPU 上: 跳过 torch.compile() 和 warm-up ---")
        return model

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

    warmup_type = "JIT 编译/缓存" if enable_compile else "Eager 模式缓存"
    print(f"--- [PID {PID}] 正在执行 {warmup_iters} 次 WARM-UP 运行 ({warmup_type})... ---")

    if dummy_input_for_warmup is None:
        print(f"!!! [PID {PID}] 警告: 缺少 WARM-UP 用的 dummy_input. 跳过 warm-up. !!!")
        return model

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input_for_warmup)
        torch.cuda.synchronize()

    print(f"--- [PID {PID}] WARM-UP 完成 ---")
    return model


def load_model(model_name: str, device: torch.device) -> nn.Module:
    """
    根据名称加载预训练模型并将其移动到指定设备。
    """
    print(f"--- [PID {PID}] 正在加载模型: {model_name} ... ---")
    torch.hub.set_dir('models')
    
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # ... (其他模型)
    else:
        print(f"!!! [PID {PID}] 错误: 未知的模型名称 '{model_name}'")
        sys.exit(1)
        
    model.to(device)
    print(f"--- [PID {PID}] 模型已加载到 {device} ---")
    return model


def setup_retrain_components(model: nn.Module, 
                             model_name: str, 
                             device: torch.device, 
                             num_classes: int,  # <--- 现在这个值是准确的
                             learning_rate: float) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """
    为 'retrain' 模式配置模型：冻结层、替换分类头、设置优化器和损失函数。
    """
    print(f"--- [PID {PID}] 模式: 'retrain'. 正在设置 model.train() 和解冻最后几层... ---")
    print(f"--- [PID {PID}] 目标类别数: {num_classes} ---")
    model.train() # 训练模式

    # 1. 冻结所有现有参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 替换分类头
    if model_name == 'vit_b_16':
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
        for param in model.heads.head.parameters():
            param.requires_grad = True
    elif model_name == 'resnet50' or model_name == 'resnet101':
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
    # ... (其他模型)
    
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

# --- (复制您原始的 build_preprocessing_pipelines, initialize_video_capture, get_dummy_label, print_final_results) ---
def build_preprocessing_pipelines(prep_level: str) -> Tuple[T.Compose, Dict[str, nn.Module], T.Normalize, T.ToTensor]:
    """
    [仅推理] 根据复杂度级别构建 CPU 和 GPU 预处理操作。
    """
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

    cpu_pipeline = T.Compose(cpu_prep_ops_list + [_normalize_op])
    
    return cpu_pipeline, gpu_prep_ops, _normalize_op, _to_tensor_op


def initialize_video_capture(input_path: str) -> Tuple[cv2.VideoCapture, float, float]:
    """
    [仅推理] 初始化视频捕获 (来自文件或摄像头)。
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
    """[仅训练] 为训练模式生成一个虚拟标签。"""
    return torch.randint(0, num_classes, (1,))


def print_final_results(start_time: float, 
                        end_time: float, 
                        total_items_processed: int, # 帧数或批次数
                        num_epochs: int, 
                        model_name: str, 
                        mode: str, 
                        pid: int,
                        unit: str = "帧"):
    """
    计算并打印最终的性能总结。
    """
    total_time_taken = end_time - start_time
    avg_throughput = total_items_processed / total_time_taken if total_time_taken > 0 else 0

    print("="*50)
    print(f"--- [PID {pid}] (模型: {model_name}) [Mode: {mode}] 任务完成。 ---")
    print(f"总耗时 ({num_epochs} Epochs): {total_time_taken:.4f} 秒")
    print(f"总处理 {unit}: {total_items_processed} {unit}")
    print(f"平均吞吐量 (所有 Epochs): {avg_throughput:.2f} {unit}/秒")
    print("="*50)

# ==================================================================
# 2. 核心逻辑函数 (与原版相同)
# ==================================================================

def read_batch_from_video(cap: cv2.VideoCapture, 
                          batch_size: int, 
                          is_camera: bool) -> List[np.ndarray]:
    """
    [仅推理] 从视频源 (cap) 读取一个批次的帧。
    (已简化: 移除了 mode 和 label 逻辑)
    """
    frame_buffer = []

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

    if len(frame_buffer) < batch_size: 
        frame_buffer = [] # 丢弃不足批次
        
    return frame_buffer


def preprocess_batch(frame_buffer: List[np.ndarray],
                     label_buffer: List[torch.Tensor], # [仅训练]
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
    [仅推理] 对一个批次的帧执行预处理和 HtoD 传输。
    (注意: 训练模式将跳过此函数, 因为 DataLoader 已处理)
    """
    
    # 步骤 1: 始终在 CPU 上将 Numpy 数组转换为张量
    nvtx.range_push(f"{nvtx_prefix}_A_Pre_ToTensor_Stack (CPU)")
    inputs = [ to_tensor_op(img) for img in frame_buffer ]
    inputs = torch.stack(inputs)
    nvtx.range_pop()

    # 步骤 2: 处理标签 (如果存在) - 对于推理, label_buffer 总是 []
    labels_on_device = None
    if label_buffer:
        nvtx.range_push(f"{nvtx_prefix}_A_Pre_Label_Processing")
        labels = torch.cat(label_buffer)
        labels_on_device = labels.to(inference_device, non_blocking=True)
        nvtx.range_pop()

    # 步骤 3: 执行预处理
    if prep_on_gpu:
        # 异步 HtoD -> GPU 预处理
        with torch.cuda.stream(stream):
            nvtx.range_push(f"{nvtx_prefix}_B_data_copy_HtoD (RAW Frame Batch)")
            inputs_gpu_float = inputs.to(prep_device, non_blocking=True)
            nvtx.range_pop()
            
            nvtx.range_push(f"{nvtx_prefix}_C_Preprocessing_GPU (Transforms)")
            inputs_gpu_float = gpu_prep_ops["resize"](inputs_gpu_float)
            inputs_gpu_float = gpu_prep_ops["crop"](inputs_gpu_float)
            inputs_gpu_float = gpu_prep_ops["rotation"](inputs_gpu_float)
            inputs_gpu_float = gpu_prep_ops["blur"](inputs_gpu_float)
            inputs_gpu_final = normalize_op(inputs_gpu_float)
            
            inputs_on_device = inputs_gpu_final.to(inference_device, non_blocking=True)
            nvtx.range_pop()
    else:
        # CPU 预处理 -> 异步 HtoD
        nvtx.range_push(f"{nvtx_prefix}_A_Preprocessing_Batch (CPU)")
        inputs = cpu_pipeline(inputs)
        nvtx.range_pop()

        nvtx.range_push(f"{nvtx_prefix}_B_data_copy_HtoD (Processed Batch)")
        with torch.cuda.stream(stream):
            inputs_on_device = inputs.to(inference_device, non_blocking=True)
        nvtx.range_pop()

    return inputs_on_device, labels_on_device


def execute_step(step: int,
                 mode: str,
                 model: nn.Module,
                 inputs_on_device: torch.Tensor,
                 labels_on_device: Optional[torch.Tensor],
                 use_amp: bool,
                 optimizer: Optional[torch.optim.Optimizer],
                 criterion: Optional[nn.Module],
                 scaler: torch.cuda.amp.GradScaler,
                 nvtx_prefix: str) -> float:
    """
    (通用) 执行单个推理或训练步骤。
    """
    loss_item = 0.0

    if mode == 'inference':
        model.eval() # 确保是 eval 模式
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            nvtx.range_push(f"{nvtx_prefix}_D_inference_Batch")
            outputs = model(inputs_on_device)
            nvtx.range_pop()


    
    elif mode == 'retrain':
        model.train() # 确保是 train 模式
        if optimizer is None or criterion is None or labels_on_device is None:
            print(f"!!! [PID {PID}] 错误: 训练模式缺少 optimizer, criterion, 或 labels。")
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
# 3. (修改) 特定模式的循环
# ==================================================================

def create_train_loader(args: argparse.Namespace) -> Tuple[DataLoader, int]:
    """
    [仅训练] [已修改] 创建一个从本地磁盘 (ImageFolder) 加载的 DataLoader。
    
    返回:
        DataLoader: 用于训练的 DataLoader。
        int: 数据集中自动检测到的类别数量。
    """
    print(f"--- [PID {PID}] 正在创建本地数据集 (ImageFolder) DataLoader... ---")
    
    if not args.retrain_data_path or not os.path.exists(args.retrain_data_path):
        print(f"!!! [PID {PID}] 错误: 训练数据路径 '{args.retrain_data_path}' 不存在或未提供。")
        sys.exit(1)

    # 定义标准训练预处理
    # (与 'complex' 推理预处理类似, 但通常使用 RandomResizedCrop)
    # (这里我们保持与推理 warm-up 一致, 使用 Resize + CenterCrop)
    normalize_op = T.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    
    train_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize_op
    ])

    try:
        # 使用 ImageFolder 加载数据
        train_dataset = ImageFolder(
            root=args.retrain_data_path,
            transform=train_transform
        )
        
        num_classes = len(train_dataset.classes)
        if num_classes == 0:
            raise ValueError("在路径中未找到任何类别文件夹。")

    except Exception as e:
        print(f"!!! [PID {PID}] 加载 ImageFolder 失败: {e}")
        print("请确保 '--retrain-data-path' 指向的目录结构如下:")
        print("  /path/to/dataset/")
        print("      class_a/")
        print("          img1.jpg")
        print("      class_b/")
        print("          img2.jpg")
        sys.exit(1)
    
    # DataLoader (使用多 worker)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size_retrain, 
        shuffle=True, 
        num_workers=args.num_workers, # <-- 使用参数
        pin_memory=True,
        drop_last=True, # 确保批次大小一致
        persistent_workers=True
    )
    
    print(f"--- [PID {PID}] 本地 DataLoader 已创建 ---")
    print(f"  路径: {args.retrain_data_path}")
    print(f"  类别数: {num_classes}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  批次数: {len(train_loader)} (Batch Size: {args.batch_size_retrain})")
    
    return train_loader, num_classes


def run_inference_loop(args: argparse.Namespace, 
                       model: nn.Module,
                       cap: cv2.VideoCapture,
                       prep_tools: Tuple,
                       device_tools: Tuple):
    """
    [仅推理] 运行视频推理的主循环。
    """
    NVTX_PREFIX = f"M_{args.model_name}_P_{PID}_INFER"
    BATCH_SIZE = args.batch_size_infer
    is_camera = args.input_path.isdigit()
    
    # 解包工具
    cpu_pipeline, gpu_prep_ops, normalize_op, to_tensor_op = prep_tools
    inference_device, prep_device, stream, scaler = device_tools
    
    # 确定总步数
    DEFAULT_FRAMES_TO_PROFILE = 900
    if is_camera:
        TOTAL_FRAMES_TO_PROFILE = DEFAULT_FRAMES_TO_PROFILE
    else:
        TOTAL_FRAMES_TO_PROFILE = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    NUM_STEPS_TO_PROFILE = math.ceil(TOTAL_FRAMES_TO_PROFILE / BATCH_SIZE)

    print(f"--- [PID {PID}] 开始 INFERENCE 循环... 目标 {NUM_STEPS_TO_PROFILE} 批次。")
    start_time = time.time()
    
    step = 0
    total_frames_processed = 0
    
    while step < NUM_STEPS_TO_PROFILE:
        nvtx.range_push(f"{NVTX_PREFIX}_step_{step}")

        # 1. 读取
        nvtx.range_push(f"{NVTX_PREFIX}_A_Load_Decode_Batch (CPU)")
        frame_buffer = read_batch_from_video(cap, BATCH_SIZE, is_camera)
        nvtx.range_pop()

        if not frame_buffer:
            print(f"--- [PID {PID}] 视频源结束。")
            nvtx.range_pop() # 结束 step
            break
            
        num_frames_in_batch = len(frame_buffer)

        # 2. 预处理
        inputs_on_device, _ = preprocess_batch(
            frame_buffer, [], to_tensor_op, normalize_op,
            cpu_pipeline, gpu_prep_ops, args.prep_on_gpu,
            prep_device, inference_device, stream, NVTX_PREFIX
        )

        # 3. 执行
        with torch.cuda.stream(stream):
            loss_item = execute_step(
                step,
                mode='inference',
                model=model,
                inputs_on_device=inputs_on_device,
                labels_on_device=None,
                use_amp=args.use_amp,
                optimizer=None,
                criterion=None,
                scaler=scaler,
                nvtx_prefix=NVTX_PREFIX
            )

        # 4. 记录与同步
        nvtx.range_push(f"{NVTX_PREFIX}_E_Sync_and_Log")
        if (step + 1) % 10 == 0:
            if stream:
                stream.synchronize()
            print(f"--- [PID {PID}] Analyzed batch {step+1}/{NUM_STEPS_TO_PROFILE}")
        nvtx.range_pop()

        nvtx.range_pop() # 结束 step
        step += 1
        total_frames_processed += num_frames_in_batch

    # --- 最终结果 ---
    if stream:
        torch.cuda.synchronize()
    end_time = time.time()
    
    print_final_results(
        start_time, end_time, total_frames_processed, 
        1, args.model_name, args.mode, PID, unit="帧"
    )
    cap.release()


def run_retrain_loop(args: argparse.Namespace, 
                     model: nn.Module,
                     train_loader: DataLoader, # <--- 现在是真实的 DataLoader
                     optimizer: torch.optim.Optimizer,
                     criterion: nn.Module,
                     device_tools: Tuple,
                     num_classes: int):
    """
    [仅训练] 运行重训练的主循环。
    """
    
    NVTX_PREFIX = f"M_{args.model_name}_P_{PID}_RETRAIN"
    
    # 解包工具
    inference_device, _, stream, scaler = device_tools
    
    print(f"--- [PID {PID}] 开始 RETRAIN 循环... 共 {args.retrain_epochs} Epochs。")
    # --- [新添加] Retrain Warm-Up ---
    if args.infer_on_gpu:
        print(f"--- [PID {PID}] 执行 RETRAIN warm-up (3 iterations)... ---")
        input_shape = (args.batch_size_retrain, 3, 224, 224)
        dummy_input = torch.randn(*input_shape, device=inference_device)
        dummy_labels = torch.randint(0, num_classes, (args.batch_size_retrain,), device=inference_device)  # num_classes 来自 setup_retrain_components
        for _ in range(3):  # 可调整为 5 次
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                optimizer.zero_grad()
                outputs = model(dummy_input)
                loss = criterion(outputs, dummy_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        torch.cuda.synchronize()
        print(f"--- [PID {PID}] RETRAIN warm-up 完成 ---")
    # --- [结束添加] ---
    start_time = time.time()
    
    total_batches_processed = 0

    for epoch in range(args.retrain_epochs):
        print(f"\n--- [PID {PID}] (Epoch {epoch+1}/{args.retrain_epochs}) ---")
        running_loss = 0.0
        
        # train_loader 现在由 num_workers 在后台加载数据
        # 主线程在这里等待数据
        nvtx.range_push(f"{NVTX_PREFIX}_epoch_{epoch}_WAIT_FOR_LOADER")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            nvtx.range_pop() # 结束 WAIT_FOR_LOADER
            nvtx.range_push(f"{NVTX_PREFIX}_epoch_{epoch}_step_{batch_idx}")

            # 1. HtoD (由 pin_memory=True 异步)
            with torch.cuda.stream(stream):
                nvtx.range_push(f"{NVTX_PREFIX}_B_HtoD")
                inputs_on_device = inputs.to(inference_device, non_blocking=True)
                labels_on_device = labels.to(inference_device, non_blocking=True)
                nvtx.range_pop()

                # 2. 执行
                loss_item = execute_step(
                    batch_idx,
                    mode='retrain',
                    model=model,
                    inputs_on_device=inputs_on_device,
                    labels_on_device=labels_on_device,
                    use_amp=args.use_amp,
                    optimizer=optimizer,
                    criterion=criterion,
                    scaler=scaler,
                    nvtx_prefix=NVTX_PREFIX
                )

            # 3. 记录与同步
            nvtx.range_push(f"{NVTX_PREFIX}_E_Sync_and_Log")
            running_loss += loss_item
            if (batch_idx + 1) % 10 == 0:
                if stream:
                    stream.synchronize()
                avg_loss = running_loss / 10
                print(f"--- [PID {PID}] [Epoch {epoch+1}, Step {batch_idx+1}] 平均损失: {avg_loss:.4f}")
                running_loss = 0.0
            nvtx.range_pop() # 结束 Sync
            
            nvtx.range_pop() # 结束 Step
            total_batches_processed += 1
            
            nvtx.range_push(f"{NVTX_PREFIX}_epoch_{epoch}_WAIT_FOR_LOADER")
        nvtx.range_pop() # 结束最后一个 WAIT_FOR_LOADER

    # --- 最终结果 ---
    if stream:
        torch.cuda.synchronize()
    end_time = time.time()
    
    print_final_results(
        start_time, end_time, total_batches_processed, 
        args.retrain_epochs, args.model_name, args.mode, PID, unit="批次"
    )

# ==================================================================
# 4. 主函数 (Main) - 逻辑分发
# ==================================================================

def main():
    """
    脚本的主执行函数。
    负责: 1. 设置 2. 加载模型 3. 根据 mode 分发到正确的循环
    """
    global PID
    
    # --- 1. 设置 ---
    args = parse_arguments()
    core_set = setup_cpu_affinity(args.core_bind) # 绑定此进程

    # --- 2. 配置与设备 ---
    MODE = args.mode
    MODEL_NAME = args.model_name

# --- [新逻辑] 根据分配的核心数自动设置 num_workers ---
    if MODE == 'retrain':
        num_assigned_cores = len(core_set)
        
        # [修改] 调整 num_workers 逻辑以避免核心竞争
        if num_assigned_cores > 1:
            # 如果分配了多个核心，为主进程保留 1 个，其余给 Workers
            new_num_workers = num_assigned_cores - 1
            print(f"--- [PID {PID}] 绑核已生效 (分配 {num_assigned_cores} 个核心)。---")
            print(f"--- [PID {PID}] 自动将 'num_workers' 从 {args.num_workers} 调整为 {new_num_workers} (保留 1 个核心给主进程) ---")
            args.num_workers = new_num_workers
        
        elif num_assigned_cores == 1:
            # 如果只分配了 1 个核心，则必须在主进程中加载数据 (num_workers=0)
            print(f"--- [PID {PID}] 警告: 仅分配 1 个核心。---")
            print(f"--- [PID {PID}] 自动将 'num_workers' 从 {args.num_workers} 调整为 0 (在主进程中加载数据) ---")
            args.num_workers = 0
            
        else:
            # 'all' 或 绑核失败, core_set 为空
            print(f"--- [PID {PID}] 未指定特定绑核 ('all' 或失败)。'num_workers' 将使用参数值: {args.num_workers} ---")
    # --- [结束新逻辑] ---
    
    use_cuda = args.infer_on_gpu and torch.cuda.is_available()
    if not use_cuda and (args.prep_on_gpu or args.infer_on_gpu):
        print(f"--- [PID {PID}] 警告: 未检测到 CUDA! 强制所有操作在 CPU 上运行。")
        args.prep_on_gpu = False
        args.infer_on_gpu = False
        args.use_amp = False
        use_cuda = False
    
    inference_device = torch.device("cuda:0" if use_cuda else "cpu")
    prep_device = torch.device("cuda:0" if args.prep_on_gpu else "cpu")

    # --- 3. 打印配置 ---
    print(f"--- [PID {PID}] 配置 (模型: {MODEL_NAME}) ---")
    print(f"运行模式: {MODE}")
    print(f"CPU 核心绑定: {args.core_bind}")
    print(f"使用 AMP: {args.use_amp}")
    print(f"推理/训练将在: {inference_device}")
    
    if MODE == 'inference':
        print(f"   批次大小 (Infer): {args.batch_size_infer}")
        print(f"   预处理将在: {prep_device}")
        print(f"   torch.compile() 已: {'启用' if args.enable_compile and use_cuda else '禁用'}")
    else: # retrain
        print(f"   批次大小 (Retrain): {args.batch_size_retrain}")
        print(f"   学习率: {args.learning_rate}, Epochs: {args.retrain_epochs}")
        print(f"   数据路径: {args.retrain_data_path}") # <--- [新]
        print(f"   Num Workers: {args.num_workers}") # <--- [新]
    print(f"------------")

    # --- 4. 加载共享资源 ---
    model = load_model(MODEL_NAME, inference_device)
    stream = Stream() if use_cuda else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and use_cuda)

    # --- 5. 根据 MODE 分发 ---
    if MODE == 'inference':
        # 加载推理特定组件
        cap, _, _ = initialize_video_capture(args.input_path)
        prep_tools = build_preprocessing_pipelines(args.prep_level)
        device_tools = (inference_device, prep_device, stream, scaler)

        # 预热模型 (仅推理)
        if use_cuda:
            cudnn.benchmark = False
            input_shape = (args.batch_size_infer, 3, 224, 224)
            dummy_input = torch.randn(*input_shape, device=inference_device)
            model = setup_model_for_inference(
                model, use_cuda, dummy_input, enable_compile=args.enable_compile
            )
        
        # 运行推理循环
        run_inference_loop(args, model, cap, prep_tools, device_tools)

    elif MODE == 'retrain':
        
        # --- [逻辑修改] ---
        # 1. 先创建 DataLoader, 这样我们才能知道 num_classes
        train_loader, num_classes = create_train_loader(args)
        
        # 2. 使用真实的 num_classes 来设置模型
        model, optimizer, criterion = setup_retrain_components(
            model, MODEL_NAME, inference_device, num_classes, args.learning_rate
        )
        # --- [结束修改] ---
        
        # (device, prep_dev, stream, scaler) - prep_dev 在训练时未使用
        device_tools = (inference_device, prep_device, stream, scaler) 
        
        # 运行重训练循环
        run_retrain_loop(args, model, train_loader, optimizer, criterion, device_tools, num_classes)


if __name__ == "__main__":
    main()