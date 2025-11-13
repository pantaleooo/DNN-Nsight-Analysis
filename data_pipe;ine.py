#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as T
import torch.cuda.nvtx as nvtx
import time
import sys
import cv2
import numpy as np
import logging # 导入 logging
from torch.cuda import Stream
from typing import List, Set, Dict, Tuple, Optional, Any

from utils import get_dummy_label # (只导入需要的函数)

# ==================================================================
# 1. 预处理流水线构建
# ==================================================================

def build_preprocessing_pipelines(prep_level: str) -> Tuple[T.Compose, Dict[str, nn.Module], T.Normalize, T.ToTensor]:
    """
    根据复杂度级别构建 CPU 和 GPU 预处理操作。
    """
    # (内部逻辑不变)
    _resize_256_op = T.Resize(256)
    _resize_224_op = T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST)
    _crop_224_op = T.CenterCrop(224)
    _normalize_op = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    _random_rotation_op = T.RandomRotation(degrees=15)
    _gaussian_blur_op = T.GaussianBlur(kernel_size=3)
    _to_tensor_op = T.ToTensor()

    logging.info(f"正在构建 '{prep_level}' 预处理流水线...")

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

# ==================================================================
# 2. 视频 I/O
# ==================================================================

def initialize_video_capture(input_path: str) -> Tuple[cv2.VideoCapture, float, float, bool, bool]:
    """
    初始化视频捕获 (来自文件、摄像头或流)。
    返回: (cap, fps, total_frames, is_camera, is_stream)
    """
    try:
        is_camera = False
        is_stream = False
        
        if input_path.isdigit():
            cap = cv2.VideoCapture(int(input_path))  # 摄像头索引
            is_camera = True
            source_type = "摄像头"
        elif input_path.startswith("rtsp://") or input_path.startswith("http://"):
            cap = cv2.VideoCapture(input_path)  # RTSP/HTTP 流
            is_stream = True
            source_type = "网络流"
        else:
            cap = cv2.VideoCapture(input_path)  # 视频文件
            source_type = "视频文件"
            
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频源: {input_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 摄像头和流没有总帧数
        total_frames = float('inf') if is_camera or is_stream else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"成功加载 {source_type}: {input_path}")
        logging.info(f"   > FPS: {fps}, 总帧数: {total_frames}")
        
        return cap, fps, total_frames, is_camera, is_stream
        
    except Exception as e:
        logging.error(f"加载视频路径时发生错误: {e}")
        sys.exit(1)


def read_batch_from_video(cap: cv2.VideoCapture, 
                          batch_size: int, 
                          mode: str, 
                          num_classes: int, 
                          is_camera: bool,
                          is_stream: bool) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
    """
    从视频源 (cap) 读取一个批次的帧。
    """
    frame_buffer = []
    label_buffer = []

    while len(frame_buffer) < batch_size and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_camera or is_stream:
                time.sleep(0.001)  # 避免摄像头/流忙等待
                continue
            else:
                break  # 视频文件结束
                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        frame_buffer.append(frame)
        if mode == 'retrain':
            label_buffer.append(get_dummy_label(num_classes))

    # 修复：如果读取的帧数少于 batch_size，这是一个不完整的批次，应丢弃
    if len(frame_buffer) < batch_size: 
        frame_buffer = []
        label_buffer = []
        
    return frame_buffer, label_buffer

# ==================================================================
# 3. 批处理
# ==================================================================

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
    """
    # (内部逻辑不变)
    
    # 步骤 1: 始终在 CPU 上将 Numpy 数组转换为张量
    inputs = [ to_tensor_op(img) for img in frame_buffer ]
    inputs = torch.stack(inputs)

    # 步骤 2: 处理标签 (如果存在)
    labels_on_device = None
    if label_buffer:
        labels = torch.cat(label_buffer)
        labels_on_device = labels.to(inference_device, non_blocking=True) if labels is not None else None

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
        inputs_on_device = inputs.to(inference_device, non_blocking=True)
        nvtx.range_pop()

    return inputs_on_device, labels_on_device
