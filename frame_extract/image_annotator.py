#!/usr/bin/env python3

"""
一个使用预训练的高精度 PyTorch 模型 (如 Faster R-CNN) 
对图像文件夹进行对象检测和标注的工具。

功能:
- 从输入目录加载所有图像。
- 使用 'fasterrcnn_resnet50_fpn_v2' 或 'retinanet_resnet50_fpn_v2' 等高精度模型进行推理。
- 在图像上绘制边界框 (Bounding Box) 和类别标签。
- 将标注后的图像保存到输出目录。
- 支持 GPU (CUDA) 加速 (如果可用)。
- 可自定义置信度阈值以过滤低可信度的检测结果。
"""

import torch
import torchvision
import cv2
import os
import sys
import logging
import argparse
import random
from pathlib import Path
from torchvision.transforms import functional as F
from typing import List

# --- [COCO 类别标签] ---
# TorchVision 的默认检测模型在 COCO 数据集上训练
# 包含 91 个类别 (含 'background')
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# --- [日志记录器] ---
def setup_logging():
    """
    配置全局日志记录器 (借鉴自 utils.py)
    """
    log_format = "--- %(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.getLogger().handlers = [] 
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- [参数解析] ---
def parse_arguments() -> argparse.Namespace:
    """
    配置并解析脚本的命令行参数 (借鉴自 config.py)
    """
    parser = argparse.ArgumentParser(description="高精度图像标注工具 (对象检测)")
    
    parser.add_argument(
        '-i', '--input_dir', 
        type=str, 
        required=True,
        help="包含待标注图像的输入目录。"
    )
    
    parser.add_argument(
        '-o', '--output_dir', 
        type=str, 
        required=True,
        help="用于保存已标注图像的输出目录。"
    )
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='fasterrcnn_resnet50_fpn_v2',
        choices=['fasterrcnn_resnet50_fpn_v2', 'retinanet_resnet50_fpn_v2', 'ssd300_vgg16'],
        help="要使用的预训练检测模型。"
    )
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help="用于显示的置信度阈值 (0.0 到 1.0)。"
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help="运行推理的设备 ('auto', 'cuda:0', 'cpu')。"
    )
    
    return parser.parse_args()

# --- [核心标注逻辑] ---

def load_detection_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """
    加载预训练的 torchvision 检测模型。
    (借鉴自 models.py 中的 load_model 概念)
    """
    logging.info(f"正在加载高精度模型: {model_name} ...")
    
    if model_name == 'fasterrcnn_resnet50_fpn_v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
    elif model_name == 'retinanet_resnet50_fpn_v2':
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        )
    elif model_name == 'ssd300_vgg16':
        model = torchvision.models.detection.ssd300_vgg16(
            weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
        )
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
    
    model.to(device)
    model.eval()  # 切换到评估模式 (借鉴自 models.py)
    
    logging.info(f"模型已加载并移至 {device}，已设置为评估模式 (eval())。")
    return model

def draw_annotations(image: cv2.Mat, predictions: dict, threshold: float) -> cv2.Mat:
    """
    在 OpenCV 图像上绘制检测结果 (边界框、标签、置信度)。
    """
    # 为不同类别生成随机颜色
    colors = {}
    
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    for i, (box, label_id, score) in enumerate(zip(boxes, labels, scores)):
        if score < threshold:
            continue
            
        # 获取类别名称
        label_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f"ID:{label_id}"
        
        # 获取该类别的颜色
        if label_name not in colors:
            colors[label_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = colors[label_name]
        
        # 转换 box 坐标为整数
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        text = f"{label_name}: {score:.2f}"
        
        # 绘制文本背景
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
        
        # 绘制文本
        cv2.putText(image, text, (x1, y1 - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # 黑色字体
        
    return image

def process_images(
    input_dir: Path, 
    output_dir: Path, 
    model: torch.nn.Module, 
    device: torch.device, 
    threshold: float
):
    """
    遍历输入目录，对每张图片进行推理和标注，并保存结果。
    """
    
    # 查找所有支持的图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(ext)))
        
    if not image_files:
        logging.warning(f"在 {input_dir} 中未找到任何图像文件。")
        return

    logging.info(f"在 {input_dir} 中找到了 {len(image_files)} 张图像。开始处理...")
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, image_path in enumerate(image_files):
        logging.info(f"--- 正在处理图像 {i+1}/{len(image_files)}: {image_path.name} ---")
        
        try:
            # 1. 加载图像 (使用 CV2，借鉴自 data_pipeline.py)
            # 我们保留 BGR 格式的副本用于绘图
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                logging.warning(f"无法读取图像 {image_path.name}，跳过。")
                continue
            
            # 转换为 RGB (CV2 默认 BGR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. 预处理 (借鉴自 data_pipeline.py 的 ToTensor)
            # T.ToTensor() 会将 (H, W, C) 的 np.array [0, 255] 
            # 转换为 (C, H, W) 的 torch.Tensor [0.0, 1.0]
            image_tensor = F.to_tensor(image_rgb).to(device)
            
            # 3. 执行推理 (借鉴自 worker.py 的 execute_step)
            with torch.no_grad():
                predictions = model([image_tensor])
            
            # `predictions` 是一个 list，我们取第一个元素
            prediction_data = predictions[0]
            
            # 4. 绘制标注
            # 我们在 BGR 图像上绘制
            annotated_image = draw_annotations(image_bgr, prediction_data, threshold)
            
            # 5. 保存结果
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), annotated_image)
            
        except Exception as e:
            logging.error(f"处理图像 {image_path.name} 时发生错误: {e}")

    logging.info("="*50)
    logging.info(f"所有图像处理完毕。已标注图像保存在: {output_dir.resolve()}")
    logging.info("="*50)

# --- [主函数] ---
def main():
    """
    脚本的主执行函数。
    """
    setup_logging()
    args = parse_arguments()
    
    # --- 1. 确定设备 (借鉴自 worker.py) ---
    if args.device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    if not torch.cuda.is_available() and device.type == 'cuda':
        logging.warning("未检测到 CUDA! 强制在 CPU 上运行。")
        device = torch.device("cpu")
        
    logging.info(f"将使用设备: {device}")
    
    # --- 2. 路径设置 ---
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.is_dir():
        logging.error(f"输入路径不是一个有效的目录: {input_path}")
        sys.exit(1)
        
    # --- 3. 加载模型 ---
    try:
        model = load_detection_model(args.model_name, device)
    except Exception as e:
        logging.error(f"加载模型 {args.model_name} 失败: {e}")
        sys.exit(1)
        
    # --- 4. 执行处理 ---
    try:
        process_images(
            input_dir=input_path,
            output_dir=output_path,
            model=model,
            device=device,
            threshold=args.threshold
        )
    except KeyboardInterrupt:
        logging.info("\n操作被用户中断。")
    except Exception as e:
        logging.error(f"发生未预料的错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()