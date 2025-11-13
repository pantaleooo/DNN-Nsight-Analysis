#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchvision import models
import sys
import logging # 导入 logging
from typing import Tuple, Optionalimport re # 用于解析 split_point
from typing import Tuple, Optional, List # 确保 List 被导入


class SplitModelWrapper(nn.Module):
    """一个封装了 CPU-GPU 跨层分割逻辑的模块。"""
    def __init__(self, part1: nn.Module, part2: nn.Module, part1_device: torch.device, part2_device: torch.device):
        super().__init__()
        logging.info(f"SplitModel: Part 1 将在 {part1_device} 运行。")
        logging.info(f"SplitModel: Part 2 将在 {part2_device} 运行。")
        
        # 确保模型处于评估模式 (如果用于推理)
        self.part1 = part1.to(part1_device).eval()
        self.part2 = part2.to(part2_device).eval()
        
        self.part1_device = part1_device
        self.part2_device = part2_device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 假设输入 x 已经在 self.part1_device (例如 CPU)
        
        # 1. Part 1 (CPU)
        nvtx.range_push("SplitModel_Part1_CPU")
        x = self.part1(x)
        nvtx.range_pop()
        
        # 2. Transfer C-to-G
        # (注意: 即使是 non_blocking, 下一步 GPU 操作也会等待它)
        nvtx.range_push("SplitModel_Transfer_CtoG")
        x = x.to(self.part2_device, non_blocking=True)
        nvtx.range_pop()
        
        # 3. Part 2 (GPU)
        # (它将在 worker.py 的 execute_step 中的 AMP 上下文下运行)
        nvtx.range_push("SplitModel_Part2_GPU")
        x = self.part2(x)
        nvtx.range_pop()
        
        return x

def _create_split_model(model: nn.Module, 
                        model_name: str, 
                        split_point_str: str, 
                        cpu_device: torch.device, 
                        gpu_device: torch.device) -> nn.Module:
    """
    实际创建 SplitModelWrapper 的辅助函数。
    这需要对模型结构有深入了解。
    """
    logging.info(f"正在尝试为 {model_name} 在 '{split_point_str}' 处分割模型...")
    
    try:
        if model_name.startswith('resnet'):
            # 示例：ResNet50。假设 split_point_str = 'layer3'
            # ResNet 结构: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
            if split_point_str == 'layer3':
                part1_layers: List[nn.Module] = [
                    model.conv1, model.bn1, model.relu, model.maxpool,
                    model.layer1, model.layer2, model.layer3
                ]
                part1 = nn.Sequential(*part1_layers)
                
                part2_layers: List[nn.Module] = [
                    model.layer4, model.avgpool, nn.Flatten(1), model.fc
                ]
                part2 = nn.Sequential(*part2_layers)
                
                return SplitModelWrapper(part1, part2, cpu_device, gpu_device)
            else:
                logging.error(f"不支持 ResNet 在 '{split_point_str}' 处分割。")

        elif model_name.startswith('vit'):
            # ViT 的分割非常复杂，因为它不是纯粹的 Sequential。
            # 例如 'encoder.layers.9'
            logging.error(f"ViT 模型的自动分割 (如 '{split_point_str}') 暂不支持。需要手动重定义模型。")
        elif model_name.startswith('mobilenet'):
            logging.error(f"MobileNet 模型的自动分割 (如 '{split_point_str}') 暂不支持。")
    except Exception as e:
        logging.error(f"创建分割模型时出错: {e}")

    logging.warning(f"无法为 {model_name} 创建分割点 '{split_point_str}'。将在主 GPU 设备上运行完整模型。")
    return model.to(gpu_device) # 回退



def load_model(model_name: str, 
               device: torch.device, 
               split_point: Optional[str] = None,
               cpu_device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    根据名称加载预训练模型并将其移动到指定设备。
    如果提供了 split_point，将尝试创建跨层分割模型。
    'device' 参数应为主 GPU 设备。
    """
    logging.info(f"正在加载模型: {model_name} ...")
    torch.hub.set_dir('models')
    
    try:
        if model_name == 'vit_b_16':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            logging.error(f"未知的模型名称 '{model_name}'")
            sys.exit(1)
    except Exception as e:
        logging.error(f"加载模型 {model_name} 失败: {e}")
        sys.exit(1)
        
    # --- 新增分割逻辑 ---
    if split_point:
        # 检查 split_point 是否适用于此模型
        # 格式: 'resnet50:layer3'
        model_tag_match = re.match(r"^(.*?):(.*)$", split_point)
        
        if model_tag_match and model_tag_match.group(1) == model_name:
            split_layer_name = model_tag_match.group(2)
            # `device` 在这里是 GPU 设备
            return _create_split_model(model, model_name, split_layer_name, cpu_device, device)
        elif model_tag_match:
            logging.info(f"跳过分割点 '{split_point}' (适用于 {model_tag_match.group(1)}，但当前是 {model_name})。")

    # --- 回退 (无分割) ---
    model.to(device)
    logging.info(f"模型已加载到 {device}")
    return model


def setup_model_for_inference(model: nn.Module, 
                              use_cuda_infer: bool, 
                              dummy_input_for_warmup: Optional[torch.Tensor], 
                              enable_compile: bool = False, 
                              warmup_iters: int = 3) -> nn.Module:
    """
    为推理准备模型：可选编译 (torch.compile) 和预热 (warm-up)。
    """
    
    # --- 新增：检查是否为分割模型 ---
    if isinstance(model, SplitModelWrapper):
        logging.info("检测到 SplitModelWrapper。")
        
        # 确保 dummy_input 在 CPU (part1_device)
        if dummy_input_for_warmup is not None:
            dummy_input_cpu = dummy_input_for_warmup.to(model.part1_device)
        else:
            logging.warning("缺少 WARM-UP 用的 dummy_input。")
            return model # 跳过预热

        if enable_compile and use_cuda_infer:
            logging.info("正在尝试 torch.compile() GPU 部分 (part2)...")
            try:
                # 只编译 GPU 部分
                model.part2 = torch.compile(model.part2)
                logging.info("torch.compile() part2 成功")
            except Exception as e:
                logging.error(f"torch.compile() 分割模型的 part2 失败: {e}")
        
        # 预热分割模型
        logging.info(f"正在执行 {warmup_iters} 次 WARM-UP 运行 (SplitModel)...")
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(dummy_input_cpu)
        
        torch.cuda.synchronize() # 确保 GPU 操作完成
        logging.info("SplitModel WARM-UP 完成")
        return model
    
    # --- 以下是原始逻辑 (适用于非分割模型) ---
    
    if not use_cuda_infer:
        logging.info("推理在 CPU 上: 跳过 torch.compile() 和 warm-up")
        model.eval() # 确保 eval
        return model

    model.eval() # 确保 eval

    # 1. 可选的 torch.compile()
    if enable_compile:
        # ... (原始逻辑不变)
        logging.info("CUDA 推理: 正在启用 torch.compile()")
        try:
            model = torch.compile(model)
            logging.info("torch.compile() 成功")
        except Exception as e:
            logging.error(f"torch.compile() 失败: {e}")
            print("--- 将回退到 Eager 模式运行 ---")
    else:
        logging.info("torch.compile() 已禁用, 在 Eager 模式下运行")

    # 2. 预热 (Warm-up)
    # ... (原始逻辑不变)
    warmup_type = "JIT 编译/缓存" if enable_compile else "Eager 模式缓存"
    logging.info(f"正在执行 {warmup_iters} 次 WARM-UP 运行 ({warmup_type})...")

    if dummy_input_for_warmup is None:
        logging.warning("缺少 WARM-UP 用的 dummy_input. 跳过 warm-up.")
        return model

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input_for_warmup)
        # 确保所有预热操作完成
        torch.cuda.synchronize()

    logging.info("WARM-UP 完成")
    return model


def setup_retrain_components(model: nn.Module, 
                             model_name: str, 
                             device: torch.device, 
                             num_classes: int, 
                             learning_rate: float) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """
    为 'retrain' 模式配置模型：冻结层、替换分类头、设置优化器和损失函数。
    """
    logging.info("模式: 'retrain'. 正在设置 model.train() 和解冻最后几层...")
    model.train()

    # 1. 冻结所有现有参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 替换分类头
    try:
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

        elif model_name == 'efficientnet_b0':
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            for param in model.classifier[1].parameters():
                param.requires_grad = True
        elif model_name == 'mobilenet_v2':
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            for param in model.classifier[1].parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"未知的模型 {model_name} 无法配置 retrain head")
    except Exception as e:
        logging.error(f"替换模型 {model_name} 的分类头时失败: {e}")
        sys.exit(1)
    
    # 3. 将新头移动到设备
    model.to(device)
    
    # 4. 设置优化器和损失函数
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss().to(device)
    
    logging.info("微调设置完成。优化器将更新以下参数:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   > {name}") # 保持 print，这是格式化数据
    print(f"------------------")
    
    return model, optimizer, criterion
