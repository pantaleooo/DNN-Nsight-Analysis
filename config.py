#!/usr/bin/env python3

import argparse
import logging
import os # 需要 os 来获取 getpid

def parse_arguments() -> argparse.Namespace:
    """
    配置并解析脚本的命令行参数。
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
    parser.add_argument('--warmup-iters', type=int, default=3,
                        help="在主循环开始前用于预热和 JIT 编译的迭代次数。")

    # --- 数据与预处理参数 ---
    parser.add_argument('--input_path', type=str, default='video.mp4',
                        help="视频文件路径、摄像头索引或 RTSP/HTTP 流地址")
    parser.add_argument('--prep-level', type=str, default='complex',
                        choices=['simple', 'medium', 'complex'],
                        help="预处理的复杂度级别。")
    parser.add_argument('--max-steps', type=int, default=None,
                        help="要分析的最大批次数。默认 (None) 表示处理整个视频源 (如果是文件) 或使用默认值 (如果是流)。")

    # --- 训练特定参数 ---
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help="用于重训练的学习率")
    parser.add_argument('--retrain-epochs', type=int, default=3,
                        help="重训练的 Epoch 数量")
    parser.add_argument('--split-point', type=str, default=None,
                        help="(实验性) DNN 跨层分割点。格式: 'model_name:layer_name'。"
                             "例如: 'resnet50:layer3'。")

    args = parser.parse_args()

    if args.mode == 'retrain' and args.split_point:
        logging.warning(f"--- [PID {os.getpid()}] 警告: 跨层分割 (split-point) 与 'retrain' 模式暂不兼容。将忽略分割点。 ---")
        # 或者直接 sys.exit(1)
        args.split_point = None

    # --- 参数验证 ---
    if args.mode == 'retrain' and args.enable_compile:
        # 在日志系统配置前发出警告
        logging.warning(f"--- [PID {os.getpid()}] 警告: 训练模式下 torch.compile() 已被强制禁用。 ---")
        args.enable_compile = False

    return args
