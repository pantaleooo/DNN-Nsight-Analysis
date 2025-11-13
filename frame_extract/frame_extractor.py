#!/usr/bin/env python3

"""
一个用于从视频文件中提取帧并将其保存为图像的工具。

功能:
- 指定要提取的目标 FPS (每秒保存几帧)。
- 自定义输出图像的格式 (jpg, png) 和文件名前缀。
- 自动创建输出目录。
- 提供清晰的日志记录。
"""

import cv2
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

def setup_logging():
    """
    配置一个简单的日志记录器，用于向控制台输出信息。
    (借鉴自您 utils.py 中的 setup_logging 概念)
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

def parse_arguments() -> argparse.Namespace:
    """
    配置并解析脚本的命令行参数。
    (借鉴自您 config.py 中的 argparse 实践)
    """
    parser = argparse.ArgumentParser(description="视频帧提取工具")
    
    parser.add_argument(
        '-i', '--input_video', 
        type=str, 
        required=True,
        help="要处理的输入视频文件路径。"
    )
    
    parser.add_argument(
        '-o', '--output_dir', 
        type=str, 
        required=True,
        help="用于保存提取帧的目录。"
    )
    
    parser.add_argument(
        '--fps', 
        type=float, 
        default=1.0,
        help="要提取的目标帧率 (每秒保存的图片数量)。例如，输入 0.5 表示每 2 秒保存一帧。"
    )
    
    parser.add_argument(
        '--prefix', 
        type=str, 
        default='frame',
        help="保存图片的文件名前缀 (例如 'frame_000001.jpg')。"
    )
    
    parser.add_argument(
        '--format', 
        type=str, 
        default='jpg',
        choices=['jpg', 'png'],
        help="保存图片的格式 (jpg 或 png)。"
    )
    
    return parser.parse_args()

def extract_frames(
    video_path: str, 
    output_dir: Path, 
    target_fps: float, 
    prefix: str, 
    img_format: str
):
    """
    执行视频帧提取的核心逻辑。

    Args:
        video_path (str): 视频文件路径。
        output_dir (Path): `pathlib.Path` 对象，指向输出目录。
        target_fps (float): 每秒要保存的帧数。
        prefix (str): 保存文件的统一前缀。
        img_format (str): 'jpg' 或 'png'。
    """
    
    # --- 1. 初始化视频捕获 ---
    # (借鉴自您 data_pipeline.py 中的 initialize_video_capture)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"无法打开视频文件: {video_path}")
        return

    # 获取源视频的 FPS
    source_video_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_video_fps == 0:
        logging.warning("无法读取源视频 FPS，将默认使用 30 FPS 进行计算。")
        source_video_fps = 30

    logging.info(f"成功打开视频: {video_path}")
    logging.info(f"   > 源视频帧率: {source_video_fps:.2f} FPS")
    logging.info(f"   > 目标提取帧率: {target_fps} 帧/秒")

    # --- 2. 计算帧间隔 ---
    # 如果 target_fps 是 1，源 FPS 是 30，则 frame_interval 约等于 30 (每 30 帧保存一次)
    # 如果 target_fps 是 0.5，源 FPS 是 30，则 frame_interval 约等于 60 (每 60 帧保存一次)
    if target_fps <= 0:
        logging.error(f"目标 FPS 必须大于 0。")
        return
        
    frame_interval = int(round(source_video_fps / target_fps))
    if frame_interval < 1:
        frame_interval = 1
        logging.warning(f"目标 FPS ({target_fps}) 高于源 FPS ({source_video_fps})。将保存每一帧。")
    
    logging.info(f"   > 计算的帧间隔: 每 {frame_interval} 帧保存 1 帧。")

    # --- 3. 创建输出目录 ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"帧将保存到: {output_dir.resolve()}")
    except Exception as e:
        logging.error(f"创建输出目录 {output_dir} 失败: {e}")
        cap.release()
        return

    # --- 4. 循环读取和保存帧 ---
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        # (借鉴自您 data_pipeline.py 中的 read_batch_from_video)
        ret, frame = cap.read()
        
        if not ret:
            logging.info("视频处理完毕。")
            break
        
        # 检查是否达到了保存该帧的条件
        if frame_count % frame_interval == 0:
            # 构造输出文件名，例如: frame_000001.jpg
            file_name = f"{prefix}_{saved_count:06d}.{img_format}"
            save_path = output_dir / file_name
            
            try:
                cv2.imwrite(str(save_path), frame)
                saved_count += 1
            except Exception as e:
                logging.error(f"保存帧 {file_name} 失败: {e}")
                # 即使保存失败也继续处理下一帧
        
        frame_count += 1
        
        # 简单的进度报告
        if saved_count > 0 and (saved_count % 100 == 0):
            logging.info(f"   ... 已保存 {saved_count} 帧 ...")

    # --- 5. 清理 ---
    cap.release()
    logging.info("="*50)
    logging.info("提取完成。")
    logging.info(f"总共读取帧: {frame_count}")
    logging.info(f"总共保存帧: {saved_count}")
    logging.info("="*50)


def main():
    """
    脚本的主执行函数。
    """
    setup_logging()
    args = parse_arguments()
    
    # 使用 pathlib 来处理路径，更现代且健壮
    output_path = Path(args.output_dir)
    video_path = args.input_video
    
    try:
        extract_frames(
            video_path=video_path,
            output_dir=output_path,
            target_fps=args.fps,
            prefix=args.prefix,
            img_format=args.format
        )
    except KeyboardInterrupt:
        logging.info("\n操作被用户中断。")
    except Exception as e:
        logging.error(f"发生未预料的错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()