#!/usr/bin/env python3

import os
import torch
import sys
import time
import logging
import csv
from typing import Set, Dict, List
import platform  # [+] 新增
# --- [全局 Worker PID 标记] ---
PID = os.getpid()

# --- [日志记录器设置] ---
def setup_logging(level=logging.INFO, model_name=""):
    """配置全局日志记录器"""
    prefix = f"[PID {PID}]"
    if model_name:
        prefix += f" (Model: {model_name})"
    
    # 格式化
    log_format = f"--- {prefix} %(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 清除任何现有的处理器，防止重复日志
    logging.getLogger().handlers = [] 
    
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)] # 默认输出到控制台
    )
    logging.info("日志系统初始化完成。")

# --- [CPU 亲和性] ---

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
                logging.error(f"无法解析核心范围 '{part}': {e}")
        else:
            try:
                core_set.add(int(part))
            except Exception as e:
                logging.error(f"无法解析核心 ID '{part}': {e}")
    return core_set


def setup_cpu_affinity(core_bind_string: str) -> Set[int]:
    """
    根据配置字符串设置当前进程的 CPU 亲和性，并设置 Torch 线程数。
    """
    core_set = set()
    if core_bind_string.lower() == "all":
        logging.info("绑核已跳过 (配置为 'all')")
        logging.info("允许 Torch 使用默认线程数")
        return core_set

    try:
        core_set = parse_core_list(core_bind_string)
        if not core_set:
            raise ValueError("解析后的核心 set 为空。")

        os.sched_setaffinity(PID, core_set)
        num_threads_to_use = len(core_set)
        
        logging.info(f"绑核成功。进程已被绑定到 CPU 核心: {core_set}")
        logging.info(f"正在设置 Torch 线程数 = {num_threads_to_use}")
        torch.set_num_threads(num_threads_to_use)

    except Exception as e:
        logging.error(f"绑核失败 (Cores: {core_bind_string}) - 错误: {e}")
        logging.info("允许 Torch 使用默认线程数")
        return set()
    
    return core_set


def log_system_info():
    """
    记录关键的系统和硬件信息。
    """
    logging.info("="*50)
    logging.info("--- 收集系统/硬件环境信息 ---")
    try:
        logging.info(f"   > 操作系统 (OS): {platform.system()} {platform.release()}")
        logging.info(f"   > Python 版本: {platform.python_version()}")
        logging.info(f"   > PyTorch 版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            logging.info("   > CUDA 可用: 是")
            logging.info(f"   > CUDA 编译版本: {torch.version.cuda}")
            logging.info(f"   > GPU 设备数量: {torch.cuda.device_count()}")
            
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            total_mem_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            logging.info(f"   > 当前 GPU (ID {current_device}): {gpu_name}")
            logging.info(f"   > GPU 总显存: {total_mem_gb:.2f} GB")
        else:
            logging.info("   > CUDA 可用: 否 (将在 CPU 上运行)")
            
    except Exception as e:
        logging.error(f"   > 收集系统信息时出错: {e}")
    finally:
        logging.info("="*50)

# --- [指标收集器] ---

class MetricsCollector:
    """
    用于收集和导出每个批次详细性能指标的类。
    """
    def __init__(self, model_name: str, pid: int, output_dir: str = "."):
        self.records: List[Dict] = []
        self.model_name = model_name
        self.pid = pid
        
        # 定义输出文件名
        self.output_filename = os.path.join(
            output_dir, 
            f"M_{model_name}_P_{pid}_metrics.csv"
        )
        # 定义 CSV 文件的列名
        self.fieldnames = [
            "step", 
            "read_latency_ms", 
            "prep_latency_ms", 
            "exec_latency_ms", 
            "total_step_latency_ms", 
            "loss"
        ]
        logging.info(f"指标收集器已初始化。结果将保存到 {self.output_filename}")

    def add_step_record(self, record: dict):
        """添加一个批次的指标记录"""
        self.records.append(record)

    def export_to_csv(self):
        """将所有收集到的指标导出到 CSV 文件"""
        if not self.records:
            logging.warning("没有收集到指标，跳过 CSV 导出。")
            return

        logging.info(f"正在导出 {len(self.records)} 条详细指标到 {self.output_filename}...")
        try:
            with open(self.output_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.records)
            logging.info("指标导出完成。")
        except Exception as e:
            logging.error(f"导出 CSV 时发生错误: {e}")


# --- [其他辅助函数] ---

def get_dummy_label(num_classes: int) -> torch.Tensor:
    """为训练模式生成一个虚拟标签。"""
    return torch.randint(0, num_classes, (1,))


def print_final_results(start_time: float, 
                        end_time: float, 
                        total_frames_processed: int, 
                        num_epochs: int, 
                        model_name: str, 
                        mode: str):
    """
    计算并打印最终的性能总结。
    """
    total_time_taken = end_time - start_time
    avg_fps = total_frames_processed / total_time_taken if total_time_taken > 0 else 0

    print("="*50)
    # 使用 logging 记录摘要
    logging.info(f"(模型: {model_name}) [Mode: {mode}] 任务完成。")
    # 保持 print 用于清晰的最终输出
    print(f"总耗时 ({num_epochs} Epochs): {total_time_taken:.4f} 秒")
    print(f"总处理帧 (所有 Epochs): {total_frames_processed} 帧")
    print(f"平均吞吐量 (所有 Epochs): {avg_fps:.2f} 帧/秒 (FPS)")
    print("="*50)
