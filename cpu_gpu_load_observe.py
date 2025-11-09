import time
import torch
import numpy as np
import pandas as pd
import multiprocessing
import threading
import queue
import signal
import os
import random

# --- 1. 实验配置 (可调整的参数) ---

CONFIG = {
    # 实验时长
    "EXPERIMENT_DURATION_S": 30.0,
    
    # 推理任务 (T_I) 配置
    "INFERENCE_CONCURRENCY": 4,      # 模拟并发的推理请求数
    "SLO_MS": 100.0,                 # 推理SLO目标 (毫秒)
    
    # P_I (推理预处理) - CPU密集型
    "P_I_CPU_WORK_ITERATIONS": 20_000_000, # 调整此值以模拟约 20-30ms 的CPU工作
    
    # E_I (推理执行) - GPU密集型
    "E_I_GPU_WORK_MATRIX_SIZE": 2048,   # 调整此值以模拟约 30-50ms 的GPU工作

    # 重训练任务 (T_R) 配置
    "RETRAINING_INTERVAL_S": 7.0,     # 每隔多久触发一次重训练
    
    # P_R (重训练预处理) - CPU密集型
    "P_R_CPU_WORK_ITERATIONS": 150_000_000, # 调整此值以模拟约 2-3 秒的重度CPU工作

    # P_R (重训练预处理) - GPU密集型 (用于策略C)
    "P_R_GPU_WORK_MATRIX_SIZE": 6144,   # 调整此值以模拟在GPU上 1-2 秒的工作
    
    # E_R (重训练执行) - GPU密集型
    "E_R_GPU_WORK_MATRIX_SIZE": 12288,  # 调整此值以模拟 4-5 秒的重度GPU训练
}

# --- 2. 资源模拟工具 ---

def simulate_cpu_work(log_queue, tag, iterations):
    """通过执行数学运算来真实地消耗CPU时间"""
    pid = os.getpid()
    start_time = time.monotonic()
    
    # 忙等待循环：这是模拟CPU密集型工作的关键
    _ = 0
    for i in range(iterations):
        _ = (i * i + 123.456) * (789.012 - i)
        
    end_time = time.monotonic()
    duration_ms = (end_time - start_time) * 1000
    
    log_queue.put({
        "timestamp": end_time,
        "type": tag,
        "pid": pid,
        "duration_ms": duration_ms
    })
    return duration_ms

def simulate_gpu_work(log_queue, tag, matrix_size):
    """通过执行矩阵乘法来真实地消耗GPU时间"""
    pid = os.getpid()
    if not torch.cuda.is_available():
        log_queue.put({
            "timestamp": time.monotonic(),
            "type": "ERROR",
            "pid": pid,
            "message": "CUDA not available. GPU work simulated with sleep."
        })
        # 如果没有GPU，则降级为sleep，但这不会消耗资源
        time.sleep(matrix_size / 2048.0 * 0.05) 
        return
        
    device = torch.device("cuda")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    try:
        # 在GPU上创建数据
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        
        start_event.record()
        # 执行GPU密集型工作
        c = torch.matmul(a, b)
        end_event.record()
        
        # 关键：同步CPU和GPU，确保GPU工作已完成
        torch.cuda.synchronize()
        duration_ms = start_event.elapsed_time(end_event)
        
        log_queue.put({
            "timestamp": time.monotonic(),
            "type": tag,
            "pid": pid,
            "duration_ms": duration_ms
        })
        return duration_ms

    except torch.cuda.OutOfMemoryError:
        log_queue.put({
            "timestamp": time.monotonic(),
            "type": "ERROR",
            "pid": pid,
            "message": f"CUDA OOM with matrix size {matrix_size}. Reduce workload."
        })
    except Exception as e:
        log_queue.put({
            "timestamp": time.monotonic(),
            "type": "ERROR",
            "pid": pid,
            "message": f"GPU work failed: {e}"
        })

# --- 3. 并发工作流 (Workers) ---

def inference_worker(request_id, log_queue, stop_event):
    """
    模拟一个完整的端到端推理请求 (T_I = P_I + E_I)
    在一个单独的进程中运行。
    """
    if stop_event.is_set():
        return
        
    pid = os.getpid()
    e2e_start = time.monotonic()
    
    # 阶段 1: 推理预处理 (CPU)
    p_i_duration = simulate_cpu_work(
        log_queue, 
        "P_I", 
        CONFIG["P_I_CPU_WORK_ITERATIONS"]
    )
    
    # 阶段 2: 推理执行 (GPU)
    e_i_duration = simulate_gpu_work(
        log_queue, 
        "E_I", 
        CONFIG["E_I_GPU_WORK_MATRIX_SIZE"]
    )
    
    e2e_end = time.monotonic()
    e2e_duration_ms = (e2e_end - e2e_start) * 1000
    
    # 上报端到端延迟
    log_queue.put({
        "timestamp": e2e_end,
        "type": "T_I_E2E",
        "pid": pid,
        "request_id": request_id,
        "duration_ms": e2e_duration_ms,
        "p_i_duration_ms": p_i_duration,
        "e_i_duration_ms": e_i_duration,
        "slo_violation": e2e_duration_ms > CONFIG["SLO_MS"]
    })

def load_generator(log_queue, stop_event):
    """
    模拟一个具有固定并发数(CONCURRENCY)的推理负载生成器。
    这模拟了一个“闭环”系统，总是有N个请求在处理中。
    """
    request_id = 0
    pool = multiprocessing.Pool(processes=CONFIG["INFERENCE_CONCURRENCY"])
    
    while not stop_event.is_set():
        try:
            # 持续提交任务以保持并发水平
            pool.apply_async(inference_worker, (request_id, log_queue, stop_event))
            request_id += 1
            # 增加一个小的随机延迟，使请求到达不那么均匀
            time.sleep(random.uniform(0.001, 0.01)) 
        except Exception as e:
            if not stop_event.is_set():
                print(f"[LoadGen] Error: {e}")
                
    pool.close()
    pool.join()
    print("[LoadGen] Shutting down.")

def retraining_worker(policy, log_queue):
    """
    模拟一个完整的重训练任务 (T_R = P_R + E_R)
    策略 B: P_R on CPU
    策略 C: P_R on GPU
    """
    pid = os.getpid()
    log_queue.put({
        "timestamp": time.monotonic(),
        "type": "T_R_START",
        "pid": pid,
        "policy": policy
    })
    
    p_r_duration, e_r_duration = 0, 0
    
    if policy == "B_CPU_BLIND":
        # --- 策略 B: 重训练预处理在 CPU 上运行 ---
        # 这将与 P_I (推理预处理) 发生冲突
        p_r_duration = simulate_cpu_work(
            log_queue, 
            "P_R", 
            CONFIG["P_R_CPU_WORK_ITERATIONS"]
        )
    
    elif policy == "C_PACS_LIKE":
        # --- 策略 C: 重训练预处理在 GPU 上运行 ---
        # 这将与 E_I (推理执行) 发生冲突，但保护了CPU
        p_r_duration = simulate_gpu_work(
            log_queue, 
            "P_R", 
            CONFIG["P_R_GPU_WORK_MATRIX_SIZE"]
        )
    
    # 两个策略都在GPU上执行重训练
    e_r_duration = simulate_gpu_work(
        log_queue, 
        "E_R", 
        CONFIG["E_R_GPU_WORK_MATRIX_SIZE"]
    )
    
    log_queue.put({
        "timestamp": time.monotonic(),
        "type": "T_R_END",
        "pid": pid,
        "p_r_duration_ms": p_r_duration,
        "e_r_duration_ms": e_r_duration
    })

def log_processor(log_queue, stop_event, all_logs):
    """
    一个单独的线程，用于安全地从队列中收集所有日志。
    """
    while not stop_event.is_set():
        try:
            log_entry = log_queue.get(timeout=0.1)
            all_logs.append(log_entry)
        except queue.Empty:
            # 检查 stop_event 是否已设置，以便在主线程结束后退出
            if stop_event.is_set() and log_queue.empty():
                break
        except Exception as e:
            print(f"[LogProcessor] Error: {e}")
            
    # 清空队列中剩余的日志
    while not log_queue.empty():
        try:
            all_logs.append(log_queue.get_nowait())
        except queue.Empty:
            break
    print("[LogProcessor] Shutting down.")


def run_experiment(policy):
    """
    执行单个策略实验的主函数。
    """
    print("\n" + "="*50)
    print(f"🚀 [Experiment] Staging Policy: {policy}")
    print("="*50)

    # multiprocessing.Manager 用于在进程间共享 `all_logs` 列表
    manager = multiprocessing.Manager()
    all_logs = manager.list()
    log_queue = manager.Queue()
    stop_event = manager.Event()

    # 启动日志收集器
    log_thread = threading.Thread(target=log_processor, args=(log_queue, stop_event, all_logs))
    log_thread.start()

    # 启动推理负载生成器
    load_gen_proc = multiprocessing.Process(target=load_generator, args=(log_queue, stop_event))
    load_gen_proc.start()
    
    print(f"[Main] Inference load started with {CONFIG['INFERENCE_CONCURRENCY']} concurrent workers.")
    
    # --- 策略逻辑 ---
    retraining_procs = []
    
    if policy == "A_BASELINE":
        # 策略 A: 只运行推理
        print("[Main] Running BASELINE. No retraining will be triggered.")
        time.sleep(CONFIG["EXPERIMENT_DURATION_S"])
        
    elif policy in ["B_CPU_BLIND", "C_PACS_LIKE"]:
        # 策略 B & C: 周期性地触发重训练
        start_time = time.monotonic()
        while time.monotonic() - start_time < CONFIG["EXPERIMENT_DURATION_S"]:
            # 清理已完成的重训练进程
            retraining_procs = [p for p in retraining_procs if p.is_alive()]
            
            # 模拟“调度器”触发
            if not retraining_procs: # 仅当没有重训练在运行时
                print(f"[Main] Triggering retraining task for policy {policy}...")
                p = multiprocessing.Process(target=retraining_worker, args=(policy, log_queue))
                p.start()
                retraining_procs.append(p)
                
            time.sleep(CONFIG["RETRAINING_INTERVAL_S"])
            
    # --- 实验结束，开始清理 ---
    print("\n[Main] Experiment duration ended. Signaling all processes to stop...")
    stop_event.set()

    # 等待所有进程结束
    load_gen_proc.join(timeout=10)
    if load_gen_proc.is_alive():
        print("[Main] Forcing load generator termination.")
        load_gen_proc.terminate()
        
    for p in retraining_procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            
    # 等待日志收集器完成
    log_thread.join(timeout=5)
    
    print(f"[Main] Experiment for {policy} finished. Processing {len(all_logs)} log entries.")
    
    # --- 4. 报告结果 ---
    if not all_logs:
        print("[Error] No logs were collected.")
        return

    # 转换为 Pandas DataFrame 进行分析
    df = pd.DataFrame(list(all_logs))
    
    # 提取关键的推理性能数据
    e2e_df = df[df["type"] == "T_I_E2E"].copy()
    
    if e2e_df.empty:
        print("[Error] No E2E inference logs found.")
        return

    e2e_df["duration_ms"] = pd.to_numeric(e2e_df["duration_ms"])
    
    # 计算关键指标
    avg_latency = e2e_df["duration_ms"].mean()
    p95_latency = e2e_df["duration_ms"].quantile(0.95)
    p99_latency = e2e_df["duration_ms"].quantile(0.99)
    total_requests = len(e2e_df)
    slo_violations = e2e_df["slo_violation"].sum()
    slo_violation_rate = (slo_violations / total_requests) * 100
    throughput = total_requests / CONFIG["EXPERIMENT_DURATION_S"]

    print("\n" + "-"*50)
    print(f"📊 [Results] Report for Policy: {policy}")
    print(f"   Total Requests Served: {total_requests:,.0f}")
    print(f"   Avg. Throughput (req/s): {throughput:.2f}")
    print(f"   Avg. E2E Latency (ms): {avg_latency:.2f}")
    print(f"   p95 E2E Latency (ms): {p95_latency:.2f}")
    print(f"   p99 E2E Latency (ms): {p99_latency:.2f}")
    print(f"   SLO Violations (> {CONFIG['SLO_MS']} ms): {slo_violations:,.0f}")
    print(f"   SLO Violation Rate: {slo_violation_rate:.2f} %")
    print("-" * 50)
    
    # 返回 p99 延迟以进行跨策略比较
    return p99_latency


# --- 5. 主执行函数 ---

def main():
    # 确保子进程在 CUDA 上是安全的
    multiprocessing.set_start_method("spawn", force=True)

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("="*50)
        print("⚠️ WARNING: No CUDA GPU detected. ⚠️")
        print("   GPU work will be simulated with time.sleep().")
        print("   This will NOT accurately demonstrate CPU vs GPU contention.")
        print("   Please run on a machine with a CUDA-enabled GPU.")
        print("="*50)
    else:
        print(f"✅ Found CUDA Device: {torch.cuda.get_device_name(0)}")

    # 运行三个策略
    results = {}
    results["A_BASELINE"] = run_experiment("A_BASELINE")
    results["B_CPU_BLIND"] = run_experiment("B_CPU_BLIND")
    results["C_PACS_LIKE"] = run_experiment("C_PACS_LIKE")

    print("\n\n" + "#"*60)
    print("### Final Experiment Summary (p99 Latency) ###")
    print(f"SLO Target: {CONFIG['SLO_MS']:.2f} ms")
    print(f"  Policy A (Baseline):   {results['A_BASELINE']:.2f} ms")
    print(f"  Policy B (CPU-Blind):  {results['B_CPU_BLIND']:.2f} ms   <--- 预期此值最高 (CPU冲突)")
    print(f"  Policy C (PACS-like):  {results['C_PACS_LIKE']:.2f} ms   <--- 预期此值接近基线 (冲突已解决)")
    print("#"*60)

if __name__ == "__main__":
    main()
