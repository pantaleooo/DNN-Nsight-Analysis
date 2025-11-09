#!/bin/bash

# ==========================================================
# set_cpu_freq.sh (可配置版)
#
# 1. 将指定范围 CPU 核心设置为 'userspace' governor。
# 2. 将它们的最大(max)和最小(min)频率锁定到目标频率。
#
# 重要: 请使用 sudo 运行此脚本。
# 用法: sudo ./set_cpu_freq.sh
# ==========================================================

# --- [ CONFIGURATION ] ---
# 修改以下变量来自定义范围和频率

# 起始 CPU 核心 ID
START_CORE=4

# 终止 CPU 核心 ID
END_CORE=19

# 目标频率 (例如: "3.4GHz", "2.8GHz")
TARGET_FREQ="3.7GHz"
# -------------------------


# 检查脚本是否以 root 权限运行
if [ "$EUID" -ne 0 ]; then 
  echo "错误: 请以 root 权限 (sudo) 运行此脚本。"
  echo "用法: sudo ./set_cpu_freq.sh"
  exit 1
fi

echo "--- [ 配置 ] ---"
echo "目标核心: $START_CORE 到 $END_CORE"
echo "目标频率: $TARGET_FREQ"
echo "------------------"

echo "正在为 $START_CORE 到 $END_CORE 号 CPU 核心设置 'userspace' governor..."

# 循环 1: 设置所有核心的 governor
for ((i=$START_CORE; i<=$END_CORE; i++)); do
  echo "设置核心 $i -> userspace"
  # -c 指定核心, -g 指定 governor
  cpufreq-set -c $i -g userspace
done

echo "--------------------------------"
echo "正在为 $START_CORE 到 $END_CORE 号 CPU 核心设置 $TARGET_FREQ 频率..."

# 循环 2: 设置所有核心的频率
for ((i=$START_CORE; i<=$END_CORE; i++)); do
  echo "设置核心 $i -> $TARGET_FREQ (min/max)"
  # -d 是 --min (最小频率), -u 是 --max (最大频率)
  cpufreq-set -c $i -d $TARGET_FREQ
  cpufreq-set -c $i -u $TARGET_FREQ
done

echo "--------------------------------"
echo "所有核心配置完成。"
