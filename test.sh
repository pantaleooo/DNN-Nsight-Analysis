# # # 运行 nsys 来分析串行推理脚本
# # sudo nsys profile -t cuda,nvtx,osrt --stats=true -q -o report_serial.nsys-rep \
# #     --force-overwrite true \
# #     /home/ubuntu/uv-envs/torch-env/bin/python experiment_1_serial.py

# # 运行 nsys 来分析串行推理脚本
# sudo nsys profile -t cuda,nvtx,osrt -o report_serial.nsys-rep \
#     --force-overwrite true \
#     /home/ubuntu/uv-envs/torch-env/bin/python experiment_1_serial.py


#!/bin/bash
# [!!! 新增 !!!]
# --- 0. 自动清理 ---
# 每次运行前，都强制杀死任何残留的 nsys 代理 (nsd) 僵尸进程
# 这可以防止 "Connection to the Agent lost" 错误
echo "--- 正在清理旧的 nsys 代理进程 (nsd)... ---"
sudo pkill -9 nsd
sudo pkill -9 nsys
sleep 5  # 等待 5 秒，让进程完全终止
echo "--- 清理完成 ---"
# --------------------

# --- 1. 在这里配置您的实验 ---
PREP_DEVICE="cpu"   # "cpu" 或 "gpu"
INFER_DEVICE="gpu"  # "cpu" 或 "gpu"
BATCH_SIZE=4
CORE_BIND="5-9_15-19"      # 要绑定的核心 (例如 "16", "0-3")。使用 "all" 表示不绑定。
COMPILE="true"      # "true" 或 "false"

# --- 2. 定义路径 ---
# (请确保这些路径正确)
PYTHON_EXE="/home/ubuntu/uv-envs/torch-env/bin/python"
SCRIPT_NAME="experiment_1_serial.py" # 假设 Python 脚本在同一目录

# --- 3. 构建 Python 参数 ---
PYTHON_ARGS=""
if [ "$PREP_DEVICE" == "gpu" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --prep-on-gpu"
fi
if [ "$INFER_DEVICE" == "gpu" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --infer-on-gpu"
fi
if [ "$COMPILE" == "true" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --enable-compile"
fi
PYTHON_ARGS="$PYTHON_ARGS --batch-size $BATCH_SIZE"

PYTHON_ARGS="$PYTHON_ARGS --core-bind $CORE_BIND"

# --- 4. 构建 nsys 命令 ---
# 动态生成报告名称
REPORT_BASENAME="profile_result/report_prep-${PREP_DEVICE}_infer-${INFER_DEVICE}_bs-${BATCH_SIZE}_compile-${COMPILE}_cores-${CORE_BIND}"
NSYS_REP_FILE="${REPORT_BASENAME}.nsys-rep"
SQLITE_FILE="${REPORT_BASENAME}.sqlite"

# 要分析的目标命令
CMD_TO_RUN="$PYTHON_EXE $SCRIPT_NAME $PYTHON_ARGS"

# nsys 分析器命令
NSYS_CMD="sudo nsys profile -t cuda,nvtx,osrt -o $NSYS_REP_FILE --force-overwrite true"



LAUNCH_CMD="$NSYS_CMD $CMD_TO_RUN"


# --- 6. 执行 ---
echo "--- 正在运行实验 (第 1 步: 分析)... ---"
echo "  配置: Prep=$PREP_DEVICE, Infer=$INFER_DEVICE, BS=$BATCH_SIZE, Compile=$COMPILE, Cores=$CORE_BIND"
echo "  输出 (rep): $NSYS_REP_FILE"
echo "  正在执行: $LAUNCH_CMD"
echo "--------------------------"

# 2> /dev/null 用于隐藏 nsys 的进度条和导入消息
eval $LAUNCH_CMD

# [!!! 新增 !!!] --- 7. 同步导出 SQLITE ---
echo "--- 分析完成 (第 2 步: 导出 SQLite)... ---"
echo "  输入: $NSYS_REP_FILE"
echo "  输出: $SQLITE_FILE"

# 也需要 sudo，因为它要读取 root 拥有的 .nsys-rep 文件
# 同样使用 2> /dev/null 隐藏进度
sudo nsys export -t sqlite -o $SQLITE_FILE $NSYS_REP_FILE --force-overwrite true

# ---------------------------------------------
echo "--- 实验完成 (rep 和 sqlite 文件均已生成) ---"

# --- 0. 自动清理 ---
# 每次运行前，都强制杀死任何残留的 nsys 代理 (nsd) 僵尸进程
# 这可以防止 "Connection to the Agent lost" 错误
echo "--- 正在清理旧的 nsys 代理进程 (nsd)... ---"
sudo pkill -9 nsd
sudo pkill -9 nsys
echo "--- 清理完成 ---"
# --------------------

# DRSW_CMD="python draw_experiment_1_serial.py -i $SQLITE_FILE"
# eval $DRAW_CMD
