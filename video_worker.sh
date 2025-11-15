#!/bin/bash
set -euo pipefail

# =========================
# 0. 清理旧的 nsys 代理
# =========================
echo "--- 正在清理旧的 nsys 代理进程 (nsd)... ---"
sudo pkill -9 nsd 2>/dev/null || true
sudo pkill -9 nsys 2>/dev/null || true
sleep 2  # 等待 2 秒，让进程完全终止
echo "--- 清理完成 ---"



# =========================
# 1. Python/脚本路径
# =========================
PYTHON_EXE="/home/ubuntu/uv-envs/torch-env/bin/python"
SCRIPT_NAME="video_worker.py" # 确保与本脚本同目录

# [!!! 新增 !!!] 辅助函数: 用于将数组转换为 "a_b_c" 格式的字符串
# 用法: join_by _ "a" "b" "c" -> "a_b_c"
join_by() {
  local d=$1 # 分隔符
  shift
  local f=$1 # 第一个元素
  shift
  printf %s "$f" "${@/#/$d}"
}

# =========================
# 2. N-Worker 配置
# =========================
# 定义每个 worker 使用的模型    "vit_b_16, resnet50, efficientnet_b0, resnet101
MODELS=(
    "vit_b_16"
    "vit_b_16"
)

# 定义每个 worker 绑定的核心
CORE_BINDS=(
    "4-5"
    "6-7"
)

# 定义每个 worker 使用的预处理级别
# 可选: 'simple', 'medium', 'complex'
PREP_LEVELS=(
    "complex"
    "complex"
)

# 检查配置是否匹配
NUM_WORKERS=${#MODELS[@]}
if [ $NUM_WORKERS -ne ${#CORE_BINDS[@]} ] || [ $NUM_WORKERS -ne ${#PREP_LEVELS[@]} ]; then
  echo "配置错误: MODELS($NUM_WORKERS), CORE_BINDS(${#CORE_BINDS[@]}), PREP_LEVELS(${#PREP_LEVELS[@]}) 数量须相同"
  exit 1
fi
echo "--- 将启动 $NUM_WORKERS 个并发 Worker ---"


# =========================
# 3. 静态实验参数（所有 Worker 共享）
# =========================
PREP_DEVICE="gpu"   # "cpu" 或 "gpu"
INFER_DEVICE="gpu"  # "cpu" 或 "gpu"
BATCH_SIZE=2  # Increased default to match code changes
COMPILE="true"      # "true" 或 "false"
INPUT_PATH="Bellevue_150th_Eastgate__2017-09-10_19-08-25_clip.mp4"  # Changed to video input path
# --- 输出与实验配置 ---
BASE_OUTPUT_DIR="profile_results"         # 实验结果的根目录
EXPERIMENT_NAME="video_pipeline_v1"       # 本次实验的顶层名称 (例如 "resolution_vs_CPU-GPU")



echo "==========================================================="
echo "--- 正在运行: $NUM_WORKERS 个 Worker 并发测试 ---"
echo "--- 实验名称: $EXPERIMENT_NAME ---"
echo "==========================================================="


# =========================
# 4. 构建 Worker 命令
# =========================

PYTHON_ARGS_SHARED=""
if [ "$PREP_DEVICE" == "gpu" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --prep-on-gpu"; fi
if [ "$INFER_DEVICE" == "gpu" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --infer-on-gpu"; fi
if [ "$COMPILE" == "true" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --enable-compile"; fi
PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --batch-size $BATCH_SIZE --input_path $INPUT_PATH"  # Changed to --input_path

WORKER_CMDS=()
echo "--- Worker 配置 ---"
for (( i=0; i<$NUM_WORKERS; i++ )); do
  MODEL=${MODELS[i]}
  CORE_BIND=${CORE_BINDS[i]}
  PREP_LEVEL=${PREP_LEVELS[i]}

  echo "  Worker $i: 模型=$MODEL, 核心=$CORE_BIND, 预处理=$PREP_LEVEL"

  PYTHON_ARGS_WORKER=""
  PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --model-name $MODEL"
  PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --core-bind $CORE_BIND"
  PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --prep-level $PREP_LEVEL"

  WORKER_CMD_I="$PYTHON_EXE $SCRIPT_NAME $PYTHON_ARGS_SHARED $PYTHON_ARGS_WORKER"
  WORKER_CMDS+=("$WORKER_CMD_I")
done
echo "--------------------------"


# 构建实际执行命令与打印命令
CMD_TO_RUN=""
LOG_CMD_TO_RUN=""
for cmd in "${WORKER_CMDS[@]}"; do
  CMD_TO_RUN+="$cmd & "
  LOG_CMD_TO_RUN+="    $cmd & \n"
done
CMD_TO_RUN+="wait"
LOG_CMD_TO_RUN+="    wait"




# =========================
# 5. nsys 命令与输出目录
# =========================
# --- 5a. 提取动态名称 ---

# 从输入路径提取文件名 (例如: video.mp4 -> video)
INPUT_BASENAME=$(basename "$INPUT_PATH")
INPUT_NAME="${INPUT_BASENAME%.*}" # 移除扩展名

# 将数组转换为字符串 (例如: "vit_b_16" "vit_b_16" -> "vit_b_16_vit_b_16")
MODELS_STR=$(join_by _ "${MODELS[@]}")
# 将核心绑定转换为字符串 (例如: "4-5" "6-7" -> "4-5_6-7")
CORES_STR=$(join_by _ "${CORE_BINDS[@]}")
# 将预处理级别转换为字符串 (例如: "complex" "complex" -> "complex_complex")
PREP_LEVELS_STR=$(join_by _ "${PREP_LEVELS[@]}")

# --- 5b. 构建多级目录 ---
# 结构: <基础目录>/<实验名>/<输入名>/<N-workers>/
REPORT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/${INPUT_NAME}/${NUM_WORKERS}_workers"
echo "--- 正在创建输出目录: $REPORT_DIR ---"
mkdir -p "$REPORT_DIR"

# --- 5c. 构建详细的文件名 ---
# 结构: <参数 1>-<值 1>_<参数 2>-<值 2>...
REPORT_FILENAME="prep-${PREP_DEVICE}_infer-${INFER_DEVICE}_bs-${BATCH_SIZE}_compile-${COMPILE}"
REPORT_FILENAME+="_models-${MODELS_STR}"
REPORT_FILENAME+="_cores-${CORES_STR}"
REPORT_FILENAME+="_preps-${PREP_LEVELS_STR}"

# --- 5d. 最终路径 ---
REPORT_BASENAME="${REPORT_DIR}/${REPORT_FILENAME}"
NSYS_REP_FILE="${REPORT_BASENAME}.nsys-rep"
SQLITE_FILE="${REPORT_BASENAME}.sqlite"

NSYS_CMD="sudo nsys profile -t cuda,nvtx,osrt -o $NSYS_REP_FILE --force-overwrite true"



# =========================
# 6. 启动 NSYS + Workers
# =========================
# 最终启动命令: nsys 启动一个 bash, bash 再启动 N 个并行的 worker
LAUNCH_CMD="$NSYS_CMD bash -c '$CMD_TO_RUN'"

# --- 执行 ---
echo "  配置: BS=$BATCH_SIZE, Compile=$COMPILE"
echo "  输出 (rep): $NSYS_REP_FILE"
echo "  正在执行: $NSYS_CMD bash -c \"... ($NUM_WORKERS 个 worker) ...\""
echo "  命令详情:"
echo -e "$LOG_CMD_TO_RUN"
echo "--------------------------"

# 在后台启动整个 workload（NSYS 控制下）
bash -c "$LAUNCH_CMD" &
NSYS_MAIN_PID=$!

# 等待 NSYS+workload 结束
wait "${NSYS_MAIN_PID}" || true


# =========================
# 7. 导出 NSYS SQLite
# =========================
echo "--- 分析完成 (导出 SQLite)... ---"
echo "  输入: $NSYS_REP_FILE"
echo "  输出: $SQLITE_FILE"
sudo nsys export -t sqlite -o "$SQLITE_FILE" "$NSYS_REP_FILE" --force-overwrite true

echo "--- $NUM_WORKERS-Worker 并发实验完成 ---"
echo "结果目录："
echo "  NSYS: $NSYS_REP_FILE"








# #!/bin/bash
# set -euo pipefail

# # =========================
# # 0. 清理旧的 nsys 代理
# # =========================
# echo "--- 正在清理旧的 nsys 代理进程 (nsd)... ---"
# sudo pkill -9 nsd 2>/dev/null || true
# sudo pkill -9 nsys 2>/dev/null || true
# sleep 2  # 等待 2 秒，让进程完全终止
# echo "--- 清理完成 ---"



# # =========================
# # 1. Python/脚本路径
# # =========================
# PYTHON_EXE="/home/ubuntu/uv-envs/torch-env/bin/python"
# SCRIPT_NAME="model_worker.py" # # 确保与本脚本同目录


# # =========================
# # 2. N-Worker 配置
# # =========================
# # 定义每个 worker 使用的模型    "vit_b_16, resnet50, efficientnet_b0, resnet101
# MODELS=(
#     "vit_b_16"
#     "vit_b_16"
# )

# # 定义每个 worker 绑定的核心
# CORE_BINDS=(
#     "4-7"
#     "4-7"
# )

# # 定义每个 worker 使用的预处理级别
# # 可选: 'simple', 'medium', 'complex'
# PREP_LEVELS=(
#     "medium"
#     "medium"
# )

# # 检查配置是否匹配
# NUM_WORKERS=${#MODELS[@]}
# if [ $NUM_WORKERS -ne ${#CORE_BINDS[@]} ] || [ $NUM_WORKERS -ne ${#PREP_LEVELS[@]} ]; then
#   echo "配置错误: MODELS($NUM_WORKERS), CORE_BINDS(${#CORE_BINDS[@]}), PREP_LEVELS(${#PREP_LEVELS[@]}) 数量须相同"
#   exit 1
# fi
# echo "--- 将启动 $NUM_WORKERS 个并发 Worker ---"


# # =========================
# # 3. 静态实验参数（所有 Worker 共享）
# # =========================
# PREP_DEVICE="gpu"   # "cpu" 或 "gpu"
# INFER_DEVICE="gpu"  # "cpu" 或 "gpu"
# BATCH_SIZE=4
# COMPILE="true"      # "true" 或 "false"
# DATASET_PATH="datasets/celeba_hq_3564-2880/val/"

# echo "==========================================================="
# echo "--- 正在运行: $NUM_WORKERS 个 Worker 并发测试 ---"
# echo "==========================================================="


# # =========================
# # 4. 构建 Worker 命令
# # =========================

# PYTHON_ARGS_SHARED=""
# if [ "$PREP_DEVICE" == "gpu" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --prep-on-gpu"; fi
# if [ "$INFER_DEVICE" == "gpu" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --infer-on-gpu"; fi
# if [ "$COMPILE" == "true" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --enable-compile"; fi
# PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --batch-size $BATCH_SIZE --dataset_path $DATASET_PATH"

# WORKER_CMDS=()
# echo "--- Worker 配置 ---"
# for (( i=0; i<$NUM_WORKERS; i++ )); do
#   MODEL=${MODELS[i]}
#   CORE_BIND=${CORE_BINDS[i]}
#   PREP_LEVEL=${PREP_LEVELS[i]}

#   echo "  Worker $i: 模型=$MODEL, 核心=$CORE_BIND, 预处理=$PREP_LEVEL"

#   PYTHON_ARGS_WORKER=""
#   PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --model-name $MODEL"
#   PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --core-bind $CORE_BIND"
#   PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --prep-level $PREP_LEVEL"

#   WORKER_CMD_I="$PYTHON_EXE $SCRIPT_NAME $PYTHON_ARGS_SHARED $PYTHON_ARGS_WORKER"
#   WORKER_CMDS+=("$WORKER_CMD_I")
# done
# echo "--------------------------"


# # 构建实际执行命令与打印命令
# CMD_TO_RUN=""
# LOG_CMD_TO_RUN=""
# for cmd in "${WORKER_CMDS[@]}"; do
#   CMD_TO_RUN+="$cmd & "
#   LOG_CMD_TO_RUN+="    $cmd & \n"
# done
# CMD_TO_RUN+="wait"
# LOG_CMD_TO_RUN+="    wait"




# # =========================
# # 5. nsys 命令与输出目录
# # =========================
# REPORT_DIR="profile_result/resolution_vs_CPU-GPU/${NUM_WORKERS}_workers"
# mkdir -p "$REPORT_DIR"

# REPORT_BASENAME="${REPORT_DIR}/report_prep-${PREP_DEVICE}_infer-${INFER_DEVICE}_bs-${BATCH_SIZE}_compile-${COMPILE}"
# NSYS_REP_FILE="${REPORT_BASENAME}.nsys-rep"
# SQLITE_FILE="${REPORT_BASENAME}.sqlite"

# NSYS_CMD="sudo nsys profile -t cuda,nvtx,osrt -o $NSYS_REP_FILE --force-overwrite true"


# # =========================
# # 6. 启动 NSYS + Workers
# # =========================
# # 最终启动命令: nsys 启动一个 bash, bash 再启动 N 个并行的 worker
# LAUNCH_CMD="$NSYS_CMD bash -c '$CMD_TO_RUN'"

# # --- 执行 ---
# echo "  配置: BS=$BATCH_SIZE, Compile=$COMPILE"
# echo "  输出 (rep): $NSYS_REP_FILE"
# echo "  正在执行: $NSYS_CMD bash -c \"... ($NUM_WORKERS 个 worker) ...\""
# echo "  命令详情:"
# echo -e "$LOG_CMD_TO_RUN"
# echo "--------------------------"

# # 在后台启动整个 workload（NSYS 控制下）
# bash -c "$LAUNCH_CMD" &
# NSYS_MAIN_PID=$!

# # 等待 NSYS+workload 结束
# wait "${NSYS_MAIN_PID}" || true


# # =========================
# # 7. 导出 NSYS SQLite
# # =========================
# echo "--- 分析完成 (导出 SQLite)... ---"
# echo "  输入: $NSYS_REP_FILE"
# echo "  输出: $SQLITE_FILE"
# sudo nsys export -t sqlite -o "$SQLITE_FILE" "$NSYS_REP_FILE" --force-overwrite true

# echo "--- $NUM_WORKERS-Worker 并发实验完成 ---"
# echo "结果目录："
# echo "  NSYS: $NSYS_REP_FILE"