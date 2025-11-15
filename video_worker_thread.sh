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
# [!!] 确保这指向您正确的 Python 脚本
SCRIPT_NAME="video_worker_thread.py" 

# 辅助函数: 用于将数组转换为 "a_b_c" 格式的字符串
join_by() {
  local d=$1 # 分隔符
  shift
  local f=$1 # 第一个元素
  shift
  printf %s "$f" "${@/#/$d}"
}

# =========================
# 2. N-Worker 配置
# [!!! 核心修改 !!!]
# =========================

# 定义每个 worker 的模式: 'inference' 或 'retrain'
MODES=(
    "retrain"
)

# [!! 新增 !!] 为每个 worker 定义数据路径
# - inference worker: 视频文件路径
# - retrain worker: 训练数据集路径 (或 "NA" 如果使用 dummy data)
DATA_PATHS=(
    "datasets/celeba_hq_partData/train/"
    # "Bellevue_150th_Eastgate__2017-09-10_19-08-25_clip.mp4"
    # "NA" # 或者，如果 retrain worker 仍使用 dummy data，请使用 "NA"
)

# 定义每个 worker 使用的模型 "vit_b_16, resnet50, efficientnet_b0, resnet101
MODELS=(
    "resnet50"
)

# 定义每个 worker 绑定的核心
CORE_BINDS=(
    "2"  # 推理工人核心
)

# 定义每个 worker 的 BATCH_SIZE
BATCH_SIZES=(
    8
)

# 预处理级别 (retrain worker 不使用此参数, 用 "NA" 占位)
PREP_LEVELS=(
    "complex" 
)

# 检查配置是否匹配
NUM_WORKERS=${#MODES[@]}
if [ $NUM_WORKERS -ne ${#MODELS[@]} ] || \
   [ $NUM_WORKERS -ne ${#CORE_BINDS[@]} ] || \
   [ $NUM_WORKERS -ne ${#BATCH_SIZES[@]} ] || \
   [ $NUM_WORKERS -ne ${#DATA_PATHS[@]} ] || \
   [ $NUM_WORKERS -ne ${#PREP_LEVELS[@]} ]; then
  echo "配置错误: 所有配置数组 (MODES, MODELS, DATA_PATHS, etc.) 的数量必须相同"
  echo "  MODES: ${#MODES[@]}"
  echo "  DATA_PATHS: ${#DATA_PATHS[@]}"
  echo "  MODELS: ${#MODELS[@]}"
  echo "  CORE_BINDS: ${#CORE_BINDS[@]}"
  echo "  BATCH_SIZES: ${#BATCH_SIZES[@]}"
  echo "  PREP_LEVELS: ${#PREP_LEVELS[@]}"
  exit 1
fi
echo "--- 将启动 $NUM_WORKERS 个并发 Worker ---"


# =========================
# 3. 静态实验参数
# [!!! 核心修改 !!!]
# =========================
# --- 通用 GPU/性能设置 ---
PREP_DEVICE="cpu"   # "cpu" 或 "gpu"
INFER_DEVICE="gpu"  # "cpu" 或 "gpu"
USE_AMP="false"      # "true" 或 "false" (新增)

# --- 'inference' worker 共享的参数 ---
COMPILE="false"      # "true" 或 "false"
# [!! 移除 !!] 静态 INPUT_PATH 已移至上面的 DATA_PATHS 数组

# --- 'retrain' worker 共享的参数 ---
LEARNING_RATE="1e-4"
RETRAIN_EPOCHS=3

# --- 输出与实验配置 ---
BASE_OUTPUT_DIR="profile_results"          # 实验结果的根目录
EXPERIMENT_NAME="Infer_vs_Retrain_v1"      # 本次实验的顶层名称


echo "==========================================================="
echo "--- 正在运行: $NUM_WORKERS 个 Worker 并发测试 ---"
echo "--- 实验名称: $EXPERIMENT_NAME ---"
echo "==========================================================="


# =========================
# 4. 构建 Worker 命令
# [!!! 核心修改 !!!]
# =========================

# 构建所有 worker 共享的参数 (例如 GPU 设置)
PYTHON_ARGS_SHARED=""
if [ "$PREP_DEVICE" == "gpu" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --prep-on-gpu"; fi
if [ "$INFER_DEVICE" == "gpu" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --infer-on-gpu"; fi
if [ "$USE_AMP" == "true" ]; then PYTHON_ARGS_SHARED="$PYTHON_ARGS_SHARED --use-amp"; fi


WORKER_CMDS=()
echo "--- Worker 配置 ---"
for (( i=0; i<$NUM_WORKERS; i++ )); do
  # 从数组中获取 worker 的配置
  MODE=${MODES[i]}
  MODEL=${MODELS[i]}
  CORE_BIND=${CORE_BINDS[i]}
  BATCH_SIZE=${BATCH_SIZES[i]}
  PREP_LEVEL=${PREP_LEVELS[i]}
  DATA_PATH=${DATA_PATHS[i]} # [!! 新增 !!]

  echo "  Worker $i: Mode=$MODE, Model=$MODEL, Cores=$CORE_BIND, BS=$BATCH_SIZE, Path=$DATA_PATH"

  # 构建此 worker 的特定参数
  PYTHON_ARGS_WORKER=""
  PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --mode $MODE"
  PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --model-name $MODEL"
  PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --core-bind $CORE_BIND"

  # [!!] 根据 MODE 添加特定参数
  if [ "$MODE" == "inference" ]; then
    PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --batch-size-infer $BATCH_SIZE"
    PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --prep-level $PREP_LEVEL"
    # [!! 修改 !!] 使用 DATA_PATH 变量
    if [ "$DATA_PATH" != "NA" ]; then
        PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --input_path $DATA_PATH"
    fi
    if [ "$COMPILE" == "true" ]; then
        PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --enable-compile"
    fi
    
  elif [ "$MODE" == "retrain" ]; then
    PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --batch-size-retrain $BATCH_SIZE"
    PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --learning-rate $LEARNING_RATE"
    PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --retrain-epochs $RETRAIN_EPOCHS"
    # [!! 新增 !!] 假设 Python 脚本接受 --retrain_data_path
    # 如果您的脚本不接受此参数，可以安全地删除下面两行
    if [ "$DATA_PATH" != "NA" ]; then
        PYTHON_ARGS_WORKER="$PYTHON_ARGS_WORKER --retrain-data-path $DATA_PATH"
    fi
  else
    echo "错误: Worker $i 的模式 '$MODE' 未知。"
    exit 1
  fi

  # 组合成最终命令
  WORKER_CMD_I="$PYTHON_EXE $SCRIPT_NAME $PYTHON_ARGS_SHARED $PYTHON_ARGS_WORKER"
  WORKER_CMDS+=("$WORKER_CMD_I")
done
echo "--------------------------"


# 构建实际执行命令与打印命令
CMD_TO_RUN=""
LOG_CMD_TO_RUN=""
for cmd in "${WORKER_CMDS[@]}"; do
  CMD_TO_RUN+="$cmd & "
  LOG_CMD_TO_RUN+="      $cmd & \n"
done
CMD_TO_RUN+="wait"
LOG_CMD_TO_RUN+="      wait"




# =========================
# 5. nsys 命令与输出目录
# [!!! 核心修改 !!!]
# =========================
# --- 5a. 提取动态名称 ---

# [!! 修改 !!] 使用第一个 worker 的路径作为目录名
# (如果第一个是 "NA", 目录名将是 "NA", 这是 OK 的)
FIRST_DATA_PATH=${DATA_PATHS[0]}
INPUT_BASENAME=$(basename "$FIRST_DATA_PATH")
INPUT_NAME="${INPUT_BASENAME%.*}" # 移除扩展名 (例如 .mp4)

# [!!] 将 *所有* 数组转换为字符串，用于文件名
MODES_STR=$(join_by _ "${MODES[@]}")
MODELS_STR=$(join_by _ "${MODELS[@]}")
CORES_STR=$(join_by _ "${CORE_BINDS[@]}")
PREP_LEVELS_STR=$(join_by _ "${PREP_LEVELS[@]}")
BATCH_SIZES_STR=$(join_by _ "${BATCH_SIZES[@]}")

# [!! 新增 !!] 为文件名创建安全的数据路径字符串
# (提取 basename 并替换 . , / 等)
DATA_PATH_NAMES=()
for p in "${DATA_PATHS[@]}"; do
    b=$(basename "$p") # 获取 "video.mp4" 或 "retrain_dataset_dir"
    b_noext="${b%.*}"  # 移除最后一个 . 扩展名
    b_safe="${b_noext//\//_}" # 替换 / 为 _
    DATA_PATH_NAMES+=("${b_safe//./_}") # 替换 . 为 _
done
DATA_PATHS_STR=$(join_by _ "${DATA_PATH_NAMES[@]}")


# --- 5b. 构建多级目录 ---
# [!! 修改 !!] 使用从 DATA_PATHS[0] 派生的 INPUT_NAME
REPORT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/${INPUT_NAME}/${NUM_WORKERS}_workers"
echo "--- 正在创建输出目录: $REPORT_DIR ---"
mkdir -p "$REPORT_DIR"

# --- 5c. 构建详细的文件名 ---
REPORT_FILENAME="prep-${PREP_DEVICE}_infer-${INFER_DEVICE}_compile-${COMPILE}_amp-${USE_AMP}"
REPORT_FILENAME+="_modes-${MODES_STR}"
REPORT_FILENAME+="_models-${MODELS_STR}"
REPORT_FILENAME+="_cores-${CORES_STR}"
REPORT_FILENAME+="_bs-${BATCH_SIZES_STR}"
# [!! 新增 !!]
REPORT_FILENAME+="_datapaths-${DATA_PATHS_STR}" 

# --- 5d. 最终路径 ---
REPORT_BASENAME="${REPORT_DIR}/${REPORT_FILENAME}"
NSYS_REP_FILE="${REPORT_BASENAME}.nsys-rep"
SQLITE_FILE="${REPORT_BASENAME}.sqlite"

NSYS_CMD="sudo nsys profile -t cuda,nvtx,osrt -o $NSYS_REP_FILE --force-overwrite true"



# =========================
# 6. 启动 NSYS + Workers
# =========================
LAUNCH_CMD="$NSYS_CMD bash -c '$CMD_TO_RUN'"

# --- 执行 ---
echo "  配置: Compile=$COMPILE, AMP=$USE_AMP"
echo "  输出 (rep): $NSYS_REP_FILE"
echo "  正在执行: $NSYS_CMD bash -c \"... ($NUM_WORKERS 个 worker) ...\""
echo "  命令详情:"
echo -e "$LOG_CMD_TO_RUN"
echo "--------------------------"

bash -c "$LAUNCH_CMD" &
NSYS_MAIN_PID=$!

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