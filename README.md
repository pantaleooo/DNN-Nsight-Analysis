# 🚀 视频流 DNN 性能分析框架 📊

本项目是一个为 PyTorch 设计的、用于深度神经网络 (DNN) 模型在视频流上进行推理或训练的性能分析框架。它深度集成了 NVIDIA Nsight Systems (nsys) 和 NVTX 标记，允许对复杂的视频处理流水线（从数据加载、预处理到模型执行）进行逐帧的瓶颈分析。

---

## ✨ 核心功能

### 📈 深度性能剖析
- 在 `worker.py`, `models.py`, `data_pipeline.py` 中深度集成了 `torch.cuda.nvtx` 标记
- 可与 `nsys` 结合使用，精确定位 GPU/CPU 瓶颈

### ⛓️ 混合流水线
- 支持灵活地将预处理 (`--prep-on-gpu`) 和模型执行 (`--infer-on-gpu`) 分配到 CPU 或 GPU 运行
- 测试不同硬件配置下的性能表现

### ⚡ 现代 PyTorch 支持
- **JIT 编译**: 支持 `torch.compile()` (`--enable-compile`) 以优化推理速度
- **自动混合精度**: 支持 `torch.cuda.amp` (`--use-amp`) 以加速 GPU 运算

### 🎯 CPU 亲和性
- 支持将 Worker 进程绑定到特定的 CPU 核心 (`--core-bind`)
- 减少多进程间的资源抢占，确保分析结果的稳定性（见 `utils.py`）

### 🔄 双模式运行
- **inference**: 纯推理性能分析
- **retrain**: 模拟轻量级"在线学习"或"微调"场景，分析包含反向传播的性能

### 📊 指标导出
- 自动将每个批次的详细延迟（读取、预处理、执行）和损失（训练模式下）记录到唯一的 CSV 文件中
- 便于后续分析（见 `utils.MetricsCollector`）

### 🔬 实验性功能
- 包含 `SplitModelWrapper`（见 `models.py`）
- 用于实验性地将模型（如 ResNet）在 CPU 和 GPU 之间进行跨层分割

### 🛠️ 辅助工具
- 额外提供 `frame_extract/` 目录
- 用于从视频中提取帧 (`frame_extractor.py`) 并使用高精度模型进行目标检测标注 (`image_annotator.py`)

---

## 📂 项目结构

```
.  
├── frame_extract/  
│   ├── frame_extractor.py      # 工具：从视频中按指定 FPS 提取帧  
│   ├── frame_extractor.sh      # 示例：如何使用 frame_extractor  
│   ├── image_annotator.py      # 工具：使用检测模型(如FasterRCNN)标注图像  
│   └── image_annotator.sh      # 示例：如何使用 image_annotator  
│  
├── config.yaml                 # (示例) 用于多 Worker 启动器配置  
├── worker.py                   # 核心入口：单个 DNN 分析 Worker  
├── config.py                   # 负责解析 worker.py 的所有命令行参数  
├── data_pipeline.py            # 负责视频 I/O 和复杂的预处理流水线  
├── models.py                   # 负责加载模型、编译、微调设置和跨层分割  
├── utils.py                    # 负责日志、CPU绑核、系统信息和指标收集  
├── README.md                   # (本项目文档)  
└── requirements.txt            # (项目依赖)
```

---

## ⚙️ 安装

### 1. 克隆本项目
```bash
git clone [your-repo-url]  
cd [your-repo-name]
```

### 2. 安装 Python 依赖
建议在虚拟环境中安装：
```bash
pip install -r requirements.txt
```

### 3. (可选) 安装性能剖析工具
确保已安装 NVIDIA Nsight Systems (`nsys`)，以便进行深度性能剖析。

---

## 🚀 使用方法

本项目主要通过 `worker.py` 脚本运行。`config.yaml` 和 `run_multi_model.sh`（在旧 README 中提到）是用于编排**多个** `worker.py` 实例的，但单个 Worker 也可以独立运行。

### 1. 运行单个 Worker (基础)

`worker.py` 接受大量参数以控制其行为。

#### 关键参数（来自 `config.py`）

| 参数 | 描述 |
|------|------|
| `--model-name` | 要加载的模型（例如 `vit_b_16`, `resnet50`） |
| `--input_path` | 视频源（文件路径, 摄像头 ID '0', 或 'rtsp://...' 流） |
| `--mode` | inference（默认）或 retrain |
| `--batch-size` | 批次大小 |
| `--prep-level` | 预处理复杂度（simple, medium, complex） |
| `--prep-on-gpu` | 在 GPU 上运行预处理 |
| `--infer-on-gpu` | 在 GPU 上运行模型（如果为 False，则在 CPU 运行） |
| `--core-bind` | 绑定的 CPU 核心（例如 0-3, all） |
| `--enable-compile` | （仅推理）启用 `torch.compile()` |
| `--use-amp` | 启用自动混合精度 |
| `--max-steps` | 限制运行的总批次数 |
| `--split-point` | （实验性）跨层分割点，格式 `model_name:layer_name`（例如 `resnet50:layer3`） |

#### 示例：在 GPU 上运行 ResNet50 推理并启用编译

```bash
python worker.py \
    --model-name "resnet50" \
    --input_path "video.mp4" \
    --batch-size 16 \
    --prep-level "medium" \
    --core-bind "4-7" \
    --infer-on-gpu \
    --prep-on-gpu \
    --enable-compile \
    --use-amp \
    --max-steps 500
```

### 2. 运行多 Worker 与性能剖析 (高级)

`config.yaml` 文件定义了如何启动多个 Worker。您需要一个封装脚本（例如 `run_multi_model.sh`，此文件未提供）来解析此 YAML 并启动 `nsys` 和多个 `worker.py` 进程。

#### 使用 `nsys` 进行剖析（概念）

```bash
# 这是一个概念性示例，展示了如何启动 nsys  
# 您的 run_multi_model.sh 脚本可能会自动执行此操作  
nsys profile \
    --output "profile_result/my_profile" \
    --trace=cuda,nvtx,osrt,cpu \
    --force-overwrite=true \
    python worker.py [上面的参数...]
```

这将生成一个 `.qdrep` 文件，您可以使用 Nsight Systems GUI 打开它，查看带有 NVTX 标记的详细时间线。

### 3. 使用辅助工具 (`frame_extract/`)

这些是独立的工具，用于准备数据或进行标注。

#### 示例 1: 提取视频帧

```bash
# 从视频中每 2 秒提取 1 帧 (fps=0.5)，保存为 png  
python frame_extract/frame_extractor.py \
    -i "video.mp4" \
    -o "output_frames" \
    --fps 0.5 \
    --prefix "shot_" \
    --format "png"
```

#### 示例 2: 标注图像（对象检测）

```bash
# 使用 FasterR-CNN 标注 'output_frames' 目录中的所有图像  
python frame_extract/image_annotator.py \
    -i "output_frames" \
    -o "annotated_frames" \
    --model_name "fasterrcnn_resnet50_fpn_v2" \
    --threshold 0.7
```

---

## 🔬 核心组件详解

### `data_pipeline.py`

#### 主要功能
- **initialize_video_capture**: 智能处理视频文件、摄像头（如 '0'）或 RTSP/HTTP 流
- **read_batch_from_video**: 使用 `cv2.VideoCapture` 高效读取帧，并（在 retrain 模式下）生成虚拟标签
- **build_preprocessing_pipelines**: 根据 `prep_level` (simple, medium, complex) 构建 `torchvision.transforms` 组合
- **preprocess_batch**: 关键函数，根据 `prep_on_gpu` 标志决定是在 CPU 上执行 `cpu_pipeline`，还是将原始张量传输到 GPU (`prep_device`) 并执行 `gpu_prep_ops`

### `models.py`

#### 主要功能
- **load_model**: 从 `torchvision.models` 加载预训练模型
- **setup_model_for_inference**: 负责模型 `eval()`、`torch.compile()` 和使用 `dummy_input` 进行预热 (Warm-up)
- **setup_retrain_components**: 负责模型 `train()`、冻结骨干网络、替换分类头 (`model.fc` 或 `model.heads`) 以进行微调
- **SplitModelWrapper**: （实验性）一个 `nn.Module` 包装器，它将模型分为 `part1` (CPU) 和 `part2` (GPU)，并在 `forward` 过程中自动处理张量在设备间的传输

### `utils.py`

#### 主要功能
- **setup_cpu_affinity**: 在 Linux 上使用 `os.sched_setaffinity` 将进程绑定到特定 CPU 核心，并通过 `torch.set_num_threads` 控制 Pytorch 的线程数
- **MetricsCollector**: 一个类，用于在内存中累积每一步的性能指标（字典），并在程序结束时调用 `export_to_csv` 将其持久化
- **print_final_results**: 计算并打印总运行时间、总帧数和平均 FPS

---

## 📊 结果分析

运行 `worker.py` 后，您将获得：

### 1. 控制台输出
- **启动时的系统信息** (`log_system_info`)
- **详细的配置信息**
- **运行时的进度日志**
- **结束时的平均吞吐量 (FPS) 总结**

### 2. CSV 指标文件
- 将生成一个名为 `M_{model_name}_P_{pid}_metrics.csv` 的文件
- **列名**: `step`, `read_latency_ms`, `prep_latency_ms`, `exec_latency_ms`, `total_step_latency_ms`, `loss`
- 该文件可用于在 Excel 或 Python (Pandas) 中进行详细的延迟分析

### 3. Nsight 报告 (.qdrep)（如果使用 `nsys` 运行）
- 用于在 Nsight Systems GUI 中进行最深入的瓶颈分析
- 查看 CUDA kernel、NVTX 范围和 CPU/GPU 交互

---

## 📝 注意事项

1. **性能测试建议**:
   - 在进行正式性能测试前，建议先运行几次以确保系统稳定
   - 对于 GPU 测试，确保其他进程不会占用过多 GPU 资源

2. **Nsight 使用**:
   - 确保 `nsys` 版本与您的 CUDA 版本兼容
   - 大型模型可能需要增加系统内存以避免 OOM 错误

3. **CPU 绑核**:
   - `--core-bind` 参数仅在 Linux 系统上有效
   - 建议为每个 Worker 分配独立的 CPU 核心集

4. **实验性功能**:
   - `SplitModelWrapper` 仍处于实验阶段，性能可能不稳定
   - 跨层分割可能会引入额外的设备间通信开销

---

## 🤝 贡献

欢迎提交 issues 和 pull requests 来改进这个项目。

---

## 📄 许可证
本项目基于 **MIT 许可证** 开源，您可以自由使用、修改和分发本项目的代码，前提是遵守以下条款：

1. 在所有副本或衍生作品中保留原始的版权声明和许可证文本；
2. 作者和项目贡献者不对本项目的使用提供任何担保，不承担因使用本项目导致的任何责任。

完整的许可证文本可查看：[MIT License](https://opensource.org/licenses/MIT)


