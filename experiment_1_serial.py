import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torchvision import models
import torch.cuda.nvtx as nvtx
import time
from PIL import Image
import os
import argparse # [!!! 新增 !!!]
import math

def parse_core_list(core_string):
    """
    将 "0-3,7,16" 这样的字符串解析为一个核心 ID 的 set {0, 1, 2, 3, 7, 16}
    """
    core_set = set()
    parts = core_string.split('_')
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                core_set.update(range(start, end + 1))
            except Exception as e:
                print(f"!!! 无法解析核心范围 '{part}': {e}")
        else:
            try:
                core_set.add(int(part))
            except Exception as e:
                print(f"!!! 无法解析核心 ID '{part}': {e}")
    return core_set



# --- [!!! 新增 !!!] ---
# --- 1. ARGPARSE (从 Bash 接收参数) ---
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="运行 DNN 分析实验。")
parser.add_argument('--prep-on-gpu', action='store_true', 
                    help="在 GPU 上运行预处理。")
parser.add_argument('--infer-on-gpu', action='store_true', 
                    help="在 GPU 上运行推理。")
parser.add_argument('--batch-size', type=int, default=1, 
                    help="用于分析的批次大小。")
parser.add_argument('--enable-compile', action='store_true', 
                    help="为推理启用 torch.compile()。")
parser.add_argument('--core-bind', type=str, default="all", 
                    help="要绑定的核心 (例如 '16', '0-3', 'all')。")
args = parser.parse_args()
# ---------------------------------------------------



# ### --- 0. 绑定 CPU 核心 (来自 Python 内部) --- ###
# ---------------------------------------------------
if args.core_bind.lower() != "all":
    try:
        pid = os.getpid()
        core_set = parse_core_list(args.core_bind)
        if not core_set:
            raise ValueError("解析后的核心 set 为空。")
            
        os.sched_setaffinity(pid, core_set)
        print(f"--- 绑核成功 ---")
        print(f"进程 {pid} 已被绑定到 CPU 核心: {core_set}")
        print(f"------------------")
    except OSError as e:
        print(f"!!! 绑核失败 (Cores: {args.core_bind}) !!!")
        print(f"错误: {e}")
        print(f"提示: CPU 亲和性 (Affinity) 是一个 Linux/Unix 特性。")
        print(f"------------------")
    except AttributeError:
        print(f"!!! 绑核失败 (Cores: {args.core_bind}) !!!")
        print(f"os.sched_setaffinity 仅在 Linux/Unix 上可用。")
        print(f"------------------")
else:
    print(f"--- 绑核已跳过 (配置为 'all') ---")


# -----------------------------------------------------------------
# 封装编译和预热逻辑
# [!!! 修改 !!!]: 添加 'enable_compile' 参数
# -----------------------------------------------------------------
def setup_model_for_inference(model, use_cuda_infer, dummy_input_for_warmup, enable_compile=False, warmup_iters=3):
    """
    为推理准备模型：
    1. (如果 enable_compile=True) 应用 torch.compile()
    2. 执行预热运行
    """
    
    # 如果不在 GPU 上推理，则跳过所有优化
    if not use_cuda_infer:
        print("--- 推理在 CPU 上: 跳过 torch.compile() 和 warm-up ---")
        return model # 返回原始模型

    # --- 1. 应用 torch.compile() (如果启用) ---
    if enable_compile:
        print("--- CUDA 推理: 正在启用 torch.compile() ---")
        try:
            model = torch.compile(model)
            print("--- torch.compile() 成功 ---")
        except Exception as e:
            print(f"!!! torch.compile() 失败: {e}")
            print("--- 将回退到 Eager 模式运行 ---")
    else:
        print("--- torch.compile() 已禁用, 在 Eager 模式下运行 ---")
            
    # --- 2. 执行 Warm-up ---
    # (即使在 Eager 模式下，预热对于稳定 GPU 时钟也很有用)
    warmup_type = "JIT 编译/缓存" if enable_compile else "Eager 模式缓存"
    print(f"--- 正在执行 {warmup_iters} 次 WARM-UP 运行 ({warmup_type})... ---")
    
    if dummy_input_for_warmup is None:
        print("!!! 警告: 缺少 WARM-UP 用的 dummy_input. 跳过 warm-up. !!!")
        return model

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input_for_warmup)
        
        # 同步以确保所有编译/预热运行已完成
        torch.cuda.synchronize()
    
    print("--- WARM-UP 完成 ---")
    return model
# -----------------------------------------------------------------


### --- 1. CONFIGURATION --- ###
# 所有变量现在都来自 args
# ----------------------------------
PREP_ON_GPU = args.prep_on_gpu
INFER_ON_GPU = args.infer_on_gpu
BATCH_SIZE = args.batch_size
ENABLE_COMPILE = args.enable_compile # 用于传递给 setup 函数
# ----------------------------------


# [!!! 修改 !!!]: 动态计算 Step 数量
TOTAL_IMAGES_TO_PROFILE = 192
NUM_STEPS_TO_PROFILE = math.ceil(TOTAL_IMAGES_TO_PROFILE / BATCH_SIZE)


### --- 2. SETUP DEVICES & MODEL --- ###

# 2a. 设置推理设备
use_cuda_infer = INFER_ON_GPU and torch.cuda.is_available()
inference_device = torch.device("cuda:0" if use_cuda_infer else "cpu")

if not torch.cuda.is_available() and (PREP_ON_GPU or INFER_ON_GPU):
    print("警告: 未检测到 CUDA! 强制所有操作在 CPU 上运行。")
    PREP_ON_GPU = False
    INFER_ON_GPU = False

print(f"--- 配置 ---")
print(f"批次大小 (Batch Size): {BATCH_SIZE}")
print(f"目标图像数: {TOTAL_IMAGES_TO_PROFILE}")
print(f"计算的 Steps: {NUM_STEPS_TO_PROFILE}")
print(f"预处理将在: {'GPU' if PREP_ON_GPU else 'CPU'}")
print(f"推理将在: {inference_device}")
print(f"torch.compile() 已: {'启用' if ENABLE_COMPILE and INFER_ON_GPU else '禁用'}")
print(f"------------")


# 2b. 设置预处理设备
use_cuda_prep = PREP_ON_GPU 
prep_device = torch.device("cuda:0" if use_cuda_prep else "cpu")

# 2c. 加载并移动模型到目标推理设备
torch.hub.set_dir('models')
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model.to(inference_device)
model.eval()
if use_cuda_infer:
    cudnn.benchmark = True

# 预处理流程 (resize 256, centercrop 224) 决定了 224x224
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224, device=inference_device)

# [!!! 修改 !!!]: 传递 ENABLE_COMPILE 标志
model = setup_model_for_inference(
    model, 
    use_cuda_infer, 
    dummy_input, 
    enable_compile=ENABLE_COMPILE
)


### --- 3. LOAD TRANSFORMS --- ###



### 简单化预处理 ###
# # 将 "Resize(256) -> CenterCrop(224)" 合并为单个 Resize((224, 224))，并使用最快的 NEAREST_NEIGHBOR 插值算法
# resize_crop_op = T.Resize((224, 224), interpolation=Image.NEAREST)
# normalize_op = T.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
# to_tensor_op = T.ToTensor() 


### 中等预处理 ######
# resize_op = T.Resize(256)   
# crop_op = T.CenterCrop(224)
# normalize_op = T.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
# to_tensor_op = T.ToTensor() 



### 复杂预处理 ######
# 将所有 CPU 预处理操作组合 (Compose) 在一起，T.Compose 会按顺序执行它们
_resize_op = T.Resize(256)
_crop_op = T.CenterCrop(224)
_to_tensor_op = T.ToTensor() 
_normalize_op = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
_random_rotation_op = T.RandomRotation(degrees=15)
_gaussian_blur_op = T.GaussianBlur(kernel_size=3)
cpu_prep_pipeline = T.Compose([
    _resize_op,
    _crop_op,
    _random_rotation_op,  # <-- 新增的重度计算
    _gaussian_blur_op,    # <-- 新增的重度计算
    _to_tensor_op,
    _normalize_op
])
#################################################

to_rgb_op = T.Lambda(lambda img: img.convert('RGB'))
pil_to_tensor_op = T.PILToTensor() 





### --- 4. LOAD IMAGE PATHS --- ###
# (无变化)
try:
    dataset_path = 'datasets/celeba_hq/train/' # 示例路径
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=None)
    image_paths = [path for path, _ in dataset.samples]
    num_images = len(image_paths)
    if num_images == 0: raise FileNotFoundError("目录中没有图像")
    print(f"从 {dataset_path} 加载了 {num_images} 张图像路径。")
except Exception as e:
    print(f"加载数据集路径时发生错误: {e}")
    exit()



### --- 5. ANALYSIS LOOP --- ###
# (无变化)
print(f"开始分析循环... 总共 {NUM_STEPS_TO_PROFILE} 个批次。")
start_time = time.time()

for step in range(NUM_STEPS_TO_PROFILE):
    nvtx.range_push(f"step_batch_{step}")

    if not PREP_ON_GPU:
        # ---
        # [SCENARIO 1/3]: 预处理在 CPU 上运行
        # ---
        nvtx.range_push("A_Preprocessing_Batch (CPU)")
        batch_tensors_cpu = []
        for i in range(BATCH_SIZE):
            img_idx = (step * BATCH_SIZE + i) % num_images
            img_path = image_paths[img_idx]
            img = Image.open(img_path)
            img = to_rgb_op(img)

            # # 简单预处理
            # img = resize_op(img)
            # img = crop_op(img)
            # tensor = to_tensor_op(img) # PIL -> [C, H, W] float
            # tensor = normalize_op(tensor)
            
            # # 中等预处理
            # img = resize_op(img)
            # img = crop_op(img)
            # tensor = to_tensor_op(img) # PIL -> [C, H, W] float
            # tensor = normalize_op(tensor)

            # # 复杂预处理
            tensor = cpu_prep_pipeline(img)


            batch_tensors_cpu.append(tensor)


        nvtx.range_push("Stack_Batch_CPU")
        inputs_on_cpu = torch.stack(batch_tensors_cpu)
        nvtx.range_pop() # Stack_Batch_CPU
        nvtx.range_pop() # A_Preprocessing_Batch
        
        # ---
        # [HtoD Transfer]: 传输【已处理】的小张量
        # ---
        nvtx.range_push("B_data_copy_HtoD (Processed Batch)")
        inputs_on_device = inputs_on_cpu.to(inference_device, non_blocking=True)
        nvtx.range_pop() # B_data_copy

    else:
        # ---
        # [SCENARIO 2/4]: 预处理在 GPU 上运行
        # ---
        nvtx.range_push("A_Preprocessing_CPU (Decode)")
        batch_tensors_cpu_uint8 = []
        for i in range(BATCH_SIZE):
            img_idx = (step * BATCH_SIZE + i) % num_images
            img_path = image_paths[img_idx]
            
            img = Image.open(img_path)
            img = to_rgb_op(img)
            # 关键: 只转为 uint8 Tensor, 不做变换
            tensor_cpu_uint8 = pil_to_tensor_op(img) 
            batch_tensors_cpu_uint8.append(tensor_cpu_uint8)

        nvtx.range_push("Stack_Batch_CPU")
        inputs_cpu_uint8 = torch.stack(batch_tensors_cpu_uint8)
        nvtx.range_pop() # Stack_Batch_CPU
        nvtx.range_pop() # A_Preprocessing_CPU
        
        # ---
        # [HtoD Transfer]: 传输【原始】的大张量 (uint8)
        # ---
        nvtx.range_push("B_data_copy_HtoD (RAW Image Batch)")
        inputs_gpu_uint8 = inputs_cpu_uint8.to(prep_device, non_blocking=True)
        nvtx.range_pop() # B_data_copy
        
        # ---
        # [GPU PREP]: 在 GPU 上执行变换
        # ---
        nvtx.range_push("C_Preprocessing_GPU (Transforms)")
        inputs_gpu_float = inputs_gpu_uint8.to(dtype=torch.float32) / 255.0
        inputs_gpu_float = resize_op(inputs_gpu_float)
        inputs_gpu_float = crop_op(inputs_gpu_float)
        inputs_gpu_final = normalize_op(inputs_gpu_float)
        
        # 确保数据在正确的设备上进行推理
        inputs_on_device = inputs_gpu_final.to(inference_device, non_blocking=True)
        nvtx.range_pop() # C_Preprocessing_GPU
        
    # ---
    # [INFERENCE]: 在 inference_device (CPU 或 GPU) 上运行
    # ---
    with torch.no_grad():
        nvtx.range_push("D_inference_Batch")
        outputs = model(inputs_on_device)
        nvtx.range_pop() # D_inference
    
    # ---
    # [SYNC]: 如果此 step 中使用了 CUDA，则同步
    # ---
    nvtx.range_push("E_Sync")
    if PREP_ON_GPU or INFER_ON_GPU:
        torch.cuda.synchronize()
    nvtx.range_pop() # E_Sync
    
    nvtx.range_pop() # step_batch_N
    
    if (step + 1) % 10 == 0:
        print(f"Analyzed batch {step+1}/{NUM_STEPS_TO_PROFILE}")


### --- 6. RESULTS --- ###
# (无变化)
if PREP_ON_GPU or INFER_ON_GPU:
    torch.cuda.synchronize()
end_time = time.time()

# [!!! 修改 !!!]: 这里的计算现在将反映实际处理的图像
# (例如, BATCH_SIZE=16, steps=13, total=208)
total_images_processed = NUM_STEPS_TO_PROFILE * BATCH_SIZE
total_time_taken = end_time - start_time
fps = total_images_processed / total_time_taken

print("="*50)
print("分析完成。")
print(f"总耗时 ({NUM_STEPS_TO_PROFILE} 批): {total_time_taken:.4f} 秒")
print(f"总处理图像: {total_images_processed} 张")
print(f"平均吞吐量: {fps:.2f} 帧/秒 (FPS)")
print("="*50)



# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.utils.data
# import torchvision
# import torchvision.transforms as T
# from torchvision import models
# import torch.cuda.nvtx as nvtx
# import time
# from PIL import Image
# import os

# ### --- 0. 绑定 CPU 核心 --- ###
# # ---
# # 指定你希望这个脚本运行在哪个 CPU 核心上
# # 你的 i9-10900X 有 10 个核心 (0-9)
# # 让我们选择一个核心，例如 5
# CORE_ID_TO_USE = {16} # 这是一个包含核心 ID 的集合
# # ---
# try:
#     pid = os.getpid()
#     os.sched_setaffinity(pid, CORE_ID_TO_USE)
#     print(f"--- 绑核成功 ---")
#     print(f"进程 {pid} 已被绑定到 CPU 核心: {CORE_ID_TO_USE}")
#     print(f"------------------")
# except OSError as e:
#     print(f"!!! 绑核失败 !!!")
#     print(f"错误: {e}")
#     print(f"提示: CPU 亲和性 (Affinity) 是一个 Linux/Unix 特性。")
#     print(f"------------------")
# except AttributeError:
#     print(f"!!! 绑核失败 !!!")
#     print(f"os.sched_setaffinity 仅在 Linux/Unix 上可用。")
#     print(f"------------------")




# # -----------------------------------------------------------------
# # 封装编译和预热逻辑
# # -----------------------------------------------------------------
# def setup_model_for_inference(model, use_cuda_infer, dummy_input_for_warmup, warmup_iters=3):
#     """
#     为推理准备模型：
#     1. (如果 use_cuda_infer=True) 应用 torch.compile()
#     2. (如果 use_cuda_infer=True) 执行预热运行
#     """
    
#     # 如果不在 GPU 上推理，则跳过所有优化
#     if not use_cuda_infer:
#         print("--- 推理在 CPU 上: 跳过 torch.compile() 和 warm-up ---")
#         return model # 返回原始模型

#     # --- 1. 应用 torch.compile() ---
#     print("--- CUDA 推理: 正在启用 torch.compile() ---")
#     try:
#         # JIT 编译模型以优化性能
#         # 这会融合算子并减少 CPU 启动开销
#         model = torch.compile(model)
#         print("--- torch.compile() 成功 ---")
#     except Exception as e:
#         print(f"!!! torch.compile() 失败: {e}")
#         print("--- 将回退到 Eager 模式运行 ---")
#         # 注意：即使编译失败，我们仍然可以预热 Eager 模式
        
#     # --- 2. 执行 Warm-up ---
#     print(f"--- 正在执行 {warmup_iters} 次 WARM-UP 运行 (用于 JIT 编译/缓存)... ---")
#     if dummy_input_for_warmup is None:
#         print("!!! 警告: 缺少 WARM-UP 用的 dummy_input. 跳过 warm-up. !!!")
#         return model

#     with torch.no_grad():
#         for _ in range(warmup_iters):
#             _ = model(dummy_input_for_warmup)
        
#         # 同步以确保所有编译/预热运行已完成
#         torch.cuda.synchronize()
    
#     print("--- WARM-UP 完成 ---")
#     return model
# # -----------------------------------------------------------------







# ### --- 1. CONFIGURATION --- ###
# # ---
# # 切换这些标志来进行 2x2 实验
# # ---
# # 实验 1: CPU 预处理 + GPU 推理 (你之前的 "CPU 瓶颈" 脚本)
# # PREP_ON_GPU = False
# # INFER_ON_GPU = True

# # 实验 2: GPU 预处理 + GPU 推理 (你之前的 "PCIe 瓶颈" 脚本)
# # PREP_ON_GPU = True
# # INFER_ON_GPU = True

# # 实验 3: CPU 预处理 + CPU 推理 (CPU 满载)
# # PREP_ON_GPU = False
# # INFER_ON_GPU = False

# # 实验 4: GPU 预处理 + CPU 推理 (HtoD -> DtoH 传输)
# # PREP_ON_GPU = True
# # INFER_ON_GPU = False

# # ---
# # 批处理和步数配置
# # ---
# # (我强烈建议使用 >= 32 的 BATCH_SIZE 来观察吞吐量瓶颈)
# BATCH_SIZE = 1
# NUM_STEPS_TO_PROFILE = 50 



# ### --- 2. SETUP DEVICES & MODEL --- ###

# # 2a. 设置推理设备
# # 根据 INFER_ON_GPU 标志决定模型在 CPU 还是 GPU 上
# use_cuda_infer = INFER_ON_GPU and torch.cuda.is_available()
# inference_device = torch.device("cuda:0" if use_cuda_infer else "cpu")

# if not torch.cuda.is_available() and (PREP_ON_GPU or INFER_ON_GPU):
#     print("警告: 未检测到 CUDA! 强制所有操作在 CPU 上运行。")
#     PREP_ON_GPU = False
#     INFER_ON_GPU = False

# print(f"--- 配置 ---")
# print(f"批次大小 (Batch Size): {BATCH_SIZE}")
# print(f"预处理将在: {'GPU' if PREP_ON_GPU else 'CPU'}")
# print(f"推理将在: {inference_device}")
# print(f"------------")


# # 2b. 设置预处理设备
# # (这只在 PREP_ON_GPU = True 时有意义)
# use_cuda_prep = PREP_ON_GPU # 依赖于上面的 CUDA 检查
# prep_device = torch.device("cuda:0" if use_cuda_prep else "cpu")

# # 2c. 加载并移动模型到目标推理设备
# torch.hub.set_dir('models')
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# model.to(inference_device)
# model.eval()
# if use_cuda_infer:
#     cudnn.benchmark = True

# # 预处理流程 (resize 256, centercrop 224) 决定了 224x224
# dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224, device=inference_device)
# model = setup_model_for_inference(model, use_cuda_infer, dummy_input)




# ### --- 3. LOAD TRANSFORMS --- ###
# # 定义所有需要的
# # 它们可以处理 CPU (PIL) 或 GPU (Tensor) 输入
# resize_op = T.Resize(256)
# crop_op = T.CenterCrop(224)
# normalize_op = T.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
# to_rgb_op = T.Lambda(lambda img: img.convert('RGB'))
# pil_to_tensor_op = T.PILToTensor() # 仅将 PIL 转为 uint8 Tensor
# to_tensor_op = T.ToTensor() # PIL -> float Tensor (含 /255)


# ### --- 4. LOAD IMAGE PATHS --- ###
# # (假设你已下载并解压 'celeba_hq' 到 'datasets/celeba_hq/train')
# try:
#     dataset_path = 'datasets/celeba_hq/train/' # 示例路径
#     dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=None)
#     image_paths = [path for path, _ in dataset.samples]
#     num_images = len(image_paths)
#     if num_images == 0: raise FileNotFoundError("目录中没有图像")
#     print(f"从 {dataset_path} 加载了 {num_images} 张图像路径。")
# except Exception as e:
#     print(f"加载数据集路径时发生错误: {e}")
#     exit()



# ### --- 5. ANALYSIS LOOP --- ###
# print(f"开始分析循环... 总共 {NUM_STEPS_TO_PROFILE} 个批次。")
# start_time = time.time()

# for step in range(NUM_STEPS_TO_PROFILE):
#     nvtx.range_push(f"step_batch_{step}")

#     if not PREP_ON_GPU:
#         # ---
#         # [SCENARIO 1/3]: 预处理在 CPU 上运行
#         # ---
#         nvtx.range_push("A_Preprocessing_Batch (CPU)")
#         batch_tensors_cpu = []
#         for i in range(BATCH_SIZE):
#             img_idx = (step * BATCH_SIZE + i) % num_images
#             img_path = image_paths[img_idx]
            
#             img = Image.open(img_path)
#             img = to_rgb_op(img)
#             img = resize_op(img)
#             img = crop_op(img)
#             tensor = to_tensor_op(img) # PIL -> [C, H, W] float
#             tensor = normalize_op(tensor)
#             batch_tensors_cpu.append(tensor)

#         nvtx.range_push("Stack_Batch_CPU")
#         inputs_on_cpu = torch.stack(batch_tensors_cpu)
#         nvtx.range_pop() # Stack_Batch_CPU
#         nvtx.range_pop() # A_Preprocessing_Batch
        
#         # ---
#         # [HtoD Transfer]: 传输【已处理】的小张量
#         # (如果推理也在 CPU，这将是一个 CPU-to-CPU 拷贝, 很快)
#         # ---
#         nvtx.range_push("B_data_copy_HtoD (Processed Batch)")
#         inputs_on_device = inputs_on_cpu.to(inference_device, non_blocking=True)
#         nvtx.range_pop() # B_data_copy

#     else:
#         # ---
#         # [SCENARIO 2/4]: 预处理在 GPU 上运行
#         # ---
#         nvtx.range_push("A_Preprocessing_CPU (Decode)")
#         batch_tensors_cpu_uint8 = []
#         for i in range(BATCH_SIZE):
#             img_idx = (step * BATCH_SIZE + i) % num_images
#             img_path = image_paths[img_idx]
            
#             img = Image.open(img_path)
#             img = to_rgb_op(img)
#             # 关键: 只转为 uint8 Tensor, 不做变换
#             tensor_cpu_uint8 = pil_to_tensor_op(img) 
#             batch_tensors_cpu_uint8.append(tensor_cpu_uint8)

#         nvtx.range_push("Stack_Batch_CPU")
#         inputs_cpu_uint8 = torch.stack(batch_tensors_cpu_uint8)
#         nvtx.range_pop() # Stack_Batch_CPU
#         nvtx.range_pop() # A_Preprocessing_CPU
        
#         # ---
#         # [HtoD Transfer]: 传输【原始】的大张量 (uint8)
#         # ---
#         nvtx.range_push("B_data_copy_HtoD (RAW Image Batch)")
#         inputs_gpu_uint8 = inputs_cpu_uint8.to(prep_device, non_blocking=True)
#         nvtx.range_pop() # B_data_copy
        
#         # ---
#         # [GPU PREP]: 在 GPU 上执行变换
#         # ---
#         nvtx.range_push("C_Preprocessing_GPU (Transforms)")
#         inputs_gpu_float = inputs_gpu_uint8.to(dtype=torch.float32) / 255.0
#         inputs_gpu_float = resize_op(inputs_gpu_float)
#         inputs_gpu_float = crop_op(inputs_gpu_float)
#         inputs_gpu_final = normalize_op(inputs_gpu_float)
        
#         # 确保数据在正确的设备上进行推理
#         # (如果 Prep 在 GPU, Infer 在 CPU, 这会触发 DtoH 传输)
#         inputs_on_device = inputs_gpu_final.to(inference_device, non_blocking=True)
#         nvtx.range_pop() # C_Preprocessing_GPU
        
#     # ---
#     # [INFERENCE]: 在 inference_device (CPU 或 GPU) 上运行
#     # ---
#     with torch.no_grad():
#         nvtx.range_push("D_inference_Batch")
#         outputs = model(inputs_on_device)
#         nvtx.range_pop() # D_inference
    
#     # ---
#     # [SYNC]: 如果此 step 中使用了 CUDA，则同步
#     # ---
#     nvtx.range_push("E_Sync")
#     if PREP_ON_GPU or INFER_ON_GPU:
#         # 任何一个在 GPU 上运行都需要同步以获得准确的 NVTX 范围
#         torch.cuda.synchronize()
#     nvtx.range_pop() # E_Sync
    
#     nvtx.range_pop() # step_batch_N
    
#     if (step + 1) % 10 == 0:
#         print(f"Analyzed batch {step+1}/{NUM_STEPS_TO_PROFILE}")


# ### --- 6. RESULTS --- ###
# if PREP_ON_GPU or INFER_ON_GPU:
#     torch.cuda.synchronize()
# end_time = time.time()

# total_images_processed = NUM_STEPS_TO_PROFILE * BATCH_SIZE
# total_time_taken = end_time - start_time
# fps = total_images_processed / total_time_taken

# print("="*50)
# print("分析完成。")
# print(f"总耗时 ({NUM_STEPS_TO_PROFILE} 批): {total_time_taken:.4f} 秒")
# print(f"总处理图像: {total_images_processed} 张")
# print(f"平均吞吐量: {fps:.2f} 帧/秒 (FPS)")
# print("="*50)