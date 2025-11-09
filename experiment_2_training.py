import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torchvision import models
import torch.cuda.nvtx as nvtx
import time
import os

# --- 1. 设置设备为 CUDA ---
torch.hub.set_dir('models')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备 (训练脚本): {device}")

if device.type == 'cpu':
    print("错误: 未检测到 CUDA! 无法运行此实验。")
    exit()

# --- 2. 定义模型和优化器 ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.to(device)
model.train() # <--- 设置为训练模式
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# --- 3. 定义 *复杂* 的训练预处理流水线 ---
# (故意增加 CPU 负载)
transform_pipeline = T.Compose([
    T.RandomResizedCrop(224),       # <--- CPU 密集
    T.ColorJitter(brightness=0.2),  # <--- CPU 密集
    T.RandomHorizontalFlip(),       # <--- CPU 密集
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# --- 4. 设置 DataLoader ---
try:
    dataset_path = 'datasets/stanford_cars/train'
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到 StanfordCars 训练文件夹: {dataset_path}")
        exit()

    print(f"正在从 ImageFolder 加载数据集 (训练脚本): {dataset_path}")
    dataset = torchvision.datasets.ImageFolder(
        root=dataset_path,
        transform=transform_pipeline  # <--- 应用复杂的变换
    )
    
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,   # <--- 使用批处理
        shuffle=True, 
        num_workers=4    # <--- 使用多进程加载数据
    )
    
    print(f"DataLoader 加载完成。")

except Exception as e:
    print(f"加载 StanfordCars (训练) 时发生错误: {e}")
    exit()


# --- 5. 训练分析循环 ---
NUM_EPOCHS = 5 # 模拟 5 个 epoch 的微调
print(f"开始训练循环... 总共 {NUM_EPOCHS} 个 Epochs。")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch: {epoch+1}/{NUM_EPOCHS}")
    nvtx.range_push(f"epoch_{epoch}")
    
    # 我们只分析前 50 步来模拟
    for step, data in enumerate(trainloader):
        nvtx.range_push(f"train_step_{step}")

        # --- (A) 等待数据 (CPU 密集, 主线程空闲) ---
        nvtx.range_push("wait_for_batch (CPU Preprocessing Wait)")
        # data = next(data_iter) # dataloader 已经帮我们做了
        nvtx.range_pop() # wait_for_batch

        # --- (B) HtoD 拷贝 ---
        nvtx.range_push("data_copy_HtoD (Train)")
        inputs, labels = data[0].to(device=device), data[1].to(device=device)
        nvtx.range_pop() # data_copy_HtoD

        # --- (C) GPU 训练 ---
        nvtx.range_push("train_step (GPU Compute)")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        nvtx.range_pop() # train_step (GPU Compute)
        
        # --- (D) 同步 ---
        nvtx.range_push("cuda_Sync (Train)")
        torch.cuda.synchronize()
        nvtx.range_pop() # cuda_Sync
        
        nvtx.range_pop() # train_step_
        
        if (step + 1) % 10 == 0:
            print(f"  ...step {step+1}, Loss: {loss.item():.4f}")

        # 为演示目的，每个 epoch 只跑 50 步
        if step + 1 >= 50:
            break
            
    nvtx.range_pop() # epoch_

print("="*50)
print("训练脚本分析完成。")
print("="*50)