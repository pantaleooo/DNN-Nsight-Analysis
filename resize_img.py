from PIL import Image
import os

def batch_resize_images(input_dir, output_dir, target_width, target_height):
    """
    批量读取一个文件夹中的所有图片，将其缩放到指定尺寸，
    并保存到另一个文件夹中。

    参数:
    input_dir (str): 原始图像的文件夹路径。
    output_dir (str): 保存缩放后图像的文件夹路径。
    target_width (int): 目标宽度。
    target_height (int): 目标高度。
    """
    
    # 1. 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 定义支持的图片文件扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    print(f"开始处理文件夹: {input_dir}")
    print(f"将保存到: {output_dir}")
    
    count_processed = 0
    count_skipped = 0
    
    # 3. 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件扩展名是否在我们的有效列表中
        if filename.lower().endswith(valid_extensions):
            # 4. 构建完整的文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # 5. 读取、缩放和保存 (与上一版代码相同)
                with Image.open(input_path) as img:
                    # 使用高质量的 LANCZOS 滤镜进行缩放
                    resized_img = img.resize((target_width, target_height), Image.LANCZOS)
                    
                    # 确保在保存时转换回 'RGB'（如果原始是 RGBA 等）
                    # 尤其是保存为 JPG 时，它不支持 Alpha 通道
                    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                        if resized_img.mode != 'RGB':
                            resized_img = resized_img.convert('RGB')
                            
                    resized_img.save(output_path)
                    print(f"[处理成功] {filename} -> {output_path}")
                    count_processed += 1
                    
            except Exception as e:
                print(f"[处理失败] {filename} | 错误: {e}")
                count_skipped += 1
        else:
            # print(f"[跳过] {filename} (非图片文件)")
            count_skipped += 1

    print("\n--- 批量处理完成 ---")
    print(f"成功处理图片: {count_processed} 张")
    print(f"跳过文件: {count_skipped} 个")

# --- 如何使用 ---

# 1. 定义你的路径和目标尺寸
input_folder = "datasets/celeba_hq/val/male"   # 替换为你的原始图片文件夹
output_folder = "datasets/celeba_hq_60-70/val/male" # 替换为你希望保存的文件夹
new_width = 60
new_height = 70

# 2. 运行批量处理函数
batch_resize_images(input_folder, output_folder, new_width, new_height)