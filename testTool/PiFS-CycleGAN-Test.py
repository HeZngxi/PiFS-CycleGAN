# CTpredictor/FEMTremendous.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import math # 用于计算批次数量
import sys

# 将 CTpredictor 的父目录添加到 Python 路径中，以便能够导入 CTpredictor.networks
# 假设 FEMTremendous.py 和 networks 目录都在 CTpredictor 目录下
# 确保这一行在 import networks 之前
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks import FEMnetwork # 导入 FEMnetwork.py 中定义的模块和函数

def tensor2im(input_image, imtype=np.uint8):
    """
    将一个 PyTorch Tensor 转换成可以显示的 PIL Image。
    针对批量处理和单通道/多通道输出进行优化。
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data.cpu().float()
    else:
        # 如果不是Tensor，假设已经是Numpy数组，直接返回
        return input_image

    # 移除批次维度，如果存在的话。
    # 这里假设我们总是处理一张图片，因为在批量处理循环中我们会逐个提取。
    if image_tensor.ndim == 4: # (B, C, H, W)
        image_tensor = image_tensor[0] # 取批次中的第一张

    image_numpy = image_tensor.numpy() # (C, H, W)
    
    # 如果是单通道图像，将其复制成三通道以方便显示，但最终保存时会转回单通道
    if image_numpy.shape[0] == 1:
        # 如果原始就是灰度，且输出是灰度，不需要转RGB
        # 这里仅是为了内部处理，后续转PIL会根据mode='L'处理
        pass # 不再进行tile操作，由PIL的mode='L'处理
    
    # 将颜色通道从第一个维度移到最后一个维度 (H, W, C)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    
    # 将图像值从 [-1, 1] 范围转换到 [0, 255]
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    
    # 裁剪值到有效范围 [0, 255]
    image_numpy = np.clip(image_numpy, 0, 255)
    
    return image_numpy.astype(imtype)


def batch_process():
    input_nc = 1           # 输入图像通道数，如CT通常为1
    output_nc = 1          # 输出图像通道数，如EUS通常为1
    ngf = 64               # 生成器第一层卷积的滤波器数量
    # 明确指定使用 ResnetGeneratorWithFEM，对应 networks.py 中的 define_G
    # 根据您提供的 define_G，netG='resnet_fem_9blocks' 应该对应 ResnetGeneratorWithFEM
    netG = 'resnet_fem_9blocks' # 这里假设您的 define_G 会识别这个字符串来构建 FEM 生成器
    norm = 'instance'      # 归一化类型
    use_dropout = False    # 是否使用 dropout
    n_blocks = 9           # ResNetBlock 的数量，与您的 FEM 生成器配置对应
    
    # 权重路径
    model_path = './checkpoints/FEM-Less2.pth' # 示例路径
    
    # 输入和输出文件夹路径
    input_dir = r"E:\HZX-experiment\Data\25404-trainCT-1229-Newest" # 您的输入目录
    output_dir = r"E:\HZX-experiment\Data\FakeEUS\FEMLess2\FEM-Less2"
    
    # 支持的图片格式
    img_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')

    # 批量处理大小
    batch_size = 32
    image_size = 256
    # ------------------------------------------

    # 确定设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}")
    if device.type == 'cuda':
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("警告: 未检测到CUDA GPU，或PyTorch未配置GPU支持，将使用CPU进行处理。")
    
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")

    # 2. 加载模型
    print(f"正在加载模型: {model_path}")
    # 使用 FEMnetwork.py 中的 define_G 函数来创建生成器
    model = FEMnetwork.define_G(
        input_nc, output_nc, ngf, netG, norm,
        use_dropout, init_type='normal', init_gain=0.02
    )

    try:
        # 尝试加载权重，如果模型训练时使用了DataParallel，权重会有'module.'前缀
        state_dict = torch.load(model_path, map_location=str(device))
        
        # 检查并移除'module.'前缀
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        
    except FileNotFoundError:
        print(f"错误: 模型文件未找到在 {model_path}")
        return
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return
        
    model.eval() # 设置为评估模式，关闭 dropout 和 BatchNorm 的统计更新
    model.to(device)
    print("模型加载完成!")

    # 3. 定义预处理转换
    transform_list = []
    if input_nc == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.extend([
        transforms.Resize((image_size, image_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # 针对单通道图像
    ])
    transform = transforms.Compose(transform_list)

    # 4. 获取文件列表
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_extensions)]
    if not img_files:
        print(f"未在 {input_dir} 找到任何图像文件。支持的格式: {img_extensions}")
        return

    print(f"找到 {len(img_files)} 张待处理图片")

    # 5. 循环处理（批量）
    with torch.no_grad(): # 在此上下文中不计算梯度，节省内存和计算
        num_batches = math.ceil(len(img_files) / batch_size)

        for i in tqdm(range(num_batches), desc="处理批次进度"):
            batch_start_idx = i * batch_size
            batch_end_idx = min((i + 1) * batch_size, len(img_files))
            
            current_batch_files = img_files[batch_start_idx:batch_end_idx]
            batch_input_tensors = []
            successful_filenames_in_batch = [] # 记录成功加载的文件名

            for filename in current_batch_files:
                img_path = os.path.join(input_dir, filename)
                try:
                    # 读取图像，统一转换为RGB，如果需要单通道再由transform处理
                    raw_img = Image.open(img_path).convert('RGB') 
                    input_tensor = transform(raw_img).unsqueeze(0) # 增加一个批次维度
                    batch_input_tensors.append(input_tensor)
                    successful_filenames_in_batch.append(filename)
                except Exception as e:
                    print(f"\n警告: 处理文件 {filename} 时出错: {e}，跳过该文件。")
                    continue
            
            if not batch_input_tensors:
                # 如果当前批次没有成功加载的图片，跳过
                continue

            # 将列表中的单张图片 tensor 合并成一个批次 tensor
            input_batch = torch.cat(batch_input_tensors, 0).to(device)

            # 模型推理
            output_batch = model(input_batch) # output_batch 形状为 (B, C, H, W)

            # 后处理并保存批次中的每个结果
            for j in range(output_batch.shape[0]):
                # 获取批次中对应的输出 tensor
                single_output_tensor = output_batch[j].unsqueeze(0) # 增加一个批次维度，适配 tensor2im
                
                output_numpy = tensor2im(single_output_tensor)
                
                # 根据 output_nc 保存为灰度图 ('L') 或 RGB ('RGB')
                if output_nc == 1:
                    # tensor2im 现在会返回 (H, W, 1) 或 (H, W)，取第一个通道
                    output_pil = Image.fromarray(output_numpy.squeeze(), mode='L') 
                elif output_nc == 3:
                    output_pil = Image.fromarray(output_numpy)
                else:
                    # 其他情况，例如非标准通道数，可能需要特定处理
                    output_pil = Image.fromarray(output_numpy.squeeze())
                
                original_filename = successful_filenames_in_batch[j]
                save_name = os.path.splitext(original_filename)[0] + "_FakeEUS.png"
                save_path = os.path.join(output_dir, save_name)
                
                output_pil.save(save_path)

    print(f"\n任务完成! 所有结果已保存至: {output_dir}")

if __name__ == '__main__':
    batch_process()