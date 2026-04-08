"""
推理脚本
单张图像或批量图像复原
"""
import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_restoration_model
from src.utils import load_config


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """加载模型"""
    config = load_config(config_path)
    model = create_restoration_model(config).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"已加载权重: {checkpoint_path}")
    else:
        print("警告: 使用随机初始化权重")

    model.eval()
    return model, config


def process_single_image(model, image_path: str, output_path: str, device: torch.device):
    """处理单张图像"""
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # 预处理
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output, info = model(img_tensor)

    # 后处理
    output_np = output.cpu().numpy()[0].transpose(1, 2, 0)
    output_np = np.clip(output_np, 0, 1)

    # 调整尺寸
    if output_np.shape[:2][::-1] != original_size:
        output_np = cv2.resize(output_np, original_size)

    # 保存
    output_img = (output_np * 255).astype(np.uint8)
    Image.fromarray(output_img).save(output_path)

    print(f"已保存至: {output_path}")
    print(f"迭代次数: {info['iterations']}")


def process_batch(model, input_dir: str, output_dir: str, device: torch.device):
    """批量处理图像"""
    os.makedirs(output_dir, exist_ok=True)

    extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])

    print(f"找到 {len(image_files)} 张图像")

    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f'restored_{filename}')

        print(f"\n处理 [{i+1}/{len(image_files)}]: {filename}")
        process_single_image(model, input_path, output_path, device)


def main():
    parser = argparse.ArgumentParser(description='水下图像复原推理')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output', type=str, required=True, help='输出图像路径或目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model, config = load_model(args.config, args.checkpoint, device)

    # 处理图像
    if os.path.isfile(args.input):
        process_single_image(model, args.input, args.output, device)
    elif os.path.isdir(args.input):
        process_batch(model, args.input, args.output, device)
    else:
        print(f"错误: 输入路径不存在: {args.input}")


if __name__ == '__main__':
    main()
