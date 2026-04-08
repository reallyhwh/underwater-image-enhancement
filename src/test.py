"""
测试脚本 - 评估模型效果
"""
import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.h_estimation_net import HEstimationNet
from src.models.denoising_net import DenoisingNet
from src.models.admm_framework import ADMMFramework
from src.utils.metrics import calculate_psnr, calculate_ssim, calculate_uiqm
from src.data.dataset import get_transforms


def load_models(config, device):
    """加载训练好的模型"""
    
    # H估计网络
    h_net = HEstimationNet(
        in_channels=6,
        out_channels=3,
        base_channels=64,
        num_blocks=8
    ).to(device)
    
    h_path = os.path.join(config['project_root'], config['model']['h_estimation'])
    if os.path.exists(h_path):
        h_net.load_state_dict(torch.load(h_path, map_location=device))
        print(f"Loaded H estimation model from {h_path}")
    else:
        print(f"Warning: H estimation model not found at {h_path}")
    
    # 去噪网络
    denoise_net = DenoisingNet(
        in_channels=config['image']['channels'],
        wavelet='db4'
    ).to(device)
    
    denoise_path = os.path.join(config['project_root'], config['model']['denoising'])
    if os.path.exists(denoise_path):
        denoise_net.load_state_dict(torch.load(denoise_path, map_location=device))
        print(f"Loaded denoising model from {denoise_path}")
    else:
        print(f"Warning: Denoising model not found at {denoise_path}")
    
    return h_net, denoise_net


def test_single_image(image_path, admm_framework, device, transform):
    """测试单张图像"""
    
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    # 预处理
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # 复原
    with torch.no_grad():
        restored = admm_framework.restore(image)
    
    return image, restored


def test_model(config):
    """测试模型"""
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    h_net, denoise_net = load_models(config, device)
    
    # 创建ADMM框架
    admm_framework = ADMMFramework(h_net, denoise_net, config)
    admm_framework.h_net.eval()
    admm_framework.denoise_net.eval()
    
    # 测试数据目录
    test_dir = os.path.join(config['project_root'], 'data/processed/test')
    input_dir = os.path.join(test_dir, 'input')
    target_dir = os.path.join(test_dir, 'target')
    
    # 转换
    transform = get_transforms('test', config['image']['size'])
    
    # 如果有目标图像，进行定量评估
    if os.path.exists(input_dir) and os.path.exists(target_dir):
        print("Running quantitative evaluation...")
        
        psnr_values = []
        ssim_values = []
        uiqm_values = []
        
        # 获取测试图像列表
        from src.data.dataset import PairedUnderwaterDataset
        test_dataset = PairedUnderwaterDataset(input_dir, target_dir, transform)
        
        for idx in tqdm(range(len(test_dataset)), desc="Testing"):
            input_img, target_img = test_dataset[idx]
            input_img = input_img.unsqueeze(0).to(device)
            target_img = target_img.unsqueeze(0).to(device)
            
            # 复原
            with torch.no_grad():
                restored = admm_framework.restore(input_img)
            
            # 评估
            psnr = calculate_psnr(restored, target_img)
            ssim = calculate_ssim(restored, target_img)
            uiqm = calculate_uiqm(restored)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            uiqm_values.append(uiqm)
        
        # 打印结果
        print("\n" + "="*50)
        print("Test Results:")
        print(f"  PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"  SSIM: {np.mean(ssim_values):.4f}")
        print(f"  UIQM: {np.mean(uiqm_values):.4f}")
        print("="*50)
        
        # 检查是否达到目标
        if np.mean(psnr_values) >= config['eval']['psnr_threshold']:
            print(f"✓ PSNR target ({config['eval']['psnr_threshold']} dB) achieved!")
        if np.mean(ssim_values) >= config['eval']['ssim_threshold']:
            print(f"✓ SSIM target ({config['eval']['ssim_threshold']}) achieved!")
    
    else:
        print("Test data not found. Please prepare test images in data/processed/test/input/")
        print("Running demo on sample image...")
        
        # 演示模式：处理单张图像
        demo_image = os.path.join(config['project_root'], 'data/raw/test.jpg')
        if os.path.exists(demo_image):
            from PIL import Image
            
            image = cv2.imread(demo_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transform(image)
            image = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                restored = admm_framework.restore(image)
            
            # 保存结果
            result_dir = os.path.join(config['project_root'], 'results/test/images')
            os.makedirs(result_dir, exist_ok=True)
            
            # 反归一化
            restored_np = restored.squeeze().cpu().numpy()
            restored_np = np.transpose(restored_np, (1, 2, 0))
            restored_np = (restored_np * 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(result_dir, 'restored.jpg'), 
                       cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR))
            
            print(f"Result saved to {result_dir}/restored.jpg")
        else:
            print("No demo image found.")


if __name__ == '__main__':
    import yaml
    from PIL import Image
    
    # 加载配置
    config_path = 'configs/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 测试
    test_model(config)
