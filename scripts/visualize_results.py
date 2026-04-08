"""
可视化模型效果 - 对比原图、输出和目标图
"""
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.denoising_net import DenoisingNet
from src.utils.metrics import calculate_psnr, calculate_ssim


def visualize_results():
    """可视化去噪网络效果"""
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = DenoisingNet(in_channels=3, wavelet='db4').to(device)
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'models/denoising/best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # 数据路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_input_dir = os.path.join(project_root, 'data/processed/UIEB/test/input')
    test_target_dir = os.path.join(project_root, 'data/processed/UIEB/test/target')
    
    # 获取测试图像
    image_files = [f for f in os.listdir(test_input_dir) if f.endswith('.png')][:5]
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 创建输出目录
    output_dir = os.path.join(project_root, 'results/visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing {len(image_files)} test images...")
    
    psnr_list = []
    ssim_list = []
    
    for img_name in image_files:
        # 读取输入图像（解决中文路径）
        input_path = os.path.join(test_input_dir, img_name)
        input_img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_pil = Image.fromarray(input_img)
        input_tensor = transform(input_pil).unsqueeze(0).to(device)
        
        # 读取目标图像
        target_path = os.path.join(test_target_dir, img_name)
        target_img = cv2.imdecode(np.fromfile(target_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_pil = Image.fromarray(target_img)
        target_tensor = transform(target_pil).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # 计算指标
        psnr = calculate_psnr(output_tensor, target_tensor)
        ssim = calculate_ssim(output_tensor, target_tensor)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        
        # 转换为图像
        output_np = output_tensor.squeeze().cpu().numpy()
        output_np = np.transpose(output_np, (1, 2, 0))
        output_np = (output_np * 255).astype(np.uint8)
        
        # 创建对比图
        comparison = np.hstack([
            cv2.resize(input_img, (256, 256)),
            output_np,
            cv2.resize(target_img, (256, 256))
        ])
        
        # 添加标签
        comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.putText(comparison, 'Input', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Output', (266, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Target', (522, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f'PSNR: {psnr:.2f}  SSIM: {ssim:.4f}', (10, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 保存
        output_path = os.path.join(output_dir, f'comparison_{img_name}')
        cv2.imencode('.png', comparison)[1].tofile(output_path)
        print(f"  {img_name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    
    # 打印平均指标
    print(f"\n{'='*50}")
    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"{'='*50}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    visualize_results()
