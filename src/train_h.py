"""
H估计网络训练脚本 - 无数据训练
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.h_estimation_net import HEstimationNet, NoiseGenerator


def train_h_estimation(config):
    """训练H估计网络
    
    核心思路：使用随机噪声训练，网络学习"退化算子的逆变换"
    """
    
    # 设置随机种子
    torch.manual_seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = HEstimationNet(
        in_channels=6,  # 输入: 噪声 + 零图像
        out_channels=3,
        base_channels=64,
        num_blocks=8
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['train']['h_estimation']['lr'])
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 日志目录
    log_dir = os.path.join(config['project_root'], 'logs', 'h_estimation')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 噪声生成器
    noise_gen = NoiseGenerator()
    image_size = config['image']['size']
    
    # 训练参数
    epochs = config['train']['h_estimation']['epochs']
    batch_size = config['train']['h_estimation']['batch_size']
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # 每个epoch多个batch
        num_batches = 100
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx in pbar:
            # 生成随机噪声输入
            noise_input = noise_gen.generate_batch(batch_size, 3, image_size, image_size).to(device)
            
            # 目标：让网络学习恒等映射（输出 ≈ 输入）
            # 这是无数据训练的核心：网络学习"保持"信息的能力
            target = noise_input.clone()
            
            # 组合输入: [noise, zero]
            zero_input = torch.zeros_like(noise_input)
            combined_input = torch.cat([noise_input, zero_input], dim=1)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(combined_input)
            
            # 损失: 输出应该接近输入（恒等映射）
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 平均损失
        avg_loss = epoch_loss / num_batches
        
        # 记录日志
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # 打印
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        # 保存模型
        if (epoch + 1) % 50 == 0:
            save_path = os.path.join(config['project_root'], 'models', 'h_estimation', f'checkpoint_epoch_{epoch+1}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
    
    # 保存最终模型
    final_path = os.path.join(config['project_root'], 'models', 'h_estimation', 'best_model.pth')
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")
    
    writer.close()
    
    return model


if __name__ == '__main__':
    import yaml
    
    # 加载配置
    config_path = 'configs/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 训练
    train_h_estimation(config)
