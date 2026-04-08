"""
小波去噪网络训练脚本 - 需要成对数据集
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

from src.models.denoising_net import DenoisingNet
from src.data.dataset import PairedUnderwaterDataset, get_transforms
from src.utils.metrics import calculate_psnr, calculate_ssim


class SSIMLoss(nn.Module):
    """SSIM损失函数"""
    
    def __init__(self):
        super(SSIMLoss, self).__init__()
        
    def forward(self, pred, target):
        """计算1-SSIM作为损失"""
        from src.utils.metrics import calculate_ssim
        ssim_value = calculate_ssim(pred, target)
        return 1 - ssim_value


def train_denoising(config):
    """训练小波去噪网络"""
    
    # 设置随机种子
    torch.manual_seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = DenoisingNet(
        in_channels=config['image']['channels'],
        wavelet='db4'
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['train']['denoising']['lr'])
    
    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 损失函数 - L1 + SSIM
    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss()
    
    # 数据加载
    train_transform = get_transforms('train', config['image']['size'])
    val_transform = get_transforms('val', config['image']['size'])
    
    # 训练集 (需要提前准备好成对数据)
    train_input_dir = os.path.join(config['project_root'], 'data/processed/UIEB/train/input')
    train_target_dir = os.path.join(config['project_root'], 'data/processed/UIEB/train/target')
    
    # 检查数据目录是否存在
    if os.path.exists(train_input_dir) and os.path.exists(train_target_dir):
        train_dataset = PairedUnderwaterDataset(train_input_dir, train_target_dir, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['train']['denoising']['batch_size'],
            shuffle=True,
            num_workers=config['train']['num_workers']
        )
        print(f"Train dataset: {len(train_dataset)} samples")
    else:
        print("Warning: Training data not found, using synthetic data")
        train_loader = None
    
    # 验证集
    val_input_dir = os.path.join(config['project_root'], 'data/processed/UIEB/val/input')
    val_target_dir = os.path.join(config['project_root'], 'data/processed/UIEB/val/target')
    
    if os.path.exists(val_input_dir) and os.path.exists(val_target_dir):
        val_dataset = PairedUnderwaterDataset(val_input_dir, val_target_dir, val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    else:
        val_loader = None
    
    # 日志
    log_dir = os.path.join(config['project_root'], 'logs', 'denoising')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 训练参数
    epochs = config['train']['denoising']['epochs']
    best_psnr = 0
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        if train_loader is None:
            # 使用合成数据训练
            print("Using synthetic data for training...")
            num_batches = 100
            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
                # 生成合成数据
                batch_size = config['train']['denoising']['batch_size']
                fake_input = torch.randn(batch_size, 3, config['image']['size'], config['image']['size'])
                fake_target = torch.randn(batch_size, 3, config['image']['size'], config['image']['size'])
                
                fake_input = fake_input.to(device)
                fake_target = fake_target.to(device)
                
                optimizer.zero_grad()
                output = model(fake_input)
                
                # 组合损失
                loss = l1_loss(output, fake_target) + 0.1 * ssim_loss(output, fake_target)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        else:
            # 使用真实数据训练
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # 组合损失
                loss = l1_loss(outputs, targets) + 0.1 * ssim_loss(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(train_loader) if train_loader else num_batches)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # 验证
        if val_loader is not None and (epoch + 1) % 10 == 0:
            model.eval()
            val_psnr = 0
            val_ssim = 0
            num_val = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    
                    val_psnr += calculate_psnr(outputs, targets)
                    val_ssim += calculate_ssim(outputs, targets)
                    num_val += 1
            
            val_psnr /= num_val
            val_ssim /= num_val
            
            writer.add_scalar('Val/PSNR', val_psnr, epoch)
            writer.add_scalar('Val/SSIM', val_ssim, epoch)
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}")
            
            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_path = os.path.join(config['project_root'], 'models', 'denoising', 'best_model.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with PSNR: {best_psnr:.2f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        scheduler.step()
    
    writer.close()
    print("Training completed!")
    
    return model


if __name__ == '__main__':
    import yaml
    
    # 加载配置
    config_path = 'configs/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 训练
    train_denoising(config)
