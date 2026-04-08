"""
端到端训练脚本 - H网络和去噪网络联合训练

核心改进：
1. 两个网络串联：输入 → H网络 → 去噪网络 → 输出
2. 整体优化：梯度同时更新两个网络
3. 网络配合：H网络知道输出要给去噪网络用
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

from src.models.h_estimation_net import HEstimationNet
from src.models.denoising_net import DenoisingNet
from src.data.dataset import PairedUnderwaterDataset, get_transforms
from src.utils.metrics import calculate_psnr, calculate_ssim


class SSIMLoss(nn.Module):
    """SSIM损失函数"""

    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        """计算1-SSIM作为损失"""
        ssim_value = calculate_ssim(pred, target)
        return 1 - ssim_value


class EndToEndModel(nn.Module):
    """端到端模型：H网络 + 去噪网络"""

    def __init__(self, h_net, denoise_net):
        super(EndToEndModel, self).__init__()
        self.h_net = h_net
        self.denoise_net = denoise_net

    def forward(self, x):
        """
        端到端前向传播

        Args:
            x: 输入退化图像 (B, 3, H, W)

        Returns:
            output: 最终复原图像 (B, 3, H, W)
            h_output: H网络中间输出 (用于分析)
        """
        # H网络处理
        h_output = self.h_net(x)

        # 去噪网络处理
        output = self.denoise_net(h_output)

        return output, h_output


def train_end_to_end(config):
    """端到端训练"""

    # 设置随机种子
    torch.manual_seed(config["train"]["seed"])
    np.random.seed(config["train"]["seed"])

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建网络
    h_net = HEstimationNet(
        in_channels=3,  # 直接输入退化图像
        out_channels=3,
        base_channels=64,
        num_blocks=8,
    ).to(device)

    denoise_net = DenoisingNet(in_channels=3, wavelet="db4").to(device)

    # 端到端模型
    model = EndToEndModel(h_net, denoise_net)

    # 注意：端到端训练从头开始，不加载预训练权重
    # 因为H网络输入通道数不同（原来是6通道，现在是3通道）
    # 且端到端训练需要两个网络相互配合，从头训练效果更好

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    h_params = sum(p.numel() for p in h_net.parameters())
    d_params = sum(p.numel() for p in denoise_net.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"  - H network: {h_params:,}")
    print(f"  - Denoising network: {d_params:,}")

    # 优化器 - 同时优化两个网络
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["denoising"]["lr"])

    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 损失函数
    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss()

    # 数据加载
    train_transform = get_transforms("train", config["image"]["size"])
    val_transform = get_transforms("val", config["image"]["size"])

    # 训练集
    train_input_dir = os.path.join(
        config["project_root"], "data/processed/UIEB/train/input"
    )
    train_target_dir = os.path.join(
        config["project_root"], "data/processed/UIEB/train/target"
    )

    if os.path.exists(train_input_dir) and os.path.exists(train_target_dir):
        train_dataset = PairedUnderwaterDataset(
            train_input_dir, train_target_dir, train_transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["train"]["denoising"]["batch_size"],
            shuffle=True,
            num_workers=config["train"]["num_workers"],
        )
        print(f"Train dataset: {len(train_dataset)} samples")
    else:
        print("Error: Training data not found!")
        print(f"Expected: {train_input_dir}")
        return None

    # 验证集
    val_input_dir = os.path.join(
        config["project_root"], "data/processed/UIEB/val/input"
    )
    val_target_dir = os.path.join(
        config["project_root"], "data/processed/UIEB/val/target"
    )

    if os.path.exists(val_input_dir) and os.path.exists(val_target_dir):
        val_dataset = PairedUnderwaterDataset(
            val_input_dir, val_target_dir, val_transform
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        print(f"Val dataset: {len(val_dataset)} samples")
    else:
        val_loader = None
        print("Warning: Validation data not found")

    # 日志
    log_dir = os.path.join(config["project_root"], "logs", "end_to_end")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 训练参数
    epochs = config["train"]["denoising"]["epochs"]
    best_psnr = 0

    print(f"\n{'=' * 60}")
    print("Starting End-to-End Training")
    print(f"{'=' * 60}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {config['train']['denoising']['batch_size']}")
    print(f"Learning rate: {config['train']['denoising']['lr']}")
    print(f"{'=' * 60}\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_l1 = 0
        epoch_ssim = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 端到端前向传播
            outputs, h_outputs = model(inputs)

            # 计算损失
            loss_l1 = l1_loss(outputs, targets)
            loss_ssim = ssim_loss(outputs, targets)

            # 总损失：L1 + 0.1*SSIM
            loss = loss_l1 + 0.1 * loss_ssim

            # 反向传播 - 同时更新两个网络
            loss.backward()
            optimizer.step()

            # 记录
            epoch_loss += loss.item()
            epoch_l1 += loss_l1.item()
            epoch_ssim += loss_ssim.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "l1": f"{loss_l1.item():.4f}",
                    "ssim": f"{loss_ssim.item():.4f}",
                }
            )

        # 平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_l1 = epoch_l1 / len(train_loader)
        avg_ssim = epoch_ssim / len(train_loader)

        # 记录日志
        writer.add_scalar("Loss/total", avg_loss, epoch)
        writer.add_scalar("Loss/l1", avg_l1, epoch)
        writer.add_scalar("Loss/ssim", avg_ssim, epoch)

        # 验证
        if val_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            val_psnr = 0
            val_ssim = 0
            num_val = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs, _ = model(inputs)

                    val_psnr += calculate_psnr(outputs, targets)
                    val_ssim += calculate_ssim(outputs, targets)
                    num_val += 1

            val_psnr /= num_val
            val_ssim /= num_val

            writer.add_scalar("Val/PSNR", val_psnr, epoch)
            writer.add_scalar("Val/SSIM", val_ssim, epoch)

            print(f"\nEpoch [{epoch + 1}/{epochs}]")
            print(
                f"  Train Loss: {avg_loss:.6f} (L1: {avg_l1:.6f}, SSIM: {avg_ssim:.6f})"
            )
            print(f"  Val PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}")

            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr

                # 保存完整模型
                save_dir = os.path.join(config["project_root"], "models", "end_to_end")
                os.makedirs(save_dir, exist_ok=True)

                # 保存端到端模型
                torch.save(
                    {
                        "epoch": epoch,
                        "h_net_state_dict": h_net.state_dict(),
                        "denoise_net_state_dict": denoise_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "psnr": val_psnr,
                        "ssim": val_ssim,
                    },
                    os.path.join(save_dir, "best_model.pth"),
                )

                # 分别保存两个网络（方便单独使用）
                torch.save(h_net.state_dict(), os.path.join(save_dir, "h_net_best.pth"))
                torch.save(
                    denoise_net.state_dict(),
                    os.path.join(save_dir, "denoise_net_best.pth"),
                )

                print(f"  Best model saved! PSNR: {best_psnr:.2f} dB")
        else:
            print(f"\nEpoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            save_dir = os.path.join(config["project_root"], "models", "end_to_end")
            torch.save(
                {
                    "epoch": epoch,
                    "h_net_state_dict": h_net.state_dict(),
                    "denoise_net_state_dict": denoise_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

        scheduler.step()

    writer.close()

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(
        f"Model saved to: {os.path.join(config['project_root'], 'models/end_to_end/')}"
    )
    print(f"{'=' * 60}\n")

    return model


def test_end_to_end(config, model_path=None):
    """测试端到端模型"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建网络
    h_net = HEstimationNet(
        in_channels=3, out_channels=3, base_channels=64, num_blocks=8
    ).to(device)
    denoise_net = DenoisingNet(in_channels=3, wavelet="db4").to(device)

    # 加载模型
    if model_path is None:
        model_path = os.path.join(
            config["project_root"], "models/end_to_end/best_model.pth"
        )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        h_net.load_state_dict(checkpoint["h_net_state_dict"])
        denoise_net.load_state_dict(checkpoint["denoise_net_state_dict"])
        print(f"Loaded model from {model_path}")
        if "psnr" in checkpoint:
            print(f"Model PSNR: {checkpoint['psnr']:.2f} dB")
    else:
        print(f"Model not found: {model_path}")
        return

    # 创建端到端模型
    model = EndToEndModel(h_net, denoise_net)
    model.eval()

    # 测试数据
    test_input_dir = os.path.join(
        config["project_root"], "data/processed/UIEB/test/input"
    )
    test_target_dir = os.path.join(
        config["project_root"], "data/processed/UIEB/test/target"
    )

    if not os.path.exists(test_input_dir):
        print(f"Test data not found: {test_input_dir}")
        return

    test_transform = get_transforms("test", config["image"]["size"])
    test_dataset = PairedUnderwaterDataset(
        test_input_dir, test_target_dir, test_transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 测试
    total_psnr = 0
    total_ssim = 0

    print(f"\nTesting on {len(test_dataset)} images...")

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _ = model(inputs)

            total_psnr += calculate_psnr(outputs, targets)
            total_ssim += calculate_ssim(outputs, targets)

    avg_psnr = total_psnr / len(test_dataset)
    avg_ssim = total_ssim / len(test_dataset)

    print(f"\nTest Results:")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")

    return avg_psnr, avg_ssim


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="End-to-End Training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--model", type=str, default=None, help="Path to model checkpoint"
    )
    args = parser.parse_args()

    # 加载配置
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        train_end_to_end(config)
    else:
        test_end_to_end(config, args.model)
