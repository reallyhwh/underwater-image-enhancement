"""
评价指标模块 - PSNR, SSIM, UIQM
"""
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr(img1, img2, data_range=1.0):
    """计算峰值信噪比 PSNR
    
    Args:
        img1: 图像1 (numpy array 或 tensor)
        img2: 图像2 (numpy array 或 tensor)
        data_range: 数据范围 (对于归一化到[0,1]的图像为1.0)
        
    Returns:
        psnr值 (dB)
    """
    # 转换为numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # 处理batch维度
    if img1.ndim == 4:
        psnr_values = []
        for i in range(img1.shape[0]):
            p = psnr(img1[i], img2[i], data_range=data_range)
            psnr_values.append(p)
        return np.mean(psnr_values)
    
    return psnr(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, data_range=1.0):
    """计算结构相似度 SSIM
    
    Args:
        img1: 图像1 (numpy array 或 tensor)
        img2: 图像2 (numpy array 或 tensor)
        data_range: 数据范围
        
    Returns:
        ssim值 [0, 1]
    """
    # 转换为numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # 处理batch维度
    if img1.ndim == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            # 转换为 (H, W, C) 格式
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            s = ssim(im1, im2, data_range=data_range, channel_axis=2)
            ssim_values.append(s)
        return np.mean(ssim_values)
    
    # 单张图像
    if img1.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    
    return ssim(img1, img2, data_range=data_range, channel_axis=2)


def calculate_uiqm(image):
    """计算水下图像质量评价指标 UIQM
    
    UIQM = 0.0282 * UICM + 0.2953 * UISM + 0.6763 * UCONM
    
    Args:
        image: 水下图像 (numpy array, 值范围 [0, 1])
        
    Returns:
        uiqm值
    """
    # 转换为numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # 处理维度
    if image.ndim == 4:
        image = image[0]  # 取第一张
    
    if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
    
    # UICM - 色彩对比度测度
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    
    # 计算各通道均值和标准差
    r_mean, g_mean, b_mean = np.mean(R), np.mean(G), np.mean(B)
    r_std, g_std, b_std = np.std(R), np.std(G), np.std(B)
    
    # UICM 简化计算
    uicm = np.sqrt((r_mean - g_mean)**2 + (r_mean - b_mean)**2 + (g_mean - b_mean)**2)
    
    # UISM - 锐度测度 (使用梯度)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Sobel 梯度
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    uism = np.mean(gradient_magnitude)
    
    # UCONM - 对比度测度
    uconm = np.std(gray)
    
    # UIQM 综合
    uiqm = 0.0282 * uicm + 0.2953 * uism + 0.6763 * uconm
    
    return uiqm


def evaluate_image(restored, target):
    """综合评价图像质量
    
    Args:
        restored: 复原图像
        target: 目标清晰图像
        
    Returns:
        包含各指标的字典
    """
    metrics = {
        'psnr': calculate_psnr(restored, target),
        'ssim': calculate_ssim(restored, target),
    }
    
    # UIQM 只对复原图像计算
    metrics['uiqm'] = calculate_uiqm(restored)
    
    return metrics


def evaluate_batch(predictions, targets):
    """批量评价
    
    Args:
        predictions: 预测图像 (B, C, H, W)
        targets: 目标图像 (B, C, H, W)
        
    Returns:
        平均指标字典
    """
    psnr_values = []
    ssim_values = []
    uiqm_values = []
    
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        pred = predictions[i:i+1]
        tgt = targets[i:i+1]
        
        m = evaluate_image(pred, tgt)
        psnr_values.append(m['psnr'])
        ssim_values.append(m['ssim'])
        uiqm_values.append(m['uiqm'])
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'uiqm': np.mean(uiqm_values)
    }
