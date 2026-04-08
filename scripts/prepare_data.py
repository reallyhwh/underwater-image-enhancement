"""
数据预处理脚本
将原始数据集转换为训练所需的格式
"""
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import shutil

# 设置随机种子
random.seed(42)
np.random.seed(42)


def cv2_imread(filepath):
    """解决OpenCV无法读取中文路径的问题"""
    cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2_imwrite(filepath, img):
    """解决OpenCV无法写入中文路径的问题"""
    cv2.imencode('.png', img)[1].tofile(filepath)


def resize_and_crop(image, target_size=256):
    """调整图像大小并裁剪到目标尺寸"""
    h, w = image.shape[:2]
    
    # 等比例缩放
    if h > w:
        new_h = target_size * h // w
        new_w = target_size
    else:
        new_w = target_size * w // h
        new_h = target_size
    
    image = cv2.resize(image, (new_w, new_h))
    
    # 中心裁剪
    h, w = image.shape[:2]
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    
    return image[start_h:start_h+target_size, start_w:start_w+target_size]


def process_uieb(input_dir, output_dir, target_size=256):
    """处理 UIEB 数据集
    
    目录结构:
    input_dir/
        underwater/      # 水下图像
        reference/       # 清晰参考图
    """
    print("Processing UIEB dataset...")
    
    underwater_dir = os.path.join(input_dir, 'underwater')
    reference_dir = os.path.join(input_dir, 'reference')
    
    # 检查目录
    if not os.path.exists(underwater_dir):
        print(f"Warning: {underwater_dir} not found")
        return
    
    # 获取图像列表
    images = [f for f in os.listdir(underwater_dir) 
              if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(images)} images")
    
    # 创建输出目录
    train_input_dir = os.path.join(output_dir, 'train', 'input')
    train_target_dir = os.path.join(output_dir, 'train', 'target')
    val_input_dir = os.path.join(output_dir, 'val', 'input')
    val_target_dir = os.path.join(output_dir, 'val', 'target')
    test_input_dir = os.path.join(output_dir, 'test', 'input')
    test_target_dir = os.path.join(output_dir, 'test', 'target')
    
    for d in [train_input_dir, train_target_dir, val_input_dir, val_target_dir, 
              test_input_dir, test_target_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 划分数据集 (8:1:1)
    random.shuffle(images)
    train_images = images[:int(len(images)*0.8)]
    val_images = images[int(len(images)*0.8):int(len(images)*0.9)]
    test_images = images[int(len(images)*0.9):]
    
    def process_split(image_list, input_subdir, target_subdir):
        for img_name in tqdm(image_list, desc=os.path.basename(input_subdir)):
            # 读取水下图像（使用cv2_imread解决中文路径问题）
            underwater_path = os.path.join(underwater_dir, img_name)
            try:
                underwater_img = cv2.imdecode(np.fromfile(underwater_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error reading {underwater_path}: {e}")
                continue
            
            if underwater_img is None:
                continue
            
            # 查找对应的参考图像
            base_name = os.path.splitext(img_name)[0]
            ref_name = base_name + '.jpg'
            reference_path = os.path.join(reference_dir, ref_name)
            
            if not os.path.exists(reference_path):
                # 尝试其他扩展名
                for ext in ['.png', '.jpeg']:
                    ref_name = base_name + ext
                    reference_path = os.path.join(reference_dir, ref_name)
                    if os.path.exists(reference_path):
                        break
            
            if os.path.exists(reference_path):
                try:
                    reference_img = cv2.imdecode(np.fromfile(reference_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"Error reading {reference_path}: {e}")
                    continue
            else:
                # 如果没有参考图，跳过
                continue
            
            if reference_img is None:
                continue
            
            # 调整大小和裁剪
            try:
                underwater_img = resize_and_crop(underwater_img, target_size)
                reference_img = resize_and_crop(reference_img, target_size)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
            
            # 保存（使用cv2.imencode解决中文路径问题）
            try:
                output_path = os.path.join(input_subdir, img_name)
                cv2.imencode('.png', underwater_img)[1].tofile(output_path)
                
                output_path = os.path.join(target_subdir, img_name)
                cv2.imencode('.png', reference_img)[1].tofile(output_path)
            except Exception as e:
                print(f"Error saving {img_name}: {e}")
    
    # 处理各数据集
    process_split(train_images, train_input_dir, train_target_dir)
    process_split(val_images, val_input_dir, val_target_dir)
    process_split(test_images, test_input_dir, test_target_dir)
    
    print(f"UIEB processing completed!")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")


def process_euvp(input_dir, output_dir, target_size=256):
    """处理 EUVP 数据集
    
    目录结构:
    input_dir/
        Paired/
            trainA/    # 水下图像
            trainB/    # 清晰图像
    """
    print("Processing EUVP dataset...")
    
    trainA_dir = os.path.join(input_dir, 'Paired', 'trainA')
    trainB_dir = os.path.join(input_dir, 'Paired', 'trainB')
    
    if not os.path.exists(trainA_dir):
        print(f"Warning: {trainA_dir} not found")
        return
    
    # 获取图像列表
    images = [f for f in os.listdir(trainA_dir) 
              if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(images)} image pairs")
    
    # 创建输出目录
    train_input_dir = os.path.join(output_dir, 'train', 'input')
    train_target_dir = os.path.join(output_dir, 'train', 'target')
    val_input_dir = os.path.join(output_dir, 'val', 'input')
    val_target_dir = os.path.join(output_dir, 'val', 'target')
    
    for d in [train_input_dir, train_target_dir, val_input_dir, val_target_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 划分数据集 (9:1)
    random.shuffle(images)
    train_images = images[:int(len(images)*0.9)]
    val_images = images[int(len(images)*0.9):]
    
    def process_split(image_list, input_subdir, target_subdir):
        for img_name in tqdm(image_list, desc=os.path.basename(input_subdir)):
            # 读取图像（使用cv2.imdecode解决中文路径问题）
            input_path = os.path.join(trainA_dir, img_name)
            target_path = os.path.join(trainB_dir, img_name)
            
            try:
                input_img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                target_img = cv2.imdecode(np.fromfile(target_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                continue
            
            if input_img is None or target_img is None:
                continue
            
            # 调整大小和裁剪
            try:
                input_img = resize_and_crop(input_img, target_size)
                target_img = resize_and_crop(target_img, target_size)
            except:
                continue
            
            # 保存（使用cv2.imencode解决中文路径问题）
            try:
                output_path = os.path.join(input_subdir, img_name)
                cv2.imencode('.png', input_img)[1].tofile(output_path)
                
                output_path = os.path.join(target_subdir, img_name)
                cv2.imencode('.png', target_img)[1].tofile(output_path)
            except:
                continue
    
    process_split(train_images, train_input_dir, train_target_dir)
    process_split(val_images, val_input_dir, val_target_dir)
    
    print(f"EUVP processing completed!")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")


def merge_datasets(data_dir, output_dir):
    """合并多个数据集"""
    print("Merging datasets...")
    
    # 合并训练集
    train_input_dir = os.path.join(output_dir, 'train', 'input')
    train_target_dir = os.path.join(output_dir, 'train', 'target')
    
    # 遍历所有子目录
    for subdir in ['UIEB', 'EUVP']:
        sub_input = os.path.join(data_dir, subdir, 'train', 'input')
        sub_target = os.path.join(data_dir, subdir, 'train', 'target')
        
        if os.path.exists(sub_input):
            for f in os.listdir(sub_input):
                src = os.path.join(sub_input, f)
                dst = os.path.join(train_input_dir, f)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
        
        if os.path.exists(sub_target):
            for f in os.listdir(sub_target):
                src = os.path.join(sub_target, f)
                dst = os.path.join(train_target_dir, f)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
    
    print("Datasets merged!")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument('--dataset', type=str, default='all', 
                      choices=['UIEB', 'EUVP', 'all'],
                      help='选择要处理的数据集')
    parser.add_argument('--size', type=int, default=256,
                      help='图像目标尺寸')
    
    args = parser.parse_args()
    
    # 路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    if args.dataset in ['UIEB', 'all']:
        uieb_dir = os.path.join(raw_dir, 'UIEB')
        if os.path.exists(uieb_dir):
            process_uieb(uieb_dir, os.path.join(processed_dir, 'UIEB'), args.size)
        else:
            print(f"UIEB dataset not found at {uieb_dir}")
    
    if args.dataset in ['EUVP', 'all']:
        euvp_dir = os.path.join(raw_dir, 'EUVP')
        if os.path.exists(euvp_dir):
            process_euvp(euvp_dir, os.path.join(processed_dir, 'EUVP'), args.size)
        else:
            print(f"EUVP dataset not found at {euvp_dir}")
    
    print("\n数据预处理完成!")
    print(f"处理后的数据保存在: {processed_dir}")


if __name__ == '__main__':
    main()
