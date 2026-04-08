"""
数据集下载脚本
支持下载 UIEB、EUVP 水下图像数据集
"""
import os
import urllib.request
import zipfile
import shutil

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

# 数据集 URLs
DATASET_URLS = {
    'UIEB': {
        'url': 'https://li-chongyi.github.io/proj_benchmark.html',
        'description': 'Underwater Image Enhancement Benchmark - 890张水下图像+清晰参考图',
        'note': '需要访问官网下载'
    },
    'EUVP': {
        'url': 'http://irvlab.cs.umn.edu/resources/euvp-dataset',
        'description': 'Enhancement of Underwater Visual Perception - 成对数据集',
        'note': '需要访问官网下载'
    }
}


def create_download_guide():
    """创建下载指南文件"""
    guide_content = """# 水下图像数据集下载指南

## 数据集介绍

### 1. UIEB (Underwater Image Enhancement Benchmark)
- **图像数量**: 890张水下图像 + 对应清晰参考图
- **用途**: 训练和测试水下图像增强模型
- **下载链接**: https://li-chongyi.github.io/proj_benchmark.html
- **备用下载**: https://opendatalab.org.cn/OpenDataLab/UIEB

### 2. EUVP (Enhancement of Underwater Visual Perception)
- **图像数量**: 成对数据集，包含水下/清晰图像对
- **用途**: 训练成对图像增强模型
- **下载链接**: http://irvlab.cs.umn.edu/resources/euvp-dataset
- **备用下载**: 百度AI Studio - EUVP数据集

### 3. LSUI (Large-Scale Underwater Image)
- **下载链接**: https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html

---

## 下载步骤

### 方法1: 手动下载（推荐）

1. 访问上述下载链接
2. 下载数据集压缩包
3. 解压到对应目录:
   - UIEB → `data/raw/UIEB/`
   - EUVP → `data/raw/EUVP/`

### 方法2: 使用百度网盘（国内加速）

部分数据集提供百度网盘下载，搜索 "UIEB 数据集 百度网盘" 可找到资源。

### 方法3: AI Studio 下载

百度AI Studio提供EUVP数据集下载:
https://aistudio.baidu.com/aistudio/datasetdetail/190159

---

## 数据目录结构

下载后，数据应按以下结构存放:

```
data/
└── raw/
    ├── UIEB/
    │   ├── underwater/          # 水下图像
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   └── ...
    │   └── reference/           # 清晰参考图
    │       ├── 1.jpg
    │       ├── 2.jpg
    │       └── ...
    │
    ├── EUVP/
    │   ├── Paired/              # 成对数据
    │   │   ├── trainA/          # 水下图像
    │   │   └── trainB/          # 清晰图像
    │   └── Unpaired/            # 非成对数据
    │
    └── LSUI/
        └── ...
```

---

## 数据预处理

下载完成后，运行数据预处理脚本:

```bash
python scripts/prepare_data.py
```

这将自动:
1. 裁剪图像到 256×256
2. 划分训练/验证/测试集
3. 保存到 `data/processed/` 目录
"""
    
    guide_path = os.path.join(PROJECT_ROOT, 'data', 'DOWNLOAD_GUIDE.md')
    os.makedirs(os.path.dirname(guide_path), exist_ok=True)
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"下载指南已保存到: {guide_path}")


def print_download_info():
    """打印下载信息"""
    print("="*60)
    print("水下图像数据集下载指南")
    print("="*60)
    print()
    
    for name, info in DATASET_URLS.items():
        print(f"【{name}】")
        print(f"  描述: {info['description']}")
        print(f"  链接: {info['url']}")
        print(f"  说明: {info['note']}")
        print()
    
    print("-"*60)
    print("请访问上述链接手动下载数据集")
    print("下载完成后，解压到 data/raw/ 对应目录")
    print("-"*60)
    
    # 创建下载指南
    create_download_guide()


if __name__ == '__main__':
    print_download_info()
