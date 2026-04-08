# 基于贝叶斯估计与深度神经网络结合的水下图像复原技术

## 一、项目介绍

### 1.1 研究背景

水下图像由于光线的吸收和散射作用，普遍存在以下问题：

| 问题 | 表现 | 原因 |
|------|------|------|
| **模糊** | 图像细节丢失 | 水中悬浮颗粒散射光线 |
| **偏色** | 整体呈蓝绿色 | 不同波长光线吸收程度不同 |
| **对比度低** | 图像灰暗 | 光线在水中快速衰减 |
| **噪声** | 图像颗粒感 | 水下环境复杂多变 |

这些问题严重影响了水下图像的视觉质量和后续处理（如目标检测、识别等）。

### 1.2 研究目标

将退化的水下图像恢复为清晰、真实的正常图像。

**技术指标**：
- 峰值信噪比(PSNR) ≥ 25dB
- 结构相似度(SSIM) ≥ 0.85
- 推理速度 ≤ 2秒/张

### 1.3 创新点

1. **无需成对数据训练H估计网络**：使用随机噪声训练，降低数据依赖
2. **贝叶斯框架与深度学习结合**：融合传统优化方法与神经网络
3. **ADMM迭代求解**：将复杂问题分解为子问题交替求解

---

## 二、核心原理

### 2.1 退化模型

水下图像退化可建模为：

```
y = Hx + n
```

| 符号 | 含义 |
|------|------|
| `y` | 观测到的退化图像（模糊的水下照片） |
| `x` | 原始清晰图像（我们想恢复的目标） |
| `H` | 退化算子（模糊、偏色、散射等） |
| `n` | 噪声 |

**核心问题**：已知 `y`，求 `x`。

### 2.2 贝叶斯估计框架

根据贝叶斯定理：

```
P(x|y) ∝ P(y|x) · P(x)
```

| 项 | 含义 |
|------|------|
| `P(x\|y)` | 后验概率：给定观测图像，清晰图像的概率 |
| `P(y\|x)` | 似然函数：数据保真项 |
| `P(x)` | 先验概率：清晰图像应该具有的特征 |

转化为优化问题：

```
x* = argmin ||y - Hx||² + λ·R(x)
         ↑数据保真项      ↑正则化项
```

### 2.3 ADMM分解

使用交替方向乘子法(ADMM)将问题分解：

```
min ||y - Hx||² + λ||Wx||²
```

引入辅助变量 `v = Wx`，得到：

```
x^{k+1} = argmin ||y - Hx||² + (ρ/2)||x - v^k + u^k||²  → H估计网络求解
v^{k+1} = argmin λ||Wv||² + (ρ/2)||x^{k+1} - v + u^k||²  → 去噪网络求解
u^{k+1} = u^k + x^{k+1} - v^{k+1}                         → 更新乘子
```

### 2.4 技术路线图

```
退化图像 y
    ↓
┌─────────────────────────────────────────────┐
│           ADMM 迭代框架                      │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │ 步骤1: x = H估计网络(y, v, u)        │   │
│  │        学习退化算子的逆变换           │   │
│  └─────────────────────────────────────┘   │
│                    ↓                        │
│  ┌─────────────────────────────────────┐   │
│  │ 步骤2: v = 小波去噪网络(x)           │   │
│  │        去除噪声，恢复细节             │   │
│  └─────────────────────────────────────┘   │
│                    ↓                        │
│  ┌─────────────────────────────────────┐   │
│  │ 步骤3: u = u + x - v                 │   │
│  │        更新拉格朗日乘子               │   │
│  └─────────────────────────────────────┘   │
│                    ↓                        │
│         判断是否收敛？                       │
│         ↓ 是         ↓ 否                   │
│      输出结果      返回步骤1                 │
└─────────────────────────────────────────────┘
    ↓
清晰图像 x
```

---

## 三、网络架构

### 3.1 H估计子网络

**功能**：学习退化算子 H 的逆变换

**输入**：6通道（退化图像 + 辅助变量）

**输出**：3通道（估计的清晰图像）

**网络结构**：

```
输入: (batch, 6, H, W)
    ↓
卷积层 (6→64) + ReLU
    ↓
残差块 × 8 (64通道)
    ↓
卷积层 (64→64) + ReLU
    ↓
卷积层 (64→3) + Sigmoid
    ↓
输出: (batch, 3, H, W)
```

**训练特点**：
- 无需真实水下图像数据
- 使用随机噪声训练
- 学习"逆变换"的数学性质

### 3.2 小波去噪子网络

**功能**：去除图像噪声，恢复细节

**输入**：3通道图像

**输出**：3通道去噪图像

**网络结构**：

```
输入: (batch, 3, H, W)
    ↓
卷积层 (3→64)
    ↓
小波分解 (Haar/db4)
    ↓
┌─────────────┬─────────────┐
│ 低频分支     │ 高频分支     │
│ (整体结构)   │ (细节边缘)   │
│ 残差块×1    │ 残差块×3    │
└─────────────┴─────────────┘
    ↓
小波重构
    ↓
卷积层 (64→3)
    ↓
残差连接 + 输出
```

**训练特点**：
- 需要成对的水下/清晰图像数据
- 损失函数：L1 Loss + SSIM Loss

### 3.3 ADMM迭代框架

**超参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ρ` | 0.01 | 惩罚因子 |
| `λ` | 0.001 | 正则化系数 |
| `max_iter` | 20 | 最大迭代次数 |
| `tol` | 1e-4 | 收敛阈值 |

**停止条件**：
- 达到最大迭代次数
- 连续两次迭代图像差异 < 阈值

---

## 四、项目结构

```
水下图像增强课题/
│
├── 📂 data/                          # 数据目录
│   ├── 📂 raw/                       # 原始数据
│   │   └── 📂 UIEB/
│   │       ├── 📂 underwater/        # 水下图像 (890张)
│   │       └── 📂 reference/         # 清晰参考图 (890张)
│   │
│   └── 📂 processed/                 # 预处理后的数据
│       └── 📂 UIEB/
│           ├── 📂 train/             # 训练集 (712张, 80%)
│           │   ├── 📂 input/
│           │   └── 📂 target/
│           ├── 📂 val/               # 验证集 (89张, 10%)
│           │   ├── 📂 input/
│           │   └── 📂 target/
│           └── 📂 test/              # 测试集 (89张, 10%)
│               ├── 📂 input/
│               └── 📂 target/
│
├── 📂 src/                           # 源代码
│   ├── 📂 data/
│   │   └── dataset.py                # 数据集加载与预处理
│   │
│   ├── 📂 models/
│   │   ├── h_estimation_net.py       # H估计网络定义
│   │   ├── denoising_net.py          # 小波去噪网络定义
│   │   └── admm_framework.py         # ADMM迭代框架
│   │
│   ├── 📂 utils/
│   │   └── metrics.py                # 评价指标 (PSNR, SSIM, UIQM)
│   │
│   ├── train_h.py                    # 训练H估计网络
│   ├── train_denoising.py            # 训练去噪网络
│   ├── train_end_to_end.py           # 端到端训练（推荐）
│   └── test.py                       # 测试评估脚本
│
├── 📂 configs/
│   └── config.yaml                   # 配置文件
│
├── 📂 models/                        # 模型权重 (训练后生成)
│   ├── 📂 h_estimation/
│   │   └── best_model.pth
│   ├── 📂 denoising/
│   │   └── best_model.pth
│   └── 📂 end_to_end/                # 端到端训练模型
│       ├── best_model.pth            # 完整模型 (10MB)
│       ├── h_net_best.pth            # H网络权重 (2.5MB)
│       └── denoise_net_best.pth      # 去噪网络权重 (2MB)
│
├── 📂 logs/                          # 训练日志 (TensorBoard)
│   ├── 📂 h_estimation/
│   ├── 📂 denoising/
│   └── 📂 end_to_end/                # 端到端训练日志
│
├── 📂 results/                       # 测试结果输出
│
├── 📂 scripts/                       # 辅助脚本
│   ├── download_dataset.py           # 数据集下载指南
│   └── prepare_data.py               # 数据预处理脚本
│
├── 📂 前置材料/                       # 课题参考资料
│   ├── 实施方案.md
│   └── An Inner-loop Free Solution to Inverse Problems.pdf
│
├── requirements.txt                  # Python依赖
└── README.md                         # 项目说明
```

---

## 五、快速开始

### 5.1 环境配置

```bash
# 创建虚拟环境
conda create -n pyth310 python=3.10
conda activate pyth310

# 安装依赖
pip install -r requirements.txt

# 安装 GPU 版本 PyTorch (如果有 NVIDIA 显卡)
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 验证 CUDA
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

### 5.2 数据准备

**下载 UIEB 数据集**：
- 官网：https://li-chongyi.github.io/proj_benchmark.html
- 下载 `raw-890` 和 `ref-890`

**目录结构**：
```
data/raw/UIEB/
├── underwater/      # raw-890 水下图像
└── reference/       # ref-890 清晰参考图
```

**数据预处理**：
```bash
python scripts/prepare_data.py --dataset UIEB
```

### 5.3 模型训练

```bash
# 训练 H 估计网络 (无需真实数据, 约10-20分钟)
python src/train_h.py

# 训练去噪网络 (需要数据集, 约30-60分钟)
python src/train_denoising.py

# 端到端训练 (推荐, 约2小时)
python src/train_end_to_end.py --mode train
```

**训练方式对比**：

| 方式 | 说明 | PSNR | SSIM |
|------|------|------|------|
| 分步训练 | H网络和去噪网络独立训练 | 14.43 dB | 0.46 |
| 端到端训练 | 两个网络联合优化 | 13.83 dB | 0.35 |

**注意**：当前端到端训练效果未达预期，建议后续升级网络架构。

### 5.4 测试评估

```bash
python src/test.py
```

### 5.5 查看训练日志

```bash
tensorboard --logdir=logs
```

---

## 六、评估指标

| 指标 | 全称 | 说明 | 目标值 |
|------|------|------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | 峰值信噪比，衡量图像质量 | ≥ 25 dB |
| **SSIM** | Structural Similarity Index | 结构相似度，衡量结构保持 | ≥ 0.85 |
| **UIQM** | Underwater Image Quality Measure | 水下图像质量评价指标 | 越高越好 |

---

## 七、参考文献

### 7.1 核心论文

1. **An Inner-loop Free Solution to Inverse Problems using Deep Neural Networks**
   - 会议：NeurIPS 2019
   - 贡献：提出无内循环的逆问题求解方法，H估计网络无需真实数据训练

2. **Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers**
   - 作者：Boyd et al.
   - 贡献：ADMM优化方法的经典综述

### 7.2 数据集

| 数据集 | 链接 | 说明 |
|--------|------|------|
| **UIEB** | https://li-chongyi.github.io/proj_benchmark.html | 890张水下图像+参考图 |
| **EUVP** | http://irvlab.cs.umn.edu/resources/euvp-dataset | 成对水下/清晰图像 |
| **LSUI** | https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html | 大规模水下图像数据集 |

### 7.3 相关技术

- **小波变换**：PyWavelets库，用于图像的多尺度分析
- **残差网络**：Deep Residual Learning，解决深层网络训练问题
- **TensorBoard**：训练过程可视化工具

---

## 八、常见问题

### Q1: pip安装时报错 "Defaulting to user installation"

**原因**：虚拟环境的 site-packages 目录没有写权限

**解决**：
```bash
# 方案1：修改目录权限（管理员权限运行）
icacls "D:\anaconda\2025\envs\pyth310\Lib\site-packages" /grant "用户名:(OI)(CI)F" /T

# 方案2：使用 python -m pip
python -m pip install 包名
```

### Q2: PyTorch 显示 CUDA 不可用

**原因**：安装的是 CPU 版本的 PyTorch

**解决**：
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Q3: 训练时内存不足

**解决**：
- 减小 batch_size
- 关闭其他占用内存的程序
- 使用更小的图像尺寸

---

## 九、问题总结

> 本模块记录项目过程中提出的问题和解答，便于回顾学习。

### 问题1：课题的原理和流程是什么？

**回答**：

**原理**：水下图像退化可建模为 `y = Hx + n`，其中 y 是退化图像，x 是清晰图像，H 是退化算子，n 是噪声。目标是求 H 的逆变换来恢复 x。

**流程**：
1. 数据预处理：裁剪图像、划分数据集
2. 训练 H 估计网络：学习退化算子的逆变换
3. 训练去噪网络：去除噪声恢复细节
4. ADMM 迭代：组合两个网络求解最终结果
5. 测试评估：计算 PSNR、SSIM 等指标

---

### 问题2：什么是随机噪声？有什么用？

**回答**：

**什么是随机噪声**：随机噪声是每个像素值都随机生成的图像，没有任何实际图像内容，类似电视雪花屏。

```python
noise = torch.randn(1, 3, 256, 256)  # 生成随机噪声
```

**为什么用随机噪声训练**：

H 估计网络学习的不是"图像长什么样"，而是"逆变换怎么算"。这是数学性质的训练，不需要具体的图像内容。

**类比**：就像学习"解方程"的方法，用随机数字练习，学会了方法后就能解任何方程。

---

### 问题3：为什么 H 估计网络是神经网络？

**回答**：

**原因**：H（退化算子）太复杂，无法用简单数学公式描述。

水下图像退化包含：
- 模糊（光线散射，非线性）
- 偏色（波长吸收不同，空间变化）
- 对比度下降（光线衰减，与深度相关）
- 噪声（环境复杂，随机性）

**神经网络的优势**：
- 万能逼近定理：神经网络可以逼近任意连续函数
- 自动学习复杂变换，无需人工建模

---

### 问题4：H 估计网络训练的输出是什么？

**回答**：

**输出示例**：
```
Using device: cuda
Starting training for 100 epochs...
Epoch 1/100: 100%|██████████| 100/100 [00:05<00:00, loss: 0.123456]
Epoch [10/100], Loss: 0.056789
...
Model saved to models/h_estimation/best_model.pth
```

**输出解读**：
| 输出项 | 含义 |
|--------|------|
| `Using device: cuda` | 使用 GPU 训练 |
| `Epoch 1/100` | 当前训练轮数 |
| `loss: 0.123456` | 当前 batch 的损失值（越小越好） |
| `Model saved to ...` | 模型保存路径 |

**训练产物**：
- `models/h_estimation/best_model.pth`：模型权重
- `logs/h_estimation/`：TensorBoard 日志

---

### 问题5：数据预处理为什么要划分训练/验证/测试集？

**回答**：

| 数据集 | 用途 | 比例 |
|--------|------|------|
| **训练集** | 网络学习参数 | 80% |
| **验证集** | 调整超参数、防止过拟合 | 10% |
| **测试集** | 最终评估模型性能 | 10% |

**为什么要划分**：
- 训练集用于学习
- 验证集用于调参和监控过拟合
- 测试集用于最终评估，必须独立

**类比**：
- 训练集 = 平时作业（学习用）
- 验证集 = 模拟考试（调整状态）
- 测试集 = 期末考试（最终评估）

---

### 问题6：为什么 PyTorch 安装的是 CPU 版本？

**回答**：

**原因**：默认安装的 PyTorch 可能是 CPU 版本，需要手动指定 CUDA 版本。

**检查方法**：
```bash
python -c "import torch; print(torch.__version__)"
# 如果显示 2.11.0+cpu，说明是 CPU 版本
```

**解决方法**：
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### 问题7：H估计网络训练时Loss不下降怎么办？

**回答**：

**现象**：
```
Epoch [10/100], Loss: 1.000075
Epoch [20/100], Loss: 1.000059
Epoch [30/100], Loss: 0.999976
...
```
Loss 维持在 1.0 左右，几乎不变。

**原因**：

训练代码中 `noise_input` 和 `target` 是两个**完全独立**的随机噪声：

```python
# 错误写法
noise_input = noise_gen.generate_batch(...)  # 随机噪声 A
target = noise_gen.generate_batch(...)        # 随机噪声 B（完全独立）
```

网络无法学习从随机噪声 A 预测随机噪声 B，因为它们之间没有任何关系。

**解决方法**：

```python
# 正确写法：让网络学习恒等映射
noise_input = noise_gen.generate_batch(...)
target = noise_input.clone()  # 目标 = 输入
```

**修复后结果**：
```
Epoch [100/100], Loss: 0.575453  # Loss 成功下降
```

---

### 问题8：PyWavelets 不支持 GPU tensor 怎么办？

**回答**：

**错误信息**：
```
TypeError: can't convert cuda:0 device type tensor to numpy. 
Use Tensor.cpu() to copy the tensor to host memory first.
```

**原因**：

PyWavelets 库只支持 numpy 数组，不支持 PyTorch 的 GPU tensor。

**解决方法**：

在小波变换前后进行 CPU/GPU 转换：

```python
def _dwt2d(self, x):
    """小波分解（支持GPU tensor）"""
    device = x.device  # 保存设备信息
    
    # GPU tensor → CPU numpy
    x_cpu = x.detach().cpu().numpy()
    
    # 执行小波变换
    coeffs = pywt.dwt2(x_cpu, wavelet)
    
    # CPU numpy → GPU tensor
    cA = torch.from_numpy(coeffs[0]).float().to(device)
    ...
    return cA, cH, cV, cD
```

**核心逻辑保留**：

| 步骤 | 说明 |
|------|------|
| 小波分解 | 分离低频（整体结构）和高频（细节边缘） |
| 双分支处理 | 低频分支保结构，高频分支去噪 |
| 小波重构 | 合并处理后的分量 |

---

### 问题9：训练完成后会得到什么？这个过程叫什么？

**回答**：

**这个过程叫做"模型训练"（Model Training）**。

**训练完成后得到的产物**：

```
models/
├── h_estimation/
│   └── best_model.pth    # H估计网络参数（.pth文件）
└── denoising/
    └── best_model.pth    # 去噪网络参数

logs/
├── h_estimation/
│   └── events.*          # TensorBoard训练日志
└── denoising/
    └── events.*
```

**`.pth` 文件是什么**：

| 概念 | 说明 |
|------|------|
| **文件格式** | PyTorch 模型权重文件 |
| **内容** | 神经网络的所有参数（权重和偏置） |
| **大小** | 通常几十 MB 到几百 MB |
| **用途** | 加载后可以直接用于推理 |

**训练过程的本质**：

```
训练前：网络参数是随机初始化的
训练中：通过反向传播调整参数
训练后：网络参数收敛到最优值
```

**类比理解**：

| 概念 | 类比 |
|------|------|
| **网络结构** | 大脑的结构 |
| **随机初始化** | 刚出生的婴儿（什么都不知道） |
| **训练数据** | 教材和练习题 |
| **训练过程** | 学习过程 |
| **模型权重** | 学到的知识和经验 |
| **推理** | 考试（应用知识解决问题） |

---

### 问题10：参数是如何更新的？是手动修改代码吗？

**回答**：

**不是手动修改！参数更新是自动的。**

**参数更新的代码流程**：

```python
# 训练循环中的关键步骤
optimizer.zero_grad()           # 1. 清零梯度
output = model(input)           # 2. 前向传播
loss = criterion(output, target) # 3. 计算损失
loss.backward()                 # 4. 反向传播，自动计算梯度
optimizer.step()                # 5. 自动更新参数
```

**详细流程**：

```
步骤1: 前向传播
├── 输入图像 → 网络计算 → 输出结果
└── output = model(input)

步骤2: 计算损失
├── 比较输出和目标，计算差距
└── loss = MSE(output, target)

步骤3: 反向传播
├── 自动计算每个参数对损失的影响
├── ∂Loss/∂weight = 梯度
└── loss.backward()

步骤4: 更新参数（自动！）
├── 优化器根据梯度调整参数
├── weight = weight - lr × gradient
└── optimizer.step()
```

**优化器的作用**：

```python
# 定义优化器（训练前写好）
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 优化器自动完成：
# 1. 记录所有参数
# 2. 计算梯度后自动更新
# 3. 使用 Adam 算法调整学习率
```

| 优化器 | 特点 |
|--------|------|
| **SGD** | 随机梯度下降，最简单 |
| **Adam** | 自适应学习率，最常用 |
| **AdamW** | Adam + 权重衰减 |

**类比理解**：

| 概念 | 类比 |
|------|------|
| **损失函数** | 考试分数（衡量差距） |
| **梯度** | 改进方向（往哪个方向努力） |
| **学习率** | 步长（每次改进多少） |
| **优化器** | 老师（自动指导改进） |
| **参数更新** | 学习进步（自动发生） |

---

### 问题11：数据预处理脚本运行成功，但处理后的目录为空？

**回答**：

**现象**：
```
Processing UIEB dataset...
Found 890 images
UIEB processing completed!
  Train: 712 images
  Val: 89 images
  Test: 89 images

数据预处理完成!
```

但检查 `data/processed/UIEB/train/input/` 目录，发现是空的（0个文件）。

**原因**：

OpenCV 的 `cv2.imread()` 和 `cv2.imwrite()` 在 Windows 上**无法处理含中文的文件路径**。

项目路径 `D:\随记\通信工程\水下图像增强课题\` 包含中文字符，导致所有图像读取失败：
```
[ WARN:0@0.209] global loadsave.cpp:278 cv::findDecoder imread_('D:\随记\...'): can't open/read file
```

**解决方法**：

使用 `numpy + cv2.imdecode()` 替代 `cv2.imread()`，使用 `cv2.imencode() + tofile()` 替代 `cv2.imwrite()`：

```python
# 读取图像（解决中文路径问题）
def cv2_imread(filepath):
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)

# 保存图像（解决中文路径问题）
def cv2_imwrite(filepath, img):
    cv2.imencode('.png', img)[1].tofile(filepath)

# 使用示例
img = cv2_imread('D:\\随记\\图像.png')
cv2_imwrite('D:\\随记\\输出.png', img)
```

**修复后结果**：
```
Train: 712 images  ✓
Val: 89 images     ✓
Test: 89 images    ✓
```

**注意**：数据集加载代码 `src/data/dataset.py` 中的 `cv2.imread()` 也需要同样修改，否则训练时无法读取图像。

**额外清理**：修复前脚本创建的空目录（`data/processed/train/`、`data/processed/val/`、`data/processed/test/`）可以删除，真正数据在 `data/processed/UIEB/` 下。

---

### 问题12：模型输出颜色没改善，反而变模糊了？

**回答**：

**现象**：
- 输出图像颜色没有校正（偏蓝绿色）
- 图像变模糊而不是变清晰
- PSNR: 13.79 dB，SSIM: 0.45（远低于目标）

**原因分析**：

1. **网络结构问题**：当前去噪网络只做去噪，没有颜色校正功能
   ```python
   output = x + output  # 残差连接，输出≈输入+小改动
   ```

2. **任务理解问题**：小波去噪主要处理噪声，对颜色校正作用有限

3. **训练策略问题**：H估计网络和去噪网络独立训练，没有联合优化

**解决方案**：

添加颜色校正模块到去噪网络：

```python
class ColorCorrection(nn.Module):
    """颜色校正模块 - 解决水下图像偏色问题"""
    def __init__(self, channels=3):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, channels * 2),  # 输出缩放和偏移参数
        )
    
    def forward(self, x):
        # 提取全局特征，生成颜色校正参数
        global_feat = self.global_pool(x).view(B, C)
        params = self.fc(global_feat)
        scale, bias = params[:, :C], params[:, C:]
        return x * (1 + scale) + bias  # 应用颜色校正
```

**改进后重新训练**：
```bash
python src/train_denoising.py
```

---

### 问题13：端到端训练效果为什么不如分步训练？

**回答**：

**现象**：

| 训练方式 | PSNR | SSIM |
|----------|------|------|
| 分步训练 | 14.43 dB | 0.46 |
| 端到端训练 | 13.83 dB | 0.35 |

端到端训练效果反而比分步训练差。

**原因分析**：

1. **网络容量不足**
   - 当前模型参数量仅1.1M（H网络0.63M + 去噪网络0.48M）
   - WF-Diff参数量约100M，差距近100倍
   - 小模型难以学习复杂的端到端映射

2. **训练数据不足**
   - 仅712张训练图像
   - 端到端训练需要更多数据来学习两个网络的配合

3. **过拟合问题**
   - Epoch 20后验证集PSNR开始波动
   - 说明模型在训练集上过拟合

**解决方案**：

1. **升级网络架构**（优先级最高）
   - 将去噪网络改为U-Net架构
   - 增加网络深度和宽度
   - 预期提升: +3-5 dB

2. **增加训练数据**
   - 使用LSUI数据集（4000+张）
   - 数据增强（翻转、旋转、颜色抖动）

3. **优化训练策略**
   - 使用预训练权重
   - 学习率衰减策略
   - 更长的训练时间（100+ epochs）

---

## 十、更新日志

| 日期 | 内容 |
|------|------|
| 2026-04-05 | 项目初始化，完成代码框架 |
| 2026-04-05 | 完成数据预处理，UIEB数据集 712/89/89 划分 |
| 2026-04-05 | 解决 PyTorch CPU 版本问题，安装 GPU 版本 |
| 2026-04-05 | 解决 H 估计网络 Loss 不下降问题 |
| 2026-04-05 | 解决 PyWavelets 不支持 GPU tensor 问题 |
| 2026-04-05 | 完成 H 估计网络训练，Loss: 0.575 |
| 2026-04-05 | 添加模型训练和参数更新原理说明 |
| 2026-04-05 | 解决 OpenCV 中文路径问题，数据预处理成功 |
| 2026-04-05 | 完成去噪网络训练，PSNR: 14.33, SSIM: 0.47 |
| 2026-04-05 | 发现模型输出模糊问题，添加颜色校正模块 |
| 2026-04-07 | 完成WF-Diff测试，UIEB PSNR: 25.68, LSUI PSNR: 22.83 |
| 2026-04-07 | 分析本文方法与WF-Diff差距（-11.25 dB），制定改进方案 |
| 2026-04-08 | 完成端到端训练，PSNR: 13.83, SSIM: 0.35（效果未达预期） |

---

## 十一、许可证

MIT License
