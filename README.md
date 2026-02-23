# SEMA — Jittor 版本

**SEMA**（Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning）的 **Jittor 框架完整迁移版**。

原始项目基于 PyTorch，本仓库将其完整迁移到 [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) 深度学习框架。torchvision 仅用于下载 CIFAR 数据集，timm 仅用于加载预训练 ViT 权重，训练与推理全程在 Jittor 上运行。

> 论文：*Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning* (CVPR 2025)

---

## 目录

- [项目简介](#项目简介)
- [Jittor 是什么](#jittor-是什么)
- [PyTorch → Jittor 迁移对照](#pytorch--jittor-迁移对照)
- [环境搭建](#环境搭建)
- [项目结构](#项目结构)
- [数据集准备](#数据集准备)
- [如何运行](#如何运行)
- [配置文件详解](#配置文件详解)
- [常见问题 FAQ](#常见问题-faq)
- [引用](#引用)

---

## 项目简介

SEMA 是一种基于预训练 Vision Transformer（ViT-B/16）的**持续学习（Continual Learning）**方法，核心机制如下：

| 组件 | 说明 |
|------|------|
| **Adapter（功能适配器）** | 插入每个 Transformer FFN 层后的瓶颈模块，用于参数高效微调 |
| **Representation Descriptor（RD）** | 基于 AutoEncoder 的分布偏移检测器，每个 Adapter 独立维护 |
| **自扩展机制** | 通过 z-score 检测新任务与已知分布的偏移，自动决定是否添加新 Adapter |
| **混合路由（Mixture Router）** | 通过可学习路由权重对多个 Adapter 输出加权混合，支持 Top-K 稀疏路由 |

**支持的数据集：**
- CIFAR-100（自动下载）
- ImageNet-R（需手动下载）
- ImageNet-A（需手动下载）
- VTAB（需手动下载）

---

## Jittor 是什么

[Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) 是清华大学计算机图形学组开发的基于**即时编译（JIT）**的深度学习框架。

**与 PyTorch 的相同点：**
- 动态图（eager execution）
- API 设计风格高度相似
- 支持 CUDA GPU 加速

**主要区别概览：**

| 特性 | PyTorch | Jittor |
|------|---------|--------|
| 基本数据类型 | `torch.Tensor` | `jt.Var` |
| 模型前向方法 | `forward(self, x)` | `execute(self, x)` |
| 反向传播 | `loss.backward()` + `optimizer.step()` | `optimizer.step(loss)` 一步完成 |
| 冻结参数 | `param.requires_grad = False` | `param.stop_grad()` |
| 参数是否冻结 | `param.requires_grad` | `not param.is_stop_grad()` |
| GPU 设备管理 | `tensor.to("cuda")` | `jt.flags.use_cuda = 1`（全局，无需移动） |
| 数据加载器 | `DataLoader(dataset, ...)` | `dataset.set_attrs(...)` + 直接迭代 |
| 无梯度推理 | `with torch.no_grad():` | `with jt.no_grad():` |

---

## PyTorch → Jittor 迁移对照

以下是本项目迁移中涉及的关键 API 对照，可作为其他项目迁移时的参考手册。

### 1. 模型定义

```python
# PyTorch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 10)

    def forward(self, x):          # 方法名必须是 forward
        return self.linear(x)

# ─────────────────────────────────────────────────────────
# Jittor（只需改方法名，其余几乎相同）
from jittor import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 10)

    def execute(self, x):          # 方法名必须是 execute
        return self.linear(x)
```

### 2. 训练循环

```python
# PyTorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for inputs, labels in loader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()          # 3 步
    loss.backward()
    optimizer.step()

# ─────────────────────────────────────────────────────────
# Jittor（optimizer.step(loss) 一行代替三行）
optimizer = jittor.optim.SGD(model.parameters(), lr=0.01)
for inputs, labels in loader:
    outputs = model(inputs)
    loss = nn.cross_entropy_loss(outputs, labels)
    optimizer.step(loss)           # 自动清零梯度 + 反向传播 + 更新
```

### 3. 参数冻结与解冻

```python
# PyTorch
for p in model.parameters():
    p.requires_grad = False        # 冻结
for p in model.parameters():
    p.requires_grad = True         # 解冻

# 判断是否可训练
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─────────────────────────────────────────────────────────
# Jittor
for p in model.parameters():
    p.stop_grad()                  # 冻结
for p in model.parameters():
    p.start_grad()                 # 解冻

# 判断是否可训练
trainable = sum(p.numel() for p in model.parameters() if not p.is_stop_grad())
```

### 4. 设备管理（最大差异）

```python
# PyTorch：每个 tensor/module 需手动 .to(device)
device = torch.device("cuda:0")
model = model.to(device)
tensor = tensor.to(device)

# ─────────────────────────────────────────────────────────
# Jittor：全局开关，所有计算自动走 GPU，不需要移动数据
import jittor as jt
jt.flags.use_cuda = 1             # 启用 GPU（0 = CPU）
# 不需要 .to(device)，所有操作自动在 GPU 上
```

### 5. 数据加载

```python
# PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
for images, labels in loader:
    ...

# ─────────────────────────────────────────────────────────
# Jittor：Dataset 本身就是可迭代的 loader
dataset.set_attrs(batch_size=32, shuffle=True, num_workers=4)
for images, labels in dataset:
    ...
```

### 6. 常用 Tensor 操作对照

```python
# PyTorch → Jittor
torch.tensor([1.0, 2.0])         → jt.array([1.0, 2.0])
torch.zeros(3, 4)                 → jt.zeros((3, 4))
torch.ones(3, 4)                  → jt.ones((3, 4))
torch.randn(3, 4)                 → jt.randn(3, 4)
torch.cat([a, b], dim=0)          → jt.concat([a, b], dim=0)
torch.stack([a, b], dim=0)        → jt.stack([a, b], dim=0)
x.view(B, -1)                     → x.view(B, -1)           # 相同
x.reshape(B, -1)                  → x.reshape(B, -1)        # 相同
x.transpose(1, 2)                 → x.transpose(0, 2, 1)    # 必须列出所有维度！
x.permute(0, 2, 1, 3)             → x.transpose(0, 2, 1, 3) # permute → transpose
x.contiguous()                    → x                       # Jittor 无需 contiguous
x.detach()                        → x.detach()              # 相同
x.numpy()                         → x.numpy()               # 相同（无需 .cpu() 前置）
torch.bmm(a, b)                   → jt.bmm(a, b)            # 相同
torch.softmax(x, dim=-1)          → jt.nn.softmax(x, dim=-1)
```

### 7. 参数初始化

```python
# PyTorch → Jittor
nn.init.kaiming_uniform_(w)         → nn.init.kaiming_uniform_(w)     # 相同
nn.init.zeros_(b)                   → nn.init.zero_(b)                 # 注意：zero_ 无 s
nn.init.trunc_normal_(w, std=0.02)  → nn.init.gauss_(w, 0, 0.02)      # 近似替代
nn.init.constant_(x, val)           → nn.init.constant_(x, val)       # 相同
```

### 8. 无梯度推理

```python
# PyTorch
with torch.no_grad():
    output = model(x)

# Jittor
with jt.no_grad():
    output = model(x)
```

### 9. 内存管理

```python
# PyTorch
torch.cuda.empty_cache()

# Jittor
jt.sync_all()   # 同步所有异步计算
jt.gc()         # 触发 Jittor 内存回收
```

### 10. 跨框架权重加载（timm → Jittor）

本项目通过 timm 下载 PyTorch ViT-B/16 权重，按层名逐一映射到 Jittor 模型。
核心逻辑见 [`backbone/vit_sema.py`](backbone/vit_sema.py)：

```python
import timm
import jittor as jt

torch_model = timm.create_model('vit_base_patch16_224', pretrained=True)
torch_state = torch_model.state_dict()

for name, param in jittor_model.named_parameters():
    if name in torch_state:
        param.data = jt.array(torch_state[name].cpu().numpy())
```

---

## 环境搭建

### 前置要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 20.04+）、Windows 10/11、macOS |
| Python | 3.7 – 3.11（推荐 **3.9**） |
| GPU | NVIDIA GPU + CUDA 11.x（推荐）；也可纯 CPU 运行但较慢 |
| 编译器 | g++ ≥ 5.4（Linux/macOS），MSVC（Windows） |

### 方法一：conda 一键创建（推荐）

```bash
# 1. 创建环境
conda env create -f sema_jittor_env.yaml

# 2. 激活环境
conda activate sema_jittor_env
```

### 方法二：手动安装

```bash
# 1. 创建 Python 3.9 环境
conda create -n sema_jittor python=3.9
conda activate sema_jittor

# 2. 安装 Jittor
pip install jittor>=1.3.8

# 3. 安装其余依赖
#    timm/torchvision 仅用于初始化，不参与 Jittor 训练
pip install timm==0.9.8 numpy>=1.24.0 scipy tqdm easydict Pillow torchvision
```

### 验证安装

```bash
# 检查 Jittor 版本
python -c "import jittor as jt; print('Jittor', jt.__version__)"

# 检查 CUDA
python -c "
import jittor as jt
jt.flags.use_cuda = 1
a = jt.randn(3, 3)
print(a)
print('CUDA OK!' if jt.flags.use_cuda else 'CPU only')
"

# 运行官方测试（可选）
python -m jittor.test.test_core
```

### Windows 额外步骤

Jittor 在 Windows 上需要 C++ 编译器完成 JIT 编译：

1. 下载安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 勾选 **"使用 C++ 的桌面开发"** 工作负载

本项目提供 [`setup_jittor_env.py`](setup_jittor_env.py) 自动处理 Windows 上的 DLL 路径和环境变量，
已集成在 `main.py` 第一行（`import setup_jittor_env`），直接运行 `python main.py` 即可，无需手动操作。

---

## 项目结构

```
SEMA-CL-Jittor/
│
├── main.py                      # 程序入口：解析命令行参数，调用 trainer.train()
├── trainer.py                   # 训练流程：多 seed 循环、日志、任务迭代
├── setup_jittor_env.py          # Windows Jittor 环境初始化（DLL 路径 + 环境变量）
├── sema_jittor_env.yaml         # Conda 环境配置文件
├── README.md                    # 本文档
│
├── exps/                        # 实验配置（JSON）
│   ├── sema_cifar.json          # CIFAR-100，10类/任务，共10任务
│   ├── sema_ina.json            # ImageNet-A，20类/任务，共10任务
│   ├── sema_inr_5task.json      # ImageNet-R，40类/任务，共5任务
│   ├── sema_inr_10task.json     # ImageNet-R，20类/任务，共10任务（默认）
│   ├── sema_inr_20tasks.json    # ImageNet-R，10类/任务，共20任务
│   └── sema_vtab.json           # VTAB，10类/任务，共10任务
│
├── backbone/                    # 骨干网络
│   ├── __init__.py
│   ├── vit_sema.py              # ViT-B/16 + SEMA Adapter；含 timm 权重转换逻辑
│   ├── sema_block.py            # AdapterModule（单 Adapter + RD）和 SEMAModules（多 Adapter 管理）
│   └── sema_components.py       # 基础组件：Adapter（MLP 瓶颈）、AE（AutoEncoder RD）、Records
│
├── models/                      # 学习算法
│   ├── __init__.py
│   ├── base.py                  # BaseLearner：评估、Exemplar 管理基类
│   └── sema.py                  # SEMA Learner：增量训练、分布偏移检测、自扩展决策
│
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── data.py                  # 数据集定义（iCIFAR100、iImageNetR、iImageNetA、vtab 等）
│   ├── data_manager.py          # DataManager：类顺序管理、Jittor Dataset 构建
│   ├── factory.py               # get_model()：模型名 → Learner 工厂函数
│   ├── inc_net.py               # SEMAVitNet：backbone + fc head 封装
│   └── toolkit.py               # 辅助函数：count_parameters、tensor2numpy、accuracy
│
└── data/                        # 数据集目录（部分需手动下载）
    ├── cifar-100-python/        # CIFAR-100（首次运行自动下载）
    ├── imagenet-r/              # ImageNet-R（需手动）
    ├── imagenet-a/              # ImageNet-A（需手动）
    └── vtab/                    # VTAB（需手动）
```

### 模块调用关系

```
main.py
  └─ trainer.py
       ├─ utils/factory.py  ──►  models/sema.py (Learner)
       │                              ├─ models/base.py (BaseLearner)
       │                              ├─ utils/inc_net.py (SEMAVitNet)
       │                              │     └─ backbone/vit_sema.py (ViT + SEMA)
       │                              │           └─ backbone/sema_block.py (SEMAModules)
       │                              │                 └─ backbone/sema_components.py (Adapter, AE)
       │                              └─ utils/toolkit.py
       └─ utils/data_manager.py  ──►  utils/data.py
```

---

## 数据集准备

### CIFAR-100

**自动下载**，无需任何手动操作。首次运行时自动下载到 `./data/` 目录。

### ImageNet-R

```
data/imagenet-r/
├── train/
│   ├── n01xxxxxx/   ← 每类一个子目录，ImageNet WordNet ID 命名
│   └── ...
└── test/
    ├── n01xxxxxx/
    └── ...
```

下载：[Google Drive](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view)

### ImageNet-A

```
data/imagenet-a/
├── train/
│   └── n01xxxxxx/
└── test/
    └── n01xxxxxx/
```

下载：[Google Drive](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view)

### VTAB

```
data/vtab/
├── orders          ← 类顺序文件
├── train/
│   └── class_xxx/
└── test/
    └── class_xxx/
```

下载：[Google Drive](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view)

### 共享原始项目数据（推荐，避免重复下载）

```bash
# Linux / macOS
ln -s /path/to/SEMA-CL/data /path/to/SEMA-CL-Jittor/data

# Windows PowerShell（需管理员权限）
New-Item -ItemType SymbolicLink -Path "F:\CL\SEMA-CL-Jittor\data" -Target "F:\CL\SEMA-CL\data"
```

---

## 如何运行

### 基本命令

```bash
cd SEMA-CL-Jittor

# ImageNet-R 10任务（默认实验）
python main.py --config exps/sema_inr_10task.json

# ImageNet-R 5任务 / 20任务
python main.py --config exps/sema_inr_5task.json
python main.py --config exps/sema_inr_20tasks.json

# CIFAR-100
python main.py --config exps/sema_cifar.json

# ImageNet-A
python main.py --config exps/sema_ina.json

# VTAB
python main.py --config exps/sema_vtab.json
```

### GPU 选择

在 JSON 配置文件中修改 `"device"` 字段：

```json
"device": ["0"]     // GPU 0
"device": ["1"]     // GPU 1
"device": ["-1"]    // CPU（调试用）
```

> 当前版本仅支持**单 GPU 或 CPU**，Jittor 的多卡并行方式与 PyTorch DDP 不同。

### 查看训练日志

日志同时输出到终端和文件，路径格式为：

```
logs/<model>/<dataset>/<init_cls>/<increment>/<prefix>_<seed>_<backbone>.log
```

例如默认 ImageNet-R 10任务实验：

```bash
# Linux/macOS
cat logs/sema/imagenetr/20/20/reproduce_1993_pretrained_vit_b16_224_adapter.log

# Windows PowerShell
Get-Content logs\sema\imagenetr\20\20\reproduce_1993_pretrained_vit_b16_224_adapter.log
```

---

## 配置文件详解

所有超参数通过 `exps/*.json` 配置，以下为完整说明：

### 数据集与任务

| 参数 | 类型 | 说明 |
|------|------|------|
| `dataset` | string | `cifar224` / `imagenetr` / `imageneta` / `vtab` |
| `init_cls` | int | 第一个任务的类别数 |
| `increment` | int | 后续每个任务的类别增量 |
| `shuffle` | bool | 是否随机打乱类别顺序 |
| `seed` | list[int] | 随机种子列表，每个 seed 独立完整运行一次 |
| `memory_size` | int | Exemplar 总数（SEMA 不需要 Exemplar，设为 0） |

### 模型与骨干

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_name` | string | 固定为 `"sema"` |
| `backbone_type` | string | `"pretrained_vit_b16_224_adapter"`（IN-1k 预训练）或 `"pretrained_vit_b16_224_in21k_adapter"`（IN-21k） |
| `device` | list[str] | GPU 设备编号，`["-1"]` 表示 CPU |
| `ffn_adapter_type` | string | Adapter 结构类型，固定 `"adaptmlp"` |
| `ffn_num` | int | Adapter 瓶颈维度，默认 16 |

### 训练超参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `batch_size` | int | 训练批大小；GPU 不足时减小（推荐范围 16–48） |
| `func_epoch` | int | 功能 Adapter 训练轮数 |
| `rd_epoch` | int | Representation Descriptor (RD) 训练轮数 |
| `init_lr` | float | Adapter 训练初始学习率 |
| `rd_lr` | float | RD 训练学习率 |
| `weight_decay` | float | 优化器 weight decay |
| `optimizer` | string | `"sgd"` 或 `"adam"` |

### SEMA 核心参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `exp_threshold` | float | 自扩展 z-score 阈值；越小越容易扩展新 Adapter |
| `adapt_start_layer` | int | 允许自扩展的 ViT 起始层（0-indexed，建议 9） |
| `adapt_end_layer` | int | 允许自扩展的 ViT 结束层（建议 11） |
| `rd_dim` | int | RD AutoEncoder 隐层维度 |
| `buffer_size` | int | RD 损失滑动窗口大小，用于计算均值/方差 |
| `detect_batch_size` | int | 分布偏移检测阶段的批大小（OOM 时可减小） |
| `use_topk_routing` | bool | 是否启用 Top-K 稀疏路由 |
| `top_k_adapters` | int | Top-K 路由中每次激活的 Adapter 数量 |

---

## 常见问题 FAQ

### Q1：Jittor 第一次运行非常慢？

正常现象。Jittor 是 JIT 编译框架，第一次运行时会编译 CUDA/C++ 算子并缓存到 `~/.cache/jittor/`，后续运行直接读取缓存，速度恢复正常。预热时间通常 1–5 分钟。

### Q2：出现 C++ 编译器报错？

- **Linux**：`sudo apt update && sudo apt install g++ build-essential`
- **Windows**：安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，选择"使用 C++ 的桌面开发"
- 验证安装：`python -m jittor.test.test_core`

### Q3：CUDA 不可用，只能 CPU 运行？

1. 检查驱动：`nvidia-smi`
2. 确保安装了 NVIDIA CUDA Toolkit（推荐 CUDA 11.x）
3. 临时用 CPU：配置文件中设置 `"device": ["-1"]`

### Q4：预训练 ViT 权重如何获取？

首次运行时 `backbone/vit_sema.py` 会通过 `timm` 自动下载 ViT-B/16 预训练权重（约 330MB），**需要网络连接**。权重缓存在 `~/.cache/torch/hub/`，后续运行无需重新下载。

### Q5：GPU 显存不足（OOM）？

按顺序尝试：
1. 减小 `batch_size`：48 → 24 → 16
2. 减小 `detect_batch_size`：128 → 32
3. 减小 `ffn_num`（Adapter 瓶颈维度）：16 → 8
4. 若后期任务 Adapter 数量过多（正常的自扩展行为），也可适当提高 `exp_threshold`

### Q6：如何从 PyTorch 版本迁移代码？

参考本 README [PyTorch → Jittor 迁移对照](#pytorch--jittor-迁移对照) 章节，核心改动：
1. `forward()` → `execute()`
2. 三行优化器更新 → `optimizer.step(loss)`
3. 去掉所有 `.to(device)`，全局 `jt.flags.use_cuda = 1`
4. `requires_grad = False` → `stop_grad()`
5. `x.transpose(1, 2)` → `x.transpose(0, 2, 1)`（需列出全部维度）

### Q7：多 seed 实验如何运行？

在配置文件的 `seed` 字段列出多个值，程序将**顺序**运行每个 seed 的完整实验：

```json
"seed": [1993, 42, 2023]
```

如需并行，可同时启动多个进程，每个进程使用不同的单 seed 配置文件。

---

## 引用

```bibtex
@inproceedings{SEMA,
    author    = {Wang, Huiyi and Lu, Haodong and Yao, Lina and Gong, Dong},
    title     = {Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```

---

## 致谢

- 原始 PyTorch 实现：[SEMA-CL](https://github.com/girafffee/SEMA-CL)
- Jittor 框架：[清华大学计算机图形学组](https://cg.cs.tsinghua.edu.cn/jittor/)
- 预训练模型来源：[timm](https://github.com/huggingface/pytorch-image-models)

