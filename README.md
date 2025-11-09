# Transformer from Scratch

本仓库实现了基于论文《Attention Is All You Need》的Transformer模型，支持机器翻译任务。项目包含完整的训练、推理和消融实验代码。

## 项目结构
```bash
transformer-from-scratch/
├── src/                   # 源代码
│   ├── model.py           # Transformer模型实现
│   ├── trainer.py         # 训练器
│   ├── dataloader.py      # 数据加载器
│   ├── config.py          # 配置类
│   └── utils.py           # 工具函数
├── configs/               # 配置文件
│   ├── base.yaml          # 基础配置
│   └── ablation/          # 消融实验配置
│       └── ablation_*.yaml
├── scripts/               # 脚本文件
│   ├── run.sh             # 训练脚本
│   └── ablation.sh        # 消融实验批量执行
├── results/               # 训练结果和模型保存
├── train.py               # 训练入口文件
├── inference.py           # 推理入口文件
└── requirements.txt       # 依赖包
```

## 依赖安装
```bash
pip install -r requirements.txt
```

## 快速开始
### 1. 训练模型
```bash
# 直接运行训练
python train.py --config configs/base.yaml --output_dir results/base --seed 42

# 或使用脚本训练
bash scripts/run.sh

# 消融实验训练
bash scripts/ablation.sh
```

### 2. 模型推理
```bash
python inference.py --config configs/base.yaml --model results/base/best_model.pth
```

## 文件说明

### 核心文件

- **`train.py`**: 训练入口文件，负责加载配置、初始化模型、启动训练流程
- **`inference.py`**: 推理入口文件，提供模型加载和翻译功能
- **`src/model.py`**: Transformer核心架构实现，包含Encoder、Decoder、MultiHeadAttention等模块
- **`src/trainer.py`**: 训练循环实现，包含训练和验证逻辑
- **`src/dataloader.py`**: 数据加载和预处理，支持IWSLT2017数据集

### 配置文件

- **`configs/base.yaml`**: 完整Transformer架构配置
- **`configs/ablation/`**: 消融实验配置目录
  - `ablation_no_pos_encoding.yaml`: 无位置编码实验
  - `ablation_single_head.yaml`: 单头注意力实验
  - `ablation_no_residual.yaml`: 无残差连接实验
  - `ablation_shallow.yaml`: 浅层模型实验

### 脚本文件

- **`scripts/run.sh`**: 基础训练脚本
- **`scripts/ablation.sh`**: 批量运行消融实验

## 配置说明

- **基础配置** (`base.yaml`): 完整Transformer架构，包含位置编码、多头注意力、残差连接等
- **消融实验配置**: 通过移除特定组件验证其对性能的影响