#!/bin/bash

# 消融实验批量执行脚本
cd ..
echo "开始Transformer消融实验"

# 基础模型
echo "训练基础模型..."
python train.py --config configs/ablation/ablation_base.yaml --output_dir results/ablation_base --seed 42

# 无位置编码
echo "训练无位置编码模型..."
python train.py --config configs/ablation/ablation_no_pos_encoding.yaml --output_dir results/ablation_no_pos --seed 42

# 单头注意力
echo "训练单头注意力模型..."
python train.py --config configs/ablation/ablation_single_head.yaml --output_dir results/ablation_single_head --seed 42

# 无残差连接
echo "训练无残差连接模型..."
python train.py --config configs/ablation/ablation_no_residual.yaml --output_dir results/ablation_no_residual --seed 42

# 浅层模型
echo "训练浅层模型..."
python train.py --config configs/ablation/ablation_shallow.yaml --output_dir results/ablation_shallow --seed 42

echo "所有消融实验完成"