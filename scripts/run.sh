#!/bin/bash

# 训练执行脚本
cd ..
echo "开始Transformer训练"

python train.py --config configs/base.yaml --output_dir results/base --seed 42

echo "训练完成"