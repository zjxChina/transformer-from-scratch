import argparse
import os

import torch
import torch.nn as nn
import yaml

from src.config import Config
from src.dataloader import get_dataloaders
from src.model import Transformer
from src.trainer import Trainer

os.environ['HF_DATASETS_CACHE'] = './datasets'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Train Transformer from scratch")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume from")
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"batch_size: {config.training.batch_size}")
    print(f"lr: {config.training.learning_rate}")

    # 获取数据加载器
    train_loader, val_loader, src_vocab_size, tgt_vocab_size = get_dataloaders(config.data)

    # 初始化模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        max_seq_length=config.training.max_seq_length,
        use_positional_encoding=config.model.use_positional_encoding,
        use_residual=config.model.use_residual
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 初始化优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.training.weight_decay
    )

    # 学习率调度器
    def lr_lambda(step):
        step = max(step, 1)
        return min(step ** -0.5, step * (config.training.warmup_steps ** -1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pad_token_id = train_loader.dataset.src_tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    start_epoch = 0
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    learning_rates = []

    if args.resume:
        print(f"恢复训练从检查点: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载训练历史
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_ppls = checkpoint.get('train_ppls', [])
        val_ppls = checkpoint.get('val_ppls', [])
        learning_rates = checkpoint.get('learning_rates', [])

        print(f"从第 {start_epoch} 轮继续训练")
        print(f"之前最佳验证损失: {checkpoint['val_loss']:.4f}")

    if args.resume_epoch > 0 and not args.resume:
        start_epoch = args.resume_epoch
        print(f"从第 {start_epoch} 轮开始训练（无检查点恢复）")

    # 初始化训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        pad_token_id=pad_token_id
    )

    trainer.train_losses = train_losses
    trainer.val_losses = val_losses
    trainer.train_ppls = train_ppls
    trainer.val_ppls = val_ppls
    trainer.learning_rates = learning_rates

    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        start_epoch=start_epoch  # 🆕 传入起始轮数
    )


if __name__ == "__main__":
    main()
