from typing import List, Dict

import torch
import os
import time
import numpy as np
from tqdm import tqdm
import logging

from src.utils import plot_training_curves

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self, model, optimizer,
            criterion, scheduler, device,
            output_dir="results", pad_token_id=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.pad_token_id = pad_token_id

        os.makedirs(output_dir, exist_ok=True)

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.learning_rates = []

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_tokens = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src_ids'].to(self.device)
            tgt = batch['tgt_ids'].to(self.device)
            pad_id = self.pad_token_id

            # 创建mask
            src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.create_masks(
                src, tgt[:, :-1], pad_token_id=self.pad_token_id
            )

            # 前向传播
            self.optimizer.zero_grad()

            # 输入是tgt[:, :-1]，目标是tgt[:, 1:]
            output, _ = self.model(src, tgt[:, :-1], src_mask, tgt_mask)

            # 计算损失
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            batch_tokens = (tgt[:, 1:] != pad_id).sum().item()
            total_tokens += batch_tokens

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PPL': f'{np.exp(loss.item()):.2f}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_ppl = np.exp(avg_loss)

        self.train_losses.append(avg_loss)
        self.train_ppls.append(avg_ppl)
        self.learning_rates.append(self.scheduler.get_last_lr()[0])

        return avg_loss, avg_ppl

    def validate(
            self,
            val_loader,
            epoch: int,
            save_attention_samples: bool = True,
            max_attention_samples: int = 2
    ):
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        # 用于保存注意力权重的样本
        attention_samples = []

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                pad_id = self.pad_token_id

                # 创建mask
                src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.create_masks(
                    src, tgt[:, :-1], pad_token_id=self.pad_token_id
                )

                # 前向传播，获取注意力权重
                output, attention_dict = self.model(src, tgt[:, :-1], src_mask, tgt_mask)

                # 计算损失
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )

                total_loss += loss.item()
                batch_tokens = (tgt[:, 1:] != pad_id).sum().item()
                total_tokens += batch_tokens

                # 保存注意力样本用于可视化
                if (save_attention_samples and
                        batch_idx < max_attention_samples and
                        len(attention_samples) < max_attention_samples):
                    # 选择第一个样本进行可视化
                    sample_idx = 0
                    attention_sample = {
                        'src_text': batch['src_text'][sample_idx],
                        'tgt_text': batch['tgt_text'][sample_idx],
                        'src_ids': src[sample_idx].cpu(),
                        'tgt_ids': tgt[sample_idx].cpu(),
                        'attention_dict': attention_dict,
                        'epoch': epoch,
                        'batch_idx': batch_idx
                    }
                    attention_samples.append(attention_sample)

                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PPL': f'{np.exp(loss.item()):.2f}'
                })

        avg_loss = total_loss / len(val_loader)
        avg_ppl = np.exp(avg_loss)

        if attention_samples:
            self._save_attention_samples(attention_samples, epoch)

        self.val_losses.append(avg_loss)
        self.val_ppls.append(avg_ppl)

        return avg_loss, avg_ppl

    def _save_attention_samples(
            self,
            attention_samples: List[Dict],
            epoch: int
    ) -> None:
        """保存注意力样本并生成可视化图表"""
        from src.utils import create_attention_visualization

        try:
            create_attention_visualization(
                attention_samples=attention_samples,
                output_dir=self.output_dir,
                epoch=epoch
            )
            logger.info(f"保存了 {len(attention_samples)} 个注意力样本")
        except Exception as e:
            logger.warning(f"保存注意力样本失败: {e}")

    def create_masks(self, src, tgt, pad_token_id=None):
        """创建注意力mask"""
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided either in init or method call")

        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()

        # 源mask (padding mask)
        src_mask = (src != pad_token_id).unsqueeze(1).unsqueeze(2)

        # 目标mask (padding mask + causal mask)
        tgt_pad_mask = (tgt != pad_token_id).unsqueeze(1).unsqueeze(2)
        tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=src.device)).bool().unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & tgt_causal_mask

        src_key_padding_mask = (src == pad_token_id)
        tgt_key_padding_mask = (tgt == pad_token_id)

        return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask

    def train(self, train_loader, val_loader, num_epochs, start_epoch=0):
        best_val_loss = float('inf')
        if self.val_losses:
            best_val_loss = min(self.val_losses)

        best_val_ppl = float('inf')
        if self.val_ppls:
            best_val_ppl = min(self.val_ppls)

        print(f"开始训练: 从第 {start_epoch + 1} 轮到第 {num_epochs} 轮")

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_ppl = self.train_epoch(train_loader, epoch)

            # 验证
            save_attention = (epoch == 0 or epoch == num_epochs - 1 or (epoch + 1) % 10 == 0)
            val_loss, val_ppl = self.validate(
                val_loader,
                epoch,
                save_attention_samples=save_attention,
                max_attention_samples=1
            )

            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
                f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
            )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"New best model saved with val_loss: {best_val_loss:.4f}, Val PPL: {best_val_ppl:.2f}")

            self.save_checkpoint(epoch, val_loss, is_best=False)
            print(f"Epoch {epoch + 1} checkpoint saved")

        self.plot_training_curves()
        print("Training completed!")

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ppls': self.train_ppls,
            'val_ppls': self.val_ppls,
            'learning_rates': self.learning_rates
        }

        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'best_model.pth'))
            print(f"最佳模型已保存: epoch {epoch + 1}, val_loss: {val_loss:.4f}")
        else:
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(checkpoint, checkpoint_path)

    def plot_training_curves(self):
        """绘制训练曲线"""
        plot_training_curves(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            save_path=os.path.join(self.output_dir, 'training_curves.png'),
            title=f"Training Curves - {os.path.basename(self.output_dir)}"
        )
