import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os
import logging
import pandas as pd

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_training_curves(
        train_losses: List[float],
        val_losses: List[float],
        save_path: Optional[str] = None,
        title: str = "Training and Validation Curves"
) -> None:
    """绘制训练和验证曲线"""
    plt.figure(figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    # 训练和验证损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 训练和验证困惑度曲线
    plt.subplot(1, 2, 2)
    train_ppls = [np.exp(loss) for loss in train_losses]
    val_ppls = [np.exp(loss) for loss in val_losses]
    plt.plot(epochs, train_ppls, 'b-', label='Training PPL', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_ppls, 'r-', label='Validation PPL', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练曲线已保存: {save_path}")

    plt.show()
    plt.close()


def plot_ablation_comparison(
        experiments_data: Dict[str, Dict],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
        colors: List[str] = None
) -> None:
    """绘制消融实验比较图"""
    if not experiments_data:
        logger.warning("没有实验数据可绘制")
        return

    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    plt.figure(figsize=figsize)

    # 最终验证损失比较
    plt.subplot(1, 3, 1)
    model_names = list(experiments_data.keys())
    final_val_losses = [data['val_loss'] for data in experiments_data.values()]

    bars = plt.bar(
        range(len(model_names)),
        final_val_losses,
        color=colors[:len(model_names)]
    )
    plt.xlabel('模型变体')
    plt.ylabel('最终验证损失')
    plt.title('消融实验: 最终验证损失比较')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')

    # 在柱子上添加数值
    for bar, value in zip(bars, final_val_losses):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # 验证损失曲线比较
    plt.subplot(1, 3, 2)
    for exp_name, data in experiments_data.items():
        if len(data['val_losses']) > 0:
            epochs = range(1, len(data['val_losses']) + 1)
            plt.plot(epochs, data['val_losses'], label=exp_name, linewidth=2, alpha=0.8)

    plt.xlabel('Epoch')
    plt.ylabel('验证损失')
    plt.title('消融实验: 验证损失曲线')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # 训练损失曲线比较
    plt.subplot(1, 3, 3)
    for exp_name, data in experiments_data.items():
        if len(data['train_losses']) > 0:
            epochs = range(1, len(data['train_losses']) + 1)
            plt.plot(epochs, data['train_losses'], label=exp_name, linewidth=2, alpha=0.8)

    plt.xlabel('Epoch')
    plt.ylabel('训练损失')
    plt.title('消融实验: 训练损失曲线')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"消融实验比较图已保存: {save_path}")

    plt.show()
    plt.close()


def plot_attention_heatmaps(
        sample: Dict,
        save_dir: str,
        epoch: int,
        sample_idx: int,
        max_tokens: int = 15
) -> None:
    """为样本绘制多种注意力热力图"""
    try:
        src_text = sample['src_text']
        tgt_text = sample['tgt_text']
        attention_dict = sample['attention_dict']

        # 简单的分词（实际应该使用tokenizer）
        src_tokens = src_text.split()[:max_tokens]
        tgt_tokens = tgt_text.split()[:max_tokens]

        # 绘制编码器自注意力（第一层，第一个头）
        if ('enc_self_attentions' in attention_dict and
                attention_dict['enc_self_attentions'] and
                len(attention_dict['enc_self_attentions']) > 0):

            enc_attention = attention_dict['enc_self_attentions'][0]  # 第一层
            if enc_attention is not None and len(enc_attention) > 0:
                # 取第一个头的注意力权重，并确保是2D
                enc_attn_weights = enc_attention[0].cpu().numpy()

                # 处理多头注意力的形状
                if len(enc_attn_weights.shape) == 3:
                    # (batch_size, seq_len, seq_len) -> 取第一个样本
                    enc_attn_weights = enc_attn_weights[0]
                elif len(enc_attn_weights.shape) == 4:
                    # (batch_size, num_heads, seq_len, seq_len) -> 取第一个样本第一个头
                    enc_attn_weights = enc_attn_weights[0, 0]

                # 截取到实际token长度
                actual_src_len = min(len(src_tokens), enc_attn_weights.shape[0])
                enc_attn_weights = enc_attn_weights[:actual_src_len, :actual_src_len]

                # 确保是2D数组
                if len(enc_attn_weights.shape) == 2:
                    plot_attention_weights(
                        attention_weights=enc_attn_weights,
                        src_tokens=src_tokens[:actual_src_len],
                        tgt_tokens=src_tokens[:actual_src_len],  # 自注意力，源和目标相同
                        layer=0,
                        head=0,
                        save_path=os.path.join(
                            save_dir,
                            f'enc_self_epoch{epoch}_sample{sample_idx}.png'
                        ),
                        figsize=(10, 8),
                        cmap='Blues'
                    )

        # 绘制解码器交叉注意力（第一层，第一个头）
        if ('dec_cross_attentions' in attention_dict and
                attention_dict['dec_cross_attentions'] and
                len(attention_dict['dec_cross_attentions']) > 0):

            cross_attention = attention_dict['dec_cross_attentions'][0]  # 第一层
            if cross_attention is not None and len(cross_attention) > 0:
                # 取第一个头的注意力权重
                cross_attn_weights = cross_attention[0].cpu().numpy()

                # 处理多头注意力的形状
                if len(cross_attn_weights.shape) == 3:
                    cross_attn_weights = cross_attn_weights[0]
                elif len(cross_attn_weights.shape) == 4:
                    cross_attn_weights = cross_attn_weights[0, 0]

                # 截取到实际token长度
                actual_src_len = min(len(src_tokens), cross_attn_weights.shape[1])
                actual_tgt_len = min(len(tgt_tokens), cross_attn_weights.shape[0])
                cross_attn_weights = cross_attn_weights[:actual_tgt_len, :actual_src_len]

                if len(cross_attn_weights.shape) == 2:
                    plot_attention_weights(
                        attention_weights=cross_attn_weights,
                        src_tokens=src_tokens[:actual_src_len],
                        tgt_tokens=tgt_tokens[:actual_tgt_len],
                        layer=0,
                        head=0,
                        save_path=os.path.join(
                            save_dir,
                            f'cross_epoch{epoch}_sample{sample_idx}.png'
                        ),
                        figsize=(12, 8),
                        cmap='viridis'
                    )

        # 绘制解码器自注意力（第一层，第一个头）
        if ('dec_self_attentions' in attention_dict and
                attention_dict['dec_self_attentions'] and
                len(attention_dict['dec_self_attentions']) > 0):

            dec_self_attention = attention_dict['dec_self_attentions'][0]  # 第一层
            if dec_self_attention is not None and len(dec_self_attention) > 0:
                # 取第一个头的注意力权重
                dec_self_attn_weights = dec_self_attention[0].cpu().numpy()

                # 处理多头注意力的形状
                if len(dec_self_attn_weights.shape) == 3:
                    dec_self_attn_weights = dec_self_attn_weights[0]
                elif len(dec_self_attn_weights.shape) == 4:
                    dec_self_attn_weights = dec_self_attn_weights[0, 0]

                # 截取到实际token长度
                actual_tgt_len = min(len(tgt_tokens), dec_self_attn_weights.shape[0])
                dec_self_attn_weights = dec_self_attn_weights[:actual_tgt_len, :actual_tgt_len]

                if len(dec_self_attn_weights.shape) == 2:
                    plot_attention_weights(
                        attention_weights=dec_self_attn_weights,
                        src_tokens=tgt_tokens[:actual_tgt_len],
                        tgt_tokens=tgt_tokens[:actual_tgt_len],  # 自注意力，源和目标相同
                        layer=0,
                        head=0,
                        save_path=os.path.join(
                            save_dir,
                            f'dec_self_epoch{epoch}_sample{sample_idx}.png'
                        ),
                        figsize=(10, 8),
                        cmap='Reds'
                    )

    except Exception as e:
        logger.warning(f"绘制注意力热力图失败: {e}")


def create_attention_visualization(
        attention_samples: List[Dict],
        output_dir: str,
        epoch: int
) -> None:
    """为多个注意力样本创建可视化"""
    attention_dir = os.path.join(output_dir, 'attention_plots')
    os.makedirs(attention_dir, exist_ok=True)

    for i, sample in enumerate(attention_samples):
        # 保存原始数据
        sample_path = os.path.join(
            attention_dir,
            f'attention_sample_epoch{epoch}_{i}.pth'
        )
        torch.save(sample, sample_path)

        # 生成注意力热力图
        plot_attention_heatmaps(
            sample=sample,
            save_dir=attention_dir,
            epoch=epoch,
            sample_idx=i
        )

    logger.info(f"为 {len(attention_samples)} 个样本生成了注意力热力图")


def plot_attention_weights(
        attention_weights: np.ndarray,
        src_tokens: List[str],
        tgt_tokens: List[str],
        layer: int = 0,
        head: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = 'viridis'
) -> None:
    """绘制注意力权重热力图"""

    # 形状验证
    if len(attention_weights.shape) != 2:
        logger.warning(f"注意力权重形状 {attention_weights.shape} 不是2D，无法绘制热力图")
        return

    if attention_weights.shape[0] == 0 or attention_weights.shape[1] == 0:
        logger.warning("注意力权重矩阵为空")
        return

    # 限制矩阵大小以避免内存问题
    max_display_tokens = 50
    if attention_weights.shape[0] > max_display_tokens or attention_weights.shape[1] > max_display_tokens:
        logger.info(f"注意力矩阵过大 {attention_weights.shape}，显示前{max_display_tokens}个token")
        attention_weights = attention_weights[:max_display_tokens, :max_display_tokens]
        src_tokens = src_tokens[:max_display_tokens]
        tgt_tokens = tgt_tokens[:max_display_tokens]

    plt.figure(figsize=figsize)

    # 创建热力图
    im = plt.imshow(attention_weights, cmap=cmap, aspect='auto')

    # 设置坐标轴标签
    if src_tokens and tgt_tokens:
        plt.xticks(
            range(len(src_tokens)),
            src_tokens,
            rotation=45,
            ha='right',
            fontsize=8
        )
        plt.yticks(range(len(tgt_tokens)), tgt_tokens, fontsize=8)

    plt.xlabel('Source Tokens', fontsize=12)
    plt.ylabel('Target Tokens', fontsize=12)
    plt.title(
        f'Attention Weights - Layer {layer}, Head {head}\nShape: {attention_weights.shape}',
        fontsize=14,
        fontweight='bold'
    )

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"注意力权重图已保存: {save_path}")

    plt.show()
    plt.close()


def generate_training_table(
        experiments_data: Dict[str, Dict],
        save_path: Optional[str] = None,
        include_ppl: bool = True
) -> pd.DataFrame:
    """生成训练结果表格"""
    results = []

    for exp_name, data in experiments_data.items():
        val_loss = data.get('val_loss', float('inf'))
        train_losses = data.get('train_losses', [])
        val_losses = data.get('val_losses', [])

        final_train_loss = train_losses[-1] if train_losses else float('inf')
        final_val_loss = val_losses[-1] if val_losses else float('inf')

        result_row = {
            '实验名称': exp_name,
            '最终训练损失': f"{final_train_loss:.4f}",
            '最终验证损失': f"{final_val_loss:.4f}",
            '最佳验证损失': f"{val_loss:.4f}",
            '训练轮数': len(train_losses)
        }

        if include_ppl:
            result_row.update({
                '最终训练困惑度': f"{np.exp(final_train_loss):.2f}",
                '最终验证困惑度': f"{np.exp(final_val_loss):.2f}",
                '最佳验证困惑度': f"{np.exp(val_loss):.2f}"
            })

        results.append(result_row)

    df = pd.DataFrame(results)

    print("\n训练结果汇总:")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    if save_path:
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"结果表格已保存: {save_path}")

    return df


def load_experiment_data(
        results_dir: str = "results",
        experiment_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Dict]:
    """加载所有实验数据"""
    if experiment_mapping is None:
        experiment_mapping = {
            'base': '基础模型',
            'ablation_base': '基础模型（消融）',
            'ablation_no_pos': '无位置编码',
            'ablation_single_head': '单头注意力',
            'ablation_shallow': '浅层模型',
            'ablation_no_residual': '无残差连接'
        }

    experiments_data = {}

    for exp_key, exp_name in experiment_mapping.items():
        exp_dir = os.path.join(results_dir, exp_key)
        best_model_path = os.path.join(exp_dir, 'best_model.pth')

        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location='cpu')
                experiments_data[exp_name] = {
                    'train_losses': checkpoint.get('train_losses', []),
                    'val_losses': checkpoint.get('val_losses', []),
                    'val_loss': checkpoint.get('val_loss', float('inf'))
                }
                logger.info(f"加载实验数据: {exp_name}")
            except Exception as e:
                logger.error(f"加载实验数据失败 {exp_name}: {e}")
        else:
            logger.warning(f"实验数据不存在: {exp_name}")

    return experiments_data


def generate_all_plots(
        results_dir: str = "results",
        ablation_save_path: str = "results/ablation_comparison.png",
        table_save_path: str = "results/training_results.csv"
) -> bool:
    """生成所有实验图表"""
    try:
        # 加载实验数据
        experiments_data = load_experiment_data(results_dir)

        if not experiments_data:
            logger.error("没有找到实验数据，请先运行训练")
            return False

        # 确保保存目录存在
        os.makedirs(os.path.dirname(ablation_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(table_save_path), exist_ok=True)

        # 生成消融实验比较图
        plot_ablation_comparison(
            experiments_data=experiments_data,
            save_path=ablation_save_path
        )

        # 生成训练结果表格
        generate_training_table(
            experiments_data=experiments_data,
            save_path=table_save_path
        )

        logger.info("  所有实验图表生成完成")
        logger.info(f"  消融实验比较图: {ablation_save_path}")
        logger.info(f"  训练结果表格: {table_save_path}")

        return True

    except Exception as e:
        logger.error(f"生成实验图表失败: {e}")
        return False
