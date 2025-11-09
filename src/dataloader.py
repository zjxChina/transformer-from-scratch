import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import logging

logger = logging.getLogger(__name__)


class TranslationDataset(Dataset):
    def __init__(
            self,
            split="train",
            src_tokenizer=None,
            tgt_tokenizer=None,
            max_length=128,
            vocab_size=10000,
            dataset_name="iwslt2017",
            data_dir="./datasets"
    ):
        """翻译数据集类，自动下载数据集到指定目录"""
        self.split = split
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.dataset_name = dataset_name
        self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok=True)
        self.dataset = self._load_or_download_dataset()
        self.src_tokenizer, self.tgt_tokenizer = self._initialize_tokenizers(src_tokenizer, tgt_tokenizer)

        logger.info(f"初始化 {split} 数据集完成，样本数量: {len(self.dataset)}")
        logger.info(f"源语言词汇表大小: {self.src_tokenizer.get_vocab_size()}")
        logger.info(f"目标语言词汇表大小: {self.tgt_tokenizer.get_vocab_size()}")

    def _load_or_download_dataset(self):
        """加载或下载数据集到本地目录"""
        dataset_path = os.path.join(self.data_dir, self.dataset_name, self.split)

        # 检查是否已存在本地缓存
        if os.path.exists(dataset_path):
            logger.info(f"从本地加载数据集: {dataset_path}")
            try:
                dataset = load_from_disk(dataset_path)
                logger.info(f"成功加载 {self.split} 数据集，大小: {len(dataset)}")
                return dataset
            except Exception as e:
                logger.warning(f"加载本地数据集失败: {e}，重新下载")
                # 如果加载失败，删除损坏的文件并重新下载
                import shutil
                shutil.rmtree(dataset_path, ignore_errors=True)

        # 下载数据集
        logger.info(f"下载数据集 {self.dataset_name} 到 {self.data_dir}")
        return self._download_and_save_dataset()

    def _download_and_save_dataset(self):
        """下载并保存数据集"""
        try:
            full_dataset = load_dataset(
                self.dataset_name,
                "iwslt2017-en-de",
                trust_remote_code=True
            )

            # 保存每个分割到本地
            dataset_save_dir = os.path.join(self.data_dir, self.dataset_name)
            os.makedirs(dataset_save_dir, exist_ok=True)

            for split_name, split_data in full_dataset.items():
                split_path = os.path.join(dataset_save_dir, split_name)
                split_data.save_to_disk(split_path)
                logger.info(f"已保存 {split_name} 分割到: {split_path}")

            logger.info(f"数据集下载完成，{self.split} 分割大小: {len(full_dataset[self.split])}")
            return full_dataset[self.split]

        except Exception as e:
            logger.error(f"数据集下载失败: {e}")
            raise

    def _initialize_tokenizers(self, src_tokenizer, tgt_tokenizer):
        """初始化tokenizer"""
        tokenizer_dir = os.path.join(self.data_dir, self.dataset_name, "tokenizers")

        if self.split != "train":
            if os.path.exists(os.path.join(tokenizer_dir, "src_tokenizer.json")):
                src_tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "src_tokenizer.json"))
                tgt_tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tgt_tokenizer.json"))
                logger.info("从训练集加载tokenizer")
                return src_tokenizer, tgt_tokenizer
            else:
                raise RuntimeError("验证集/测试集需要预训练tokenizer！请先运行训练集。")

        # 训练集：创建新tokenizer
        if src_tokenizer is None or tgt_tokenizer is None:
            logger.info("创建tokenizers...")

            # 准备训练数据
            src_texts = [item['translation']['en'] for item in self.dataset]
            tgt_texts = [item['translation']['de'] for item in self.dataset]

            if src_tokenizer is None:
                src_tokenizer = self._create_tokenizer(src_texts, "source")
            if tgt_tokenizer is None:
                tgt_tokenizer = self._create_tokenizer(tgt_texts, "target")

            # 保存tokenizer
            os.makedirs(tokenizer_dir, exist_ok=True)
            src_tokenizer.save(os.path.join(tokenizer_dir, "src_tokenizer.json"))
            tgt_tokenizer.save(os.path.join(tokenizer_dir, "tgt_tokenizer.json"))
            logger.info(f"Tokenizers 已保存到: {tokenizer_dir}")

        return src_tokenizer, tgt_tokenizer

    def _create_tokenizer(self, texts, name):
        """创建BPE tokenizer"""
        logger.info(f"训练 {name} tokenizer...")

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        )

        # 训练tokenizer
        tokenizer.train_from_iterator(texts, trainer)
        logger.info(f"  {name} tokenizer 训练完成，词汇表大小: {tokenizer.get_vocab_size()}")

        return tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['translation']['en']  # 英语原文
        tgt_text = item['translation']['de']  # 德语翻译

        # 编码源文本和目标文本
        src_encoding = self.src_tokenizer.encode(src_text)
        tgt_encoding = self.tgt_tokenizer.encode(tgt_text)

        # 添加特殊token并截断
        src_ids = [self.src_tokenizer.token_to_id("[SOS]")] + src_encoding.ids[:self.max_length - 2] + [
            self.src_tokenizer.token_to_id("[EOS]")]
        tgt_ids = [self.tgt_tokenizer.token_to_id("[SOS]")] + tgt_encoding.ids[:self.max_length - 2] + [
            self.tgt_tokenizer.token_to_id("[EOS]")]

        # 填充到最大长度
        src_ids = self._pad_sequence(src_ids, self.max_length, self.src_tokenizer.token_to_id("[PAD]"))
        tgt_ids = self._pad_sequence(tgt_ids, self.max_length, self.tgt_tokenizer.token_to_id("[PAD]"))

        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }

    def _pad_sequence(self, sequence, max_length, pad_id):
        """填充序列到指定长度"""
        if len(sequence) < max_length:
            sequence = sequence + [pad_id] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence


def create_masks(src, tgt, pad_token_id=0):
    """创建注意力mask"""
    batch_size, src_len = src.size()
    _, tgt_len = tgt.size()

    # 源mask (padding mask)
    src_mask = (src != pad_token_id).unsqueeze(1).unsqueeze(2)

    # 目标mask (padding mask + causal mask)
    tgt_pad_mask = (tgt != pad_token_id).unsqueeze(1).unsqueeze(2)
    tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_pad_mask & tgt_causal_mask.to(src.device)

    return src_mask, tgt_mask


def get_dataloaders(data_config, batch_size=32, max_length=128, data_dir="./datasets"):
    """获取数据加载器，自动管理数据集下载"""
    logger.info("初始化数据加载器...")

    # 创建训练集（会自动下载数据集）
    train_dataset = TranslationDataset(
        split="train",
        max_length=max_length,
        vocab_size=data_config.max_vocab_size,
        dataset_name=data_config.dataset_name,
        data_dir=data_dir
    )

    # 创建验证集
    val_dataset = TranslationDataset(
        split="validation",
        src_tokenizer=train_dataset.src_tokenizer,
        tgt_tokenizer=train_dataset.tgt_tokenizer,
        max_length=max_length,
        vocab_size=data_config.max_vocab_size,
        dataset_name=data_config.dataset_name,
        data_dir=data_dir
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    src_vocab_size = train_dataset.src_tokenizer.get_vocab_size()
    tgt_vocab_size = train_dataset.tgt_tokenizer.get_vocab_size()

    logger.info(f"数据加载器初始化完成:")
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    logger.info(f"  源语言词汇表: {src_vocab_size}")
    logger.info(f"  目标语言词汇表: {tgt_vocab_size}")
    logger.info(f"  数据集位置: {data_dir}")

    return train_loader, val_loader, src_vocab_size, tgt_vocab_size
