import torch
from src.model import Transformer
from tokenizers import Tokenizer
import argparse
import yaml
from src.config import Config
import os
import torch.nn.functional as F


class Translator:
    def __init__(self, config_path, model_path, dataset_dir="./datasets/iwslt2017"):
        # 加载配置
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = Config(**config_dict)

        # 加载tokenizer
        tokenizer_dir = os.path.join(dataset_dir, "tokenizers")
        self.src_tokenizer = Tokenizer.from_file(f"{tokenizer_dir}/src_tokenizer.json")
        self.tgt_tokenizer = Tokenizer.from_file(f"{tokenizer_dir}/tgt_tokenizer.json")

        # 初始化模型
        self.model = Transformer(
            src_vocab_size=self.src_tokenizer.get_vocab_size(),
            tgt_vocab_size=self.tgt_tokenizer.get_vocab_size(),
            d_model=self.config.model.d_model,
            num_heads=self.config.model.num_heads,
            num_layers=self.config.model.num_layers,
            d_ff=self.config.model.d_ff,
            dropout=0.0,
            max_seq_length=self.config.training.max_seq_length,
            use_positional_encoding=self.config.model.use_positional_encoding,
            use_residual=self.config.model.use_residual
        )

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 获取特殊token
        self.sos_token_id = self.src_tokenizer.token_to_id("[SOS]")
        self.eos_token_id = self.src_tokenizer.token_to_id("[EOS]")
        self.pad_token_id = self.src_tokenizer.token_to_id("[PAD]")

        print(f"翻译器加载成功")
        print(f"模型困惑度: {torch.exp(torch.tensor(checkpoint['val_loss'])):.2f}")

    def preprocess(self, text):
        """预处理输入文本"""
        encoding = self.src_tokenizer.encode(text)
        input_ids = [self.sos_token_id] + encoding.ids + [self.eos_token_id]

        # 填充/截断
        max_length = self.config.training.max_seq_length
        if len(input_ids) < max_length:
            input_ids = input_ids + [self.pad_token_id] * (max_length - len(input_ids))
        else:
            input_ids = input_ids[:max_length]

        return torch.tensor([input_ids], dtype=torch.long)

    def postprocess(self, output_ids):
        """后处理输出序列"""
        tokens = []
        for token_id in output_ids:
            token_id_val = token_id.item()
            if token_id_val in [self.eos_token_id, self.pad_token_id]:
                break
            if token_id_val != self.sos_token_id:
                token = self.tgt_tokenizer.id_to_token(token_id_val)
                if token and token != '[UNK]':
                    tokens.append(token)

        return " ".join(tokens) if tokens else ""

    def translate(self, text, max_length=50):
        with torch.no_grad():
            # 预处理
            src = self.preprocess(text).to(self.device)
            src_mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)

            # 编码
            enc_output, _ = self.model.encode(src, src_mask)

            # 生成
            tgt = torch.tensor([[self.tgt_tokenizer.token_to_id("[SOS]")]], device=self.device)
            generated_tokens = []

            for i in range(max_length):
                tgt_mask = self.model._create_causal_mask(tgt.size(1)).to(self.device)
                dec_output, _, _ = self.model.decode(tgt, enc_output, src_mask, tgt_mask)
                output = self.model.output_layer(dec_output[:, -1, :])

                # 适度的重复惩罚 (1.1)
                for token in set(generated_tokens):
                    output[0, token] = output[0, token] / 1.1

                # 温度调节 (0.8)
                output = output / 0.8
                probs = F.softmax(output, dim=-1)

                # 核采样 (top-p=0.9)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # 移除累积概率超过0.9的token
                sorted_indices_to_remove = cumulative_probs > 0.9
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[0, indices_to_remove] = 0

                # 重新归一化并采样
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 如果所有概率都为0，回退到贪婪
                    next_token = output.argmax(-1).unsqueeze(0)

                tgt = torch.cat([tgt, next_token], dim=1)
                generated_tokens.append(next_token.item())

                if next_token.item() == self.tgt_tokenizer.token_to_id("[EOS]"):
                    break

            return self.postprocess(tgt[0])


def main():
    parser = argparse.ArgumentParser(description="Transformer Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="./datasets/iwslt2017")

    args = parser.parse_args()

    translator = Translator(args.config, args.model, args.dataset_dir)

    print("翻译模式")
    print("输入 'quit' 退出")
    print("-" * 30)

    while True:
        text = input("输入英文: ").strip()
        if text.lower() == 'quit':
            break
        if not text:
            continue

        result = translator.translate(text)
        print(f"德语: {result}")


if __name__ == "__main__":
    main()
