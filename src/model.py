import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, Q, K, V, mask=None, key_padding_mask=None):
        batch_size, seq_len = Q.size(0), Q.size(1)

        # 线性变换并分头 (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数 (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用 padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq_len_k) → (batch, 1, 1, seq_len_k)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, -1e9)

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到V
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, d_k)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 线性变换
        output = self.W_o(output)

        return output, attn_weights


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_residual=True):
        super(EncoderLayer, self).__init__()
        self.use_residual = use_residual
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model) if use_residual else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_residual else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, key_padding_mask=None):
        # 多头自注意力
        attn_output, attn_weights = self.self_attention(x, x, x, mask, key_padding_mask)

        if self.use_residual:
            x = self.norm1(x + self.dropout(attn_output))
        else:
            x = self.dropout(attn_output)

        # 前馈网络
        ff_output = self.feed_forward(x)

        if self.use_residual:
            x = self.norm2(x + self.dropout(ff_output))
        else:
            x = self.dropout(ff_output)

        return x, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_residual=True):
        super(DecoderLayer, self).__init__()
        self.use_residual = use_residual

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)

        # 根据是否使用残差连接选择归一化层或恒等映射
        self.norm1 = nn.LayerNorm(d_model) if use_residual else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_residual else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if use_residual else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(
            self, x, enc_output,
            src_mask=None, tgt_mask=None,
            src_key_padding_mask=None, tgt_key_padding_mask=None
    ):
        # 自注意力
        self_attn_output, self_attn_weights = self.self_attention(
            x, x, x, tgt_mask, tgt_key_padding_mask
        )

        if self.use_residual:
            x = self.norm1(x + self.dropout(self_attn_output))
        else:
            x = self.dropout(self_attn_output)

        # 交叉注意力
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, enc_output, enc_output, src_mask, src_key_padding_mask
        )

        if self.use_residual:
            x = self.norm2(x + self.dropout(cross_attn_output))
        else:
            x = self.dropout(cross_attn_output)

        # 前馈网络
        ff_output = self.feed_forward(x)

        if self.use_residual:
            x = self.norm3(x + self.dropout(ff_output))
        else:
            x = self.dropout(ff_output)

        return x, self_attn_weights, cross_attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_length=5000,
            dropout=0.1,
            use_positional_encoding=True,
            use_residual=True
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.use_residual = use_residual

        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        else:
            self.positional_encoding = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, use_residual)
            for _ in range(num_layers)
        ])

        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_residual)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 特别初始化词嵌入层
        nn.init.normal_(self.src_embedding.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.tgt_embedding.weight, mean=0, std=self.d_model ** -0.5)

        # 输出层的特殊初始化
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        # 源语言嵌入
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)

        # 位置编码
        if self.positional_encoding is not None:
            src_embedded = self.positional_encoding(src_embedded)

        src_embedded = self.dropout(src_embedded)

        enc_output = src_embedded
        enc_self_attentions = []

        for enc_layer in self.encoder_layers:
            enc_output, enc_self_attn = enc_layer(
                enc_output,
                src_mask,
                key_padding_mask=src_key_padding_mask
            )
            enc_self_attentions.append(enc_self_attn)

        return enc_output, enc_self_attentions

    def decode(
            self, tgt, enc_output,
            src_mask=None, tgt_mask=None,
            src_key_padding_mask=None, tgt_key_padding_mask=None
    ):
        # 目标语言嵌入
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # 位置编码
        if self.positional_encoding is not None:
            tgt_embedded = self.positional_encoding(tgt_embedded)

        tgt_embedded = self.dropout(tgt_embedded)

        dec_output = tgt_embedded
        dec_self_attentions = []
        dec_cross_attentions = []

        for dec_layer in self.decoder_layers:
            dec_output, dec_self_attn, dec_cross_attn = dec_layer(
                dec_output, enc_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            dec_self_attentions.append(dec_self_attn)
            dec_cross_attentions.append(dec_cross_attn)

        return dec_output, dec_self_attentions, dec_cross_attentions

    def forward(
            self, src, tgt,
            src_mask=None, tgt_mask=None,
            src_key_padding_mask=None, tgt_key_padding_mask=None
    ):
        # 编码器
        enc_output, enc_self_attentions = self.encode(
            src, src_mask, src_key_padding_mask
        )

        # 解码器
        dec_output, dec_self_attentions, dec_cross_attentions = self.decode(
            tgt, enc_output, src_mask, tgt_mask,
            src_key_padding_mask, tgt_key_padding_mask
        )

        # 输出层
        output = self.output_layer(dec_output)

        return output, {
            'enc_self_attentions': enc_self_attentions,
            'dec_self_attentions': dec_self_attentions,
            'dec_cross_attentions': dec_cross_attentions
        }

    def generate(self, src, src_mask, max_length, start_token, end_token, device):
        """生成序列"""
        self.eval()

        # 编码源序列
        enc_output, _ = self.encode(src, src_mask)

        # 初始化目标序列
        tgt = torch.tensor([[start_token]], device=device)

        for _ in range(max_length):
            # 创建目标mask
            tgt_mask = self._create_causal_mask(tgt.size(1)).to(device)

            # 解码
            dec_output, _, _ = self.decode(tgt, enc_output, src_mask, tgt_mask)
            output = self.output_layer(dec_output[:, -1, :])

            # 获取下一个token
            next_token = output.argmax(-1).unsqueeze(0)

            # 添加到序列
            tgt = torch.cat([tgt, next_token], dim=1)

            # 如果生成了结束符则停止
            if next_token.item() == end_token:
                break

        return tgt.squeeze(0)

    def _create_causal_mask(self, size):
        """创建因果掩码"""
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)
