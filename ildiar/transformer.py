"""Self-attention building blocks used by the diarization head.

Independent implementation of the post-LayerNorm Transformer encoder described in
Mishra et al., "End to End Spoken Language Diarization with Wav2vec Embeddings", Interspeech 2023.
Module / attribute names match the training code so that trained weights load strictly.
"""
import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, precomputed for up to `max_seq_len` positions."""

    def __init__(self, max_seq_len, features_dim, device=None):
        super().__init__()
        pos = np.arange(max_seq_len, dtype=np.float64)[:, None]
        i = np.arange(features_dim, dtype=np.float64)[None, :]
        table = pos / np.power(10000.0, 2.0 * (i // 2) / features_dim)
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        self.register_buffer("pos_enc", torch.from_numpy(table).float(), persistent=False)

    def forward(self, x, seq_len=None):
        return x + self.pos_enc[: x.size(1)].unsqueeze(0).to(x.dtype)


class LayerNorm(nn.Module):
    """LayerNorm with the unbiased standard deviation and eps added outside the square root."""

    def __init__(self, d_hid, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, atten_mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        if atten_mask is not None:
            scores = scores.masked_fill(atten_mask, -1e9)
        atten = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(atten, v), atten


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, d_k, d_v, n_heads, dropout):
        super().__init__()
        self.d_k, self.d_v, self.n_heads = d_k, d_v, n_heads
        self.w_q = Linear(feat_dim, d_k * n_heads)
        self.w_k = Linear(feat_dim, d_k * n_heads)
        self.w_v = Linear(feat_dim, d_v * n_heads)
        self.attenion = ScaledDotProductAttention(d_k, dropout)      # (sic) attribute name kept for weight compatibility

    def forward(self, x, atten_mask=None):
        b = x.size(0)
        q = self.w_q(x).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, -1, self.n_heads, self.d_v).transpose(1, 2)
        if atten_mask is not None:
            atten_mask = atten_mask.unsqueeze(1)
        ctx, atten = self.attenion(q, k, v, atten_mask)
        return ctx.transpose(1, 2).contiguous().view(b, -1, self.n_heads * self.d_v), atten


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, feat_dim, d_k, d_v, n_heads, dropout):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(feat_dim, d_k, d_v, n_heads, dropout)
        self.linear = Linear(n_heads * d_v, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(feat_dim)

    def forward(self, x, atten_mask=None):
        ctx, atten = self.multihead_attention(x, atten_mask)
        return self.layernorm(self.dropout(self.linear(ctx)) + x), atten


class PositionWiseFeedForward(nn.Module):
    def __init__(self, feat_dim, d_ff=2048, dropout=0.1):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = Linear(feat_dim, d_ff)
        self.fc2 = Linear(d_ff, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(feat_dim)

    def forward(self, x):
        return self.layernorm(self.dropout(self.fc2(self.relu(self.fc1(x)))) + x)


class EncoderBlock(nn.Module):
    def __init__(self, feat_dim, d_k, d_v, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(feat_dim, d_k, d_v, n_heads, dropout)
        self.position_wise_ff = PositionWiseFeedForward(feat_dim, d_ff, dropout)

    def forward(self, x, atten_mask=None):
        out, atten = self.self_attention(x, atten_mask)
        return self.position_wise_ff(out), atten
