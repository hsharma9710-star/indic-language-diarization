"""W2V-E2E head (statistical or attention pooling + 4-block self-attention), n_lang output classes."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import EncoderBlock, LayerNorm, PositionalEncoding

FRAMES_PER_POOL = 10          # 10 wav2vec frames x 20 ms = one 200 ms bin


class _PooledHead(nn.Module):
    def __init__(self, pooled_dim, feat_dim, d_k, d_v, d_ff, n_heads, dropout, n_lang, max_seq_len):
        super().__init__()
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(pooled_dim, 3000)
        self.ln1 = LayerNorm(3000)
        self.fc2 = nn.Linear(3000, feat_dim)
        self.ln2 = LayerNorm(feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.ln3 = LayerNorm(feat_dim)
        self.fc4 = nn.Linear(feat_dim, n_lang)                    # auxiliary head (training only)
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len, feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_blocks = nn.ModuleList(
            [EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout) for _ in range(4)])
        self.output_fc = nn.Linear(feat_dim, n_lang)

    def forward(self, pooled, b, s, seq_len=None, atten_mask=None):
        x = self.ln1(F.relu(self.fc1(pooled)))
        emb = self.fc2(x).view(b, s, self.feat_dim)
        out = self.layernorm2(self.pos_encoding(self.layernorm1(emb)))
        for blk in self.attention_blocks:
            out, _ = blk(out, atten_mask)
        return self.output_fc(out), out                             # logits [B, S, n_lang], embedding [B, S, 256]


class W2V_ES(nn.Module):
    """Statistical pooling: mean || std over each 10-frame group -> 1536-d."""

    def __init__(self, wav2vec_dim=768, feat_dim=256, d_k=256, d_v=256, d_ff=2048, n_heads=4,
                 dropout=0.1, n_lang=15, max_seq_len=1024):
        super().__init__()
        self.head = _PooledHead(wav2vec_dim * 2, feat_dim, d_k, d_v, d_ff, n_heads, dropout, n_lang, max_seq_len)

    def forward(self, x, seq_len=None, atten_mask=None):
        b, t, f = x.shape
        s = t // FRAMES_PER_POOL
        x = x[:, : s * FRAMES_PER_POOL].reshape(b, s, FRAMES_PER_POOL, f)
        pooled = torch.cat((x.mean(2), x.std(2)), dim=-1).reshape(b * s, 2 * f)
        return self.head(pooled, b, s, seq_len, atten_mask)


class W2V_EA(nn.Module):
    """Attention pooling: learned softmax weights over each 10-frame group -> 768-d."""

    def __init__(self, wav2vec_dim=768, feat_dim=256, d_k=256, d_v=256, d_ff=2048, n_heads=4,
                 dropout=0.1, n_lang=15, max_seq_len=1024):
        super().__init__()
        self.attn_fc1 = nn.Linear(wav2vec_dim, wav2vec_dim)
        self.attn_fc2 = nn.Linear(wav2vec_dim, wav2vec_dim)
        self.attn_score = nn.Linear(wav2vec_dim, 1)
        self.head = _PooledHead(wav2vec_dim, feat_dim, d_k, d_v, d_ff, n_heads, dropout, n_lang, max_seq_len)

    def forward(self, x, seq_len=None, atten_mask=None):
        b, t, f = x.shape
        s = t // FRAMES_PER_POOL
        x = x[:, : s * FRAMES_PER_POOL].reshape(b * s, FRAMES_PER_POOL, f)
        h = F.relu(self.attn_fc2(F.relu(self.attn_fc1(x))))
        w = F.softmax(self.attn_score(h), dim=1)
        pooled = (h * w).sum(1)
        return self.head(pooled, b, s, seq_len, atten_mask)
