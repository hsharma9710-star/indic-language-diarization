import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
W2V-E2E language-diarization head (Mishra et al., Interspeech 2023), generalised from 3 to n_lang classes.

Statistical pooling (W2V-ES) or attention pooling (W2V-EA) over every 10 wav2vec2 frames (200 ms)
-> FC1 -> FC2 (256-d embedding) -> classification-embedding block (auxiliary head)
 + self-attention block (4 encoder blocks) -> per-bin class logits.

Same fixes as the earlier 'correctjagbandhu' study: LayerNorm instead of affine-free BatchNorm,
no sigmoid before the cross-entropy, attention pooling on the true 768-d input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ildiar.transformer import LayerNorm, PositionalEncoding, EncoderBlock

FRAMES_PER_POOL = 10


class _PooledHead(nn.Module):
    def __init__(self, pooled_dim, feat_dim, d_k, d_v, d_ff, n_heads, dropout, n_lang, max_seq_len, device):
        super().__init__()
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(pooled_dim, 3000)
        self.ln1 = LayerNorm(3000)
        self.fc2 = nn.Linear(3000, feat_dim)
        self.ln2 = LayerNorm(feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.ln3 = LayerNorm(feat_dim)
        self.fc4 = nn.Linear(feat_dim, n_lang)          # auxiliary (context-free) head
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim, device=device)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_blocks = nn.ModuleList(
            [EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout) for _ in range(4)])
        self.output_fc = nn.Linear(feat_dim, n_lang)

    def forward(self, pooled, b, s, seq_len, atten_mask, return_embedding=False):
        x = self.ln1(F.relu(self.fc1(pooled)))
        embedding = self.fc2(x)                          # [B*P, 256]
        x = self.ln2(F.relu(embedding))
        x = self.dropout(x)
        x = self.ln3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        aux_logits = self.fc4(x)
        embedding = embedding.view(b, s, self.feat_dim)
        out = self.layernorm1(embedding)
        out = self.pos_encoding(out, seq_len)
        out = self.layernorm2(out)
        for blk in self.attention_blocks:
            out, _ = blk(out, atten_mask)
        trans_logits = self.output_fc(out)               # [B, P, n_lang]
        if return_embedding:
            return trans_logits, aux_logits, out         # out = context-aware 256-d embedding per bin
        return trans_logits, aux_logits


class W2V_ES(nn.Module):
    """Statistical pooling: mean + std over each 10-frame group -> 1536-d."""

    def __init__(self, wav2vec_dim=768, feat_dim=256, d_k=256, d_v=256, d_ff=2048, n_heads=4,
                 dropout=0.1, n_lang=15, max_seq_len=1024,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.head = _PooledHead(wav2vec_dim * 2, feat_dim, d_k, d_v, d_ff, n_heads, dropout, n_lang, max_seq_len, device)

    def forward(self, x, seq_len, atten_mask, return_embedding=False):
        b, t, f = x.size()
        s = t // FRAMES_PER_POOL
        x = x[:, : s * FRAMES_PER_POOL, :].reshape(b, s, FRAMES_PER_POOL, f)
        pooled = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=-1).reshape(b * s, 2 * f)
        return self.head(pooled, b, s, seq_len, atten_mask, return_embedding)


class W2V_EA(nn.Module):
    """Attention pooling: learned softmax weights over each 10-frame group -> 768-d."""

    def __init__(self, wav2vec_dim=768, feat_dim=256, d_k=256, d_v=256, d_ff=2048, n_heads=4,
                 dropout=0.1, n_lang=15, max_seq_len=1024,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.attn_fc1 = nn.Linear(wav2vec_dim, wav2vec_dim)
        self.attn_fc2 = nn.Linear(wav2vec_dim, wav2vec_dim)
        self.attn_score = nn.Linear(wav2vec_dim, 1)
        self.head = _PooledHead(wav2vec_dim, feat_dim, d_k, d_v, d_ff, n_heads, dropout, n_lang, max_seq_len, device)

    def forward(self, x, seq_len, atten_mask, return_embedding=False):
        b, t, f = x.size()
        s = t // FRAMES_PER_POOL
        x = x[:, : s * FRAMES_PER_POOL, :].reshape(b * s, FRAMES_PER_POOL, f)
        h = F.relu(self.attn_fc2(F.relu(self.attn_fc1(x))))
        w = F.softmax(self.attn_score(h), dim=1)         # softmax over the 10 frames
        pooled = (h * w).sum(dim=1)                      # [B*P, 768]
        return self.head(pooled, b, s, seq_len, atten_mask, return_embedding)


def get_atten_mask(seq_lens, batch_size, device=None):
    max_len = max(seq_lens)
    m = torch.ones([batch_size, max_len, max_len], device=device)
    for i in range(batch_size):
        m[i, : seq_lens[i], : seq_lens[i]] = 0
    return m.bool()
