"""B1 (TF-IDF + Ridge) and the three neural regressors D1, D2, D3.

All neural models: same vocabulary, same embedding size (32), same regression
head (32 -> 32 -> 1), same target z = ln(1 + kWp).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from attention import MultiHeadSelfAttention
from common import tokenize_tr
from tokenizer import CLS, PAD

D_MODEL = 32


# ----------------------------------------------------------------- B1
def make_b1(alpha: float) -> Pipeline:
    pre = ColumnTransformer([
        ("name", TfidfVectorizer(tokenizer=tokenize_tr, lowercase=False, token_pattern=None,
                                 ngram_range=(1, 2), sublinear_tf=True), "name_clean"),
        ("district", OneHotEncoder(handle_unknown="ignore"), ["district_clean"]),
        ("year", OneHotEncoder(handle_unknown="ignore"), ["file_year"]),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


# ----------------------------------------------------------------- shared parts
class RegressionHead(nn.Module):
    """32 -> 32 (ReLU, dropout) -> 1. Parameters: (32*32+32) + (32+1) = 1089."""

    def __init__(self, d=D_MODEL, hidden=32, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sinusoidal_positions(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ----------------------------------------------------------------- D1
class D1MeanPool(nn.Module):
    """Masked mean of token embeddings -> head."""

    def __init__(self, vocab_size, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD)
        self.head = RegressionHead(dropout=dropout)

    def pooled(self, ids):
        m = (ids != PAD).float()                              # [B, T]
        h = self.emb(ids)                                     # [B, T, d]
        return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)

    def forward(self, ids):
        return self.head(self.pooled(ids))


# ----------------------------------------------------------------- D2
class D2AttentionPool(nn.Module):
    """Learned single-query attention pooling (additive / Bahdanau form):

        e_i = v^T tanh(W h_i + b),  alpha = softmax(e) over non-pad tokens,  r = sum_i alpha_i h_i

    W: 32x32 (+ b: 32), v: 32 -> 1088 pooling parameters. Tokens do NOT attend to
    each other (no contextualisation) - that is the difference from D3.
    equal_weights=True is ablation A1 (alpha_i = 1/n over non-pad tokens).
    """

    def __init__(self, vocab_size, dropout=0.0, equal_weights=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD)
        self.W = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, 1, bias=False)
        self.head = RegressionHead(dropout=dropout)
        self.equal_weights = equal_weights
        self.last_alpha = None

    def forward(self, ids):
        pad = ids == PAD                                      # [B, T]
        h = self.emb(ids)                                     # [B, T, d]
        if self.equal_weights:
            alpha = (~pad).float() / (~pad).float().sum(1, keepdim=True)
        else:
            e = self.v(torch.tanh(self.W(h))).squeeze(-1)     # [B, T]
            e = e.masked_fill(pad, torch.finfo(e.dtype).min)
            alpha = torch.softmax(e, dim=-1).masked_fill(pad, 0.0)
        self.last_alpha = alpha.detach()
        r = (alpha[..., None] * h).sum(1)                     # [B, d]
        return self.head(r)


# ----------------------------------------------------------------- D3
class PreLNEncoderBlock(nn.Module):
    """h = x + Drop(MHA(LN(x)));  y = h + Drop(FFN(LN(h)))."""

    def __init__(self, d=D_MODEL, h=4, d_ff=64, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, h, dropout=dropout)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad):
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask=pad))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class D3Transformer(nn.Module):
    """Embedding (+ fixed sinusoidal positions) -> 2 Pre-LN blocks -> final LN -> [CLS] -> head.

    use_positions=False is ablation A2. Block parameters: 2 x 8544 = 17088.
    """

    def __init__(self, vocab_size, dropout=0.0, n_layers=2, n_heads=4, d_ff=64, use_positions=True, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD)
        self.register_buffer("pe", sinusoidal_positions(max_len, D_MODEL), persistent=False)
        self.use_positions = use_positions
        self.blocks = nn.ModuleList([PreLNEncoderBlock(D_MODEL, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = RegressionHead(dropout=dropout)

    def forward(self, ids):
        pad = ids == PAD
        x = self.emb(ids)
        if self.use_positions:
            x = x + self.pe[: ids.size(1)][None]
        for blk in self.blocks:
            x = blk(x, pad)
        x = self.ln_f(x)
        assert bool((ids[:, 0] == CLS).all()), "first token must be [CLS]"
        return self.head(x[:, 0])

    def attention_maps(self):
        return [b.attn.last_weights for b in self.blocks]


def build_model(name: str, vocab_size: int, dropout: float, **kw) -> nn.Module:
    if name == "D1":
        return D1MeanPool(vocab_size, dropout)
    if name == "D2":
        return D2AttentionPool(vocab_size, dropout, equal_weights=kw.get("equal_weights", False))
    if name == "D3":
        return D3Transformer(vocab_size, dropout, use_positions=kw.get("use_positions", True))
    raise ValueError(name)


def count_parameters(model: nn.Module) -> dict:
    out = {}
    for n, p in model.named_parameters():
        top = n.split(".")[0]
        out[top] = out.get(top, 0) + p.numel()
    out["total"] = sum(p.numel() for p in model.parameters())
    return out


def to_kwp(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(z))
