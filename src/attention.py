"""Multi-head self-attention written from basic tensor operations.

torch.nn.MultiheadAttention / TransformerEncoder are NOT used here; they appear
only in tests as independent references. nn.Linear, autograd and dropout are
allowed by the assignment.

Shapes (B batch, T tokens, d model width, h heads, d_h = d / h):
    x            [B, T, d]
    qkv          [B, T, 3d]  -> q, k, v  [B, T, d] each
    q,k,v heads  [B, h, T, d_h]
    scores       [B, h, T, T]   (scaled by 1/sqrt(d_h), NOT 1/sqrt(d))
    weights      [B, h, T, T]   softmax over the LAST axis (keys)
    out          [B, h, T, d_h] -> transpose(1, 2) -> [B, T, h, d_h] -> [B, T, d]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def masked_softmax_attention(q, k, v, key_padding_mask=None, attn_mask=None, dropout=None):
    """Scaled dot-product attention for [..., T, d_h] tensors.

    key_padding_mask: bool [B, T_k], True = padding (excluded as key)
    attn_mask:        bool [T_q, T_k], True = not allowed (e.g. causal upper triangle)
    Rows whose keys are ALL masked return a zero output vector (defined behaviour).
    """
    d_h = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)
    neg = torch.finfo(scores.dtype).min
    blocked = None
    if key_padding_mask is not None:
        blocked = key_padding_mask[:, None, None, :] if scores.dim() == 4 else key_padding_mask[:, None, :]
    if attn_mask is not None:
        blocked = attn_mask if blocked is None else (blocked | attn_mask)
    if blocked is not None:
        scores = scores.masked_fill(blocked, neg)
    weights = torch.softmax(scores, dim=-1)
    if blocked is not None:
        # finfo.min keeps the softmax finite; a fully masked row becomes uniform,
        # so zero it explicitly (no NaN, no fake information).
        weights = weights.masked_fill(blocked, 0.0)
        all_blocked = blocked.all(dim=-1, keepdim=True)
        weights = weights.masked_fill(all_blocked, 0.0)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.h, self.d_h = d_model, n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)   # W_Q, W_K, W_V stacked
        self.out = nn.Linear(d_model, d_model, bias=bias)       # W_O
        self.attn_drop = nn.Dropout(dropout)
        self.last_weights = None

    def split_heads(self, t):
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.d_h).transpose(1, 2)          # [B, h, T, d_h]

    def merge_heads(self, t):
        B, h, T, d_h = t.shape
        return t.transpose(1, 2).reshape(B, T, h * d_h)                # transpose FIRST

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        B, T, d = x.shape
        assert d == self.d_model
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        assert q.shape == (B, self.h, T, self.d_h)
        o, w = masked_softmax_attention(q, k, v, key_padding_mask, attn_mask, self.attn_drop)
        self.last_weights = w.detach()
        o = self.merge_heads(o)
        assert o.shape == (B, T, d)
        return self.out(o)
