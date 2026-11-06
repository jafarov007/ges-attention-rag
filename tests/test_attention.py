"""Implementation checks for the from-scratch attention and the D1-D3 models."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attention import MultiHeadSelfAttention, masked_softmax_attention  # noqa: E402
from models import D1MeanPool, D2AttentionPool, D3Transformer, count_parameters  # noqa: E402
from tokenizer import PAD  # noqa: E402

torch.manual_seed(0)
X_M2 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

# reference values (4 decimals) from docs/06_matematik.md, M2
A_UNMASKED = [[0.4011, 0.1978, 0.4011], [0.1978, 0.4011, 0.4011], [0.2483, 0.2483, 0.5035]]
O_UNMASKED = [[0.8022, 0.5989], [0.5989, 0.8022], [0.7517, 0.7517]]
A_PADDED = [[0.6698, 0.3302, 0.0], [0.3302, 0.6698, 0.0], [0.5, 0.5, 0.0]]
A_CAUSAL = [[1.0, 0.0, 0.0], [0.3302, 0.6698, 0.0], [0.2483, 0.2483, 0.5035]]


def _ids(rows, T=None):
    T = T or max(len(r) for r in rows)
    t = torch.full((len(rows), T), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        t[i, : len(r)] = torch.tensor(r)
    return t


# ---------------------------------------------------------------- M2 values
def test_m2_unmasked():
    q = k = v = X_M2[None]
    o, a = masked_softmax_attention(q, k, v)
    assert torch.allclose(a[0], torch.tensor(A_UNMASKED), atol=1e-4)
    assert torch.allclose(o[0], torch.tensor(O_UNMASKED), atol=1e-4)


def test_m2_key_padding():
    q = k = v = X_M2[None]
    o, a = masked_softmax_attention(q, k, v, key_padding_mask=torch.tensor([[False, False, True]]))
    assert torch.allclose(a[0], torch.tensor(A_PADDED), atol=1e-4)
    # pooled output over NON-padding queries only
    assert torch.allclose(o[0, :2].mean(0), torch.tensor([0.5, 0.5]), atol=1e-4)


def test_m2_causal():
    q = k = v = X_M2[None]
    causal = torch.triu(torch.ones(3, 3, dtype=torch.bool), 1)
    o, a = masked_softmax_attention(q, k, v, attn_mask=causal)
    assert torch.allclose(a[0], torch.tensor(A_CAUSAL), atol=1e-4)
    assert torch.equal(a[0][causal], torch.zeros(3))


def test_m2_matches_torch_sdpa():
    q = k = v = X_M2[None, None]
    o, _ = masked_softmax_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(o, ref)


# ---------------------------------------------------------------- properties
@pytest.fixture
def mha():
    torch.manual_seed(1)
    return MultiHeadSelfAttention(32, 4).eval()


def test_rows_sum_to_one_and_masked_weights_zero(mha):
    x = torch.randn(3, 7, 32)
    pad = torch.zeros(3, 7, dtype=torch.bool)
    pad[0, 5:] = True
    pad[2, 3:] = True
    mha(x, key_padding_mask=pad)
    w = mha.last_weights                                         # [B, h, T, T]
    torch.testing.assert_close(w.sum(-1), torch.ones(3, 4, 7))
    assert torch.equal(w[0, :, :, 5:], torch.zeros_like(w[0, :, :, 5:]))
    assert torch.equal(w[2, :, :, 3:], torch.zeros_like(w[2, :, :, 3:]))


def test_matches_torch_multiheadattention(mha):
    ref = torch.nn.MultiheadAttention(32, 4, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(mha.qkv.weight)
        ref.in_proj_bias.copy_(mha.qkv.bias)
        ref.out_proj.weight.copy_(mha.out.weight)
        ref.out_proj.bias.copy_(mha.out.bias)
    x = torch.randn(2, 6, 32)
    pad = torch.tensor([[False] * 6, [False] * 4 + [True] * 2])
    ours = mha(x, key_padding_mask=pad)
    theirs, _ = ref(x, x, x, key_padding_mask=pad, need_weights=False)
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)


def test_wrong_head_merge_is_detected(mha):
    """Negative control (slide 9): reshape without transpose keeps the shape but mixes tokens."""
    x = torch.randn(2, 5, 32)
    q, k, v = mha.qkv(x).chunk(3, dim=-1)
    q, k, v = mha.split_heads(q), mha.split_heads(k), mha.split_heads(v)
    o, _ = masked_softmax_attention(q, k, v)
    right = mha.merge_heads(o)
    wrong = o.reshape(2, 5, 32)
    assert right.shape == wrong.shape
    assert not torch.allclose(right, wrong)


def test_wrong_scale_is_detected():
    """Negative control: scaling by sqrt(d_model) instead of sqrt(d_h) changes the result."""
    torch.manual_seed(2)
    q, k, v = torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8)
    ours, _ = masked_softmax_attention(q, k, v)
    wrong = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(32), -1) @ v
    assert not torch.allclose(ours, wrong, atol=1e-3)
    torch.testing.assert_close(ours, F.scaled_dot_product_attention(q, k, v))


def test_fully_masked_row_no_nan():
    q = k = v = torch.randn(1, 3, 4)
    o, w = masked_softmax_attention(q, k, v, key_padding_mask=torch.tensor([[True, True, True]]))
    assert not torch.isnan(o).any() and torch.equal(o, torch.zeros_like(o))
    assert torch.equal(w, torch.zeros_like(w))


def test_single_token_sequence(mha):
    x = torch.randn(1, 1, 32)
    mha(x)
    assert torch.allclose(mha.last_weights, torch.ones(1, 4, 1, 1))


def test_gradcheck_float64():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    pad = torch.tensor([[False, False, False, True]])
    assert torch.autograd.gradcheck(lambda a, b, c: masked_softmax_attention(a, b, c, key_padding_mask=pad)[0],
                                    (q, k, v), eps=1e-6, atol=1e-5)


def test_manual_finite_difference_dL_dQ():
    """dL/dQ by central differences equals autograd (M1 chain rule)."""
    torch.manual_seed(4)
    q = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    k, v = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 2, dtype=torch.float64)
    L = lambda qq: masked_softmax_attention(qq[None], k[None], v[None])[0].pow(2).sum()  # noqa: E731
    L(q).backward()
    num = torch.zeros_like(q)
    eps = 1e-6
    for i in range(3):
        for j in range(2):
            e = torch.zeros_like(q); e[i, j] = eps
            num[i, j] = (L(q.detach() + e) - L(q.detach() - e)) / (2 * eps)
    torch.testing.assert_close(q.grad, num, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("cls", [D1MeanPool, D2AttentionPool, D3Transformer])
def test_padding_invariance(cls):
    torch.manual_seed(5)
    m = cls(40).eval()
    a = _ids([[2, 3, 7, 4, 9, 5, 11, 12], [2, 3, 8, 4, 10, 5, 13]])
    b = _ids([[2, 3, 7, 4, 9, 5, 11, 12], [2, 3, 8, 4, 10, 5, 13]], T=14)
    torch.testing.assert_close(m(a), m(b))


@pytest.mark.parametrize("cls", [D1MeanPool, D2AttentionPool, D3Transformer])
def test_batch_equals_single(cls):
    torch.manual_seed(6)
    m = cls(40).eval()
    rows = [[2, 3, 7, 4, 9, 5, 11, 12, 14], [2, 3, 8, 4, 10, 5, 13]]
    batch = m(_ids(rows))
    single = torch.cat([m(_ids([r])) for r in rows])
    torch.testing.assert_close(batch, single, atol=1e-6, rtol=1e-5)


def test_d2_alpha_masked_and_normalised():
    torch.manual_seed(7)
    m = D2AttentionPool(40).eval()
    ids = _ids([[2, 3, 7, 4, 9], [2, 3, 8]])
    m(ids)
    a = m.last_alpha
    torch.testing.assert_close(a.sum(1), torch.ones(2))
    assert torch.equal(a[1, 3:], torch.zeros(2))


def test_permutation_without_positions():
    """M4: without positions, permuting non-[CLS] tokens leaves the [CLS] output unchanged."""
    torch.manual_seed(8)
    m_nopos = D3Transformer(40, use_positions=False).eval()
    m_pos = D3Transformer(40, use_positions=True).eval()
    m_pos.load_state_dict(m_nopos.state_dict())
    a = _ids([[2, 3, 7, 4, 9, 5, 11, 12]])
    b = _ids([[2, 12, 11, 5, 9, 4, 7, 3]])
    torch.testing.assert_close(m_nopos(a), m_nopos(b), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(m_pos(a), m_pos(b), atol=1e-4)


def test_parameter_counts():
    V = 100
    p1, p2, p3 = count_parameters(D1MeanPool(V)), count_parameters(D2AttentionPool(V)), count_parameters(D3Transformer(V))
    assert p1["emb"] == V * 32 and p1["head"] == 1089 and p1["total"] == V * 32 + 1089
    assert p2["W"] + p2["v"] == 1088 and p2["total"] == V * 32 + 1088 + 1089
    blk = count_parameters(D3Transformer(V).blocks[0])
    assert blk["attn"] == 4 * (32 * 32 + 32) == 4224
    assert blk["ffn"] == (32 * 64 + 64) + (64 * 32 + 32) == 4192
    assert blk["ln1"] + blk["ln2"] == 128 and blk["total"] == 8544
    assert p3["blocks"] == 17088 and p3["total"] == V * 32 + 17088 + 64 + 1089


def test_determinism_same_seed():
    def run():
        torch.manual_seed(9)
        m = D3Transformer(40, dropout=0.2).train()
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        ids = _ids([[2, 3, 7, 4, 9, 5, 11], [2, 3, 8, 4, 10, 5, 13]])
        for _ in range(3):
            opt.zero_grad(); m(ids).pow(2).mean().backward(); opt.step()
        return m.eval()(ids)
    torch.testing.assert_close(run(), run())


def test_overfit_tiny_batch():
    torch.manual_seed(10)
    m = D3Transformer(40)
    ids = _ids([[2, 3, 7, 4, 9, 5, 11], [2, 3, 8, 4, 10, 5, 13], [2, 3, 6, 4, 9, 5, 14]])
    y = torch.tensor([1.0, 3.0, 5.0])
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    for _ in range(300):
        opt.zero_grad(); loss = F.mse_loss(m(ids), y); loss.backward(); opt.step()
    assert loss.item() < 1e-2
