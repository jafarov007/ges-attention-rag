"""Numerical companions of the math section (docs/06_matematik.md).

M1: Monte-Carlo check of Var(q.k) = d_k and of softmax saturation without scaling
M2: hand-calculation matrices (unmasked / key padding / causal), NumPy vs PyTorch
M3: parameter counts and attention-matrix sizes, computed from the real modules
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from attention import masked_softmax_attention
from common import FIG, OUT_TABLES
from models import D3Transformer, count_parameters

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def np_softmax(s):
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def m2():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Q = K = V = X  # W_Q = W_K = W_V = I_2
    S = Q @ K.T / np.sqrt(2)
    A = np_softmax(S)
    S_pad = S.copy(); S_pad[:, 2] = -np.inf
    A_pad = np_softmax(S_pad)
    S_c = S + np.triu(np.full((3, 3), -np.inf), 1)
    A_c = np_softmax(S_c)
    res = {"S": S, "A": A, "O": A @ V, "A_pad": A_pad, "O_pad": A_pad @ V,
           "pooled_pad": (A_pad @ V)[:2].mean(0), "A_causal": A_c, "O_causal": A_c @ V}
    # PyTorch implementation must agree
    t = torch.tensor(X, dtype=torch.float32)[None]
    o, a = masked_softmax_attention(t, t, t)
    assert np.allclose(a[0].numpy(), A, atol=1e-6) and np.allclose(o[0].numpy(), A @ V, atol=1e-6)
    o, a = masked_softmax_attention(t, t, t, key_padding_mask=torch.tensor([[False, False, True]]))
    assert np.allclose(a[0].numpy(), A_pad, atol=1e-6)
    o, a = masked_softmax_attention(t, t, t, attn_mask=torch.triu(torch.ones(3, 3, dtype=torch.bool), 1))
    assert np.allclose(a[0].numpy(), A_c, atol=1e-6)
    for nm in ("A", "A_pad", "A_causal"):
        assert np.allclose(res[nm].sum(1), 1.0)
    rows = []
    for k, v in res.items():
        rows.append({"matrix": k, "values": np.array2string(np.round(v, 4), separator=", ").replace("\n", "")})
    pd.DataFrame(rows).to_csv(OUT_TABLES / "m2_values.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a, (title, M) in zip(ax, [("unmasked", A), ("key padding (token 3 = PAD)", A_pad), ("causal", A_c)]):
        a.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=8,
                       color="white" if M[i, j] > 0.6 else "black")
        a.set_xticks(range(3), ["k1", "k2", "k3"]); a.set_yticks(range(3), ["q1", "q2", "q3"])
        a.set_title(f"A = softmax(QK^T/sqrt(2)) - {title}", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig18_m2_attention.png", bbox_inches="tight"); plt.close(fig)
    return res


def m1_simulation(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for dk in [2, 4, 8, 16, 32, 64, 128, 256]:
        q, k = rng.standard_normal((n, dk)), rng.standard_normal((n, dk))
        dot = (q * k).sum(1)
        # 8 keys: average max softmax weight with and without scaling
        Kq = rng.standard_normal((2000, 8, dk)); qq = rng.standard_normal((2000, dk))
        logits = np.einsum("nkd,nd->nk", Kq, qq)
        pmax_raw = np_softmax(logits).max(1).mean()
        pmax_sc = np_softmax(logits / np.sqrt(dk)).max(1).mean()
        rows.append(dict(d_k=dk, var_dot=dot.var(), var_scaled=(dot / np.sqrt(dk)).var(),
                         mean_max_softmax_unscaled=pmax_raw, mean_max_softmax_scaled=pmax_sc))
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT_TABLES / "m1_variance_simulation.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3))
    ax[0].plot(df.d_k, df.var_dot, "o-", label="Var(q.k) simulated")
    ax[0].plot(df.d_k, df.d_k, "k--", lw=1, label="theory: d_k")
    ax[0].plot(df.d_k, df.var_scaled, "s-", label="Var(q.k / sqrt(d_k))")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log", base=2); ax[0].set_xlabel("d_k"); ax[0].legend(fontsize=7)
    ax[0].set_title("M1: dot-product variance grows with d_k")
    ax[1].plot(df.d_k, df.mean_max_softmax_unscaled, "o-", color="#C44E52", label="unscaled")
    ax[1].plot(df.d_k, df.mean_max_softmax_scaled, "s-", color="#55A868", label="scaled by 1/sqrt(d_k)")
    ax[1].axhline(1 / 8, color="gray", ls=":", lw=1); ax[1].set_xscale("log", base=2)
    ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("d_k"); ax[1].set_ylabel("mean max weight (8 keys)")
    ax[1].set_title("Unscaled softmax saturates (one-hot)"); ax[1].legend(fontsize=7)
    p = np.linspace(0.5, 0.999, 200)
    jnorm = [np.linalg.norm(np.diag([pp, 1 - pp]) - np.outer([pp, 1 - pp], [pp, 1 - pp])) for pp in p]
    ax[2].plot(p, jnorm, color="#4C72B0")
    ax[2].scatter([0.5, 0.99], [np.linalg.norm(np.diag([.5, .5]) - np.outer([.5, .5], [.5, .5])),
                                np.linalg.norm(np.diag([.99, .01]) - np.outer([.99, .01], [.99, .01]))], color="k", zorder=3)
    ax[2].set_xlabel("max softmax probability p (2 classes)"); ax[2].set_ylabel("||diag(a) - a a^T||_F")
    ax[2].set_title("Jacobian vanishes as softmax saturates")
    fig.tight_layout(); fig.savefig(FIG / "fig19_m1_scaling_saturation.png", bbox_inches="tight"); plt.close(fig)
    return df


def m3():
    blk = count_parameters(D3Transformer(100).blocks[0])
    rows = [
        ("Q,K,V,O projections (with bias)", "4(d^2 + d)", blk["attn"]),
        ("FFN 32->64->32", "(d d_ff + d_ff) + (d_ff d + d)", blk["ffn"]),
        ("2 x LayerNorm", "2 x 2d", blk["ln1"] + blk["ln2"]),
        ("one block", "", blk["total"]),
        ("two blocks", "", 2 * blk["total"]),
        ("final LayerNorm (Pre-LN)", "2d", 64),
        ("regression head 32->32->1", "(32*32+32)+(32+1)", 1089),
        ("token embedding", "V x 32", "depends on train vocabulary"),
    ]
    pd.DataFrame(rows, columns=["component", "rule", "parameters"]).to_csv(OUT_TABLES / "m3_parameters.csv", index=False)
    sizes = pd.DataFrame([dict(n=n, elements_per_head=n * n, elements_4_heads=4 * n * n,
                               attention_flops_QK=n * n * 32, ffn_flops=n * 32 * 64 * 2) for n in (16, 64, 256)])
    sizes.to_csv(OUT_TABLES / "m3_attention_size.csv", index=False)
    return rows, sizes


def main():
    r = m2()
    print("M2 S\n", np.round(r["S"], 4), "\nA\n", np.round(r["A"], 4), "\nO\n", np.round(r["O"], 4))
    print("pad A\n", np.round(r["A_pad"], 4), "\npooled", np.round(r["pooled_pad"], 4))
    print("causal A\n", np.round(r["A_causal"], 4), "\nO\n", np.round(r["O_causal"], 4))
    print(m1_simulation().round(3).to_string(index=False))
    rows, sizes = m3()
    for row in rows:
        print(row)
    print(sizes.to_string(index=False))


if __name__ == "__main__":
    main()
