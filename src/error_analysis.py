"""Error analysis on outer-test predictions (docs/11).

* subgroup errors (file year, district seen in training or not, power bin, facility type, group size)
* largest errors with context
* attention weights vs token-deletion effect (D2 alpha, D3 [CLS] attention), Spearman per record
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common import ABLATION_SEED, FIG, OUT_DATA, OUT_MODELS, OUT_PRED, OUT_TABLES, PROTOCOL
from models import build_model
from s04_eda import facility_type
from tokenizer import SPECIALS, UNK, Vocab, batch_encode, record_tokens

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 8})
BINS = [0, 10, 50, 150, 1000]
BIN_LABELS = ["<=10", "10-50", "50-150", ">150"]


def load_model(label, k):
    ck = torch.load(OUT_MODELS / f"{label}_outer{k}_seed{ABLATION_SEED}.pt", weights_only=False)
    v = Vocab(); v.itos = ck["itos"]; v.stoi = {t: i for i, t in enumerate(v.itos)}
    m = build_model(ck["name"], len(v.itos), ck["dropout"], **(ck["extra"] or {}))
    m.load_state_dict(ck["state"]); m.eval()
    return m, v


def predict_tokens(model, vocab, toks):
    ids, _ = batch_encode(vocab, [toks])
    with torch.no_grad():
        z = float(model(ids)[0])
    return z, ids


def attention_vs_deletion(df, ens, label, k_of):
    rows, examples = [], {}
    for r in ens.itertuples():
        rec = df.loc[r.record_id]
        model, vocab = load_model(label, k_of[r.record_id])
        toks = record_tokens(rec.district_clean, rec.file_year, rec.name_clean)
        z0, ids = predict_tokens(model, vocab, toks)
        if label == "D2":
            att = model.last_alpha[0].numpy()
        else:
            att = model.attention_maps()[-1][0].mean(0)[0].numpy()     # last layer, mean over heads, [CLS] row
        pos = [i for i, t in enumerate(toks) if t not in SPECIALS]
        dele = []
        for i in pos:
            z1, _ = predict_tokens(model, vocab, toks[:i] + toks[i + 1:])
            dele.append(abs(z1 - z0))
        a = att[pos]
        rho = spearmanr(a, dele).statistic if len(pos) > 2 and np.std(dele) > 0 and np.std(a) > 0 else np.nan
        unk = sum(vocab.stoi.get(toks[i], UNK) == UNK for i in pos)
        rows.append(dict(model=label, record_id=r.record_id, n_tokens=len(pos), spearman=rho, n_unk=unk,
                         top_attention_token=toks[pos[int(np.argmax(a))]], top_deletion_token=toks[pos[int(np.argmax(dele))]]))
        examples[r.record_id] = ([toks[i] for i in pos], a, np.array(dele), [vocab.stoi.get(toks[i], UNK) == UNK for i in pos])
    return pd.DataFrame(rows), examples


def main():
    df = pd.read_csv(OUT_DATA / "records_clean.csv").set_index("record_id")
    man = pd.read_csv(PROTOCOL / "split_manifest.csv").set_index("record_id")
    p = pd.read_csv(OUT_PRED / "outer_predictions_main.csv")
    ens = p.groupby(["model", "record_id", "entity_group_id", "outer_fold", "y"], as_index=False).y_hat.mean()
    ens["abs_err"] = (ens.y - ens.y_hat).abs()
    ens["signed_err"] = ens.y_hat - ens.y
    ens["file_year"] = ens.record_id.map(df.file_year)
    ens["type"] = ens.record_id.map(df.name_ws).map(facility_type)
    ens["power_bin"] = pd.cut(ens.y, BINS, labels=BIN_LABELS)
    gsize = df[df.in_regression].groupby("entity_group_id").size()
    ens["group_size"] = np.where(ens.entity_group_id.map(gsize) > 1, "repeated facility", "single record")
    reg = df[df.in_regression].join(man[["outer_fold"]])
    seen = {}
    for k in range(5):
        tr_d = set(reg[reg.outer_fold != k].district_clean.value_counts().loc[lambda s: s >= 2].index)  # min_freq=2
        for rid in reg[reg.outer_fold == k].index:
            seen[rid] = df.loc[rid, "district_clean"] in tr_d
    ens["district_in_vocab"] = ens.record_id.map(seen).map({True: "district token known", False: "district -> [UNK]"})

    # ---------------- subgroup table + heatmap
    sub = []
    for col in ["file_year", "district_in_vocab", "power_bin", "type", "group_size"]:
        for (m, val), g in ens.groupby(["model", col], observed=True):
            sub.append(dict(model=m, factor=col, level=str(val), n=len(g), mae=g.abs_err.mean(),
                            mean_signed_error=g.signed_err.mean()))
    sub = pd.DataFrame(sub)
    sub.round(2).to_csv(OUT_TABLES / "error_subgroups.csv", index=False)
    piv = sub[sub.model.isin(["B0", "B1", "D2", "D3"])].pivot_table(index=["factor", "level"], columns="model", values="mae")
    nn = sub[sub.model == "D3"].set_index(["factor", "level"]).n
    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.imshow(piv.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [f"{f}: {l} (n={nn.get((f, l), 0)})" for f, l in piv.index], fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("Record MAE (kWp) by subgroup, outer test, 3-seed mean prediction", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig29_error_subgroups.png", bbox_inches="tight"); plt.close(fig)

    # ---------------- signed error vs size (Jensen / shrinkage)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for m, c in [("B1", "#937860"), ("D2", "#55A868"), ("D3", "#C44E52")]:
        g = ens[ens.model == m].groupby("power_bin", observed=True).signed_err.mean()
        ax.plot(range(len(g)), g.values, "o-", color=c, label=m)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(4), [f"{b} kWp" for b in BIN_LABELS])
    ax.set_ylabel("mean (prediction - truth), kWp"); ax.legend()
    ax.set_title("Small plants over-predicted, large plants under-predicted (shrinkage + log target)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig30_signed_error_by_size.png", bbox_inches="tight"); plt.close(fig)

    # ---------------- largest errors
    d3 = ens[ens.model == "D3"].sort_values("abs_err", ascending=False).head(8).copy()
    d3["name"] = d3.record_id.map(df.name_ws); d3["district"] = d3.record_id.map(df.district_clean)
    d3["B1_pred"] = d3.record_id.map(ens[ens.model == "B1"].set_index("record_id").y_hat)
    d3["quality_flag"] = d3.record_id.map(df.quality_flag)
    d3[["record_id", "district", "name", "file_year", "y", "y_hat", "B1_pred", "abs_err", "type", "group_size",
        "district_in_vocab", "quality_flag"]].round(1).to_csv(OUT_TABLES / "largest_errors_D3.csv", index=False)

    # ---------------- attention vs token deletion
    k_of = man.outer_fold.to_dict()
    att_all = []
    ex = {}
    for label in ["D2", "D3"]:
        a, e = attention_vs_deletion(df, ens[ens.model == label], label, k_of)
        att_all.append(a); ex[label] = e
    att = pd.concat(att_all)
    att.round(3).to_csv(OUT_TABLES / "attention_vs_deletion.csv", index=False)
    summ = att.groupby("model").agg(records=("record_id", "size"), median_spearman=("spearman", "median"),
                                    mean_spearman=("spearman", "mean"),
                                    top_token_agreement=("top_attention_token", lambda s: np.nan)).reset_index()
    for i, m in enumerate(summ.model):
        sub_ = att[att.model == m]
        summ.loc[i, "top_token_agreement"] = (sub_.top_attention_token == sub_.top_deletion_token).mean()
    summ.round(3).to_csv(OUT_TABLES / "attention_vs_deletion_summary.csv", index=False)

    top5 = d3.record_id.head(5).tolist()
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.2))
    for r_, label in enumerate(["D2", "D3"]):
        for c, rid in enumerate(top5):
            toks, a, dl, unk = ex[label][rid]
            ax = axes[r_, c]
            x = np.arange(len(toks))
            ax.bar(x - 0.2, a / max(a.max(), 1e-9), 0.4, label="attention (scaled)", color="#4C72B0")
            ax.bar(x + 0.2, dl / max(dl.max(), 1e-9), 0.4, label="|delta z| when deleted (scaled)", color="#DD8452")
            ax.set_xticks(x, [t + (" [UNK]" if u else "") for t, u in zip(toks, unk)], rotation=70, fontsize=6)
            rho = att[(att.model == label) & (att.record_id == rid)].spearman.item()
            ax.set_title(f"{label} {rid}\nrho={rho:.2f}", fontsize=7)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Largest D3 errors: attention weight vs token-deletion effect (D2 alpha, D3 last-layer [CLS] attention)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig31_attention_vs_deletion.png", bbox_inches="tight"); plt.close(fig)

    print(sub[sub.model == "D3"].round(1).to_string(index=False))
    print(d3[["record_id", "district", "name", "file_year", "y", "y_hat", "B1_pred", "abs_err"]].round(1).to_string(index=False))
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
