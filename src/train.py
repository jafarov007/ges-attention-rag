"""Nested grouped cross-validation for B0, B1, D1, D2, D3 (+ ablations, leakage run).

Protocol (docs/05): outer 5 x inner 3 GroupKFold from protocol/split_manifest.csv.
Inner: 4 candidates, <=150 epochs, patience 15 on inner-val facility MAE (kWp), seed 0.
Outer: retrain the chosen candidate for median(best inner epochs) epochs on the full
outer-train part with seeds 17, 42, 2026 and predict the outer test fold ONCE.

usage:  python train.py main        # B0, B1, D1, D2, D3
        python train.py ablations   # A1, A2, A3 (needs main selections)
        python train.py leakage     # B1 with random row KFold (diagnostic only)
"""
from __future__ import annotations

import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

from common import (ABLATION_SEED, FINAL_SEEDS, INNER_SEED, OUT_DATA, OUT_MODELS, OUT_PRED, OUT_TABLES,
                    PROTOCOL, SPLIT_SEED, set_seed)
from metrics import all_metrics, facility_mae
from models import build_model, count_parameters, make_b1, to_kwp
from tokenizer import Vocab, batch_encode, record_tokens

torch.set_num_threads(1)
GRID = list(product([1e-3, 3e-4], [0.0, 0.2]))          # (lr, dropout)
ALPHAS = [0.1, 1.0, 10.0, 100.0]
MAX_EPOCHS, PATIENCE, BATCH, WD = 150, 15, 16, 1e-4
FORBIDDEN_INPUTS = {"power_kwp", "power_raw", "status_raw", "status_clean", "record_id", "excel_row",
                    "source_no", "unit_id", "entity_group_id", "quality_flag"}


def load():
    df = pd.read_csv(OUT_DATA / "records_clean.csv")
    man = pd.read_csv(PROTOCOL / "split_manifest.csv")
    df = df.merge(man.drop(columns=["entity_group_id", "unit_id", "file_year", "in_regression"]), on="record_id")
    reg = df[df.in_regression].reset_index(drop=True)
    reg["z"] = np.log1p(reg.power_kwp)
    return reg


def model_inputs(frame: pd.DataFrame, use_year=True):
    """The ONLY place where model inputs are read: district, year, cleaned name."""
    cols = ["district_clean", "file_year", "name_clean"]
    assert not FORBIDDEN_INPUTS & set(cols)
    return [record_tokens(d, y, n, use_year) for d, y, n in frame[cols].itertuples(index=False)]


# ----------------------------------------------------------------- neural training
def fit_nn(name, tr, va, lr, dropout, seed, epochs=None, use_year=True, extra=None, log_rows=None, tag=None):
    """Train on tr. If epochs is None -> early stopping on va (facility MAE, kWp)."""
    set_seed(seed)
    tok_tr = model_inputs(tr, use_year)
    vocab = Vocab(min_freq=2).fit(tok_tr)
    ids_tr, _ = batch_encode(vocab, tok_tr)
    y_tr = torch.tensor(tr.z.values, dtype=torch.float32)
    model = build_model(name, len(vocab), dropout, **(extra or {}))
    with torch.no_grad():                         # start the head at the train mean of z (train-only statistic)
        model.head.net[-1].bias.fill_(float(y_tr.mean()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)
    if va is not None:
        tok_va = model_inputs(va, use_year)
        ids_va, n_trunc = batch_encode(vocab, tok_va, max_len=max(ids_tr.size(1), max(len(t) for t in tok_va)))
    best = (np.inf, 0, None)
    stale = 0
    n_ep = epochs if epochs is not None else MAX_EPOCHS
    for ep in range(1, n_ep + 1):
        model.train()
        perm = torch.randperm(len(y_tr), generator=gen)
        tot = 0.0
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss = F.mse_loss(model(ids_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if va is None:
            continue
        model.eval()
        with torch.no_grad():
            z_va = model(ids_va).numpy()
        val_loss = float(np.mean((z_va - va.z.values) ** 2))
        val_fac = facility_mae(va.power_kwp.values, to_kwp(z_va), va.entity_group_id.values)
        if log_rows is not None:
            log_rows.append({**(tag or {}), "epoch": ep, "train_loss": tot / len(y_tr), "val_loss": val_loss,
                             "val_fac_mae": val_fac})
        if val_fac < best[0] - 1e-9:
            best, stale = (val_fac, ep, None), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return model, vocab, best


def predict_nn(model, vocab, frame, use_year=True):
    toks = model_inputs(frame, use_year)
    ids, n_trunc = batch_encode(vocab, toks)
    model.eval()
    with torch.no_grad():
        return model(ids).numpy(), n_trunc


def run_nn(name, reg, seeds, extra=None, use_year=True, fixed_selection=None, label=None):
    label = label or name
    preds, sel_rows, curves, fold_info = [], [], [], []
    for k in range(5):
        tr_all = reg[reg.outer_fold != k].reset_index(drop=True)
        te = reg[reg.outer_fold == k].reset_index(drop=True)
        col = f"inner_fold_o{k}"
        if fixed_selection is None:
            scores = {}
            for (lr, dr) in GRID:
                res = []
                for j in range(3):
                    tr, va = tr_all[tr_all[col] != j], tr_all[tr_all[col] == j]
                    _, _, best = fit_nn(name, tr, va, lr, dr, INNER_SEED, use_year=use_year, extra=extra,
                                        log_rows=curves, tag=dict(model=label, outer=k, lr=lr, dropout=dr, inner=j))
                    res.append(best)
                scores[(lr, dr)] = (np.mean([r[0] for r in res]), int(np.median([r[1] for r in res])),
                                    [r[1] for r in res])
                sel_rows.append(dict(model=label, outer=k, lr=lr, dropout=dr, inner_val_fac_mae=scores[(lr, dr)][0],
                                     best_epochs=str(scores[(lr, dr)][2]), median_epoch=scores[(lr, dr)][1]))
            (lr, dr) = min(scores, key=lambda c: scores[c][0])
            E = max(1, scores[(lr, dr)][1])
        else:
            s = fixed_selection[k]
            lr, dr, E = s["lr"], s["dropout"], s["epochs"]
        for seed in seeds:
            t0 = time.perf_counter()
            model, vocab, _ = fit_nn(name, tr_all, None, lr, dr, seed, epochs=E, use_year=use_year, extra=extra)
            secs = time.perf_counter() - t0
            z_hat, n_trunc = predict_nn(model, vocab, te, use_year)
            if seed == ABLATION_SEED:
                torch.save({"state": model.state_dict(), "itos": vocab.itos, "name": name, "extra": extra,
                            "dropout": dr}, OUT_MODELS / f"{label}_outer{k}_seed{seed}.pt")
            for r, z in zip(te.itertuples(), z_hat):
                preds.append(dict(model=label, seed=seed, outer_fold=k, record_id=r.record_id,
                                  entity_group_id=r.entity_group_id, y=r.power_kwp, z_hat=float(z),
                                  y_hat=float(to_kwp(np.array([z]))[0])))
            fold_info.append(dict(model=label, outer=k, seed=seed, lr=lr, dropout=dr, epochs=E,
                                  vocab_size=len(vocab), max_len=int(max(len(t) for t in model_inputs(tr_all, use_year))),
                                  n_truncated_test=int(n_trunc), params=count_parameters(model)["total"],
                                  train_seconds=round(secs, 3)))
        print(f"[{label}] outer {k}: lr={lr} dropout={dr} epochs={E} vocab={len(vocab)}", flush=True)
    return pd.DataFrame(preds), pd.DataFrame(sel_rows), pd.DataFrame(curves), pd.DataFrame(fold_info)


# ----------------------------------------------------------------- baselines
def run_b0(reg):
    out = []
    for k in range(5):
        tr, te = reg[reg.outer_fold != k], reg[reg.outer_fold == k]
        med = tr.power_kwp.median()
        for r in te.itertuples():
            for seed in FINAL_SEEDS:
                out.append(dict(model="B0", seed=seed, outer_fold=k, record_id=r.record_id,
                                entity_group_id=r.entity_group_id, y=r.power_kwp, z_hat=np.log1p(med), y_hat=med))
    return pd.DataFrame(out)


def b1_select(tr_all, folds):
    scores = {}
    for a in ALPHAS:
        s = []
        for j in sorted(set(folds)):
            tr, va = tr_all[folds != j], tr_all[folds == j]
            m = make_b1(a).fit(tr[["name_clean", "district_clean", "file_year"]], tr.z)
            s.append(facility_mae(va.power_kwp, to_kwp(m.predict(va[["name_clean", "district_clean", "file_year"]])),
                                  va.entity_group_id))
        scores[a] = float(np.mean(s))
    return min(scores, key=scores.get), scores


def run_b1(reg, label="B1", row_split=False):
    out, sel = [], []
    if row_split:  # leakage diagnostic: random rows, same model, same alpha search on random rows
        splits = list(KFold(5, shuffle=True, random_state=SPLIT_SEED).split(reg))
    else:
        splits = [(np.where(reg.outer_fold != k)[0], np.where(reg.outer_fold == k)[0]) for k in range(5)]
    for k, (tri, tei) in enumerate(splits):
        tr_all, te = reg.iloc[tri].reset_index(drop=True), reg.iloc[tei].reset_index(drop=True)
        if row_split:
            folds = np.zeros(len(tr_all), int)
            for j, (_, v) in enumerate(KFold(3, shuffle=True, random_state=SPLIT_SEED + k).split(tr_all)):
                folds[v] = j
        else:
            folds = tr_all[f"inner_fold_o{k}"].values
        alpha, scores = b1_select(tr_all, folds)
        sel.append(dict(model=label, outer=k, alpha=alpha, **{f"inner_fac_mae_alpha_{a:g}": v for a, v in scores.items()}))
        m = make_b1(alpha).fit(tr_all[["name_clean", "district_clean", "file_year"]], tr_all.z)
        z = m.predict(te[["name_clean", "district_clean", "file_year"]])
        g_tr = set(tr_all.entity_group_id)
        for r, zz in zip(te.itertuples(), z):
            for seed in FINAL_SEEDS:
                out.append(dict(model=label, seed=seed, outer_fold=k, record_id=r.record_id,
                                entity_group_id=r.entity_group_id, y=r.power_kwp, z_hat=float(zz),
                                y_hat=float(to_kwp(np.array([zz]))[0]), group_in_train=r.entity_group_id in g_tr))
        print(f"[{label}] fold {k}: alpha={alpha}", flush=True)
    return pd.DataFrame(out), pd.DataFrame(sel)


def main(mode):
    reg = load()
    if mode == "main":
        p0 = run_b0(reg)
        p1, s1 = run_b1(reg)
        s1.to_csv(OUT_TABLES / "selection_B1.csv", index=False)
        allp = [p0, p1]
        for name in ["D1", "D2", "D3"]:
            t0 = time.perf_counter()
            p, s, c, f = run_nn(name, reg, FINAL_SEEDS)
            print(f"{name} total {time.perf_counter() - t0:.1f}s", flush=True)
            s.to_csv(OUT_TABLES / f"selection_{name}.csv", index=False)
            c.to_csv(OUT_TABLES / f"learning_curves_{name}.csv", index=False)
            f.to_csv(OUT_TABLES / f"fold_info_{name}.csv", index=False)
            allp.append(p)
        pd.concat(allp).to_csv(OUT_PRED / "outer_predictions_main.csv", index=False)
    elif mode == "ablations":
        allp, infos = [], []
        specs = [("A1_D2_equal_weights", "D2", dict(equal_weights=True), True),
                 ("A2_D3_no_positions", "D3", dict(use_positions=False), True),
                 ("A3_D3_no_year", "D3", {}, False)]
        for label, base, extra, use_year in specs:
            fi = pd.read_csv(OUT_TABLES / f"fold_info_{base}.csv")
            fi = fi[fi.seed == ABLATION_SEED].set_index("outer")
            fixed = {k: dict(lr=fi.loc[k, "lr"], dropout=fi.loc[k, "dropout"], epochs=int(fi.loc[k, "epochs"])) for k in range(5)}
            p, _, _, f = run_nn(base, reg, [ABLATION_SEED], extra=extra, use_year=use_year, fixed_selection=fixed, label=label)
            allp.append(p); infos.append(f)
        pd.concat(allp).to_csv(OUT_PRED / "outer_predictions_ablations.csv", index=False)
        pd.concat(infos).to_csv(OUT_TABLES / "fold_info_ablations.csv", index=False)
    elif mode == "leakage":
        p, s = run_b1(reg, label="B1_rowsplit", row_split=True)
        p.to_csv(OUT_PRED / "outer_predictions_leakage.csv", index=False)
        s.to_csv(OUT_TABLES / "selection_B1_rowsplit.csv", index=False)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
