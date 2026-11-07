"""POST-HOC sensitivity analysis (protocol v1.1, added AFTER the v1.0 results were seen).

Question: how much of the large-plant under-prediction is the retransformation bias of the
log target (docs/06 M4.3)? Duan smearing: E[y + 1 | x] = exp(z_hat) * mean(exp(r)), r = z - z_hat.

The smearing factor of outer fold k uses ONLY out-of-fold residuals from the inner folds of the
outer-train part (same selected configuration and epoch count as the main model, seed 0).
Outer-test labels are not used to compute it. Nothing here changes the main results table.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ABLATION_SEED, FIG, INNER_SEED, OUT_PRED, OUT_TABLES
from metrics import all_metrics, paired_group_bootstrap
from models import make_b1
from train import fit_nn, load, predict_nn

COLS = ["name_clean", "district_clean", "file_year"]


def oof_residuals(reg, model, k):
    tr_all = reg[reg.outer_fold != k].reset_index(drop=True)
    col = f"inner_fold_o{k}"
    res = []
    if model == "B1":
        alpha = pd.read_csv(OUT_TABLES / "selection_B1.csv").set_index("outer").alpha[k]
    else:
        fi = pd.read_csv(OUT_TABLES / f"fold_info_{model}.csv")
        s = fi[(fi.outer == k) & (fi.seed == ABLATION_SEED)].iloc[0]
    for j in range(3):
        tr, va = tr_all[tr_all[col] != j], tr_all[tr_all[col] == j]
        if model == "B1":
            z = make_b1(alpha).fit(tr[COLS], tr.z).predict(va[COLS])
        else:
            m, v, _ = fit_nn(model, tr, None, float(s.lr), float(s.dropout), INNER_SEED, epochs=int(s.epochs))
            z, _ = predict_nn(m, v, va)
        res.append(va.z.values - z)
    return np.concatenate(res)


def main():
    reg = load()
    p = pd.read_csv(OUT_PRED / "outer_predictions_main.csv")
    rows, factors, corrected = [], [], []
    for model in ["B1", "D1", "D2", "D3"]:
        pm = p[p.model == model].copy()
        fac = {}
        for k in range(5):
            r = oof_residuals(reg, model, k)
            fac[k] = float(np.mean(np.exp(r)))
            factors.append(dict(model=model, outer_fold=k, smearing_factor=fac[k], oof_residual_mean=float(r.mean()),
                                oof_residual_std=float(r.std())))
        pm["y_hat_smear"] = np.maximum(0.0, np.exp(pm.z_hat) * pm.outer_fold.map(fac) - 1.0)
        ens = pm.groupby(["record_id", "entity_group_id", "y"], as_index=False)[["y_hat", "y_hat_smear"]].mean()
        ens["model"] = model
        corrected.append(ens)
        a = all_metrics(ens.y, ens.y_hat, ens.entity_group_id)
        b = all_metrics(ens.y, ens.y_hat_smear, ens.entity_group_id)
        bs = paired_group_bootstrap((ens.y - ens.y_hat_smear).values, (ens.y - ens.y_hat).values,
                                    ens.entity_group_id.values, 1000, 0)
        big = ens.y > 150
        rows.append(dict(model=model, fac_mae_v10=a["fac_mae"], fac_mae_smear=b["fac_mae"],
                         rmse_v10=a["rmse"], rmse_smear=b["rmse"], r2_v10=a["r2"], r2_smear=b["r2"],
                         log_mae_v10=a["log_mae"], log_mae_smear=b["log_mae"],
                         mae_gt150_v10=float((ens.y - ens.y_hat)[big].abs().mean()),
                         mae_gt150_smear=float((ens.y - ens.y_hat_smear)[big].abs().mean()),
                         mae_le50_v10=float((ens.y - ens.y_hat)[ens.y <= 50].abs().mean()),
                         mae_le50_smear=float((ens.y - ens.y_hat_smear)[ens.y <= 50].abs().mean()),
                         delta_fac_mae=bs["delta"], ci_low=bs["ci_low"], ci_high=bs["ci_high"]))
    out = pd.DataFrame(rows)
    out.round(3).to_csv(OUT_TABLES / "posthoc_smearing.csv", index=False)
    pd.DataFrame(factors).round(4).to_csv(OUT_TABLES / "posthoc_smearing_factors.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    x = np.arange(len(out)); w = 0.38
    for a_, (c1, c2, t) in zip(ax, [("fac_mae_v10", "fac_mae_smear", "facility MAE (kWp)"),
                                    ("mae_gt150_v10", "mae_gt150_smear", "MAE on plants > 150 kWp"),
                                    ("mae_le50_v10", "mae_le50_smear", "MAE on plants <= 50 kWp")]):
        a_.bar(x - w / 2, out[c1], w, label="v1.0 (no correction)", color="#4C72B0")
        a_.bar(x + w / 2, out[c2], w, label="Duan smearing (post-hoc)", color="#DD8452")
        a_.set_xticks(x, out.model); a_.set_title(t, fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle("POST-HOC sensitivity: retransformation bias correction (factor from inner out-of-fold residuals only)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig35_posthoc_smearing.png", bbox_inches="tight"); plt.close(fig)
    print(out.round(2).to_string(index=False))
    print(pd.DataFrame(factors).groupby("model").smearing_factor.agg(["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
