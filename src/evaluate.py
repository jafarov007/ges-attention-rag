"""Read outer-test predictions ONCE and produce result tables, bootstrap CIs and figures
for training diagnostics (docs/09) and results (docs/10)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ABLATION_SEED, FIG, OUT_PRED, OUT_TABLES
from metrics import all_metrics, facility_mae, paired_group_bootstrap

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
MAIN = ["B0", "B1", "D1", "D2", "D3"]
COL = {"B0": "#8C8C8C", "B1": "#937860", "D1": "#4C72B0", "D2": "#55A868", "D3": "#C44E52"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)


def seed_mean(p):
    return p.groupby(["model", "record_id", "entity_group_id", "outer_fold", "y"], as_index=False).y_hat.mean()


def training_figures():
    sel = pd.concat([pd.read_csv(OUT_TABLES / f"selection_{m}.csv") for m in ["D1", "D2", "D3"]])
    fi = pd.concat([pd.read_csv(OUT_TABLES / f"fold_info_{m}.csv") for m in ["D1", "D2", "D3"]])
    chosen = fi[fi.seed == ABLATION_SEED][["model", "outer", "lr", "dropout", "epochs"]]

    # learning curves of the chosen candidate (inner folds of every outer fold)
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
    for c, m in enumerate(["D1", "D2", "D3"]):
        cur = pd.read_csv(OUT_TABLES / f"learning_curves_{m}.csv")
        cm = cur.merge(chosen[chosen.model == m], on=["model", "outer", "lr", "dropout"])
        for (k, j), g in cm.groupby(["outer", "inner"]):
            ax[0, c].plot(g.epoch, g.train_loss, color="#4C72B0", alpha=0.35, lw=0.8)
            ax[0, c].plot(g.epoch, g.val_loss, color="#C44E52", alpha=0.35, lw=0.8)
            ax[1, c].plot(g.epoch, g.val_fac_mae, color=COL[m], alpha=0.4, lw=0.8)
        ax[0, c].set_title(f"{m}: log-MSE (blue train, red inner-val)", fontsize=8); ax[0, c].set_yscale("log")
        ax[1, c].set_title(f"{m}: inner-val facility MAE (kWp)", fontsize=8); ax[1, c].set_xlabel("epoch")
        for E in chosen[chosen.model == m].epochs:
            ax[1, c].axvline(E, color="k", lw=0.5, ls=":")
    fig.suptitle("Learning curves of the selected candidate, 5 outer x 3 inner folds (dotted: transferred epochs)", fontsize=9)
    save(fig, "fig20_learning_curves.png")

    agg = sel.groupby(["model", "lr", "dropout"]).inner_val_fac_mae.mean().reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    for c, m in enumerate(["D1", "D2", "D3"]):
        t = agg[agg.model == m].pivot(index="lr", columns="dropout", values="inner_val_fac_mae")
        ax[c].imshow(t.values, cmap="viridis_r"); ax[c].grid(False)
        ax[c].set_xticks(range(t.shape[1]), [f"dropout {d}" for d in t.columns])
        ax[c].set_yticks(range(t.shape[0]), [f"lr {l:g}" for l in t.index])
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                ax[c].text(j, i, f"{t.values[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9)
        ax[c].set_title(f"{m}: mean inner-val facility MAE", fontsize=8)
    save(fig, "fig21_hyperparameter_grid.png")
    agg.round(3).to_csv(OUT_TABLES / "hyperparameter_grid_mean.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    for i, m in enumerate(["D1", "D2", "D3"]):
        eps = []
        for s in sel[sel.model == m].merge(chosen[chosen.model == m], on=["model", "outer", "lr", "dropout"]).best_epochs:
            eps += [int(x) for x in s.strip("[]").split(",")]
        ax.scatter(np.full(len(eps), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(eps)), eps, color=COL[m], s=14)
        ax.scatter([i], [np.median(eps)], marker="_", s=600, color="k")
    ax.set_xticks(range(3), ["D1", "D2", "D3"]); ax.set_ylabel("best inner epoch (early stopping)")
    ax.set_title("Best epochs of the selected candidate (black: median)")
    save(fig, "fig22_best_epochs.png")
    return chosen, fi


def main():
    chosen, fi = training_figures()
    p = pd.read_csv(OUT_PRED / "outer_predictions_main.csv")
    pa = pd.read_csv(OUT_PRED / "outer_predictions_ablations.csv")
    pl = pd.read_csv(OUT_PRED / "outer_predictions_leakage.csv")
    ens = seed_mean(p)

    # ---------------- main table
    rows = []
    for m in MAIN:
        g = ens[ens.model == m]
        met = all_metrics(g.y, g.y_hat, g.entity_group_id)
        per_seed = [facility_mae(s.y, s.y_hat, s.entity_group_id) for _, s in p[p.model == m].groupby("seed")]
        met.update(fac_mae_seed_std=float(np.std(per_seed)), fac_mae_seed_min=min(per_seed), fac_mae_seed_max=max(per_seed))
        if m in ("D1", "D2", "D3"):
            f = fi[fi.model == m]
            met.update(params_mean=float(f.params.mean()), train_seconds_per_model=float(f.train_seconds.mean()),
                       epochs_median=float(f.epochs.median()), vocab_mean=float(f.vocab_size.mean()))
        rows.append(dict(model=m, **met))
    main_tab = pd.DataFrame(rows)
    main_tab.round(3).to_csv(OUT_TABLES / "results_main.csv", index=False)

    fold_rows = []
    for (m, k, s), g in p.groupby(["model", "outer_fold", "seed"]):
        fold_rows.append(dict(model=m, outer_fold=k, seed=s, fac_mae=facility_mae(g.y, g.y_hat, g.entity_group_id),
                              mae=float(np.abs(g.y - g.y_hat).mean())))
    folds = pd.DataFrame(fold_rows)
    folds.round(3).to_csv(OUT_TABLES / "results_by_fold_seed.csv", index=False)

    # ---------------- bootstrap
    def err(model, src=ens):
        g = src[src.model == model].sort_values("record_id")
        return (g.y - g.y_hat).values, g.entity_group_id.values

    pairs = [("D2", "D1"), ("D3", "D1"), ("D3", "D2"), ("D1", "B1"), ("D2", "B1"), ("D3", "B1"), ("B1", "B0"), ("D3", "B0")]
    bs = []
    for a, b in pairs:
        ea, g = err(a); eb, _ = err(b)
        bs.append(dict(comparison=f"{a} - {b}", **paired_group_bootstrap(ea, eb, g, 1000, seed=0)))
    s42 = p[p.seed == ABLATION_SEED]
    abl = []
    for lab, base in [("A1_D2_equal_weights", "D2"), ("A2_D3_no_positions", "D3"), ("A3_D3_no_year", "D3")]:
        ga = pa[pa.model == lab].sort_values("record_id"); gb = s42[s42.model == base].sort_values("record_id")
        assert (ga.record_id.values == gb.record_id.values).all()
        r = paired_group_bootstrap((ga.y - ga.y_hat).values, (gb.y - gb.y_hat).values, ga.entity_group_id.values, 1000, 0)
        m_a = all_metrics(ga.y, ga.y_hat, ga.entity_group_id); m_b = all_metrics(gb.y, gb.y_hat, gb.entity_group_id)
        abl.append(dict(ablation=lab, base=f"{base} (seed {ABLATION_SEED})", fac_mae_ablation=m_a["fac_mae"],
                        fac_mae_base=m_b["fac_mae"], log_mae_ablation=m_a["log_mae"], log_mae_base=m_b["log_mae"],
                        rmse_ablation=m_a["rmse"], rmse_base=m_b["rmse"], **{f"boot_{k}": v for k, v in r.items()}))
        bs.append(dict(comparison=f"{lab} - {base}@42", **r))
    bs = pd.DataFrame(bs); bs.round(3).to_csv(OUT_TABLES / "bootstrap_differences.csv", index=False)
    abl = pd.DataFrame(abl); abl.round(3).to_csv(OUT_TABLES / "results_ablations.csv", index=False)

    # ---------------- leakage diagnostic
    lk = seed_mean(pl)
    lk = lk.merge(pl[["record_id", "group_in_train"]].drop_duplicates(), on="record_id")
    gb1 = ens[ens.model == "B1"]
    leak_rows = [dict(split="GroupKFold (valid)", **all_metrics(gb1.y, gb1.y_hat, gb1.entity_group_id)),
                 dict(split="row KFold (LEAKAGE DIAGNOSTIC)", **all_metrics(lk.y, lk.y_hat, lk.entity_group_id))]
    for flag, g in lk.groupby("group_in_train"):
        leak_rows.append(dict(split=f"row KFold, group_in_train={flag}", **all_metrics(g.y, g.y_hat, g.entity_group_id)))
    leak = pd.DataFrame(leak_rows)
    leak["share_group_in_train"] = [0.0, lk.group_in_train.mean(), np.nan, np.nan]
    leak.round(3).to_csv(OUT_TABLES / "results_leakage.csv", index=False)

    # ---------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(MAIN))
    ax[0].bar(x, main_tab.fac_mae, color=[COL[m] for m in MAIN], yerr=main_tab.fac_mae_seed_std, capsize=3)
    for i, m in enumerate(MAIN):
        fv = folds[(folds.model == m)].groupby("outer_fold").fac_mae.mean()
        ax[0].scatter(np.full(len(fv), i) + np.linspace(-0.25, 0.25, len(fv)), fv, color="k", s=8, zorder=3)
        ax[0].text(i, main_tab.fac_mae[i] + 3, f"{main_tab.fac_mae[i]:.1f}", ha="center", fontsize=8)
    ax[0].set_xticks(x, MAIN); ax[0].set_ylabel("facility-balanced MAE (kWp)")
    ax[0].set_title("Outer-test facility MAE (bars: 3-seed mean prediction,\nerror bar: seed std, dots: outer folds)", fontsize=8)
    ax[1].bar(x, main_tab.log_mae, color=[COL[m] for m in MAIN])
    for i in range(len(MAIN)):
        ax[1].text(i, main_tab.log_mae[i] + 0.02, f"{main_tab.log_mae[i]:.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(x, MAIN); ax[1].set_ylabel("|ln(1+y) - ln(1+y_hat)|")
    ax[1].set_title("Log-scale MAE (relative error, what the loss optimises)", fontsize=8)
    save(fig, "fig23_main_results.png")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for a, m in zip(ax, ["B1", "D2", "D3"]):
        g = ens[ens.model == m]
        a.scatter(g.y, g.y_hat, s=12, color=COL[m], alpha=0.8)
        a.plot([3, 600], [3, 600], "k--", lw=0.8)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("true kWp")
        a.set_title(f"{m}: R2={all_metrics(g.y, g.y_hat, g.entity_group_id)['r2']:.2f}", fontsize=9)
    ax[0].set_ylabel("predicted kWp (outer test)")
    fig.suptitle("Predicted vs true (log axes): large plants are under-predicted (log target, Jensen)", fontsize=9)
    save(fig, "fig24_pred_vs_true.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    yy = np.arange(len(bs))[::-1]
    for y_, r in zip(yy, bs.itertuples()):
        c = "#55A868" if r.ci_high < 0 else ("#C44E52" if r.ci_low > 0 else "#4C72B0")
        ax.plot([r.ci_low, r.ci_high], [y_, y_], color=c, lw=2)
        ax.scatter([r.delta], [y_], color=c, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yy, bs.comparison, fontsize=8)
    ax.set_xlabel("difference in facility MAE (kWp), 95% paired group bootstrap")
    ax.set_title("Model differences (green: left model better, blue: indistinguishable)", fontsize=9)
    save(fig, "fig25_bootstrap_differences.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    lab = ["D2@42", "A1 equal\nweights", "D3@42", "A2 no\npositions", "A3 no\nyear"]
    fv = [abl.fac_mae_base[0], abl.fac_mae_ablation[0], abl.fac_mae_base[1], abl.fac_mae_ablation[1], abl.fac_mae_ablation[2]]
    lv = [abl.log_mae_base[0], abl.log_mae_ablation[0], abl.log_mae_base[1], abl.log_mae_ablation[1], abl.log_mae_ablation[2]]
    cols = [COL["D2"], "#A6D8A8", COL["D3"], "#E8A0A0", "#E8C0A0"]
    for a, v, t in [(ax[0], fv, "facility MAE (kWp)"), (ax[1], lv, "log-scale MAE")]:
        a.bar(range(5), v, color=cols)
        for i, vv in enumerate(v):
            a.text(i, vv * 1.01, f"{vv:.2f}", ha="center", fontsize=8)
        a.set_xticks(range(5), lab, fontsize=8); a.set_title(f"Ablations (single seed {ABLATION_SEED}): {t}", fontsize=8)
    save(fig, "fig26_ablations.png")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(["GroupKFold\n(valid)", "row KFold\n(leakage)"], leak.fac_mae[:2], color=["#55A868", "#C44E52"])
    for i, v in enumerate(leak.fac_mae[:2]):
        ax[0].text(i, v + 1.5, f"{v:.1f}", ha="center")
    ax[0].set_ylabel("B1 facility MAE (kWp)"); ax[0].set_title("Same model, two split schemes", fontsize=9)
    sub = leak.iloc[2:]
    ax[1].bar([s.split("=")[1] for s in sub.split], sub.mae, color=["#4C72B0", "#C44E52"])
    for i, (v, n) in enumerate(zip(sub.mae, sub.n)):
        ax[1].text(i, v + 1.5, f"{v:.1f} (n={n})", ha="center", fontsize=8)
    ax[1].set_xlabel("facility group also present in training?"); ax[1].set_ylabel("record MAE (kWp)")
    ax[1].set_title("Row split: error collapses only for memorised facilities", fontsize=9)
    save(fig, "fig27_leakage.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    for m in ["D1", "D2", "D3"]:
        f = folds[folds.model == m]
        for s, g in f.groupby("seed"):
            ax.plot(g.outer_fold + {"D1": -0.15, "D2": 0, "D3": 0.15}[m], g.fac_mae, "o", color=COL[m], alpha=0.7, ms=4)
        ax.plot([], [], "o", color=COL[m], label=m)
    b1 = folds[(folds.model == "B1") & (folds.seed == 17)]
    ax.plot(b1.outer_fold, b1.fac_mae, "s", color=COL["B1"], label="B1", ms=5)
    ax.set_xticks(range(5)); ax.set_xlabel("outer fold"); ax.set_ylabel("facility MAE (kWp)")
    ax.set_title("Fold-to-fold spread dominates seed-to-seed spread", fontsize=9); ax.legend(fontsize=7, ncol=4)
    save(fig, "fig28_fold_seed_spread.png")

    print(main_tab.round(3).to_string(index=False))
    print(bs.round(3).to_string(index=False))
    print(abl.round(3).to_string(index=False))
    print(leak.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
