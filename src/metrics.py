"""Metrics: facility-balanced MAE (primary), record-level metrics, paired group bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd


def facility_mae(y, yhat, groups) -> float:
    """Mean over groups of the mean absolute error inside each group."""
    d = pd.DataFrame({"e": np.abs(np.asarray(y) - np.asarray(yhat)), "g": np.asarray(groups)})
    return float(d.groupby("g").e.mean().mean())


def all_metrics(y, yhat, groups) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    resid = y - yhat
    ss_res, ss_tot = float((resid ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "fac_mae": facility_mae(y, yhat, groups),
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "log_mae": float(np.abs(np.log1p(y) - np.log1p(np.maximum(yhat, 0))).mean()),
        "n": int(len(y)),
        "groups": int(len(set(groups))),
    }


def group_errors(err, groups) -> pd.Series:
    return pd.DataFrame({"e": err, "g": groups}).groupby("g").e.mean()


def paired_group_bootstrap(err_a, err_b, groups, n_boot=1000, seed=0):
    """CI of MAE_fac(A) - MAE_fac(B) by resampling facility groups with replacement."""
    ga, gb = group_errors(np.abs(err_a), groups), group_errors(np.abs(err_b), groups)
    ga, gb = ga.sort_index().values, gb.sort_index().values
    rng = np.random.default_rng(seed)
    G = len(ga)
    idx = rng.integers(0, G, size=(n_boot, G))
    diffs = ga[idx].mean(1) - gb[idx].mean(1)
    return {"delta": float(ga.mean() - gb.mean()), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_a_better": float((diffs < 0).mean())}
