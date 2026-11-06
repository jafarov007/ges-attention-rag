"""Step 5 — frozen split manifest (nested grouped CV).

outer: GroupKFold(5, shuffle, seed 2026) on entity_group_id
inner: GroupKFold(3, shuffle, seed 2026 + k) inside every outer-train part
Quarantined records get a fold (their group's fold) but are never used for fitting
or scoring (in_regression == False).

Output: protocol/split_manifest.csv (+ sha256 printed for the protocol)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from common import FIG, OUT_DATA, OUT_TABLES, PROTOCOL, SPLIT_SEED, sha256

N_OUTER, N_INNER = 5, 3


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    man = df[["record_id", "entity_group_id", "unit_id", "file_year", "in_regression"]].copy()
    man["outer_fold"] = -1
    gkf = GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SPLIT_SEED)
    for k, (_, te) in enumerate(gkf.split(man, groups=man.entity_group_id)):
        man.iloc[te, man.columns.get_loc("outer_fold")] = k
    for k in range(N_OUTER):
        col = f"inner_fold_o{k}"
        man[col] = -1
        tr = man.index[man.outer_fold != k]
        ig = GroupKFold(n_splits=N_INNER, shuffle=True, random_state=SPLIT_SEED + k)
        for j, (_, va) in enumerate(ig.split(man.loc[tr], groups=man.loc[tr, "entity_group_id"])):
            man.loc[tr[va], col] = j
    return man


def check_manifest(man: pd.DataFrame) -> None:
    assert (man.outer_fold >= 0).all()
    for k in range(N_OUTER):
        te = set(man[man.outer_fold == k].entity_group_id)
        tr = set(man[man.outer_fold != k].entity_group_id)
        assert not te & tr, f"outer {k}: group leakage"
        sub = man[man.outer_fold != k]
        col = f"inner_fold_o{k}"
        assert (sub[col] >= 0).all() and (man[man.outer_fold == k][col] == -1).all()
        for j in range(N_INNER):
            assert not set(sub[sub[col] == j].entity_group_id) & set(sub[sub[col] != j].entity_group_id)


def main():
    df = pd.read_csv(OUT_DATA / "records_clean.csv")
    man = build_manifest(df)
    check_manifest(man)
    man.to_csv(PROTOCOL / "split_manifest.csv", index=False)

    reg = man[man.in_regression].merge(df[["record_id", "power_kwp"]], on="record_id")
    rows = []
    for k in range(N_OUTER):
        te = reg[reg.outer_fold == k]
        rows.append(dict(outer_fold=k, test_records=len(te), test_groups=te.entity_group_id.nunique(),
                         train_records=int((reg.outer_fold != k).sum()),
                         train_groups=reg[reg.outer_fold != k].entity_group_id.nunique(),
                         test_median_kwp=te.power_kwp.median(),
                         test_years=dict(te.file_year.value_counts().sort_index())))
    fold_tab = pd.DataFrame(rows)
    fold_tab.to_csv(OUT_TABLES / "outer_folds.csv", index=False)

    # leakage diagnostic preview: share of test records whose group is in train under row-KFold
    shared = []
    for tr, te in KFold(5, shuffle=True, random_state=SPLIT_SEED).split(reg):
        g_tr = set(reg.iloc[tr].entity_group_id)
        shared.append(reg.iloc[te].entity_group_id.isin(g_tr).mean())
    leak = pd.DataFrame({"split": ["row KFold (random rows)", "GroupKFold (entity groups)"],
                         "share_test_records_with_group_in_train": [np.mean(shared), 0.0]})
    leak.to_csv(OUT_TABLES / "shared_group_share.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].bar(fold_tab.outer_fold.astype(str), fold_tab.test_records, label="records", color="#4C72B0")
    ax[0].bar(fold_tab.outer_fold.astype(str), fold_tab.test_groups, label="groups", color="#55A868", width=0.5)
    ax[0].set_title("Outer test folds: size"); ax[0].set_xlabel("outer fold"); ax[0].legend(fontsize=7)
    ax[1].boxplot([reg[reg.outer_fold == k].power_kwp for k in range(N_OUTER)], labels=[str(k) for k in range(N_OUTER)])
    ax[1].set_yscale("log"); ax[1].set_title("Outer test folds: power (kWp)"); ax[1].set_xlabel("outer fold")
    ax[2].bar(["row KFold", "GroupKFold"], leak.share_test_records_with_group_in_train, color=["#C44E52", "#55A868"])
    for i, v in enumerate(leak.share_test_records_with_group_in_train):
        ax[2].text(i, v + 0.02, f"{v:.0%}", ha="center")
    ax[2].set_ylim(0, 1); ax[2].set_title("Test records whose facility\ngroup is also in train", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig17_split_folds.png", bbox_inches="tight"); plt.close(fig)

    print(fold_tab.to_string(index=False))
    print(leak.to_string(index=False))
    print("manifest sha256:", sha256(PROTOCOL / "split_manifest.csv"))


if __name__ == "__main__":
    main()
