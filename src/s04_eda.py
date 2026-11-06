"""Step 4 — descriptive audit / EDA figures and tables (no modelling decision is
taken from test folds here; everything is descriptive on the full inventory).

Figures (English text) -> docs/figures/, tables -> outputs/tables/
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from common import FIG, OUT_DATA, OUT_LOGS, OUT_TABLES, basic_clean, tokenize_tr, tr_lower

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
YEARS = [2022, 2023, 2024, 2025]
YC = {2022: "#4C72B0", 2023: "#DD8452", 2024: "#55A868", 2025: "#C44E52"}
NOTE = "Inventory snapshot per source file - not new installations, not production."


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def facility_type(name: str) -> str:
    t = tr_lower(name)
    if "itfaiye" in t:
        return "Fire station"
    if re.search(r"tuvalet", t):
        return "Toilet"
    if re.search(r"park|orman|koru|bahçe|gölet", t):
        return "Park / forest"
    if re.search(r"spor|yüzme|havuz|pist|buz", t):
        return "Sports / pool"
    if re.search(r"kültür|sahne|davet|toplantı", t):
        return "Culture / events"
    return "Other building"


def main():
    df = pd.read_csv(OUT_DATA / "records_clean.csv")
    log = pd.read_csv(OUT_LOGS / "ingest_log.csv")
    clog = pd.read_csv(OUT_LOGS / "cleaning_log.csv")
    mlog = pd.read_csv(OUT_LOGS / "matching_log.csv")
    df["type"] = df.name_ws.map(facility_type)
    df["log_power"] = np.log1p(df.power_kwp)

    # ---------- 01 audit ----------
    log["year"] = log.source_file.str[:4].astype(int)
    ct = log.pivot_table(index="year", columns="action", values="excel_row", aggfunc="count", fill_value=0)
    ct = ct.reindex(columns=["record", "header", "title", "description", "empty"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bottom = np.zeros(len(ct))
    for col, c in zip(ct.columns, ["#4C72B0", "#8172B3", "#937860", "#DA8BC3", "#BBBBBB"]):
        ax.bar(ct.index.astype(str), ct[col], bottom=bottom, label=col, color=c)
        bottom += ct[col].values
    for i, (y, n) in enumerate(ct.record.items()):
        ax.text(i, ct.loc[y].sum() + 0.8, f"{n} records", ha="center", fontsize=8)
    ax.set_ylabel("Excel rows"); ax.set_title("Raw Excel rows by detected row type")
    ax.legend(ncol=5, fontsize=7, loc="upper left"); ax.set_ylim(0, bottom.max() * 1.15)
    save(fig, "fig01_rows_by_type.png")
    ct.to_csv(OUT_TABLES / "audit_row_types.csv")

    summ = df.groupby("file_year").agg(records=("record_id", "size"), districts=("district_clean", "nunique"),
                                       total_kwp=("power_kwp", "sum"), median_kwp=("power_kwp", "median"),
                                       min_kwp=("power_kwp", "min"), max_kwp=("power_kwp", "max"))
    summ.round(1).to_csv(OUT_TABLES / "summary_by_year.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].bar(summ.index.astype(str), summ.records, color=[YC[y] for y in summ.index])
    ax[0].set_title("Records per source file"); ax[0].set_ylabel("records")
    ax[1].bar(summ.index.astype(str), summ.total_kwp, color=[YC[y] for y in summ.index])
    ax[1].set_title("Sum of 'Güç' per file (kWp)"); ax[1].set_ylabel("kWp")
    fig.suptitle(NOTE, fontsize=7, y=0.02)
    save(fig, "fig02_records_power_per_year.png")

    # ---------- 02 cleaning ----------
    rc = clog.rule_id.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(rc.index, rc.values, color="#4C72B0")
    for i, v in enumerate(rc.values):
        ax.text(v + 1, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("changed cells"); ax.set_title("Cleaning rules applied (raw value kept in *_raw)")
    save(fig, "fig03_cleaning_rules.png")

    st = df.pivot_table(index="file_year", columns="status_clean", values="record_id", aggfunc="count", fill_value=0)
    st = st.reindex(columns=["aktif", "pasif", "karma", "belirsiz"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3))
    bottom = np.zeros(len(st))
    for col, c, lab in zip(st.columns, ["#55A868", "#C44E52", "#DD8452", "#8C8C8C"],
                           ["active", "passive", "mixed ('Aktif - Pasif')", "ambiguous ('(Aktif / Pasif)')"]):
        ax.bar(st.index.astype(str), st[col], bottom=bottom, label=lab, color=c)
        bottom += st[col].values
    ax.set_title("Status after cleaning (not a model input)"); ax.set_ylabel("records"); ax.legend(fontsize=7)
    save(fig, "fig04_status_by_year.png")
    st.to_csv(OUT_TABLES / "status_by_year.csv")

    pats = {"'GES' suffix": r"\bges\b", "'(GES)'": r"\(ges\)", "'Güneş Enerji Santrali'": r"güneş enerji santrali",
            "'İBB' prefix": r"\bibb\b", "'Çatısı'": r"çatısı", "district name in Mekan": None}
    mat = []
    for lab, p in pats.items():
        row = []
        for y in YEARS:
            sub = df[df.file_year == y]
            if p is None:
                v = np.mean([tr_lower(d) in tr_lower(n) for d, n in zip(sub.district_clean, sub.name_ws)])
            else:
                v = sub.name_ws.map(lambda s: bool(re.search(p, tr_lower(s)))).mean()
            row.append(v)
        mat.append(row)
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), YEARS); ax.set_yticks(range(len(pats)), list(pats))
    for i in range(mat.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.6 else "black")
    ax.grid(False); ax.set_title("Naming templates leak the file year (share of names)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    save(fig, "fig05_year_template_leak.png")
    pd.DataFrame(mat, index=list(pats), columns=YEARS).round(3).to_csv(OUT_TABLES / "year_template_share.csv")

    # ---------- 03 matching ----------
    ent = df.groupby("entity_group_id").agg(n=("record_id", "size"), years=("file_year", "nunique"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    v = ent.n.value_counts().sort_index()
    ax[0].bar(v.index.astype(str), v.values, color="#4C72B0"); ax[0].set_title("Records per evaluation group")
    ax[0].set_xlabel("records in group"); ax[0].set_ylabel("groups")
    v = ent.years.value_counts().sort_index()
    ax[1].bar(v.index.astype(str), v.values, color="#55A868"); ax[1].set_title("Distinct file years per group")
    ax[1].set_xlabel("file years"); ax[1].set_ylabel("groups")
    fig.suptitle(f"{len(df)} records -> {df.unit_id.nunique()} units -> {len(ent)} evaluation groups", fontsize=9)
    save(fig, "fig06_group_sizes.png")

    gy = df.groupby("entity_group_id").file_year.apply(set)
    ov = np.array([[sum(1 for s in gy if a in s and b in s) for b in YEARS] for a in YEARS])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(ov, cmap="Greens"); ax.grid(False)
    ax.set_xticks(range(4), YEARS); ax.set_yticks(range(4), YEARS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, ov[i, j], ha="center", va="center", fontsize=9)
    ax.set_title("Evaluation groups shared between files")
    save(fig, "fig07_year_overlap.png")
    pd.DataFrame(ov, index=YEARS, columns=YEARS).to_csv(OUT_TABLES / "group_year_overlap.csv")

    cand = mlog[mlog.token_set_ratio.notna()].copy()
    cand["same_district"] = cand.district_a == cand.district_b
    fig, ax = plt.subplots(figsize=(6.4, 3))
    for dec, c in [("different", "#C44E52"), ("same_group", "#DD8452"), ("same_unit", "#55A868")]:
        s = cand[(cand.decision == dec) & cand.same_district].token_set_ratio
        ax.hist(s, bins=np.arange(68, 102, 2), alpha=0.7, label=f"{dec} (n={len(s)})", color=c)
    ax.set_xlabel("token_set_ratio of match keys (same district candidates)"); ax.set_ylabel("pairs")
    ax.set_title("Similarity alone does not decide: rules + manual review"); ax.legend(fontsize=7)
    save(fig, "fig08_similarity_by_decision.png")

    # ---------- 04 EDA ----------
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(df.power_kwp, bins=30, color="#4C72B0"); ax[0].set_xlabel("kWp"); ax[0].set_title("Power (raw)")
    ax[1].hist(df.log_power, bins=30, color="#55A868"); ax[1].set_xlabel("ln(1 + kWp)"); ax[1].set_title("Power (log target)")
    for a in ax:
        a.set_ylabel("records")
    save(fig, "fig09_power_hist.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    data = [df[df.file_year == y].power_kwp for y in YEARS]
    bp = ax.boxplot(data, labels=[str(y) for y in YEARS], patch_artist=True, showfliers=True)
    for p, y in zip(bp["boxes"], YEARS):
        p.set_facecolor(YC[y]); p.set_alpha(0.6)
    ax.set_yscale("log"); ax.set_ylabel("kWp (log scale)"); ax.set_title("Power by source file")
    save(fig, "fig10_power_by_year.png")

    order = df.groupby("type").power_kwp.median().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = [df[df.type == t].power_kwp for t in order]
    ax.boxplot(data, labels=[f"{t}\n(n={len(d)})" for t, d in zip(order, data)])
    ax.set_yscale("log"); ax.set_ylabel("kWp (log scale)")
    ax.set_title("Facility type (keyword in name) strongly separates capacity")
    save(fig, "fig11_power_by_type.png")
    df.groupby("type").power_kwp.describe().round(1).to_csv(OUT_TABLES / "power_by_type.csv")

    dc = df.district_clean.value_counts()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(len(dc)), dc.values, color="#4C72B0")
    ax.set_xticks(range(len(dc)), dc.index, rotation=90, fontsize=7)
    ax.set_ylabel("records"); ax.set_title(f"Records per district ({len(dc)} districts, {int((dc == 1).sum())} seen once)")
    save(fig, "fig12_district_freq.png")

    lens = df.apply(lambda r: len(tokenize_tr(r.name_clean)) + len(tokenize_tr(r.district_clean)), axis=1)
    seq = lens + 5  # [CLS] [ILCE] [YIL] year [MEKAN]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.hist(seq, bins=np.arange(seq.min() - 0.5, seq.max() + 1.5), color="#8172B3")
    ax.axvline(64, color="red", ls="--"); ax.text(seq.max() * 0.98, 5, "cap = 64 (never reached)", ha="right", fontsize=7, color="red")
    ax.set_xlabel("tokens per model sequence (incl. special tokens)"); ax.set_ylabel("records")
    ax.set_title(f"Sequence length: min {seq.min()}, median {int(seq.median())}, max {seq.max()}")
    save(fig, "fig13_sequence_length.png")

    rows = []
    for uid, g in df[df.in_regression].groupby("unit_id"):
        if g.file_year.nunique() > 1:
            g = g.sort_values("file_year")
            rows.append(dict(unit_id=uid, name=g.name_ws.iloc[-1], years=",".join(map(str, g.file_year)),
                             powers=",".join(f"{p:g}" for p in g.power_kwp),
                             max_abs_diff=g.power_kwp.max() - g.power_kwp.min(),
                             rel_diff=(g.power_kwp.max() - g.power_kwp.min()) / g.power_kwp.mean()))
    wu = pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)
    wu.to_csv(OUT_TABLES / "within_unit_variation.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(wu.max_abs_diff, bins=20, color="#937860")
    ax.set_xlabel("max - min power of the same unit across files (kWp)"); ax.set_ylabel("units")
    ax.set_title(f"Noise floor: {len(wu)} repeated units, mean |diff| = {wu.max_abs_diff.mean():.1f} kWp, "
                 f"{(wu.max_abs_diff == 0).mean():.0%} identical", fontsize=8)
    save(fig, "fig14_within_unit_variation.png")

    tf = TfidfVectorizer(tokenizer=tokenize_tr, lowercase=False, token_pattern=None, ngram_range=(1, 2))
    X = tf.fit_transform((df.district_clean + " " + df.name_clean).tolist())
    Z = TruncatedSVD(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=df.log_power, cmap="viridis", s=18)
    fig.colorbar(sc, ax=ax, label="ln(1 + kWp)")
    for t in ["Fire station", "Park / forest", "Toilet"]:
        m = (df.type == t).values
        ax.scatter(Z[m, 0], Z[m, 1], facecolors="none", edgecolors="red" if t == "Fire station" else "black", s=40, lw=0.6)
    ax.set_xlabel("SVD 1"); ax.set_ylabel("SVD 2")
    ax.set_title("TF-IDF name space (TruncatedSVD, descriptive only)\nred rings: fire stations, black rings: parks/toilets", fontsize=8)
    save(fig, "fig15_tfidf_svd.png")

    tok_year = {}
    for y in YEARS:
        for n in df[df.file_year == y].name_clean:
            for t in set(tokenize_tr(n)):
                tok_year.setdefault(t, {yy: 0 for yy in YEARS})[y] += 1
    ty = pd.DataFrame(tok_year).T
    ty["total"] = ty.sum(axis=1)
    ty["n_years"] = (ty[YEARS] > 0).sum(axis=1)
    single = ty[(ty.n_years == 1) & (ty.total >= 4)].sort_values("total", ascending=False)
    ty.sort_values("total", ascending=False).to_csv(OUT_TABLES / "token_year_counts.csv")
    frequent = ty.sort_values("total", ascending=False).head(15)
    share = frequent[YEARS].div(df.groupby("file_year").size(), axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.imshow(share.values, cmap="Purples", aspect="auto", vmin=0, vmax=1); ax.grid(False)
    ax.set_yticks(range(len(share)), share.index); ax.set_xticks(range(4), YEARS)
    for i in range(len(share)):
        for j in range(4):
            ax.text(j, i, f"{share.values[i, j]:.0%}", ha="center", va="center", fontsize=7)
    ax.set_title("Top tokens after cleaning: share of names per file\n('çatısı' still marks 2023-2025, 'ibb' 2024)", fontsize=8)
    save(fig, "fig16_token_year_share.png")

    print(summ.round(1).to_string())
    print("types:", df.type.value_counts().to_dict())
    print("seq len", seq.min(), seq.median(), seq.max(), "| districts", len(dc), "singletons", int((dc == 1).sum()))
    print("within-unit: n", len(wu), "mean diff", round(wu.max_abs_diff.mean(), 2), "identical", round((wu.max_abs_diff == 0).mean(), 3))
    print("single-year frequent tokens:", single.index.tolist())


if __name__ == "__main__":
    main()
