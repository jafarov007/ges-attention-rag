"""Step 2 — rule-based cleaning. Raw columns are never overwritten.

Rules (ids are referenced in outputs/logs/cleaning_log.csv and in
docs/02_veri_temizleme_on_isleme.md):

R-WS-01   NFC + non-breaking space -> space + whitespace collapse + strip
R-DIST-01 explicit district spelling corrections from protocol/district_corrections.csv
R-NAME-02 remove year-specific naming templates ("Güneş Enerji Santrali", "(GES)", "GES")
R-STAT-01 'Aktif' -> aktif
R-STAT-02 'Pasif' -> pasif
R-STAT-03 'Aktif - Pasif' -> karma        (both words, cannot be resolved)
R-STAT-04 '(Aktif / Pasif)' -> belirsiz  (column template text copied into a record)
R-UNIT-01 unit must be 'kWp'
"""
from __future__ import annotations

import pandas as pd

from common import OUT_DATA, OUT_LOGS, PROTOCOL, basic_clean, strip_boilerplate, tr_lower

STATUS_RULES = {
    "aktif": ("aktif", "R-STAT-01"),
    "pasif": ("pasif", "R-STAT-02"),
    "aktif - pasif": ("karma", "R-STAT-03"),
    "(aktif / pasif)": ("belirsiz", "R-STAT-04"),
}


def main():
    df = pd.read_csv(OUT_DATA / "records_raw.csv")
    corr = pd.read_csv(PROTOCOL / "district_corrections.csv")
    corr_map = dict(zip(corr.raw_value, zip(corr.normalized_value, corr.rule_id)))
    log, flags = [], {rid: [] for rid in df.record_id}

    def note(rid, field, raw, clean, rule):
        if str(raw) != str(clean):
            log.append((rid, field, raw, clean, rule))

    dist_clean, name_ws, name_clean, stat_clean, unit_clean = [], [], [], [], []
    for r in df.itertuples():
        # district
        d = basic_clean(r.district_raw)
        note(r.record_id, "district", r.district_raw, d, "R-WS-01")
        if d != r.district_raw:
            flags[r.record_id].append("whitespace_fixed")
        if d in corr_map:
            new, rule = corr_map[d]
            note(r.record_id, "district", d, new, rule)
            flags[r.record_id].append("district_typo_fixed")
            d = new
        dist_clean.append(d)
        # name
        n = basic_clean(r.name_raw)
        note(r.record_id, "name", r.name_raw, n, "R-WS-01")
        if n != r.name_raw and "whitespace_fixed" not in flags[r.record_id]:
            flags[r.record_id].append("whitespace_fixed")
        name_ws.append(n)
        nc = strip_boilerplate(n)
        note(r.record_id, "name_clean", n, nc, "R-NAME-02")
        name_clean.append(nc)
        # status
        s = tr_lower(basic_clean(r.status_raw))
        if s not in STATUS_RULES:
            raise ValueError(f"unknown status {r.status_raw!r} in {r.record_id}")
        sc, rule = STATUS_RULES[s]
        note(r.record_id, "status", r.status_raw, sc, rule)
        if sc == "belirsiz":
            flags[r.record_id].append("status_ambiguous")
        if sc == "karma":
            flags[r.record_id].append("status_mixed")
        stat_clean.append(sc)
        # unit
        u = basic_clean(r.unit_raw)
        if u != "kWp":
            flags[r.record_id].append("unit_issue")
            log.append((r.record_id, "unit", r.unit_raw, u, "R-UNIT-01"))
        unit_clean.append(u)

    df["district_clean"] = dist_clean
    df["name_ws"] = name_ws          # whitespace-normalised full name (for RAG / display)
    df["name_clean"] = name_clean    # model text: boilerplate removed
    df["status_clean"] = stat_clean
    df["unit_clean"] = unit_clean
    df["quality_flag"] = [";".join(flags[r]) if flags[r] else "ok" for r in df.record_id]

    assert (df.unit_clean == "kWp").all()
    assert (df.name_clean.str.len() > 0).all()
    assert ((df.file_year == 2025) & (df.status_clean == "pasif")).sum() == 15
    assert ((df.file_year == 2025) & (df.status_clean == "karma")).sum() == 2
    assert ((df.file_year == 2024) & (df.status_clean == "belirsiz")).sum() == 1

    df.to_csv(OUT_DATA / "records_clean_stage2.csv", index=False)
    pd.DataFrame(log, columns=["record_id", "field", "raw", "clean", "rule_id"]) \
        .to_csv(OUT_LOGS / "cleaning_log.csv", index=False)
    print(pd.Series([r for *_, r in log]).value_counts().to_string())
    print(df.quality_flag.value_counts().to_string())


if __name__ == "__main__":
    main()
