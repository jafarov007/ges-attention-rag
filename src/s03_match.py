"""Step 3 — facility matching across file years (power and status are NEVER used).

Two identities are produced:
* unit_id          : records that are the same physical PV unit
* entity_group_id  : connected component of "possibly the same facility" links
                     (same_unit + same_group). GroupKFold uses this id.

Procedure
1. match_key: Turkish lower case, year templates removed, style words removed
   (ibb, çatısı, bina, binası), the record's own district tokens removed,
   unit numbers ('1. Etap', 'Etap 2', '(3)') extracted into unit_no.
2. candidate pairs: same district & token_set_ratio >= 70, or different
   district & ratio >= 90 (cross-district scan catches relabelled districts).
3. automatic proposal (see AUTO RULES below), then manual decisions from
   protocol/matching_decisions.csv override it. A pair whose proposal is
   'review' must have a manual decision, otherwise the script fails.
4. union-find -> unit_id / entity_group_id; quality flags; quarantine.
"""
from __future__ import annotations

import re
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz

from common import OUT_DATA, OUT_LOGS, PROTOCOL, strip_boilerplate, tokenize_tr

STYLE_WORDS = {"ibb", "çatısı", "bina", "binası"}
UNIT_PATTERNS = [r"(\d+)\s*\.\s*etap", r"etap\s*(\d+)", r"\((\d+)\)"]


def unit_number(name: str):
    low = name.lower()
    for p in UNIT_PATTERNS:
        m = re.search(p, low)
        if m:
            return int(m.group(1))
    return None


def match_key(name: str, district: str) -> str:
    text = strip_boilerplate(name)
    for p in UNIT_PATTERNS:
        text = re.sub(p, " ", text, flags=re.I)
    toks = tokenize_tr(text)
    dist = set(tokenize_tr(district))
    toks = [t for t in toks if t not in STYLE_WORDS and t not in dist and t != "etap"]
    return " ".join(toks)


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


SITES: list[tuple[str, str]] = []  # (district, site_key) from protocol/site_subunits.csv


def site_descriptor(rec):
    for district, key in SITES:
        if rec.district_clean == district and rec.match_key.startswith(key):
            return key, rec.match_key[len(key):].strip()
    return None, None


def auto_rule(a, b, score):
    """AUTO RULES (documented in docs/03_tesis_eslestirme.md)."""
    if a.district_clean != b.district_clean:
        return "different", "A1 different district"
    if a.file_year == b.file_year:
        return "different", "A2 same file, presumed different units"
    sa, da = site_descriptor(a)
    sb, db = site_descriptor(b)
    if sa is not None and sa == sb:
        if da == db and a.unit_no == b.unit_no:
            return "same_unit", "A0 declared multi-unit site, same sub-unit descriptor and number"
        return "different", "A0 declared multi-unit site, different sub-unit"
    if a.unit_no is not None and b.unit_no is not None and a.unit_no != b.unit_no:
        return "different", "A3 different stage/sub-unit numbers"
    ta, tb = a.match_key.split(), b.match_key.split()
    if ta != tb and min(len(ta), len(tb)) == 1:
        return "review", "A7 generic one-word key is a subset of a specific name"
    if (a.unit_no is None) != (b.unit_no is None):
        if score >= 90:
            return "same_group", "A4 one side has a stage number -> stage/whole relation"
        return "review", "A4b stage number on one side but names differ"
    if score >= 90:
        return "same_unit", "A5 same district, ratio >= 90"
    return "review", "A6 same district, 70 <= ratio < 90"


def main():
    full = pd.read_csv(OUT_DATA / "records_clean_stage2.csv")
    sites = pd.read_csv(PROTOCOL / "site_subunits.csv")
    SITES[:] = list(zip(sites.district, sites.site_key))
    # Matching must not see power or status (assignment rule) -> drop them explicitly.
    m = full[["record_id", "file_year", "district_clean", "name_ws"]].copy()
    assert not {"power_kwp", "power_raw", "status_raw", "status_clean"} & set(m.columns)
    m["unit_no"] = [unit_number(n) for n in m.name_ws]
    m["match_key"] = [match_key(n, d) for n, d in zip(m.name_ws, m.district_clean)]
    m["unit_no"] = m["unit_no"].astype("object").where(m["unit_no"].notna(), None)

    rows = list(m.itertuples(index=False))
    cands = []
    for a, b in combinations(rows, 2):
        ratio = fuzz.token_set_ratio(a.match_key, b.match_key)
        same_d = a.district_clean == b.district_clean
        if (same_d and ratio >= 70) or (not same_d and ratio >= 90):
            prop, why = auto_rule(a, b, ratio)
            cands.append(dict(record_id_a=a.record_id, record_id_b=b.record_id,
                              key_a=a.match_key, key_b=b.match_key,
                              district_a=a.district_clean, district_b=b.district_clean,
                              token_set_ratio=round(ratio, 1),
                              token_sort_ratio=round(fuzz.token_sort_ratio(a.match_key, b.match_key), 1),
                              auto_decision=prop, auto_reason=why))
    cand = pd.DataFrame(cands)

    dec = pd.read_csv(PROTOCOL / "matching_decisions.csv", comment="#")
    dec_map = {(r.record_id_a, r.record_id_b): r for r in dec.itertuples()}
    final, conf, basis, rationale = [], [], [], []
    for r in cand.itertuples():
        d = dec_map.pop((r.record_id_a, r.record_id_b), None)
        if d is not None:
            final.append(d.decision); conf.append(d.confidence); basis.append("manual"); rationale.append(d.rationale)
        elif r.auto_decision == "review":
            raise SystemExit(f"needs manual decision: {r.record_id_a} ~ {r.record_id_b} ({r.key_a} | {r.key_b})")
        else:
            final.append(r.auto_decision)
            conf.append("high" if r.auto_decision in ("same_unit", "different") else "medium")
            basis.append("auto"); rationale.append(r.auto_reason)
    cand["decision"], cand["confidence"], cand["basis"], cand["rationale"] = final, conf, basis, rationale
    # manual decisions for pairs that were not generated as candidates (low similarity)
    extra = []
    for (a, b), d in dec_map.items():
        extra.append(dict(record_id_a=a, record_id_b=b, key_a=m.set_index("record_id").match_key[a],
                          key_b=m.set_index("record_id").match_key[b], decision=d.decision,
                          confidence=d.confidence, basis="manual (not a candidate)", rationale=d.rationale))
    log = pd.concat([cand, pd.DataFrame(extra)], ignore_index=True)

    uf_unit, uf_group = UnionFind(m.record_id), UnionFind(m.record_id)
    for r in log.itertuples():
        if r.decision == "same_unit":
            uf_unit.union(r.record_id_a, r.record_id_b)
            uf_group.union(r.record_id_a, r.record_id_b)
        elif r.decision == "same_group":
            uf_group.union(r.record_id_a, r.record_id_b)

    def ids(uf, prefix):
        roots = {rid: uf.find(rid) for rid in m.record_id}
        order = {root: i for i, root in enumerate(sorted(set(roots.values())))}
        return {rid: f"{prefix}{order[root]:03d}" for rid, root in roots.items()}

    unit_id, group_id = ids(uf_unit, "U"), ids(uf_group, "G")
    full["unit_no"] = m["unit_no"].values
    full["match_key"] = m["match_key"].values
    full["unit_id"] = full.record_id.map(unit_id)
    full["entity_group_id"] = full.record_id.map(group_id)

    # quality flags derived from the matching
    flags = {rid: ([] if f == "ok" else f.split(";")) for rid, f in zip(full.record_id, full.quality_flag)}
    for uid, g in full.groupby("unit_id"):
        if g.file_year.duplicated().any():
            for rid in g[g.file_year.duplicated(keep=False)].record_id:
                flags[rid].append("conflict_same_year")
        if g.district_clean.nunique() > 1:
            for rid in g.record_id:
                flags[rid].append("district_conflict")
    for r in log[log.decision == "same_group"].itertuples():
        for rid in (r.record_id_a, r.record_id_b):
            if "phase_merge_candidate" not in flags[rid] and "örtüşme" not in str(r.rationale):
                flags[rid].append("phase_merge_candidate")
            if "örtüşme" in str(r.rationale) and "name_overlap_other_unit" not in flags[rid]:
                flags[rid].append("name_overlap_other_unit")
    full["quality_flag"] = [";".join(flags[r]) if flags[r] else "ok" for r in full.record_id]
    full["in_regression"] = ~full.quality_flag.str.contains("conflict_same_year")

    # a unit can never hold two different file-year records except the documented conflict
    assert full[~full.in_regression].record_id.tolist() == ["GES-2024-013", "GES-2024-015"], \
        full[~full.in_regression].record_id.tolist()

    full.to_csv(OUT_DATA / "records_clean.csv", index=False)
    log.to_csv(OUT_LOGS / "matching_log.csv", index=False)
    ent = (full.groupby("entity_group_id")
           .agg(n_records=("record_id", "size"), n_units=("unit_id", "nunique"),
                years=("file_year", lambda s: ",".join(map(str, sorted(set(s))))),
                districts=("district_clean", lambda s: ",".join(sorted(set(s)))),
                record_ids=("record_id", ",".join),
                example_name=("name_ws", "first"))
           .reset_index())
    ent.to_csv(OUT_LOGS / "entities.csv", index=False)
    print(f"records={len(full)} units={full.unit_id.nunique()} groups={full.entity_group_id.nunique()}")
    print("decisions:", log.decision.value_counts().to_dict(), "| basis:", log.basis.value_counts().to_dict())
    print("group size dist:", ent.n_records.value_counts().sort_index().to_dict())
    print(full.quality_flag.value_counts().to_string())


if __name__ == "__main__":
    main()
