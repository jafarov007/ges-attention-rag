"""Knowledge base (one document per raw record, all 153 incl. quarantined), BM25 retriever
and the whitelisted calculation tool used by R2.

The KB contains power values and is kept apart from the regression code: it reads
records_clean.csv only, never predictions or splits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DATA, tokenize_tr, tr_lower  # noqa: E402

FLAG_TEXT = {
    "conflict_same_year": "UYARI: aynı dosyada aynı tesis için çelişkili ikinci kayıt var",
    "district_conflict": "UYARI: bu tesis farklı dosyalarda farklı ilçe etiketiyle geçiyor",
    "phase_merge_candidate": "NOT: etap/bütün ilişkisi olası kayıt",
    "name_overlap_other_unit": "NOT: adı başka bir birimin adıyla örtüşüyor",
    "status_ambiguous": "NOT: Durum alanı '(Aktif / Pasif)' sütun açıklaması, aktiflik bilgisi değil",
    "status_mixed": "NOT: Durum alanında hem Aktif hem Pasif yazıyor",
}


def load_records() -> pd.DataFrame:
    return pd.read_csv(OUT_DATA / "records_clean.csv")


def doc_text(r) -> str:
    flags = [FLAG_TEXT[f] for f in str(r.quality_flag).split(";") if f in FLAG_TEXT]
    return (f"[{r.record_id}] dosya: {r.source_file} | sayfa: {r.sheet} | Excel satırı: {r.excel_row} | "
            f"dosya yılı (kaynak sürümü, kurulum yılı değil): {r.file_year} | ilçe: {r.district_clean} | "
            f"mekan: {r.name_ws} | güç: {r.power_kwp:g} kWp | durum (ham): {r.status_raw}"
            + (" | " + " ; ".join(flags) if flags else ""))


class Retriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.docs = [doc_text(r) for r in self.df.itertuples()]
        corpus = [tokenize_tr(f"{r.district_clean} {r.name_ws} {r.file_year} {r.status_raw}") for r in self.df.itertuples()]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int):
        scores = self.bm25.get_scores(tokenize_tr(query))
        order = scores.argsort()[::-1][:k]
        return [(self.df.record_id[i], self.docs[i], float(scores[i])) for i in order]


# ----------------------------------------------------------------- whitelisted tool (R2)
ALLOWED_INTENTS = {"lookup", "count", "sum", "max", "min", "list", "compare_years", "unanswerable"}
ALLOWED_FILTERS = {"file_year", "district", "status", "name_contains", "min_power"}
STATUS_MAP = {"aktif": "aktif", "pasif": "pasif", "karma": "karma", "belirsiz": "belirsiz"}


def validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("plan is not an object")
    intent = str(plan.get("intent", "")).lower()
    if intent not in ALLOWED_INTENTS:
        raise ValueError(f"intent not allowed: {intent}")
    filters = plan.get("filters") or {}
    bad = set(filters) - ALLOWED_FILTERS
    if bad:
        raise ValueError(f"filters not allowed: {bad}")
    clean = {}
    for k, v in filters.items():
        if v in (None, "", "null"):
            continue
        if k == "file_year":
            v = int(v)
            if v not in (2022, 2023, 2024, 2025):
                raise ValueError("file_year outside 2022-2025")
        if k == "status":
            v = STATUS_MAP.get(tr_lower(str(v)))
            if v is None:
                raise ValueError("status not allowed")
        if k == "min_power":
            v = float(v)
        clean[k] = v
    years = [int(y) for y in plan.get("years", []) or []]
    return {"intent": intent, "filters": clean, "years": years}


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    s = df
    if "file_year" in f:
        s = s[s.file_year == f["file_year"]]
    if "district" in f:
        s = s[s.district_clean.map(tr_lower) == tr_lower(f["district"].strip())]
    if "status" in f:
        s = s[s.status_clean == f["status"]]
    if "min_power" in f:
        s = s[s.power_kwp >= f["min_power"]]
    if "name_contains" in f:
        needle = tokenize_tr(f["name_contains"])
        keep = [all(t in tokenize_tr(n) for t in needle) for n in s.name_ws]
        s = s[keep]
    return s


def record_view(r) -> dict:
    return {"record_id": r.record_id, "dosya": r.source_file, "excel_satiri": int(r.excel_row), "dosya_yili": int(r.file_year),
            "ilce": r.district_clean, "mekan": r.name_ws, "guc_kwp": float(r.power_kwp), "durum_ham": r.status_raw,
            "uyarilar": [FLAG_TEXT[x] for x in str(r.quality_flag).split(";") if x in FLAG_TEXT]}


def run_tool(df: pd.DataFrame, plan: dict) -> dict:
    """Executes ONLY whitelisted pandas operations on the FULL table (never on top-k)."""
    p = validate_plan(plan)
    if p["intent"] == "unanswerable":
        return {"intent": "unanswerable", "result": None, "record_ids": [], "n_rows": 0}
    if p["intent"] == "compare_years":
        out = {}
        ids = []
        for y in p["years"]:
            s = apply_filters(df, {**p["filters"], "file_year": y})
            out[str(y)] = {"kayit": len(s), "guc_toplami_kwp": round(float(s.power_kwp.sum()), 2)}
            ids += s.record_id.tolist()
        if len(p["years"]) == 2:
            a, b = (out[str(y)]["guc_toplami_kwp"] for y in p["years"])
            out["fark_kwp"] = round(a - b, 2)
        return {"intent": "compare_years", "result": out, "record_ids": ids, "n_rows": len(ids)}
    s = apply_filters(df, p["filters"])
    res = {"intent": p["intent"], "filters": p["filters"], "n_rows": int(len(s)), "record_ids": s.record_id.tolist()}
    if p["intent"] == "count":
        res["result"] = int(len(s))
    elif p["intent"] == "sum":
        res["result"] = round(float(s.power_kwp.sum()), 2)
    elif p["intent"] in ("max", "min") and len(s):
        r = s.loc[s.power_kwp.idxmax() if p["intent"] == "max" else s.power_kwp.idxmin()]
        res["result"] = record_view(r)
    else:  # lookup / list
        res["result"] = [record_view(r) for r in s.head(20).itertuples()]
    if p["intent"] in ("lookup", "list", "count", "sum"):
        res["distinct_districts"] = sorted(set(s.district_clean))
    return res
