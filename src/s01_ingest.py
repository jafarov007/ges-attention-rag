"""Step 1 — read the four raw Excel files without modifying them.

* header row is detected by content (the row that contains 'İlçe'), never assumed;
* columns are mapped by name ('Ölçü Birimi' == 'Ölçü / Birim');
* a row is a record only if its 'Güç' cell is numeric;
* every skipped row is written to outputs/logs/ingest_log.csv with a reason.

Outputs: outputs/data/records_raw.csv, outputs/logs/ingest_log.csv,
         outputs/logs/raw_manifest.csv
"""
from __future__ import annotations

import json
import re

import openpyxl
import pandas as pd

from common import EXPECTED_COUNTS, EXPECTED_TOTALS, OUT_DATA, OUT_LOGS, RAW_DIR, basic_clean, sha256, tr_lower

COLUMN_MAP = {
    "no": "source_no",
    "ilçe": "district_raw",
    "mekan": "name_raw",
    "güç": "power_raw",
    "ölçü / birim": "unit_raw",
    "ölçü birimi": "unit_raw",
    "durum": "status_raw",
}


def _norm_header(v) -> str:
    return tr_lower(basic_clean(v))


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def ingest_file(path):
    year = int(re.match(r"(\d{4})", path.name).group(1))
    wb = openpyxl.load_workbook(path, data_only=True)
    if len(wb.worksheets) != 1:
        raise ValueError(f"{path.name}: expected exactly one sheet")
    ws = wb.worksheets[0]
    manifest = {
        "source_file": path.name,
        "file_year": year,
        "sha256": sha256(path),
        "sheet": ws.title,
        "max_row": ws.max_row,
        "max_column": ws.max_column,
        "merged_cells": ";".join(str(r) for r in ws.merged_cells.ranges),
        "openpyxl": openpyxl.__version__,
    }
    records, log = [], []
    colmap = None
    for row in ws.iter_rows():
        r = row[0].row
        values = [c.value for c in row]
        raw_json = json.dumps([None if v is None else str(v) for v in values], ensure_ascii=False)
        if colmap is None:
            heads = [_norm_header(v) for v in values]
            if "ilçe" in heads:
                colmap = {i: COLUMN_MAP[h] for i, h in enumerate(heads) if h in COLUMN_MAP}
                missing = {"district_raw", "name_raw", "power_raw", "unit_raw", "status_raw"} - set(colmap.values())
                if missing:
                    raise ValueError(f"{path.name}: header missing {missing}")
                log.append((path.name, ws.title, r, "header", "column header row", raw_json))
            elif all(v is None for v in values):
                log.append((path.name, ws.title, r, "empty", "empty row before header", raw_json))
            else:
                log.append((path.name, ws.title, r, "title", "title text above header", raw_json))
            continue
        rec = {colmap[i]: values[i] for i in colmap}
        if all(v is None for v in values):
            log.append((path.name, ws.title, r, "empty", "empty row", raw_json))
            continue
        if not _is_number(rec.get("power_raw")):
            log.append((path.name, ws.title, r, "description",
                        f"non-numeric power cell {rec.get('power_raw')!r} (unit/description row)", raw_json))
            continue
        rec.update({
            "record_id": f"GES-{year}-{r:03d}",
            "source_file": path.name,
            "sheet": ws.title,
            "excel_row": r,
            "file_year": year,
        })
        records.append(rec)
        log.append((path.name, ws.title, r, "record", "numeric power", raw_json))
    if colmap is None:
        raise ValueError(f"{path.name}: header row not found")
    return records, log, manifest


def main():
    files = sorted(RAW_DIR.glob("*.xlsx"))
    assert len(files) == 4, f"expected 4 raw files, found {len(files)}"
    all_rec, all_log, manifests = [], [], []
    for f in files:
        rec, log, man = ingest_file(f)
        all_rec += rec
        all_log += log
        manifests.append(man)
    df = pd.DataFrame(all_rec)
    cols = ["record_id", "source_file", "sheet", "excel_row", "file_year", "source_no",
            "district_raw", "name_raw", "power_raw", "unit_raw", "status_raw"]
    df = df.reindex(columns=cols)
    df["power_kwp"] = df["power_raw"].astype(float)

    # --- hard checks against the audited facts ---
    counts = df.groupby("file_year").size().to_dict()
    assert counts == EXPECTED_COUNTS, counts
    totals = df.groupby("file_year")["power_kwp"].sum().round(1).to_dict()
    for y, t in EXPECTED_TOTALS.items():
        assert abs(totals[y] - t) < 0.01, (y, totals[y], t)
    assert df["record_id"].is_unique

    df.to_csv(OUT_DATA / "records_raw.csv", index=False)
    pd.DataFrame(all_log, columns=["source_file", "sheet", "excel_row", "action", "reason", "raw_values"]) \
        .to_csv(OUT_LOGS / "ingest_log.csv", index=False)
    pd.DataFrame(manifests).to_csv(OUT_LOGS / "raw_manifest.csv", index=False)
    lg = pd.DataFrame(all_log, columns=["f", "s", "r", "action", "reason", "v"])
    print(f"records: {len(df)}  per year: {counts}")
    print("skipped rows:", lg[lg.action != "record"].groupby(["f", "action"]).size().to_dict())


if __name__ == "__main__":
    main()
