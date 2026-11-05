"""Data pipeline tests (run after src/s01-s03)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT_DATA, OUT_LOGS, RAW_DIR, sha256, strip_boilerplate, tr_lower  # noqa: E402

RAW_SHA = {
    "2022-yl-kurulumu-tamamlanan-gune-enerji-sistemleri.xlsx": "e2983072e179f8bc540ba664089840a6f270cd282010e9fa8bc9d68256c80b95",
    "2023_kurulumu-tamamlanan-gune-enerji-sistemleri.xlsx": "b760df1f9d02a6365b2759386be6f7307d7b23583f92ac102cc71433ab552078",
    "2024-yl-kurulumu-tamamlanan-gune-enerji-sistemleri.xlsx": "7c170eb8fb93609e2d7b2de7d4432a5b8521227d56542a0e4771db2aae77ebe8",
    "2025-yl-kurulumu-tamamlanan-gune-enerji-sistemleri.xlsx": "32ba496529aa6805faea1c10a834967497b04c198f9dffc54d1e4d8bc47d0115",
}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(OUT_DATA / "records_clean.csv")


def test_raw_files_unchanged():
    for name, h in RAW_SHA.items():
        assert sha256(RAW_DIR / name) == h, name


def test_counts_and_totals(df):
    assert df.groupby("file_year").size().to_dict() == {2022: 20, 2023: 11, 2024: 46, 2025: 76}
    tot = df.groupby("file_year").power_kwp.sum().round(1).to_dict()
    assert tot == {2022: 3834.8, 2023: 984.7, 2024: 3887.0, 2025: 8418.0}


def test_ids_unique_and_traceable(df):
    assert df.record_id.is_unique
    assert not df.duplicated(["source_file", "excel_row"]).any()
    assert (df.record_id == "GES-" + df.file_year.astype(str) + "-" + df.excel_row.map("{:03d}".format)).all()


def test_every_skipped_row_logged():
    log = pd.read_csv(OUT_LOGS / "ingest_log.csv")
    man = pd.read_csv(OUT_LOGS / "raw_manifest.csv")
    # every physical row up to max_row appears exactly once in the log
    for m in man.itertuples():
        rows = log[log.source_file == m.source_file].excel_row
        assert sorted(rows) == list(range(1, m.max_row + 1)), m.source_file


def test_status_counts(df):
    s25 = df[df.file_year == 2025].status_clean.value_counts().to_dict()
    assert s25 == {"aktif": 59, "pasif": 15, "karma": 2}
    assert df[df.file_year == 2024].status_clean.value_counts().to_dict() == {"aktif": 45, "belirsiz": 1}
    assert df.loc[df.record_id == "GES-2024-002", "status_clean"].item() == "belirsiz"


def test_district_typo(df):
    r = df[df.record_id == "GES-2022-005"].iloc[0]
    assert r.district_raw == "Atnavutköy" and r.district_clean == "Arnavutköy"
    assert "Atnavutköy" not in set(df.district_clean)
    assert not df.district_clean.str.endswith(" ").any()


def test_turkish_lower():
    assert tr_lower("İBB IŞIK") == "ibb ışık"


def test_boilerplate_removal_removes_year_templates():
    assert strip_boilerplate("Davet ve Toplantı Salonu Güneş Enerji Santrali (GES)") == "Davet ve Toplantı Salonu"
    assert strip_boilerplate("Fatih Spor Kompleksi") == "Fatih Spor Kompleksi"
    assert strip_boilerplate("Hoşdere Park Alanı GES") == "Hoşdere Park Alanı"
    assert strip_boilerplate("Orhangazi Şehir Parkı- Su Deposu Üzeri (1) Güneş Enerji Santrali") == \
        "Orhangazi Şehir Parkı - Su Deposu Üzeri (1)"


def test_no_year_template_tokens_left(df):
    low = df.name_clean.map(tr_lower)
    assert not low.str.contains(r"\bges\b|güneş enerji santrali", regex=True).any()


def test_matching_groups(df):
    g = df.set_index("record_id").entity_group_id
    u = df.set_index("record_id").unit_id
    # stages: different units, same evaluation group
    assert u["GES-2022-012"] != u["GES-2022-013"]
    assert g["GES-2022-012"] == g["GES-2022-013"] == g["GES-2025-016"] == g["GES-2023-010"]
    assert u["GES-2022-013"] == u["GES-2023-010"]  # '2. Etap' == '(2)'
    # Maltepe sub-units stay separate
    assert len({u[f"GES-2024-0{r}"] for r in range(29, 35)}) == 6
    assert len({g[f"GES-2024-0{r}"] for r in range(29, 35)}) == 6
    assert u["GES-2024-030"] == u["GES-2025-051"]
    # same name, different district -> different
    assert g["GES-2025-003"] != g["GES-2025-062"]
    assert g["GES-2024-002"] != g["GES-2024-026"]
    # İPA / Florya: different unit, same group
    assert u["GES-2022-002"] == u["GES-2025-011"] != u["GES-2025-014"]
    assert g["GES-2022-002"] == g["GES-2025-014"]
    # generic 'Büyükçekmece İtfaiye' is not Kumburgaz / Mimar Sinan
    assert g["GES-2024-011"] not in {g["GES-2025-022"], g["GES-2025-023"]}
    # Kumburgaz across districts
    assert u["GES-2023-008"] == u["GES-2024-010"] == u["GES-2025-022"]
    assert df.entity_group_id.nunique() == 76


def test_quarantine(df):
    q = df[~df.in_regression]
    assert sorted(q.record_id) == ["GES-2024-013", "GES-2024-015"]
    assert sorted(q.power_kwp) == [25.0, 29.0]  # kept, not averaged
    assert df.in_regression.sum() == 151


def test_matching_does_not_read_power():
    src = (Path(__file__).resolve().parents[1] / "src" / "s03_match.py").read_text(encoding="utf-8")
    assert 'full[["record_id", "file_year", "district_clean", "name_ws"]]' in src
