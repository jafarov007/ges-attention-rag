"""Build the 30-question gold set. Every numeric / evidence answer is computed
from records_clean.csv with an explicit filter spec (no hand-typed numbers).

Output: protocol/qa_gold.jsonl  (split = dev | test). The test part is locked
before any prompt or k is chosen; its sha256 is recorded in docs/05.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DATA, PROTOCOL, sha256  # noqa: E402

df = pd.read_csv(OUT_DATA / "records_clean.csv")


def sel(**f):
    s = df
    for k, v in f.items():
        if k == "name_contains":
            s = s[s.name_ws.str.contains(v, case=False, regex=False)]
        elif k == "min_power":
            s = s[s.power_kwp >= v]
        else:
            s = s[s[k] == v]
    return s


Q = []


def add(qid, split, cat, question, answer, numeric=None, tol=None, evidence=(), answerable=True):
    Q.append(dict(qid=qid, split=split, category=cat, question=question, gold_answer=answer,
                  gold_numeric=numeric, tolerance=tol, evidence_record_ids=list(evidence),
                  answerable=answerable, paraphrase_group=qid))


def single(qid, split, q, field="power_kwp", **f):
    s = sel(**f)
    assert len(s) == 1, (qid, len(s))
    r = s.iloc[0]
    if field == "power_kwp":
        add(qid, split, "single", q, f"{r.power_kwp:g} kWp ({r.record_id}, {r.source_file}, Excel satırı {r.excel_row})",
            float(r.power_kwp), 0.0, [r.record_id])
    else:
        add(qid, split, "single", q, f"{r.status_raw} ({r.record_id}, Excel satırı {r.excel_row})", None, None, [r.record_id])


# ---------------- TEST (locked) ----------------
single("T01", "test", "2024 dosyasında Tuzla Bahar Merkezi GES'in gücü kaç kWp?", file_year=2024, district_clean="Tuzla", name_contains="Bahar Merkezi")
single("T02", "test", "2022 dosyasında Silivri Kapı Buz Pisti GES'in gücü nedir?", file_year=2022, name_contains="Buz Pisti")
single("T03", "test", "2025 dosyasında Kadıköy Filiz Akın Kız Öğrenci Yurdu GES'in gücü kaç kWp?", file_year=2025, name_contains="Filiz Akın")
single("T04", "test", "2023 dosyasında Avcılar Sahil Tuvaleti GES'in gücü nedir?", file_year=2023, name_contains="Sahil Tuvaleti")
single("T05", "test", "2024 dosyasında Maltepe Orhangazi Şehir Parkı Su Deposu Üzeri (2) GES'in gücü kaç kWp?", file_year=2024, name_contains="Su Deposu Üzeri (2)")
single("T06", "test", "2025 dosyasında Pendik Yenişehir İtfaiye Eğitim Binası GES'in durumu nedir?", field="status", file_year=2025, name_contains="Yenişehir")

s = sel(file_year=2025, status_clean="pasif")
add("T07", "test", "aggregate", "2025 dosyasında Durum alanında açıkça 'Pasif' yazan kaç kayıt var?",
    f"{len(s)} kayıt ('Aktif - Pasif' yazan 2 karma kayıt sayılmaz)", len(s), 0, s.record_id)
s = sel(file_year=2024, district_clean="Maltepe")
add("T08", "test", "aggregate", "2024 dosyasındaki Maltepe kayıtlarının toplam gücü kaç kWp?",
    f"{s.power_kwp.sum():g} kWp ({len(s)} kayıt)", float(s.power_kwp.sum()), 0.01, s.record_id)
s = sel(file_year=2025, district_clean="Kartal")
add("T09", "test", "aggregate", "2025 dosyasında Kartal ilçesinde kaç kayıt var?", f"{len(s)} kayıt", len(s), 0, s.record_id)
s = sel(file_year=2022)
add("T10", "test", "aggregate", "2022 dosyasındaki Güç sütununun toplamı kaç kWp?",
    f"{s.power_kwp.sum():.1f} kWp ({len(s)} kayıt; benzersiz tesis kapasitesi değildir)", round(float(s.power_kwp.sum()), 1), 0.05, s.record_id)
a, b = sel(file_year=2022), sel(file_year=2023)
add("T11", "test", "aggregate", "2022 ve 2023 dosyalarının Güç toplamları arasındaki fark kaç kWp?",
    f"{a.power_kwp.sum() - b.power_kwp.sum():.1f} kWp (2022: {a.power_kwp.sum():.1f}, 2023: {b.power_kwp.sum():.1f})",
    round(float(a.power_kwp.sum() - b.power_kwp.sum()), 1), 0.05, list(a.record_id) + list(b.record_id))
s = sel(file_year=2025, min_power=300)
add("T12", "test", "aggregate", "2025 dosyasında gücü 300 kWp ve üzeri olan kaç kayıt var?", f"{len(s)} kayıt", len(s), 0, s.record_id)

s = sel(file_year=2024, name_contains="Ömerli")
add("T13", "test", "conflict", "2024 dosyasında Çekmeköy Ömerli İtfaiye GES için tek bir güç değeri verilebilir mi?",
    "Hayır. 2024 dosyasında aynı ad ve ilçeyle iki kayıt var: " +
    "; ".join(f"{r.power_kwp:g} kWp (Excel satırı {r.excel_row}, {r.record_id})" for r in s.itertuples()) +
    ". Değerler çelişkili; tek değer verilemez.", None, None, s.record_id)
s = sel(name_contains="Kumburgaz")
add("T14", "test", "conflict", "Kumburgaz İtfaiye GES hangi ilçededir?",
    "Kaynaklar çelişiyor: " + "; ".join(f"{r.file_year}: {r.district_clean} ({r.record_id})" for r in s.itertuples()),
    None, None, s.record_id)
s = sel(district_clean="Başakşehir", name_contains="Engelliler Merkezi")
add("T15", "test", "conflict", "Başakşehir Engelliler Merkezi GES'in gücü kaç kWp?",
    "Tek değer yok (yıl/etap belirsiz): " + "; ".join(f"{r.file_year} '{r.name_clean}': {r.power_kwp:g} kWp ({r.record_id})" for r in s.itertuples()),
    None, None, s.record_id)
s = sel(file_year=2025, name_contains="Mezarlığı Hizmet ve Gasilhane")
add("T16", "test", "conflict", "2025 dosyasında 'Mezarlığı Hizmet ve Gasilhane Bina Çatısı' GES'in gücü nedir?",
    "Bu ad 2025'te iki farklı ilçede geçiyor: " + "; ".join(f"{r.district_clean} {r.power_kwp:g} kWp ({r.record_id})" for r in s.itertuples()),
    None, None, s.record_id)

add("T17", "test", "unanswerable", "2025 dosyasındaki tesislerin yıllık elektrik üretimi toplam kaç kWh?",
    "Yanıtlanamaz: dosyalar yalnız kurulu güç (kWp) içerir, üretim (kWh) içermez.", answerable=False)
add("T18", "test", "unanswerable", "Tuzla Bahar Merkezi GES hangi tarihte devreye alındı?",
    "Yanıtlanamaz: devreye alma tarihi yok; dosya yılı kurulum tarihi değildir.", answerable=False)
add("T19", "test", "unanswerable", "Fatih Spor Kompleksi GES'in yatırım maliyeti ne kadardır?",
    "Yanıtlanamaz: maliyet bilgisi dosyalarda yok.", answerable=False)
add("T20", "test", "unanswerable", "2026 yılında İstanbul'da kaç yeni GES kurulacak?",
    "Yanıtlanamaz: dosyalar gelecek planı içermez.", answerable=False)

# ---------------- DEV ----------------
single("D01", "dev", "2024 dosyasında Şişli Muhsin Ertuğrul Sahnesi GES'in gücü kaç kWp?", file_year=2024, name_contains="Muhsin Ertuğrul")
single("D02", "dev", "2025 dosyasında Eyüpsultan Kemerburgaz İtfaiye GES'in gücü nedir?", file_year=2025, name_contains="Kemerburgaz")
single("D03", "dev", "2023 dosyasında Güngören Davet ve Toplantı Salonu GES'in durumu nedir?", field="status", file_year=2023, name_contains="Davet")
s = sel(file_year=2024)
add("D04", "dev", "aggregate", "2024 dosyasında kaç kayıt var?", f"{len(s)} kayıt", len(s), 0, s.record_id)
s = sel(file_year=2025, district_clean="Sancaktepe")
add("D05", "dev", "aggregate", "2025 dosyasında Sancaktepe kayıtlarının toplam gücü kaç kWp?",
    f"{s.power_kwp.sum():g} kWp ({len(s)} kayıt)", float(s.power_kwp.sum()), 0.01, s.record_id)
s = sel(file_year=2023)
r = s.loc[s.power_kwp.idxmax()]
add("D06", "dev", "aggregate", "2023 dosyasında en yüksek güce sahip kayıt hangisi?",
    f"{r.district_clean} {r.name_clean}: {r.power_kwp:g} kWp ({r.record_id})", float(r.power_kwp), 0, [r.record_id])
s = sel(file_year=2025, name_contains="Florya")
add("D07", "dev", "conflict", "2025 dosyasında Florya Atatürk Kent Ormanı GES aktif mi?",
    f"Kesin söylenemez: Durum alanında '{s.iloc[0].status_raw}' yazıyor ({s.iloc[0].record_id}).", None, None, s.record_id)
s = sel(file_year=2024, district_clean="Ataşehir", name_contains="İtfaiye")
add("D08", "dev", "conflict", "2024 dosyasında Ataşehir İBB İtfaiye GES aktif mi?",
    f"Kesin söylenemez: Durum alanında '{s.iloc[0].status_raw}' yazıyor; bu sütun açıklaması, aktiflik bilgisi değil ({s.iloc[0].record_id}).",
    None, None, s.record_id)
add("D09", "dev", "unanswerable", "Silivri Kapı Buz Pisti GES'te hangi marka panel kullanılıyor?",
    "Yanıtlanamaz: ekipman bilgisi dosyalarda yok.", answerable=False)
add("D10", "dev", "unanswerable", "2024 dosyasındaki GES'lerin toplam CO2 tasarrufu ne kadar?",
    "Yanıtlanamaz: emisyon/üretim bilgisi dosyalarda yok.", answerable=False)


def main():
    test = [q for q in Q if q["split"] == "test"]
    dev = [q for q in Q if q["split"] == "dev"]
    cats = pd.Series([q["category"] for q in test]).value_counts().to_dict()
    assert len(test) == 20 and len(dev) == 10, (len(test), len(dev))
    assert cats == {"single": 6, "aggregate": 6, "conflict": 4, "unanswerable": 4}, cats
    with open(PROTOCOL / "qa_gold.jsonl", "w", encoding="utf-8") as f:
        for q in Q:
            f.write(json.dumps(q, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    with open(PROTOCOL / "qa_test_locked.jsonl", "w", encoding="utf-8") as f:
        for q in test:
            f.write(json.dumps(q, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    for q in Q:
        print(q["qid"], q["category"], "|", q["gold_answer"][:110])
    print("locked test sha256:", sha256(PROTOCOL / "qa_test_locked.jsonl"))


if __name__ == "__main__":
    main()
