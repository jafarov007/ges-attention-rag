"""Defence helper: trace ONE record end to end.

    python trace_record.py GES-2024-013

Excel row -> ingest log -> cleaning rules -> matching decisions / groups -> split fold and
leakage check -> model tokens ([UNK] marked) -> outer-test predictions -> D3 attention -> RAG evidence.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from common import ABLATION_SEED, OUT_DATA, OUT_LOGS, OUT_MODELS, OUT_PRED, PROTOCOL
from tokenizer import SPECIALS, UNK, Vocab, batch_encode, record_tokens


def h(title):
    print("\n" + "=" * 8 + f" {title} " + "=" * (70 - len(title)))


def main(rid):
    df = pd.read_csv(OUT_DATA / "records_clean.csv").set_index("record_id")
    if rid not in df.index:
        raise SystemExit(f"unknown record_id {rid}")
    r = df.loc[rid]

    h("1. HAM EXCEL SATIRI")
    log = pd.read_csv(OUT_LOGS / "ingest_log.csv")
    row = log[(log.source_file == r.source_file) & (log.excel_row == r.excel_row)].iloc[0]
    man = pd.read_csv(OUT_LOGS / "raw_manifest.csv").set_index("source_file").loc[r.source_file]
    print(f"dosya   : {r.source_file}  (sha256 {man.sha256[:16]}...)\nsayfa   : {r.sheet}\nsatır   : {r.excel_row}")
    print(f"hücreler: {row.raw_values}\nkarar   : {row.action} ({row.reason})")

    h("2. TEMİZLEME")
    cl = pd.read_csv(OUT_LOGS / "cleaning_log.csv")
    for c in cl[cl.record_id == rid].itertuples():
        print(f"[{c.rule_id}] {c.field}: {c.raw!r} -> {c.clean!r}")
    print(f"ilçe={r.district_clean} | model metni={r.name_clean!r} | güç={r.power_kwp:g} kWp | "
          f"durum={r.status_raw}->{r.status_clean} | bayraklar={r.quality_flag}")

    h("3. EŞLEŞTİRME")
    grp = df[df.entity_group_id == r.entity_group_id]
    print(f"unit_id={r.unit_id}  entity_group_id={r.entity_group_id}  in_regression={r.in_regression}")
    for o in grp.itertuples():
        print(f"   {o.Index} {o.file_year} {o.district_clean:<13} unit={o.unit_id} {o.name_clean}")
    ml = pd.read_csv(OUT_LOGS / "matching_log.csv")
    sub = ml[((ml.record_id_a == rid) | (ml.record_id_b == rid)) & (ml.decision != "different")]
    for m in sub.itertuples():
        print(f"   karar: {m.record_id_a} ~ {m.record_id_b}: {m.decision} ({m.basis}, {m.confidence}) - {m.rationale}")

    h("4. BÖLME VE SIZINTI KONTROLÜ")
    sp = pd.read_csv(PROTOCOL / "split_manifest.csv").set_index("record_id")
    k = int(sp.loc[rid, "outer_fold"])
    same_group_in_train = sp[(sp.entity_group_id == r.entity_group_id) & (sp.outer_fold != k)]
    print(f"dış test katı = {k}; aynı gruptan eğitimde kayıt sayısı = {len(same_group_in_train)} (0 olmalı)")
    assert len(same_group_in_train) == 0

    h("5. MODEL GİRDİSİ (dış kat eğitim sözlüğü, min_freq=2)")
    reg = df[df.in_regression].join(sp[["outer_fold"]])
    tr = reg[reg.outer_fold != k]
    vocab = Vocab(2).fit([record_tokens(d, y, n) for d, y, n in tr[["district_clean", "file_year", "name_clean"]].itertuples(index=False)])
    toks = record_tokens(r.district_clean, r.file_year, r.name_clean)
    print(" ".join(t if t in SPECIALS or vocab.stoi.get(t, UNK) != UNK else f"{t}->[UNK]" for t in toks))

    h("6. DIŞ TEST TAHMİNLERİ")
    if not r.in_regression:
        print("karantinada: regresyonda kullanılmadı (eğitim de test de değil)")
    else:
        p = pd.read_csv(OUT_PRED / "outer_predictions_main.csv")
        p = p[p.record_id == rid]
        for m, g in p.groupby("model", sort=False):
            print(f"{m}: tohum ort. {g.y_hat.mean():7.1f} kWp  (tohumlar: {', '.join(f'{v:.1f}' for v in g.y_hat.unique())})"
                  f"  gerçek {r.power_kwp:g}  |hata| {abs(g.y_hat.mean() - r.power_kwp):.1f}")
        ck = torch.load(OUT_MODELS / f"D3_outer{k}_seed{ABLATION_SEED}.pt", weights_only=False)
        from models import build_model
        v = Vocab(); v.itos = ck["itos"]; v.stoi = {t: i for i, t in enumerate(v.itos)}
        model = build_model("D3", len(v.itos), ck["dropout"]); model.load_state_dict(ck["state"]); model.eval()
        ids, _ = batch_encode(v, [toks])
        with torch.no_grad():
            model(ids)
        att = model.attention_maps()[-1][0].mean(0)[0].numpy()
        order = np.argsort(att)[::-1][:5]
        print("D3 (tohum 42) son katman [CLS] attention, ilk 5: " + ", ".join(f"{toks[i]}={att[i]:.2f}" for i in order))
        print("(attention ağırlığı açıklama değildir; bkz. docs/11 token silme karşılaştırması)")

    h("7. RAG KANITI")
    sys.path.insert(0, str(OUT_DATA.parents[1] / "src" / "rag"))
    from kb import Retriever, doc_text, run_tool
    print(doc_text(df.reset_index().set_index("record_id", drop=False).loc[rid].rename(None)))
    q = f"{r.file_year} dosyasında {r.district_clean} {r.name_clean} GES'in gücü nedir?"
    hits = Retriever(df.reset_index()).search(q, 3)
    print(f"BM25 ({q!r}) ilk 3: {[x[0] for x in hits]}  -> kayıt ilk 3'te: {rid in [x[0] for x in hits]}")
    tool = run_tool(df.reset_index(), {"intent": "lookup", "filters": {"file_year": int(r.file_year), "district": r.district_clean}})
    print(f"araç (aynı yıl + ilçe, tüm tablo): {tool['n_rows']} kayıt, bu kayıt dahil: {rid in tool['record_ids']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "GES-2024-013")
