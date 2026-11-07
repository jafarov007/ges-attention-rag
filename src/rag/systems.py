"""R0 (no evidence), R1 (BM25 RAG), R2 (RAG + whitelisted calculation tool).
Same LLM, same question, same generation budget (answer: max_new_tokens = 320)."""
from __future__ import annotations

import json
import re
import time

from kb import Retriever, run_tool

ANSWER_TOKENS = 320
PLAN_TOKENS = 120

BASE = ("Sen İstanbul Büyükşehir Belediyesi güneş enerji sistemi (GES) envanter dosyaları (2022-2025) hakkında "
        "soru yanıtlayan bir asistansın. Türkçe ve kısa yanıt ver. Bilmediğin veya kanıtı olmayan bilgiyi uydurma; "
        "emin değilsen bunu açıkça söyle.")

R1_SYSTEM = BASE + (
    " Yalnız verilen KANIT kayıtlarını kullan. Her olgusal bilgiden sonra köşeli parantez içinde kaynağın record_id'sini "
    "yaz, örneğin [GES-2024-013]. Aynı soru için birden fazla kayıt veya çelişkili değer varsa hepsini kaynaklarıyla "
    "birlikte ver, tek değer uydurma. Kanıt soruyu yanıtlamıyorsa 'Verilen kayıtlarda bu bilgi yok' de. "
    "Durum sorularında kaydın 'durum (ham)' değerini aynen aktar; 'Aktif - Pasif' veya '(Aktif / Pasif)' yazıyorsa "
    "durumun kesin olmadığını söyle. Kayıttaki UYARI ve NOT ifadelerini yanıtına ekle. "
    "Dosyalar yalnız ilçe, mekan adı, kurulu güç (kWp) ve durum içerir; elektrik üretimi (kWh), maliyet, tarih, "
    "marka bilgisi içermez. Dosya yılı kurulum yılı değildir.")

R2_SYSTEM = R1_SYSTEM + (
    " HESAP SONUCU, tüm tablo üzerinde kodla hesaplanmıştır; sayım ve toplamları oradan al, kendin hesaplama. "
    "HESAP SONUCU 'unanswerable' ise veya sonuç boşsa bilginin dosyalarda olmadığını söyle.")

PLANNER_SYSTEM = """Görevin: kullanıcı sorusunu bir JSON sorgu planına çevirmek. Yalnız JSON yaz, açıklama yazma.
Tablo alanları: file_year (2022, 2023, 2024, 2025), district (ilçe adı), name (mekan adı), power_kwp (kurulu güç), status (aktif, pasif, karma, belirsiz).
Tabloda OLMAYAN bilgiler: elektrik üretimi (kWh), maliyet, devreye alma tarihi, panel markası, CO2, gelecek planları.
Şema: {"intent": "lookup|count|sum|max|min|list|compare_years|unanswerable", "filters": {"file_year": sayı, "district": "ilçe", "status": "aktif|pasif|karma|belirsiz", "name_contains": "mekan adından ayırt edici sözcükler", "min_power": sayı}, "years": [yıl, yıl]}
Kullanılmayan filtreleri yazma.
Örnekler:
Soru: 2023 dosyasında Şile'de kaç kayıt var? -> {"intent": "count", "filters": {"file_year": 2023, "district": "Şile"}}
Soru: 2025 dosyasında Gölet Parkı GES'in gücü nedir? -> {"intent": "lookup", "filters": {"file_year": 2025, "name_contains": "Gölet Parkı"}}
Soru: 2022 dosyasında Bakırköy kayıtlarının toplam gücü kaç kWp? -> {"intent": "sum", "filters": {"file_year": 2022, "district": "Bakırköy"}}
Soru: 2024 ve 2025 dosyalarının güç toplamları farkı nedir? -> {"intent": "compare_years", "filters": {}, "years": [2024, 2025]}
Soru: 2024 dosyasında en düşük güçlü kayıt hangisi? -> {"intent": "min", "filters": {"file_year": 2024}}
Soru: 2025 dosyasında 100 kWp ve üzeri kaç kayıt var? -> {"intent": "count", "filters": {"file_year": 2025, "min_power": 100}}
Soru: Sefaköy Yüzme Havuzu GES'in yıllık üretimi kaç kWh? -> {"intent": "unanswerable"}"""


def evidence_block(hits) -> str:
    return "\n".join(d for _, d, _ in hits)


def parse_plan(text: str):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in planner output")
    return json.loads(m.group(0))


class Systems:
    def __init__(self, llm, df, k: int):
        self.llm, self.df, self.k = llm, df, k
        self.ret = Retriever(df)

    def r0(self, q):
        t0 = time.perf_counter()
        ans, meta = self.llm.chat(BASE, f"SORU: {q}", ANSWER_TOKENS)
        return dict(system="R0", answer=ans, retrieved=[], plan=None, tool=None, seconds=time.perf_counter() - t0, llm=[meta])

    def r1(self, q):
        t0 = time.perf_counter()
        hits = self.ret.search(q, self.k)
        ans, meta = self.llm.chat(R1_SYSTEM, f"KANIT:\n{evidence_block(hits)}\n\nSORU: {q}", ANSWER_TOKENS)
        return dict(system="R1", answer=ans, retrieved=[h[0] for h in hits], plan=None, tool=None,
                    seconds=time.perf_counter() - t0, llm=[meta])

    def r2(self, q):
        t0 = time.perf_counter()
        hits = self.ret.search(q, self.k)
        plan_txt, m1 = self.llm.chat(PLANNER_SYSTEM, f"Soru: {q} ->", PLAN_TOKENS)
        plan, tool, err = None, None, None
        try:
            plan = parse_plan(plan_txt)
            tool = run_tool(self.df, plan)
        except Exception as e:  # invalid or non-whitelisted plan -> no tool result, evidence only
            err = f"{type(e).__name__}: {e}"
        tool_txt = json.dumps(tool, ensure_ascii=False) if tool is not None else f"HESAP YAPILAMADI ({err})"
        if tool is not None and len(tool_txt) > 6000:
            tool_txt = tool_txt[:6000] + " ...(kısaltıldı)"
        user = f"HESAP SONUCU:\n{tool_txt}\n\nKANIT:\n{evidence_block(hits)}\n\nSORU: {q}"
        ans, m2 = self.llm.chat(R2_SYSTEM, user, ANSWER_TOKENS)
        return dict(system="R2", answer=ans, retrieved=[h[0] for h in hits], plan=plan_txt, plan_error=err,
                    tool=tool, seconds=time.perf_counter() - t0, llm=[m1, m2])
