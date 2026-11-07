# 12 — Kanıta Dayalı Soru-Cevap (RAG)

> İlgili kod: [`src/rag/kb.py`](../src/rag/kb.py) (bilgi tabanı, BM25, hesap aracı), [`src/rag/llm.py`](../src/rag/llm.py), [`src/rag/systems.py`](../src/rag/systems.py), [`src/rag/eval_rag.py`](../src/rag/eval_rag.py), [`src/rag/build_gold.py`](../src/rag/build_gold.py)
> Altın set: [`protocol/qa_gold.jsonl`](../protocol/qa_gold.jsonl) · Kilitli test: [`protocol/qa_test_locked.jsonl`](../protocol/qa_test_locked.jsonl)

> **Durum:** Bilgi tabanı, getirici, hesap aracı, üç sistem ve puanlama hazır. Geliştirme (dev) setinde k seçimi ve prompt geliştirme yapıldı. **Kilitli 20 test sorusu henüz çalıştırılmadı** (`python eval_rag.py test`).

## 1. Bilgi tabanı (regresyondan ayrı)
- **153 satır belgesi**, karantinadaki Ömerli kayıtları dahil. Her belgede `record_id`, dosya adı, sayfa, Excel satırı, dosya yılı (“kurulum yılı değil” notuyla), ilçe, tam mekan adı, güç, ham durum ve kalite uyarıları (çelişki, ilçe çelişkisi, etap, belirsiz/karma durum) yer alır.
- KB yalnız `records_clean.csv` okur. Regresyon tahminlerine, bölmelere veya modellere erişmez. **RAG başarısı regresyon başarısı olarak sunulmaz.**

## 2. LLM (sıfırdan eğitim yok)
| Alan | Değer |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (Hugging Face, Apache-2.0) |
| Nicemleme | yok, fp16 |
| Donanım | NVIDIA RTX 2050 (4 GB), GPU belleği ≈ 3,1 GB |
| Üretim | greedy (`do_sample=False`), cevap `max_new_tokens=320`, plan `120` |
| Ortam | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (aksi hâlde k=10 promptlarında bellek yetmiyor) |
| Kart | `outputs/tables/rag_llm_card.json` |

## 3. Üç sistem (aynı model, aynı soru, aynı üretim bütçesi)
- **R0, kanıtsız LLM:** yalnız soru.
- **R1, RAG:** BM25 (`rank_bm25`, Türkçe-duyarlı tokenizasyon; ilçe + ad + yıl + durum indekslenir) ile ilk k belge. Prompt: her olguya `[record_id]` atfı, çelişkide tüm değerler, kanıt yoksa “bu bilgi yok”.
- **R2, RAG + hesap aracı:**
  1. LLM soruyu JSON **sorgu planına** çevirir (`lookup | count | sum | max | min | list | compare_years | unanswerable`; filtreler `file_year, district, status, name_contains, min_power`).
  2. `validate_plan` beyaz liste dışındaki intent/filtreleri reddeder. `run_tool` yalnız izinli pandas filtre/sayım/toplamlarını **tüm tablo** üzerinde çalıştırır (ilk-k üzerinden toplam yok). **LLM'in ürettiği kod hiçbir zaman çalıştırılmaz.**
  3. Hesap sonucu, eşleşen tüm `record_id`'ler ve BM25 kanıtı ile LLM yanıtı yazar.
- Her sorgu için plan, getirilen `record_id`'ler, hesap sonucu, yanıt ve süre `outputs/rag_runs/*.json`'a kaydedilir (git dışı).

## 4. Puanlama
- **Otomatik kurallar** (`eval_rag.py`, testten önce yazıldı): sayısal toleransla eşleşme, çelişki sorularında tüm değerlerin geçmesi, çekimserlik ifadeleri, atıf çıkarımı (`GES-YYYY-RRR`), sayısal iddiaların kanıtla desteklenmesi.
- **Nihai etiket elle verilir:** test yanıtlarının hepsi okunur, `outputs/tables/rag_manual_review.csv` yazılır. LLM hakem kullanılmaz.
- Metrikler: tekil sorularda Recall@k, doğruluk, sayısal doğruluk, atıf doğruluğu, kanıt kapsamı, doğru çekimserlik, gereksiz ret. Boş paydalar “uygulanamaz”.

## 5. Geliştirme seti: k seçimi ve prompt geliştirme (yalnız dev)
| k | Dev doğruluk (R1) | Recall@k (tekil) |
|---|---|---|
| **3** | **0,5** | 1,0 |
| 5 | 0,4 | 1,0 |
| 10 | 0,3 | 1,0 |

Seçim kuralı: en yüksek dev doğruluğu → recall → küçük k. **Seçilen k = 3.** Daha çok belge, küçük modelin dikkatini dağıtıp yanlış satırdan sayı çekmesine yol açtı.

Dev'de görülen sorun ve düzeltme: R1, “Aktif - Pasif” ve “(Aktif / Pasif)” kayıtları için yalnız “Aktif” yanıtı verdi. Prompta “ham durumu aynen aktar, karma/belirsizse kesin olmadığını söyle, UYARI/NOT ifadelerini ekle” talimatı eklendi. Puanlayıcıdaki çekimserlik ifadesine “bilgisi yok” kalıbı eklendi.

Dev setinde üç sistem (k = 3, otomatik puan, n = 10):

| Sistem | Doğruluk | Sayısal doğruluk | Toplama | Doğru çekimserlik | Gereksiz ret |
|---|---|---|---|---|---|
| R0 | 0,1 | 0,0 | 0,00 | 0,0 | 0,25 |
| R1 | 0,5 | 0,4 | 0,00 | 1,0 | 0,25 |
| R2 | **0,7** | **0,6** | 0,33 | 1,0 | 0,125 |

Toplama sorularında R1'in sıfır alması beklenen bir sonuç: ilk 3 belgeden toplam/sayım yapılamaz. R2'nin hesap aracı bu sınıfta fark yaratıyor. Bu dev sonuçlarıdır; H3'ün sınanması kilitli test setiyle yapılacak.
