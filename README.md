# BM618 Ödev 2 — Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM

İstanbul'daki 2022–2025 güneş enerji sistemi (GES) envanter dosyaları üzerinde iki iş:

- **A. Kurulu güç kestirimi:** Tesis adı, ilçe ve dosya yılından `Güç (kWp)` kestirimi. B0/B1 referansları ve sıfırdan yazılmış attention ile D1/D2/D3. Değerlendirme görülmemiş tesislere göre (gruplu iç içe CV).
- **B. Kanıta dayalı soru-cevap:** Bilgi tabanı + BM25 + izinli hesap aracı + yerel instruction LLM (R0/R1/R2).

**Araştırma sorusu:** Az sayıda ve yıllar arasında tekrar eden tesis kaydında attention tabanlı bir model hangi koşullarda basit yöntemlerden daha yararlıdır; aynı veri üzerinde bir dil modeli hangi sorulara güvenilir biçimde yanıt verebilir?

## Belgeler — hangi konu hangi dosyada?

| # | Belge | İçerik |
|---|---|---|
| 00 | [docs/00_PLAN.md](docs/00_PLAN.md) | Başlangıç planı (ödev yönergesinden çıkarılan tüm adımlar) |
| 01 | [docs/01_veri_denetimi.md](docs/01_veri_denetimi.md) | Ham dosya yapıları, sha256, başlık tespiti, satır günlüğü, 153 kayıt, kalite bulguları Q1–Q18 |
| 02 | [docs/02_veri_temizleme_on_isleme.md](docs/02_veri_temizleme_on_isleme.md) | Kural kimlikli temizleme (R-WS, R-DIST, R-NAME, R-STAT, R-UNIT), yıl kalıbı sızıntısı, veri sözlüğü |
| 03 | [docs/03_tesis_eslestirme.md](docs/03_tesis_eslestirme.md) | İki seviyeli kimlik (`unit_id` / `entity_group_id`), kurallar A0–A7, 12 manuel karar, karantina |
| 04 | [docs/04_kesifsel_veri_analizi.md](docs/04_kesifsel_veri_analizi.md) | Hedef dağılımı, tür sinyali, TF-IDF SVD, ilçe seyrekliği, dizi uzunluğu, gürültü tabanı |
| 05 | [docs/05_protokol_ve_bolme.md](docs/05_protokol_ve_bolme.md) | Dondurulmuş protokol v1.0: girdi sözleşmesi, bölme manifesti, bütçe, metrikler, **hipotezler**, ablasyonlar, kilitli soru seti |
| 06 | [docs/06_matematik.md](docs/06_matematik.md) | M1–M4: boyutlar, varyans, softmax Jacobian'ı, ∂L/∂Q, M2 elle hesap, parametre/maliyet, Pre/Post-LN, permütasyon, log-hedef |
| 07 | [docs/07_tokenizasyon_ve_girdi.md](docs/07_tokenizasyon_ve_girdi.md) | Dizi biçimi, önekli alan tokenları, yalnız eğitimden sözlük, `min_freq=2` / `[UNK]` |
| 08 | [docs/08_model_mimarisi.md](docs/08_model_mimarisi.md) | B1, D1, D2, D3 mimarileri, sıfırdan MHA, parametre dökümü, 23 uygulama testi |
| 09 | [docs/09_egitim_ve_deney_duzeni.md](docs/09_egitim_ve_deney_duzeni.md) | İç içe CV prosedürü, öğrenme eğrileri, hiperparametre seçimi, maliyet (eğitim öncesi/sırası/sonrası) |
| 10 | [docs/10_sonuclar_ve_ablasyon.md](docs/10_sonuclar_ve_ablasyon.md) | Ana sonuçlar, bootstrap güven aralıkları, kat/tohum dağılımı, ablasyonlar, sızıntı tanısı, hipotezlerin akıbeti |
| 11 | [docs/11_hata_analizi.md](docs/11_hata_analizi.md) | En büyük hatalar, alt gruplar, attention ↔ token silme |
| 12 | [docs/12_rag.md](docs/12_rag.md) | Bilgi tabanı, LLM kartı, R0/R1/R2, puanlama, dev k seçimi |

Görseller `docs/figures/` altında (fig01–fig32) ve ilgili belgelere gömülü. Görsel içi yazılar İngilizce.

## Özet bulgular (regresyon)

| Model | Tesis dengeli MAE (kWp) | log-MAE | R² |
|---|---|---|---|
| B0 medyan | 87,7 | 1,08 | −0,14 |
| B1 TF-IDF + Ridge | 81,4 | 0,92 | −0,16 |
| D1 ortalama havuzlama | 80,1 | 0,89 | −0,14 |
| D2 attention havuzlama | 78,3 | 0,82 | −0,03 |
| D3 Transformer | **76,2** | **0,79** | −0,10 |

- Yalnız **D3 − B0** farkı %95 grup bootstrap aralığında anlamlı. D2 − D1 ve D3 − B1 ayırt edilemiyor (H1 yanlışlandı).
- Rastgele **satır** bölmesinde B1'in hatası 28,8 kWp'ye düşüyor (R² 0,77). Test kayıtlarının %72,8'inin tesisi eğitimde de bulunuyor → **sızıntı** (H2 doğrulandı).
- Hataların çoğu, eğitimde benzeri olmayan 450–574 kWp'lik birkaç tesisten geliyor ([11](docs/11_hata_analizi.md)).

## Klasör yapısı
```
data/raw/              ham Excel (repoda YOK; aşağıya bakın)
protocol/              dondurulmuş kararlar: düzeltmeler, eşleştirme kararları, bölme manifesti, altın sorular
src/                   s01_ingest → s02_clean → s03_match → s04_eda → s05_splits → train → evaluate → error_analysis
src/rag/               build_gold, kb (BM25 + araç), llm, systems (R0/R1/R2), eval_rag
tests/                 veri hattı (12) + attention/model (23) testleri
docs/                  belgeler ve figürler
outputs/logs, tables/  denetim günlükleri ve sonuç tabloları (repoda)
outputs/data, predictions, models, rag_runs/   üretilen veri, tahmin ve ağırlıklar (repoda YOK)
```

## Kurulum ve çalıştırma
```bash
python3 -m venv .venv --system-site-packages && source .venv/bin/activate
pip install -r requirements.txt      # + transformers/accelerate (RAG için)
# ham verileri data/raw/ altına koyun (sha256: outputs/logs/raw_manifest.csv)
cd src
python s01_ingest.py && python s02_clean.py && python s03_match.py && python s04_eda.py && python s05_splits.py
python math_checks.py
python train.py main && python train.py ablations && python train.py leakage
python evaluate.py && python error_analysis.py
cd rag && python build_gold.py && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_rag.py dev
cd ../.. && python -m pytest tests -q
```
Ana deneyin tamamı tek CPU iş parçacığında ≈ 80 saniye sürer.

## Veri kaynağı
İBB açık veri: “2022/2023/2024/2025 yılı kurulumu tamamlanan güneş enerji sistemleri” (4 xlsx). Dosyalar repoya eklenmedi; doğruluk sha256 ile kontrol edilir (`tests/test_data_pipeline.py::RAW_SHA`).
