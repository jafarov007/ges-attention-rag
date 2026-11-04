# BM618 Ödev 2 — Uçtan Uca Çalışma Planı

**Konu:** Az verili GES envanterinde attention, Transformer ve kanıta dayalı LLM
**Veri:** `data/` içindeki 4 Excel (2022–2025). `ODEV_2/2/Odev 2/` altındaki kopyalarla md5 olarak birebir aynı.
**Bu belgenin amacı:** Veri denetiminden RAG değerlendirmesine kadar her adımda *ne yapılacağını, neden yapılacağını ve hangi tuzaklara dikkat edileceğini* tek yerde toplamak. Buradaki tüm veri gözlemleri ham Excel dosyalarına satır satır bakılarak çıkarıldı (Excel satır numaraları 1'den başlar, başlık satırı dahil).

---

## 0. Önce şunu ezberle: ödevin ruhu

1. **Skor değil yöntem puanlanıyor.** Transformer'ın medyanı veya Ridge'i geçememesi *başarılı* bir sonuç olabilir — yeter ki deney sızıntısız, kontrollü ve dürüst raporlansın. Hocanın notu: “Yanlış veya desteklenmeyen yüksek başarı puanı ödüllendirilmemelidir.”
2. **Dosya yılı ≠ kurulum yılı.** Kodda, tabloda, raporda “kurulum yılı” ifadesi geçmeyecek. Alan adı `file_year`, anlamı “kaynak sürümü”.
3. **Aynı tesis iki tarafa düşmez.** Aynı olası tesisin hiçbir kaydı dış eğitim ile dış test arasında bölünmeyecek.
4. **Güç değeri eşleştirmede kullanılmaz.** Tesis eşleştirme yalnız ad + ilçe + etap/alt birim bilgisine dayanır.
5. **Hiçbir şey silinmez, ortalanmaz.** Çelişkiler karantinaya alınır, ham tabloda etiketle yaşamaya devam eder.
6. **Her sayı izlenebilir.** Her kayıt → `source_file`, `sheet`, `excel_row`, `record_id`.
7. **Dış test skoruna bakarak hiçbir seçim yapılmaz** (model, epoch, k, prompt, eşik).
8. **Attention ağırlığı açıklama değildir.** Token silme deneyiyle karşılaştırılmadan yorumlanmaz.

---

## 1. Proje klasör yapısı (önerilen)

```
ABDUL_JAFAROV_2/
├── data/                         # HAM dosyalar — salt okunur, asla değiştirilmez
├── ODEV_2_PLAN.md                # bu belge
├── protocol/                     # 1. hafta teslimi, model eğitiminden ÖNCE dondurulur
│   ├── data_dictionary.md
│   ├── matching_rules.md
│   ├── exclusion_rules.md
│   ├── hypotheses.md
│   ├── metrics.md
│   ├── split_manifest.csv        # record_id, entity_group_id, outer_fold, inner_fold
│   ├── protocol_v1.0.md          # hepsini birleştiren dondurulmuş belge + tarih + sha256
│   └── qa_test_locked.jsonl      # 20 kilitli test sorusu (hocaya teslim edilen sürüm)
├── src/
│   ├── 01_ingest.py              # ham → records_raw.csv + ingest_log
│   ├── 02_clean.py               # temizleme → records_clean.csv
│   ├── 03_match.py               # aday çiftler + manuel karar dosyası → entity tablosu
│   ├── 04_eda.py                 # tablolar ve grafikler
│   ├── 05_splits.py              # GroupKFold manifesti
│   ├── tokenizer.py              # Türkçe-duyarlı normalizasyon + sözlük
│   ├── attention.py              # SIFIRDAN MHA (ana çözüm)
│   ├── models.py                 # B0, B1, D1, D2, D3
│   ├── train.py                  # iç arama + dış yeniden eğitim
│   ├── evaluate.py               # metrikler, bootstrap
│   ├── ablations.py
│   ├── leakage_check.py
│   ├── error_analysis.py         # büyük hatalar, attention vs token silme
│   ├── math_checks.py            # M2 elle hesap doğrulaması, sonlu fark
│   └── rag/
│       ├── build_kb.py
│       ├── retriever.py          # BM25
│       ├── tools.py              # izinli filtre/sayım/toplam
│       ├── llm.py                # yerel model sarmalayıcı
│       ├── systems.py            # R0, R1, R2
│       └── eval_rag.py
├── tests/                        # pytest: attention ve veri hattı testleri
├── outputs/
│   ├── data/                     # records_raw.csv, records_clean.csv, entities.csv
│   ├── logs/                     # ingest_log.csv, cleaning_log.csv, matching_log.csv
│   ├── predictions/              # outer_predictions_{model}_{seed}.csv
│   ├── figures/
│   └── rag_runs/                 # her sorgu için plan, record_id'ler, yanıt, süre
├── report/
│   └── rapor.md / rapor.pdf
├── requirements.txt              # sürümler sabit
└── RUN_ALL.sh                    # tek komutla tüm hattı yeniden üretir
```

---

## 2. Ortam ve yeniden üretilebilirlik

**Mevcut makine (kontrol edildi):** Python 3.12.3, torch 2.13.0+cu130 (CUDA çalışıyor), numpy 1.26.4, pandas 2.1.4, GPU = NVIDIA RTX 2050 **4 GB VRAM**, 15 GB RAM, 16 çekirdek. `scikit-learn` ve `rapidfuzz` **kurulu değil**.

- Kali'de sistem pip'ini bozmamak için sanal ortam:
  ```bash
  python3 -m venv .venv --system-site-packages && source .venv/bin/activate
  pip install scikit-learn openpyxl rapidfuzz rank-bm25 pytest
  # LLM için ikisinden biri:
  pip install llama-cpp-python      # GGUF modeller, CUDA build tercih edilir
  # veya: ollama kur + ollama pull qwen2.5:3b-instruct
  pip freeze > requirements.txt
  ```
- Modeller çok küçük (~20 bin parametre); **D1–D3 için CPU yeterli ve daha deterministik**. GPU'yu LLM'e ayır.
- Determinizm: `random`, `numpy`, `torch` tohumları; `torch.use_deterministic_algorithms(True)`; `torch.set_num_threads(1)` (küçük modelde fark etmez, tekrar üretimi kolaylaştırır). DataLoader shuffle için ayrı `torch.Generator` tohumu.
- Her koşuda bir **run kartı** kaydet: git commit (yoksa script sha256), config JSON, veri dosyalarının sha256'sı, kütüphane sürümleri, donanım, tohum, süre, *başarısız koşular dahil*. Ek alanlar: `experiment_id, data_manifest_version, split_manifest_version, parameter_count, train/val/test kayıt ve grup sayıları, primary_metric, all_metrics, runtime, peak_memory, failed_conditions`.
- **Ham dosya manifesti** (`outputs/data/raw_manifest.csv`): her dosya için `source_file, sha256, sheet adları, max_row, max_column, birleşik hücreler, okuma zamanı, openpyxl sürümü`.
- `RUN_ALL.sh` sırası: ingest → clean → match (manuel karar dosyasını okur) → splits → math_checks → tests → train (B0,B1,D1,D2,D3) → ablations → leakage → error_analysis → rag → rapor tabloları.

---

## 3. Ham veri envanteri (gerçek gözlemler)

### 3.1 Dosya yapıları — her dosya farklı!

| Dosya | Sayfa adı | Başlık satırı | Başlık dışı/açıklama satırları | Sütunlar | Sayısal kayıt | Excel satır aralığı |
|---|---|---|---|---|---|---|
| 2022 | `2022 KURULUMU TAMAMLANAN GES` | 1 | — | No, İlçe, Mekan, Güç, `Ölçü / Birim`, Durum | 20 | 2–21 |
| 2023 | `2023 KURULUMU TAMAMLANAN GES` | 2 | 1 = başlık metni, 3 = `(Aktif / Pasif)` açıklaması; A15:C15 boş birleşik hücre | No, İlçe, Mekan, Güç, `Ölçü / Birim`, Durum | 11 | 4–14 |
| 2024 | `2024 KURULUMU TAMAMLANAN GES` | 1 | — | İlçe, Mekan, Güç, `Ölçü Birimi`, Durum (**No yok**) | 46 | 2–47 |
| 2025 | `2025 KURULUMU TAMAMLANAN GES` | 1 (A1:A2, B1:B2, D1:D2 birleşik) | 2 = `(kWp)` / `(Aktif / Pasif)` birim satırı | İlçe, Mekan, Güç, `Ölçü / Birim`, Durum (**No yok**) | 76 | 3–78 |

**Sonuç:** Sabit “ilk satırı atla” mantığı çalışmaz. Doğru yöntem: `İlçe` hücresini içeren satırı başlık kabul et, sütunları **adla** eşle (`Ölçü Birimi` ↔ `Ölçü / Birim`), `Güç` sayısal olmayan her satırı “kayıt değil” olarak günlüğe yaz. Toplam = **153 kayıt** (20/11/46/76). Bu sayı hocanın notuyla birebir tutuyor; `01_ingest.py` bunu `assert` ile kontrol etmeli.

### 3.2 Doğrulanmış özet istatistikler

| Yıl | Kayıt | İlçe etiketi | Güç toplamı | Medyan | Min–Maks | Ham durum |
|---|---|---|---|---|---|---|
| 2022 | 20 | 15 | 3.834,8 | 106 | 20–574,2 | 20 Aktif |
| 2023 | 11 | 9 | 984,7 | 25 | 6,6–305 | 11 Aktif |
| 2024 | 46 | 21 | 3.887 | 36 | 3–484 | 45 Aktif, 1 belirsiz |
| 2025 | 76 | 34 | 8.418 | 63 | 3–574 | 59 Aktif, 15 Pasif, 2 karma |

Tüm veri: medyan **70 kWp**, çeyrekler ≈ 25 / 118, aralık 3–574,2. Dağılım **çok sağa çarpık**; hedefi log uzayında modellemek şart. Ondalıklı değerler yalnız 2022 (105,6; 574,2) ve 2023'te (8,5; 6,6) var; 2024–2025 değerleri tamsayıya yuvarlanmış. Bu durum aynı tesisin yıllar arasındaki küçük farklarını kısmen açıklıyor (ör. 574,2 → 574).

### 3.3 Kalite bulguları — tam liste

| # | Bulgu | Konum | Yapılacak |
|---|---|---|---|
| Q1 | İlçe yazım hatası `Atnavutköy` | 2022 r5 | `district_clean = Arnavutköy`, cleaning_log'a yaz |
| Q2 | İlçede sondaki boşluk `Sultangazi ` | 2024 r42, 2025 r69 | `strip()`; ham değer korunur |
| Q3 | Mekan adında `\xa0` (bölünmez boşluk) | 2022 r19 (Güngören) | NFC + `\xa0`→boşluk |
| Q4 | Çift boşluk, sondaki boşluklar | 2022 r8 ve birçok satır | boşlukları tekilleştir |
| Q5 | Başlık/açıklama satırları | 2023 r1, r2, r3; 2025 r1–r2; 2022/2024 r1 | `excluded_rows` günlüğüne gerekçesiyle yaz |
| Q6 | `Durum = (Aktif / Pasif)` gerçek kayıtta | 2024 r2 Ataşehir İBB İtfaiye | `status_clean = belirsiz`; aktif **sayılmaz** |
| Q7 | `Durum = Aktif - Pasif` | 2025 r14 Florya Atatürk Kent Ormanı, r17 Başakşehir İtfaiye | `status_clean = karma`; Pasif sayımına **eklenmez** |
| Q8 | Aynı tesis–yıl için iki kayıt, farklı güç | 2024 r13 (29 kWp) ve r15 (25 kWp) Çekmeköy Ömerli İtfaiye | **Karantina**: `quality_flag = conflict_same_year`, ana regresyondan çıkar, KB'de iki kayıt da kalır |
| Q9 | İlçe çelişkisi | Kumburgaz İtfaiye: 2023 r8 **Silivri**, 2024 r10 ve 2025 r22 **Büyükçekmece** | Aynı gruba al, `quality_flag = district_conflict`, güveni “orta” |
| Q10 | Etap birleşmesi | Başakşehir Engelliler Merkezi: 2022 r12 1. Etap 99 + r13 2. Etap 80; 2023 r10 “(2)” 80; 2025 r16 etapsız 179 (= 99+80) | Tek **değerlendirme grubu**; etaplar ayrı `unit_id` |
| Q11 | Olası etap birleşmesi | Ataköy: 2022 r3 “Kadın ve Aile Hizm. Müd.” 106; 2023 r9 “Kadın Koordinasyon Müd. – Etap 2” 100; 2025 r13 “Sosyal Hizmetler Daire Bşk.” 210 (≈106+100) | Tek değerlendirme grubu, güven “orta”, gerekçe yazılı |
| Q12 | Etap → bütün | Kağıthane Cemal Kamacı: 2022 r15 “2. Etap” 387; 2025 r40 etapsız 428 | Tek değerlendirme grubu, güven “orta” |
| Q13 | Ad içinde başka tesisin adı | 2022 r2 “İstanbul Planlama Ajansı (**İPA – Florya Atatürk Kent Ormanı**) GES” 20 kWp; 2024 r7 ve 2025 r14 “Florya Atatürk Kent Ormanı GES” 7 kWp; 2025 r11 “İstanbul Planlama Ajansı Çatısı” 20 | İPA 2022 ↔ İPA 2025 **aynı birim**. Florya 7 kWp **ayrı birim**, ama aynı mekânda olduğu için **aynı değerlendirme grubuna** alınır (metin sızıntısını önlemek için, muhafazakâr seçim) |
| Q14 | Aynı ad, farklı ilçe | 2025 r3 Arnavutköy ve r62 Silivri: ikisi de “Mezarlığı Hizmet ve Gasilhane Bina Çatısı GES”. 2024'te Ataşehir r2 ve Küçükçekmece r26: ikisi de “İBB İtfaiye Bina Çatısı GES” | **İlçe, eşleştirme anahtarının zorunlu parçası**. Yalnız ada göre eşleştirme yanlış birleştirir |
| Q15 | Ad stili yıla göre değişiyor | 2022 “… GES”; 2023 “… Güneş Enerji Santrali (GES)”; 2024–2025 “… Güneş Enerji Santrali”; 2024 itfaiyelerde “İBB” öneki var, 2025'te yok; 2022 adlarında ilçe öneki var (“Silivri Mezarlığı”), 2025'te yok | Kalıp ekleri **yıl bilgisini sızdırır**, bkz. §5.3 |
| Q16 | Alt birimler | Maltepe Orhangazi Şehir Parkı: Atletizm, Su Deposu (1)(2)(3), Şeflik, Tenis (2024 r29–34, 2025 r50–55) | Her biri ayrı `unit_id`; gerekçesiz birleştirilmez. Site düzeyi gruplama **yalnız duyarlılık analizi** |
| Q17 | Aynı tesiste yıllar arası güç oynaması | Kumburgaz 100→106→88; Turgut Reis 25→33; Başakşehir İtf. 25→33; İşitme Engelliler 89→98; Hoşdere 120→135; Darulacize 50→48; Soğanlık 29→24; Çamlıca 7→5 … | Eşleştirmede kullanılmaz. Raporda “indirgenemez gürültü tabanı” olarak analiz edilir, bkz. §6.4 |
| Q18 | Birim | Tüm sayısal satırlarda `kWp`. 2025 r2'de `Güç = (kWp)` metni var | Birim kWp değilse `quality_flag = unit_issue` (beklenen: 0 kayıt). Testle kontrol et |

---

## 4. Görev 1.1 — Denetlenebilir birleşik tablo

### 4.1 Şema (`outputs/data/records_clean.csv`)

| Alan | Tip | Tanım / kural |
|---|---|---|
| `record_id` | str | `GES-{file_year}-{excel_row:03d}`, ör. `GES-2024-013`. Deterministik, sıraya bağlı değil |
| `source_file` | str | Orijinal dosya adı (tam) |
| `sheet` | str | Sayfa adı (tam) |
| `excel_row` | int | 1 tabanlı Excel satır numarası (openpyxl `cell.row`) |
| `file_year` | int | Dosya adından; **anlamı “kaynak sürümü”** |
| `source_no` | int/None | 2022–2023'teki `No` sütunu (diğer yıllarda boş) |
| `district_raw` | str | Hücre değeri, dokunulmamış |
| `district_clean` | str | strip + NFC + yazım düzeltme haritası (`Atnavutköy→Arnavutköy`) |
| `name_raw` | str | Mekan hücresi, dokunulmamış |
| `name_clean` | str | §5.3'teki model metni (kalıp ekleri temizlenmiş, Türkçe karakterler ve sayılar korunmuş) |
| `match_key` | str | Yalnız eşleştirme için agresif normalize anahtar (§5.1); **modele girmez** |
| `unit_raw` | str | Ölçü/Birim |
| `power_kwp` | float | Güç; sayıya çevrilemezse satır kayıt sayılmaz ve günlüğe yazılır |
| `status_raw` | str | Ham durum |
| `status_clean` | enum | `aktif` / `pasif` / `karma` (`Aktif - Pasif`) / `belirsiz` (`(Aktif / Pasif)`) |
| `unit_id` | str | Aynı fiziksel alt birim (etap, depo no) — §5 |
| `entity_group_id` | str | **Değerlendirme grubu**: olası aynı tesis kümesinin bağlı bileşeni — §5 |
| `quality_flag` | str | `;` ile ayrılmış liste: `ok`, `district_typo_fixed`, `whitespace_fixed`, `status_ambiguous`, `status_mixed`, `conflict_same_year`, `district_conflict`, `phase_merge_candidate`, `name_overlap_other_unit` |
| `in_regression` | bool | `False` ise karantinada (Ömerli 2024 iki satır) |

### 4.2 Günlükler

- `ingest_log.csv`: `source_file, sheet, excel_row, action(header|description|empty|record), reason, raw_values_json`. Atlanan **her** satır burada.
- `cleaning_log.csv`: `record_id, field, raw, clean, rule_id`. Ör. `GES-2022-005, district, Atnavutköy, Arnavutköy, R-DIST-TYPO-01`.
- `numeric_errors.csv`: Güç dönüşüm hataları (beklenen: yalnız başlık/birim satırları).
- `district_map.csv`: `raw_value, normalized_value, rule_id, confidence, manual_review`. Yazım düzeltmeleri koda gömülmez, bu tablodan okunur. Kesin olmayan bir düzeltme `district_spelling_uncertain` bayrağı alır. `Atnavutköy→Arnavutköy` kesindir, çünkü aynı satırın Mekan alanında “Arnavutköy Mezarlığı” yazıyor.

### 4.3 Doğrulama testleri (`tests/test_ingest.py`)
- Kayıt sayısı (20, 11, 46, 76) ve toplamlar (3834,8 / 984,7 / 3887 / 8418) ±0,01.
- `record_id` benzersiz; `(source_file, excel_row)` benzersiz.
- 2025 `status_clean == pasif` sayısı **15**; `karma` 2; 2024 `belirsiz` 1.
- Hiçbir `name_clean` boş değil; tüm `unit_raw == 'kWp'`.
- Ham dosyaların sha256'sı protokoldeki değerle aynı.

---

## 5. Görev 1.2 — Tesis eşleştirme ve çelişki yönetimi

### 5.1 İki seviyeli kimlik (kritik tasarım kararı)

Hoca iki şeyi aynı anda istiyor: (a) “Etapları/alt birimleri gerekçesiz birleştirmeyin”, (b) “Aynı olası tesise ait tüm kayıtlar aynı değerlendirme grubunda kalmalı”. Bu ikisi **iki ayrı kimlikle** çözülür:

- **`unit_id`** (dar): “bu iki kayıt *kesinlikle aynı fiziksel GES birimi*”. Etaplar, Su Deposu (1)/(2)/(3) ayrı birimlerdir. RAG yanıtlarında ve “aynı tesis” sorularında kullanılır.
- **`entity_group_id`** (geniş, muhafazakâr): “bu kayıtlar arasında *aynı tesis olma ihtimali* var”. `unit_id`'ler arasındaki olası-eşleşme kenarlarının **bağlı bileşeni** (union-find). GroupKFold bunu kullanır. Örneğin Engelliler 1. Etap, 2. Etap ve 2025'teki 179 kWp'lik kayıt tek grupta olur, çünkü 2025 kaydı ikisini kapsıyor.

Bu ayrımı raporda açıkça savun. Birleştirme kararı değil, **sızıntı önleme** kararıdır.

### 5.2 Eşleştirme algoritması (`03_match.py`)

1. **match_key** üret: Türkçe-duyarlı küçük harf (`İ→i`, `I→ı`), NFC, noktalama → boşluk. Kalıp/stil sözcüklerini sil: `güneş enerji santrali`, `(ges)`, `ges`, `ibb`, `çatısı`, `bina`, `binası`. Baştaki ilçe adını sil (2022 stili). Sayıları ve `etap` sözcüğünü **koru**.
2. **Bloklama:** Aday çiftler yalnız aynı `district_clean` içinde. Ek olarak bir **ilçeler arası tarama** yap (yüksek ad benzerliği + farklı ilçe → Kumburgaz gibi vakaları yakalar ve manuel incelemeye düşürür).
3. **Benzerlik:** `rapidfuzz.fuzz.token_set_ratio` + token Jaccard. Eşikler: ≥90 → “yüksek”, 70–90 → “orta / manuel bak”, <70 → eşleşme yok (ilçeler arası taramada yalnız ≥90).
4. **Etap/alt birim kuralı:** İki kayıtta da **farklı** bir birim numarası varsa (`1. Etap` vs `2. Etap`, `(1)` vs `(2)`) → farklı `unit_id`. Bir tarafta numara yok, diğerinde var → `phase_merge_candidate`, aynı `entity_group_id`, farklı `unit_id`.
5. **Manuel karar dosyası** (`protocol/matching_decisions.csv`): `record_id_a, record_id_b, decision(same_unit|same_group|different), confidence(yüksek|orta|düşük), rationale, reviewer, date`. Otomatik öneriler bu dosyayla **ezilir**; dosya sürüm numaralı.
6. **Güç ve durum hiçbir adımda okunmaz.** `03_match.py` bu sütunları DataFrame'den düşürerek çalışsın; testle doğrula (`assert 'power_kwp' not in df_match.columns`).
7. Çıktılar: `entities.csv` (`unit_id`, `entity_group_id`, kayıt listesi, yıllar, güven, `match_status ∈ {confirmed, probable, ambiguous, separate}`) ve `matching_log.csv` (her çift: skorlar, karar, gerekçe).
8. **Kilit sırası:** Eşleştirme kararları **bölme manifesti üretilmeden önce** sürümlenip kilitlenir. Kat atamalarına veya model hatalarına bakıp eşleştirmeyi sonradan değiştirmek sızıntıdır. Değişiklik gerekirse yeni sürüm açılır, gerekçe yazılır ve tüm bölmeler yeniden üretilir.

### 5.3 Model metni (`name_clean`) ve yıl sızıntısı

Ad kalıpları yıla göre değiştiği için (`GES` ↔ `Güneş Enerji Santrali (GES)` ↔ `Güneş Enerji Santrali`), ham metin **dosya yılını kodluyor**. Bu yüzden:
- `name_clean`'de yalnız yıla özgü **kalıp ekleri** silinir: `güneş enerji santrali`, `(ges)`, `ges`. İçerik sözcükleri (`çatısı`, `itfaiye`, `park`, `yüzme havuzu` …), Türkçe karakterler ve etap/alt birim numaraları korunur.
- Bunu yapmazsan **ablasyon (3) “dosya yılını çıkar” anlamsız olur**: model yılı sonek tokenlarından yine öğrenir. Raporda bunu açıkça yaz.
- Denetim tablosu: temizlik sonrası her token için yıl dağılımı. Yalnız tek yılda görülen ve sık geçen tokenları listele (ör. `ibb` → 2024, `çatısı` → 2023–2025). Kalan yıl sinyalini kabul et ama raporla.

### 5.4 Ön eşleştirme sonucu (veriye bakılarak elle çıkarıldı; kodla yeniden üretilip doğrulanacak)

**Yıllar arası eşleşen gruplar (özet):**

| Grup | Kayıtlar (yıl: güç) | Güven | Not |
|---|---|---|---|
| İPA (+ Florya, geniş) | 2022: 20 · 2025: 20 // Florya 2024: 7 · 2025: 7 | İPA yüksek; Florya ayrı birim, aynı grup | Q13 |
| Ataköy sosyal hizmet | 2022: 106 · 2023 Etap2: 100 · 2025: 210 (Pasif) | orta | Q11 |
| Esenler Kemer Spor | 2022: 105,6 · 2025: 106 | yüksek | ad değişmiş |
| Arnavutköy Mezarlığı | 2022: 28 · 2025: 28 | yüksek | ilçe typo |
| Silivri Mezarlığı | 2022: 95 · 2025: 95 | yüksek | Q14 |
| Çatalca Kaleiçi İtf. | 2022: 64 · 2025: 63 | yüksek | |
| Ümraniye Çakmak Havuz | 2022: 112 · 2025: 112 | yüksek | |
| Sultangazi 75. Yıl | 2022: 106 · 2025: 106 | yüksek | |
| Bahçelievler Havuz | 2022: 63 · 2025: 63 | yüksek | |
| Hoşdere Park | 2022: 120 · 2025: 135 | yüksek-orta | “(Millet Bahçesi)” eklenmiş |
| Başakşehir Engelliler | 2022: 99 (1.Etap) + 80 (2.Etap) · 2023: 80 “(2)” · 2025: 179 | orta | Q10 |
| Şile Engelliler Yaz Kampı | 2022: 215 · 2025: 215 | yüksek | |
| Cemal Kamacı | 2022 (2.Etap): 387 · 2025: 428 | orta | Q12 |
| Fatih Spor Kompleksi | 2022: 550 · 2025: 550 | yüksek | |
| Silivri Kapı Buz Pisti | 2022: 574,2 · 2025: 574 | yüksek | |
| Tepeören Barınak | 2022: 452 · 2025: 451 | yüksek | |
| Güngören Davet Salonu | 2022: 305 · 2023: 305 · 2025: 305 | yüksek | 3 yıl |
| Hoca Ahmet Yesevi KM | 2022: 303 · 2023: 303 · 2025: 304 | yüksek | 3 yıl |
| Kayışdağı Darulacize | 2022: 50 · 2025: 48 | orta | “Müdürlüğü” ↔ “Ana Kampüsü” |
| Avcılar Sahil Tuvaleti | 2023: 8,5 · 2025: 9 | yüksek | |
| Başakşehir İtfaiye | 2023: 25 · 2025: 33 (karma) | yüksek | |
| Selimpaşa İtf. | 2023: 25 · 2025: 28 | yüksek | |
| Turgut Reis İtf. | 2023: 25 · 2025: 33 | yüksek | |
| Kumburgaz İtf. | 2023 (Silivri): 100 · 2024: 106 · 2025: 88 | orta | Q9 ilçe çelişkisi |
| Maçka / Topkapı tuvalet | 2023: 6,6 · 2025: 7 (her biri) | yüksek | |
| 2024 ↔ 2025 çiftleri | 2024'teki 44 grubun 43'ü 2025'te var (yalnız *Büyükçekmece İtfaiye* 2024 r11 yok) | çoğu yüksek | Ömerli: 2024 iki kayıt + 2025 bir kayıt tek grup |
| Yalnız 2025 | Bayrampaşa ×2, Kadıköy Filiz Akın, Kartal Yakacık Çocuk Eğt., Kartal Yunus Çocuk Eğt., Küçükçekmece Traker, Pendik Yenişehir İtf. Eğt. | — | 7 yeni grup |

**Beklenen sayılar:** ~**76 değerlendirme grubu** (2022'den 18 + 2023'ten 7 yeni + 2024'ten 44 yeni + 2025'ten 7 yeni; Florya ayrı sayılırsa 76, İPA ile birleşirse 75). Kayıt/grup ≈ **2,0**. Ömerli karantinasından sonra regresyonda **151 kayıt**. Bu, **tek başına en önemli bulgu**: kayıtların çoğu başka bir yılda neredeyse aynı adla ve güçle tekrar ediyor. Satır bölmesi bu yüzden ağır sızıntı üretir.

### 5.5 Karantina kuralı
- `conflict_same_year`: Aynı `unit_id` + aynı `file_year` + farklı güç → **her iki kayıt** `in_regression=False`. Grubun diğer yıllardaki kayıtları (Ömerli 2025: 28) regresyonda kalabilir. Karantinadaki kayıtlar fold atamasında yine **aynı grup** olarak işaretlenir ki ileride geri alınırsa bile sızıntı olmasın.
- Hiçbir kayıt silinmez; KB'de iki değer de çelişki etiketiyle durur.

---

## 6. Keşifsel veri analizi (EDA) — üretilecek tablo ve grafikler

Hepsi **eşleştirme ve bölmeden sonra, yalnız betimleyici** olarak. Hiçbir model kararı test katlarına bakılarak verilmez.

1. Yıl × kayıt sayısı, ilçe sayısı, güç toplamı, medyan (§3.2 tablosu) + “bu toplamlar yıllık büyüme değildir” notu.
2. Güç histogramı: ham ve `log1p` ölçekte. Çarpıklığı göster.
3. Grup büyüklüğü dağılımı: kaç grup 1, 2, 3, 4 kayıtlı? Kaç grup kaç farklı yılda?
4. **Yıllar arası tekrar matrisi**: 2022→2025, 2023→2025, 2024→2025 ortak grup sayıları (Venn/upset).
5. İlçe frekansı: kaç ilçe yalnız 1 kez görülüyor? Bu, test katında `[UNK]` oranını belirler.
6. **Tesis türü sinyali**: anahtar sözcüğe göre güç kutu grafiği. Gözlem: `itfaiye` ≈ 20–33 kWp, `park`/`koru`/`orman` ≈ 3–9, `tuvalet` ≈ 6,6–9, `spor kompleksi`/`yüzme havuzu`/`kültür merkezi` ≈ 60–574. Modelin asıl öğreneceği şey **tür → büyüklük** ilişkisi; attention'ın faydası varsa burada çıkar.
7. Token uzunluğu dağılımı (`name_clean` + ilçe): sözcük sayısı min 2, medyan 4, maks 9 civarı. Özel tokenlarla birlikte dizi uzunluğu ~8–16. **64 üst sınırına hiç ulaşılmaz**; kesilen kayıt sayısı = 0 (raporla).
8. **Gürültü tabanı:** Aynı `unit_id` içinde yıllar arası mutlak güç farkının ortalaması. Bu, en iyi senaryoda bile ulaşılabilecek hatanın kabaca alt sınırı (ör. Kumburgaz 100/106/88).
9. Durum dağılımı (yalnız betimsel; **modele girmez**).

---

## 7. Görev 1.3 — Dondurulmuş protokol (model eğitiminden ÖNCE teslim)

`protocol_v1.0.md` içeriği:
1. Veri sözlüğü (§4.1 tablosu).
2. Eşleştirme kuralları + `matching_decisions.csv` + güven dağılımı.
3. Dışlama gerekçeleri (başlık satırları, karantina) ve **sayılar**: 153 kayıt → 151 regresyon kaydı, G grup, ilçe sayısı, yıl başına kayıt.
4. Bölme manifesti: `split_manifest.csv` (outer fold 0–4, her dış eğitim için inner fold 0–2), sabit tohum (ör. 2026). Her katta grup/kayıt sayıları ve **ortak grup sayısı = 0** kontrolü.
5. Model listesi, girdi sözleşmesi, hiperparametre ızgarası, epoch politikası, tohumlar.
6. Metrikler ve birincil metrik tanımı.
7. Hipotezler ve yanlışlama ölçütleri (aşağıda).
8. Sürüm notu: Bu dosyanın sha256'sı; sonradan değişiklik → v1.1 + gerekçe.

### 7.1 Hipotezler (önceden kaydedilecek, yanlışlanabilir)

- **H1 — Attention etkisi.** *“D2 (attention pooling), D1'e (maskeli ortalama) göre tesis-dengeli MAE'yi düşürür.”*
  - Mekanizma: Tür belirleyici token (`itfaiye`, `park`, `kompleksi`) ağırlıklandırılırsa sinyal ortalamada sulanmaz.
  - **Yanlışlayan gözlem:** `Δ = MAE(D2) − MAE(D1)` için %95 eşleştirilmiş bootstrap aralığı 0'ı içeriyorsa ya da tamamen > 0 ise.
  - Ek: D3 için aynı test (D3 vs D1, D3 vs B1). **Önceden beklenti:** ~60 eğitim grubunda D3, B1'i geçemez. Bunu da önceden yaz; tutarsa “başarılı negatif sonuç” olur.
- **H2 — Gruplamanın hata kestirimine etkisi.** *“B1'in rastgele satır bölmesiyle ölçülen MAE'si, grup bölmesine göre belirgin biçimde (≥%30) düşüktür. Fark, test kayıtlarının eğitimde ortak grubu olma oranıyla açıklanır.”*
  - **Yanlışlayan gözlem:** Göreli fark <%10 ya da ortak grup oranı ile kayıt hatası arasında ilişki yok (ortak grubu olan ve olmayan test kayıtlarının MAE'leri benzer).
- **H3 — Kanıt ve hesap aracının LLM doğruluğuna etkisi.** *“Toplama/sayım sorularında sayısal doğruluk R2 > R1 > R0; yanıtlanamaz sorularda doğru çekimserlik R1, R2 ≥ R0.”*
  - **Yanlışlayan gözlem:** Toplama sorularında R2'nin sayısal doğruluğu R1'den yüksek değilse ya da R1'in atıf doğruluğu R0'dan iyi değilse.
  - Not: n=6 toplama sorusu çok az. Sonuç betimsel raporlanır, istatistiksel anlamlılık iddia edilmez.

---

## 8. Matematik bölümü — çözüm planı ve kontrol değerleri

Her türetmede **boyut + varsayım + limit durumu** yaz (slayt 4 disiplini).

### M1 — Ölçekli noktasal attention ve gradyan
- **Boyutlar:** `X ∈ ℝ^{n×d}`, `W_Q, W_K ∈ ℝ^{d×d_k}`, `W_V ∈ ℝ^{d×d_v}` → `Q, K ∈ ℝ^{n×d_k}`, `V ∈ ℝ^{n×d_v}`, `S = QKᵀ/√d_k ∈ ℝ^{n×n}`, `A = softmax_row(S+M) ∈ ℝ^{n×n}`, `O = AV ∈ ℝ^{n×d_v}`.
- **Varyans:** `qᵢ, kᵢ` bağımsız, `E=0`, `Var=1` → `E[qᵢkᵢ]=0`, `Var(qᵢkᵢ)=E[qᵢ²]E[kᵢ²]=1`. Toplam d_k bağımsız terim → `Var(q·k)=d_k`, std `√d_k`. Ölçekleme → `Var=1`.
- **Doygunluk bağlantısı:** Ölçeksiz durumda logit farkı ~√d_k kadar büyür. Softmax one-hot'a gider, `J = diag(a) − aaᵀ → 0` ve gradyan söner. Sayısal örnek: dengeli `(0.5, 0.5)` → J girdileri 0,25; doygun `(0.99, 0.01)` → 0,0099 (yaklaşık 25 kat küçük). Kırılgan varsayım: eğitim ilerledikçe bağımsızlık bozulur.
- **Softmax türevi:** `aᵢ = e^{sᵢ}/Σe^{s_k}` → bölüm kuralıyla `∂aᵢ/∂sⱼ = aᵢ(δᵢⱼ − aⱼ)`.
- **Zincir (matris biçimi):**
  `G_O = ∂L/∂O (n×d_v)`
  `∂L/∂A = G_O Vᵀ (n×n)`, `∂L/∂V = Aᵀ G_O`
  satır başına `∂L/∂Sᵢ = Aᵢ ⊙ (∂L/∂Aᵢ − ⟨∂L/∂Aᵢ, Aᵢ⟩)`, yani `G_S = A ⊙ (G_A − rowsum(G_A ⊙ A))`
  `∂L/∂Q = G_S K / √d_k (n×d_k)`, `∂L/∂K = G_Sᵀ Q / √d_k`
  `∂L/∂W_Q = Xᵀ ∂L/∂Q (d×d_k)`.
  Maskeli konumlarda `A=0` olduğu için `G_S=0`: gradyan oraya akmaz.

### M2 — Elle hesap (kontrol değerleri, NumPy ile hesaplandı)
`X = [[1,0],[0,1],[1,1]]`, `W_Q=W_K=W_V=I₂` → `Q=K=V=X`, `d_k=2`.

`S = XXᵀ/√2`:
```
[[0.7071 0.     0.7071]
 [0.     0.7071 0.7071]
 [0.7071 0.7071 1.4142]]
```
**Maskesiz** `A`:
```
[[0.4011 0.1978 0.4011]
 [0.1978 0.4011 0.4011]
 [0.2483 0.2483 0.5035]]
```
`O = AV`:
```
[[0.8022 0.5989]
 [0.5989 0.8022]
 [0.7517 0.7517]]
```
**Padding** (3. token = PAD, anahtar maskesi 3. sütunu −∞ yapar): `A` satırları `[0.6698 0.3302 0]`, `[0.3302 0.6698 0]`, `[0.5 0.5 0]`. `O` = `[0.6698 0.3302]`, `[0.3302 0.6698]`, `[0.5 0.5]`. Padding sorgusunun (3. satır) çıktısı havuzlamadan **çıkarılır** → geçerli satırların ortalaması `[0.5, 0.5]`.

**Nedensel** (üst üçgen −∞): `A` = `[1 0 0]`, `[0.3302 0.6698 0]`, `[0.2483 0.2483 0.5035]`. `O` = `[1 0]`, `[0.3302 0.6698]`, `[0.7517 0.7517]`. Son satır maskesizle aynı, çünkü son token zaten her şeyi görüyor; bunu yorumla.

**Kontroller:** satır toplamları = 1 (eval modunda, dropout kapalı), maskeli konumlar tam 0, NumPy ↔ kendi PyTorch attention'ın ↔ `F.scaled_dot_product_attention` `assert_close`.

### M3 — Parametre ve maliyet (d=32, h=4, d_ff=64, 2 blok)
| Bileşen | Hesap | Parametre |
|---|---|---|
| Q,K,V,O izdüşümü (biaslı) | 4·(32·32+32) | 4.224 |
| FFN | (32·64+64)+(64·32+32) | 4.192 |
| 2 LayerNorm | 2·(32+32) | 128 |
| **Blok** | | **8.544** |
| **2 blok** | | **17.088** |
| Pre-LN son LayerNorm | 64 | 64 |
| Token embedding | V·32 (V≈150–250 → ~5–8 bin) | fold'a bağlı, raporla |
| Konum (öğrenilen, L_max·32) / sinüzoidal | 64·32 = 2.048 / 0 | seçime bağlı |
| Regresyon başlığı (32→32→1) | (32·32+32)+(32+1) | 1.089 |

- Hocanın notu: “toplamı kullandığın başlığa göre ayrıca kontrol” → başlığın 32 gizli birimli olduğunu açıkça yaz.
- **Maliyet:** Attention skorları `QKᵀ` için `O(n²d)`, FFN `O(n·d·d_ff)`. n=64 → 4.096 eleman/başlık; n=256 → 65.536 (16 kat). Bizde n≈16 → 256 eleman. Yani burada **attention maliyeti sorun değil, veri azlığı sorun**.
- **Pre-LN:** `h = x + MHA(LN(x))`, `y = h + FFN(LN(h))` (+ sonda LN). **Post-LN:** `h = LN(x + MHA(x))`, `y = LN(h + FFN(h))`. Gradyan yolu tartışması: slayt 33–34.
- **Mimari karşılaştırma:**
  - Encoder-only: maske yalnız padding, cross-attn yok, hedef MLM/sınıflama/regresyon.
  - Decoder-only: nedensel maske, cross-attn yok, hedef sonraki token.
  - Encoder–decoder: encoder padding maskesi + decoder nedensel maske + cross-attn, hedef seq2seq.
  - **Regresyon için encoder-only** (tüm tokenlar birbirini görmeli, tek skaler çıktı). **Yanıt üretimi için decoder-only** önceden eğitilmiş instruction model.

### M4 — Sıra, kayıp, genelleme
- **Permütasyon eşdeğişimi:** P permütasyon matrisi. `Attn(PX) = softmax(PXW_Q W_Kᵀ XᵀPᵀ/√d) PXW_V = P·softmax(S)·PᵀP·XW_V = P·Attn(X)`. Softmax satır bazlı olduğu için `softmax(PSPᵀ) = P softmax(S) Pᵀ`. **Ortalama havuzlama** `1ᵀP = 1ᵀ` olduğundan değişmez. [CLS] okuması da değişmez: CLS'nin attention'ı diğer tokenların sırasına bakmaz.
- **Özellik kimliği ≠ konum:** Tesis adında sıra anlamlıdır (“Spor Kompleksi” ≠ “Kompleksi Spor”) → konumsal kodlama. Tabular alanlarda (ilçe, yıl) önemli olan **hangi alan** olduğudur → alan işaret tokenları `[ILCE]`, `[YIL]`, `[MEKAN]` (veya segment embedding). Konumsal kodlamanın taşıdığı tek “alan bilgisi” sabit sıradan gelir; ablasyon (2) bunu sınar.
- **Log hedef:** `z = ln(1+y)`. Büyük kapasitelerde mutlak hataların baskınlığını azaltır, göreli hatayı optimize eder.
- **Ters dönüşüm yanlılığı:** `ŷ = exp(ẑ)−1`, `ẑ` koşullu ortalama ise `exp(E[z]) ≤ E[exp z]` (Jensen) → **sistematik eksik kestirim** (koşullu medyanı kestirir). Lognormal varsayımında düzeltme `exp(σ²/2)`; Duan smearing. Raporda tartış, ana modelde uygulama (protokolde sabit dönüşüm: `max(0, exp(ẑ)−1)`).
- **Kayıp ≠ metrik:** Eğitim log-MSE'si göreli hatayı ölçer; kWp MAE büyük tesislere duyarlıdır. 500 kWp'lik bir tesiste %20 hata = 100 kWp; 5 kWp'de %20 = 1 kWp.
- **Bias–variance:** D3 ≈ 18k + embedding ≈ **25k parametre / ~60 bağımsız eğitim grubu** ≈ 400 parametre/grup → yüksek varyans; dropout, weight decay ve erken durdurma şart. B1'in etkin serbestlik derecesi çok daha düşük.

---

## 9. Girdi sözleşmesi ve tokenizasyon

### 9.1 Yasak girdiler (kodda assert)
`power_kwp`, `status_*`, `record_id`, `excel_row`, `source_no`, `unit_id`, `entity_group_id`, `quality_flag` ve güçten türetilmiş her şey. **Satırlar arası attention yok:** her örnek tek bir tesis kaydı, batch içindeki kayıtlar birbirini görmez.

### 9.2 Dizi formatı
```
[CLS] [ILCE] <ilçe tokenları> [YIL] <yil_2024> [MEKAN] <name_clean tokenları> [PAD]...
```
- İlçe tek token olarak (`büyükçekmece`). Yıl özel token `yil_2022..yil_2025`.
- Türkçe-duyarlı küçük harf (`str.lower()` `İ`'yi `i̇` yapar; özel eşle: `İ→i`, `I→ı`), NFC. Ayırma: `\w+` ve parantezli sayılar (`(1)` → `1`), `75.` → `75`. Etap numaraları ve Türkçe karakterler korunur.
- **Sözlük yalnız iç eğitim bölümünden** (her iç ve dış eğitim için yeniden kurulur): `[PAD]=0, [UNK]=1, [CLS], [ILCE], [YIL], [MEKAN]` + tokenlar.
- **`[UNK]` tuzağı:** Eğitimde hiç `[UNK]` görülmezse embedding'i rastgele kalır ve testteki her bilinmeyen token gürültü olur. Çözüm (protokolde önceden sabitle): **`min_freq=2`**. Eğitimde tek kez geçen tokenlar `[UNK]`'e eşlenir, böylece `[UNK]` embedding'i öğrenilir. Duyarlılık olarak `min_freq=1` tek tohumla raporlanabilir.
- `max_len` = iç eğitimdeki en uzun dizi (≈16), üst sınır 64. Kesilen kayıt sayısını yaz (beklenen 0).
- Testte görülmeyen ilçe/yıl → `[UNK]`.

### 9.3 Hedef
`z = ln(1+power_kwp)`. Tüm modellerde aynı. Tahmin: `ŷ = max(0, exp(ẑ)−1)`. Embedding boyutu 32.

---

## 10. Modeller — mimari ayrıntılar

Ortak kural: **D1–D3 aynı sözlük, aynı bölmeler, aynı hedef dönüşümü, aynı regresyon başlığı.** Parametre sayılarını tabloyla raporla.

### B0 — Medyan
Dış/iç eğitimdeki güç medyanı. Monoton dönüşüm medyanı korur; kWp'de doğrudan hesapla. 0 parametre.

**Ek referanslar (isteğe bağlı, hiperparametresiz, raporda “ek” diye ayrı satırda):**
- **B0-ilçe:** Eğitimdeki ilçe medyanı; bilinmeyen ilçe → genel eğitim medyanı.
- **B0-yıl:** Dosya yılı medyanı.
Fallback kuralı protokolde önceden yazılır, sonradan değişmez. Bu iki referans ilçe/yıl bilgisinin tek başına ne kadar sinyal taşıdığını gösterir. Böylece “metin modeli gerçekten addan mı öğreniyor?” sorusuna zemin hazırlanır.

### B1 — TF-IDF + Ridge
- `ColumnTransformer`: `TfidfVectorizer(analyzer='word', ngram_range=(1,2), sublinear_tf=True, token_pattern` Türkçe uyumlu, ön-işleme = §9.2 normalizasyonu)` ad üzerinde + `OneHotEncoder(handle_unknown='ignore')` ilçe ve yıl.
- Hedef `log1p`; `alpha ∈ {0.1, 1, 10, 100}` (≤4 değer). Seçim iç 3 katlı grup doğrulamasıyla.
- **Tüm dönüşümler `Pipeline` içinde.** Böylece TF-IDF sözlüğü ve IDF yalnız iç eğitimde öğrenilir.
- Ayrıca **rastgele satır bölmesi** ile çalıştırılır (yalnız sızıntı tanısı, §12.2).

### D1 — Maskeli ortalama + MLP
```
E = Embedding(V, 32, padding_idx=0)          # [B, T, 32]
m = (ids != PAD).float()                     # [B, T]
h̄ = Σ_t m_t E_t / Σ_t m_t                    # [B, 32]  (payda ≥1, [CLS] her zaman var)
ẑ = W2·ReLU(W1 h̄ + b1) + b2                  # 32→32→1  (dropout ReLU sonrası)
```
Parametre: V·32 + 1.089.

### D2 — Attention pooling (öğrenilen tek sorgulu havuzlama)
```
h_i = E_i                                     # [B, T, 32]
e_i = vᵀ tanh(W h_i + b)                      # W: 32×32, b: 32, v: 32 → e: [B, T]
e_i = finfo.min  (PAD konumları)              # maske softmax'tan ÖNCE
α   = softmax(e)  (T ekseninde)               # [B, T], Σα=1
r   = Σ_i α_i h_i                             # [B, 32]
ẑ   = aynı regresyon başlığı(r)
```
Parametre: V·32 + 1.088 (havuzlama) + 1.089.
**D2 ≠ D3:** D2'de **tek, öğrenilen, girdiden bağımsız** bir sorgu vektörü (v) tokenları puanlar; tokenlar birbirini görmez, temsiller bağlamsallaşmaz. D3'te her token her tokena sorgu atar (n×n), temsiller bağlamsallaşır.
**Ablasyon (1):** `α_i = m_i / Σm` (eşit ağırlık) → mimari D1'e yaklaşır ama havuzlama parametreleri yok sayılır. Yeniden eğit.

### D3 — Transformer encoder (attention SIFIRDAN)
```
x = E(ids) + P(pos)                            # [B, T, 32]; P sinüzoidal (0 param) — protokolde sabitle
for blok in 1..2:   (Pre-LN)
    x = x + Dropout(MHA(LN(x), key_padding_mask))
    x = x + Dropout(FFN(LN(x)))                # FFN: 32→64 (GELU/ReLU) →32
x = LN_final(x)
ẑ = regresyon_başlığı(x[:, 0])                 # [CLS]
```
**`attention.py`'de elle yazılacak MHA** (hazır `nn.MultiheadAttention`/`TransformerEncoder` yasak; `nn.Linear`, `nn.LayerNorm`, autograd serbest):
```
qkv  = Linear(32, 96)(x)                       # [B,T,96]
q,k,v = qkv.chunk(3, -1)                       # 3×[B,T,32]
q = q.view(B,T,4,8).transpose(1,2)             # [B,4,T,8]  (aynısı k, v)
s = q @ k.transpose(-2,-1) / sqrt(8)           # [B,4,T,T]  — √d_h, √d_model DEĞİL
s = s.masked_fill(pad_mask[:,None,None,:], finfo(s.dtype).min)   # anahtar maskesi [B,1,1,T]
a = softmax(s, -1)                             # son eksen!
a = Dropout(a)
o = a @ v                                      # [B,4,T,8]
o = o.transpose(1,2).reshape(B,T,32)           # transpose ÖNCE (slayt 9 sessiz hatası)
o = Linear(32,32)(o)
```
- Q/K/V'yi tek Linear(32,96) veya üç ayrı Linear ile yaz; parametre aynı (4.224'te dördü dahil).
- **Tamamen maskeli satır tanımı:** `[CLS]` hiç maskelenmediği için gerçek veride oluşmaz. Yine de tanımla: `finfo.min` ile softmax düzgün dağılım verir → çıktıyı `0`'a zorla (`o = o * any_valid[...,None]`). Testle göster.
- Tek tokenlı dizi (`[CLS]` + 0 token): attention kendine 1 ağırlık verir; tanımlı ve test edilir.
- Parametre: V·32 + 17.088 + 64 + 1.089 (+ öğrenilen konum seçilirse 2.048).
- **Ablasyon (2):** `P = 0` (konum yok). **Ablasyon (3):** `[YIL] yil_xxxx` tokenlarını diziden çıkar (sözlük ve maks uzunluk yeniden kurulur).

### 10.1 Uygulama doğrulama testleri (`tests/test_attention.py`) — en az 4, öneri 8
1. **Padding değişmezliği:** Diziye fazladan `[PAD]` ekle → D1/D2/D3 tahmini `assert_close` (eval modunda).
2. **Satır toplamı = 1** (eval modu, dropout kapalı; train modunda dropout yüzünden olmadığını da not et).
3. **Maskeli ağırlık = 0:** PAD sütunlarında `a == 0` (tam sıfır, `finfo.min` + softmax float32'de 0'a iner; kontrol et).
4. **M2 kontrol değerleri:** kendi attention fonksiyonun M2 matrislerini 1e-4 hassasiyetle üretir.
5. **Sonlu fark gradyanı:** float64'te `torch.autograd.gradcheck` + elle `(f(x+ε)−f(x−ε))/2ε`, `ε=1e-6`, ∂L/∂Q için.
6. **Referans karşılaştırma:** `F.scaled_dot_product_attention` ve `nn.MultiheadAttention` (ağırlıkları kopyalayarak) ile `assert_close`. Hazır modüller **yalnız testte**.
7. **Head birleştirme testi:** yanlış `reshape` (transpose'suz) farklı sonuç veriyor mu? (Slayt 9 sessiz hatası.)
8. **Permütasyon testi (M4):** konum kodlaması kapalıyken [CLS] hariç tokenları karıştır → D3 tahmini değişmez; açıkken değişir.
9. **Tamamen maskeli satır** → NaN yok, çıktı 0.
10. **Parametre sayısı testi:** `sum(p.numel())` bağımsız elle hesapla (§8 M3 tablosu + V·32) birebir eşleşir.
11. **Batch ve uzunluk:** batch>1 ve farklı uzunluktaki diziler aynı batch'te → her örnek tek başına çalıştırıldığındaki tahminle `assert_close`. Bu, batch'ler arası sızıntı olmadığını da gösterir.
12. **Determinizm:** Aynı tohumla iki koşu → aynı tahminler (tolerans 1e-6).

### 10.2 Negatif kontroller (hatayı bilerek üret, testin yakaladığını göster)
Slayt 5 “en az bir negatif kontrol” istiyor. Her biri için bir test + raporda bir satır:
| Bilerek yapılan hata | Yakalaması gereken kontrol |
|---|---|
| Attention'da transpose'suz `reshape` | head birleştirme testi (10.1-7) / referans karşılaştırma |
| PAD anahtarlarını maskelememe | padding değişmezliği testi (10.1-1) |
| `√d_model` ile ölçekleme | M2 / `F.scaled_dot_product_attention` karşılaştırması |
| `power_kwp`'yi girdi tokenı yapma | girdi sözleşmesi assert'i (§9.1); ayrıca MAE'nin şüpheli biçimde ~0'a düşmesi |
| Aynı grubu train/test'e dağıtma | `train_groups ∩ test_groups == ∅` assert'i |
| Sözlüğü/TF-IDF'i tüm veride fit etme | fit edilen nesnenin sözlüğünde yalnız eğitim tokenları var mı testi |
| RAG'de ilk-k satır üzerinden toplam | R2 kapsam kontrolü: filtre satır sayısı ≠ getirilen satır sayısı → hata |

---

## 11. Deney tasarımı — iç içe gruplu çapraz doğrulama

### 11.1 Bölmeler
- **Dış:** `GroupKFold(n_splits=5)` ve `groups=entity_group_id` (~76 grup → katta ~15 grup, ~30 kayıt). Tekrarlanabilirlik için grupları tohumlu karıştırıp atamayı `split_manifest.csv`'ye yaz. sklearn ≥1.6 `GroupKFold(shuffle=True, random_state)` destekler, yoksa elle karıştır.
- **İç:** Her dış eğitimde `GroupKFold(3)`.
- **Kontrol:** Her katta `set(train_groups) ∩ set(test_groups) == ∅` assert. Grup sayısı yetersiz kalırsa kat sayısını azalt ve gerekçe yaz; **asla satır bölmesine dönme**.
- Karantinadaki 2 Ömerli kaydı eğitimde de testte de kullanılmaz; grubunun diğer kaydı normal katta.
- İsteğe bağlı kontrol: Kat başına güç dağılımları (medyan) çok sapmasın. Stratifikasyon yapma, sadece raporla.

### 11.2 Arama bütçesi (yönerge sabitleri)
| | Değer |
|---|---|
| B1 | `alpha ∈ {0.1, 1, 10, 100}` |
| D1–D3 | `lr ∈ {1e-3, 3e-4}` × `dropout ∈ {0, 0.2}` = 4 aday |
| Optimizer | AdamW, `weight_decay=1e-4` |
| Batch | 16 |
| Epoch | ≤150, iç doğrulamada `patience=15` |
| Kayıp | log uzayında MSE |
| İç arama tohumu | tek sabit (ör. 0) |
| Dış final tohumları | 17, 42, 2026 |

### 11.3 Prosedür (sözde kod)
```
for outer k in 0..4:
    D_tr, D_te = split(outer k)
    for config c in grid:                     # 4 aday
        for inner j in 0..2:
            fit vocab/tfidf/encoders on D_tr_inner_j
            train ≤150 epoch, early stop (patience 15) on D_val_inner_j
            kaydet: best_epoch[c,j], val_metric[c,j]   # tesis-dengeli MAE (kWp)
    c* = argmin_c mean_j val_metric[c,j]
    E* = median_j best_epoch[c*, j]           # yönerge: iç epoch sayısını taşı
    for seed in (17, 42, 2026):
        fit vocab on D_tr (tam dış eğitim), train c* tam E* epoch (erken durdurma YOK)
        predict D_te → outer_predictions_{model}_{seed}.csv
# D_te'ye hiçbir aşamada bakılmaz
```
- Seçim metriği: iç doğrulama **tesis-dengeli MAE (kWp)** (birincil metrikle tutarlı). Log-kaybı ikincil olarak kaydet.
- Tablolar: kat × tohum sonuçları ayrı ayrı + birleştirilmiş dış tahminler.

---

## 12. Metrikler, belirsizlik, sızıntı tanısı

### 12.1 Metrikler
- **Birincil, tesis-dengeli MAE:** `MAE_fac = (1/G) Σ_g [ (1/n_g) Σ_{i∈g} |yᵢ − ŷᵢ| ]`, g = `entity_group_id`. Çok kaydı olan tesis ağırlığı şişirmez.
- İkincil: kayıt düzeyi MAE, RMSE, R², log-ölçek MAE (`|ln(1+y) − ln(1+ŷ)|`).
- **Birleştirme:** 5 dış katın tahminleri birleştirilip metrik bir kez hesaplanır. Kat başına ve tohum başına sonuçlar ayrıca tabloda.
- **Tohumlar bağımsız tesis değildir:** Bootstrap'ta her kayıt için 3 tohumun mutlak hatalarının **ortalaması** kullanılır (veya tohum ortalaması tahmin). Hangisi olduğu protokolde sabitlenir. Tohumlar arası std ayrı raporlanır.

### 12.2 Eşleştirilmiş grup bootstrap
```
for b in 1..1000:
    G_b = gruplardan iadeli G örnek
    Δ_b = MAE_fac(A; G_b) − MAE_fac(B; G_b)     # aynı dış tahminler
CI95 = percentile(Δ, [2.5, 97.5])
```
Karşılaştırmalar: D1–B1, D2–D1, D3–D1, D3–B1, B1–B0 ve ablasyonlar ana modele karşı. **Not yaz:** Bu aralık sabit bölmeler + eğitim prosedürü koşullu belirsizliği gösterir; yeniden bölme ve küçük örneklem (≈76 grup) belirsizliğini tam kapsamaz.

### 12.3 Sızıntı tanısı (H2)
- B1'i `KFold(5, shuffle=True)` **satır** bölmesiyle çalıştır.
- Raporla: test kayıtlarının yüzde kaçının grubu eğitimde de var (“ortak tesis oranı”, beklenen %50+). Satır bölmesi MAE vs grup bölmesi MAE. Ortak grubu olan ve olmayan test kayıtlarının ayrı MAE'leri.
- Tablo başlığına açıkça yaz: **“SIZINTI TANISI — geçerli genelleme skoru değildir.”**
- Ek tanı (isteğe bağlı): “Aynı tesisin diğer yıldaki gücünü kopyala” oracle'ı. Satır bölmesinin neden bu kadar iyi göründüğünü sezgisel olarak gösterir.

### 12.4 Üç zorunlu ablasyon
| # | Model | Değişiklik | Hiperparametre | Bölme | Tohum |
|---|---|---|---|---|---|
| A1 | D2 | α eşit (maskeli ortalama) | D2'nin her dış katta seçilmiş c*, E* | aynı dış bölmeler | tek sabit (ör. 42) |
| A2 | D3 | konumsal kodlama yok | D3'ün c*, E* | aynı | tek |
| A3 | D3 | `[YIL]` tokenları yok | D3'ün c*, E* | aynı | tek |
Karşılaştırmayı **ana modelin aynı tohumlu (42) koşusuyla** yap. Üç tohumlu ana sonuçlarla karıştırma; tabloyu ayrı tut.

### 12.5 Hata analizi
- En büyük 5+ mutlak hatalı kaydı listele: ad, ilçe, yıl, gerçek, tahmin, grup büyüklüğü, eğitimde benzer tür var mıydı?
- Hata taksonomisi önerisi: (a) nadir tür/benzersiz tesis (Tepeören, Buz Pisti), (b) etap/birleşme kaydı (Engelliler 179, Ataköy 210), (c) aynı adın farklı büyüklükleri, (d) `[UNK]` yoğun girdi, (e) yıl kayması.
- **Alt grup hata tablosu** (birleşik dış tahminlerde, tüm modeller için):
  - dosya yılına göre;
  - **eğitimde görülmüş vs görülmemiş ilçe** (`[UNK]` etkisi);
  - güç aralığına göre (≤10, 10–50, 50–150, >150 kWp; bölmeden bağımsız sabit eşikler);
  - grup büyüklüğüne göre (1 kayıtlı vs çok kayıtlı tesis);
  - tesis başına hata dağılımı (kutu grafiği).
  Alt gruplar küçük olduğundan sayıları (n) her hücrede yazılır ve yalnız betimsel yorumlanır.
- **Attention vs token silme:** Bu 5 kayıt için D2 `α` ve D3 [CLS] attention'ı (katman×başlık) çiz. Her tokenı tek tek sil (veya `[UNK]` ile değiştir), `Δŷ`'yi ölç. İki sıralamanın Spearman korelasyonunu raporla. **Düşük korelasyon beklenir ve dürüstçe yazılır**; yüksek attention = nedensel önem iddiası kurulmaz.

### 12.6 İsteğe bağlı zamansal deney
Test = 2025 kayıtları. Bunları “geçmişte görülen grup” ve “görülmeyen grup” olarak ayır. Görülmeyen test gruplarının geçmiş sürümleri eğitimden çıkarılır. İki alt kümede ayrı MAE. “Gelecek üretim tahmini” ifadesi **kullanılmaz**.

---

## 13. LLM ve RAG

### 13.1 Bilgi tabanı (KB) — regresyondan ayrı
- 153 kaydın her biri bir **satır belgesi** (karantinadakiler dahil):
  ```
  [record_id: GES-2024-013] [dosya: 2024-yl-...xlsx] [sayfa: 2024 KURULUMU TAMAMLANAN GES] [Excel satırı: 13]
  Dosya yılı (kaynak sürümü): 2024 | İlçe: Çekmeköy | Mekan: İBB Ömerli İtfaiye Bina Çatısı Güneş Enerji Santrali
  Güç: 29 kWp | Durum (ham): Aktif | Durum (temiz): aktif
  Uyarılar: conflict_same_year (aynı yıl GES-2024-015: 25 kWp) | unit_id: … | entity_group_id: …
  ```
- Metadata ayrı sütunlarda (filtre için). KB güç değerlerini içerir ve **regresyon kodundan ayrı klasörde** durur. RAG başarısı regresyon başarısı olarak sunulmaz.

### 13.2 LLM seçimi (4 GB VRAM'e göre)
- Birincil öneri: **Qwen2.5-3B-Instruct GGUF Q4_K_M** (~2 GB, tamamen GPU'da, Türkçe makul). llama.cpp veya ollama ile.
- Alternatif (daha iyi Türkçe, daha yavaş): Qwen2.5-7B-Instruct Q4_K_M, kısmi GPU offload (RAM 15 GB, şu an ~9 GB kullanımda, dikkat).
- Kaydet: model adı + sürüm/hash, nicemleme, bağlam uzunluğu, `temperature=0`, `top_p=1`, `max_tokens=384`, seed, donanım, tam prompt metinleri.
- **Sıfırdan LLM eğitme yok.** Şablon yanıta “LLM” deme. Erişim sorunu olursa ilk hafta hocaya bildir.

### 13.3 Üç sistem (aynı model, aynı soru, aynı üretim bütçesi)
- **R0 — Kanıtsız:** Yalnız soru + “bilmiyorsan söyle” sistem promptu.
- **R1 — RAG:** BM25 (`rank_bm25`, §9.2 normalizasyonu + ilçe/yıl alanları indekslenir), ilk k belge. Kaynak gösterme promptu: “Her olgusal iddiaya [record_id] ekle; kanıt yoksa ‘veride yok’ de.” `k ∈ {3, 5, 10}` **yalnız 10 geliştirme sorusuyla** seçilir.
- **R2 — RAG + hesap aracı:**
  1. LLM, soruyu **JSON sorgu planına** çevirir (şemalı, kısıtlı): `{"intent": "lookup|count|sum|compare|list|unanswerable", "filters": {"file_year": …, "district_clean": …, "status_clean": …, "name_contains": …}, "agg": "count|sum|mean|min|max|none"}`.
  2. Python planı **beyaz listeye** göre doğrular (izinli alan/işlem dışı → ret) ve pandas ile **tüm tablo** üzerinde çalıştırır. **İlk-k üzerinden toplam yasak.**
  3. Sonuç + eşleşen tüm `record_id` listesi + kapsam kontrolü (filtre sonucu satır sayısı, `name_contains` belirsizse birden fazla gruba düşüyor mu?) LLM'e verilir. LLM yanıtı yazar.
  4. LLM'in ürettiği kod **asla `exec` edilmez**.
- Her sorgu için `rag_runs/{system}/{qid}.json`: plan, getirilen record_id'ler, hesap sonucu, ham yanıt, süre (s), token sayısı.

### 13.4 Çekimserlik ve çelişki davranışı (prompta ve araca gömülü)
- Veride olmayan kavramlar (kWh üretim, maliyet, kurulum tarihi, panel markası, gelecek yıl) → “Bu dosyalar bu bilgiyi içermiyor” + gerekçe.
- Aynı tesis–yıl çelişkisi → tüm değerler ve tüm kaynaklar, tek değer uydurmadan.
- `karma`/`belirsiz` durum → “açıkça aktif/pasif denemez”.
- Dosya yılı soruları → “dosya yılı kurulum yılı anlamına gelmeyebilir” uyarısı.
- **LLM'nin izinli işleri:** getirilen kanıtı özetlemek, araçla hesaplanmış sonucu açıklamak, kaynak göstermek, reddetmek.
- **İzinsiz işler:** keyfî kod çalıştırmak; getirilmemiş satırlar hakkında sayı söylemek; **kanıt olmadan iki kaydın aynı tesis olduğunu iddia etmek** (eşleştirme yalnız `entities.csv`'den gelir); veride olmayan kWh/maliyet tahmini.

### 13.5 30 soruluk altın set (taslak; test 20'si hocaya teslimle kilitlenir)
Format (`jsonl`): `qid, split(dev|test), category(single|aggregate|conflict|unanswerable), question, gold_answer, gold_numeric, tolerance, evidence_record_ids, answerable(bool), paraphrase_group`. **Parafrazlar aynı bölmede kalır.**

Örnek adaylar (altın değerler veriden doğrulandı, kodla tekrar kontrol et):

| Kategori | Soru | Altın yanıt | Kanıt |
|---|---|---|---|
| single | 2024 dosyasında Tuzla Bahar Merkezi GES gücü? | 458 kWp | GES-2024-046 |
| single | 2022 dosyasında Silivri Kapı Buz Pisti gücü? | 574,2 kWp | GES-2022-017 |
| single | 2025'te Kadıköy Filiz Akın Kız Öğrenci Yurdu gücü? | 21 kWp | GES-2025-039 |
| single | 2023 Avcılar Sahil Tuvaleti GES gücü? | 8,5 kWp | GES-2023-004 |
| single | 2024 Maltepe Orhangazi Su Deposu Üzeri (2) gücü? | 78 kWp | GES-2024-031 |
| single | 2025 Pendik Yenişehir İtfaiye Eğitim Binası durumu? | Aktif | GES-2025-057 |
| aggregate | 2025 dosyasında açıkça Pasif yazan kaç kayıt var? | **15** (2 karma dahil değil) | 15 record_id |
| aggregate | 2024 dosyasında Maltepe kayıtlarının toplam gücü? | 326 kWp (6 kayıt) | GES-2024-029…034 |
| aggregate | 2025 dosyasında Kartal'da kaç kayıt var, toplam güç? | 5 kayıt, 246 kWp | GES-2025-041…045 |
| aggregate | 2022 dosyasının güç toplamı? | 3.834,8 kWp | 20 kayıt |
| aggregate | 2022 ve 2023 dosya toplamları farkı? | 2.850,1 kWp | 31 kayıt |
| aggregate | 2025 Maltepe toplam gücü 2024'e göre nasıl? | 324 vs 326 (−2) | 12 kayıt |
| conflict | 2024 Ömerli İtfaiye için tek bir güç değeri verilebilir mi? | Hayır: 29 (r13) ve 25 (r15) | GES-2024-013, -015 |
| conflict | Kumburgaz İtfaiye hangi ilçede? | Kaynaklar çelişiyor: 2023 Silivri, 2024–2025 Büyükçekmece | 3 kayıt |
| conflict | Başakşehir Engelliler Merkezi GES'in gücü nedir? | Yıl/etap belirsiz: 2022 99+80 (iki etap), 2023 80, 2025 179 | 4 kayıt |
| conflict | 2025'te “Mezarlığı Hizmet ve Gasilhane” GES gücü? | İki ilçede aynı ad: Arnavutköy 28, Silivri 95 | GES-2025-003, -062 |
| conflict (dev) | 2025 Florya Atatürk Kent Ormanı aktif mi? | “Aktif - Pasif” → kesinleştirilemez | GES-2025-014 |
| unanswerable | 2025 yıllık elektrik üretimi kaç kWh? | Yanıtlanamaz | — |
| unanswerable | Tuzla Bahar Merkezi GES hangi tarihte devreye alındı? | Yanıtlanamaz (dosya yılı ≠ kurulum tarihi) | — |
| unanswerable | Fatih Spor Kompleksi GES'in yatırım maliyeti? | Yanıtlanamaz | — |
| unanswerable | 2026'da kaç GES kurulacak? | Yanıtlanamaz | — |

Bölme: 10 dev (her kategoriden karışık), 20 test = 6 single + 6 aggregate + 4 conflict + 4 unanswerable. **Test seti, prompt ve k seçiminden önce hocaya teslim edilir**, sha256'sı protokole yazılır.

### 13.6 RAG değerlendirmesi
| Metrik | Tanım | Kapsam |
|---|---|---|
| Recall@k | altın kanıt record_id'lerinden en az biri/hepsi ilk k'da mı | single sorular (R1, R2) |
| Doğru yanıt oranı | elle puanlama, altın yanıta göre | tümü |
| Sayısal doğruluk | `|ŷ−y| ≤ tolerans` | sayısal sorular |
| Atıf doğruluğu | destekleyen atıf / tüm atıf | atıf içeren yanıtlar |
| Kanıt kapsamı | atıfla desteklenen olgusal iddia / tüm olgusal iddia | tümü |
| Doğru çekimserlik | yanıtlanamazda ret / yanıtlanamaz | 4 soru |
| Gereksiz ret | yanıtlanabilirde ret / yanıtlanabilir | 16 soru |
- Boş payda → “uygulanamaz” (0 değil). R0 için Recall@k uygulanamaz.
- **En az 5 hatayı elle sınıflandır:** getirme hatası, filtre/plan hatası, hesap doğru ama yanlış aktarma, halüsinasyon atıfı, gereksiz ret, çelişkiyi tek değere indirgeme.
- LLM-hakem kullanılırsa yalnız yardımcı; altın standart elle hazırlanmış yanıtlardır.

---

## 14. Rapor yapısı

1. Özet (bulgular + sınırlılıklar, 1 paragraf)
2. Veri denetimi: dosya yapıları, kalite bulguları, eşleştirme, karantina, EDA
3. Dondurulmuş protokol özeti + hipotezler (tarihli)
4. Matematik (M1–M4) türetmeler + M2 doğrulama çıktısı
5. Yöntem: girdi sözleşmesi, modeller, parametre tablosu, eğitim, bölmeler
6. Uygulama doğrulaması: test listesi + geçti/kaldı
7. Sonuçlar:
   - Tablo 1: Ana modeller, birleşik dış tahmin metrikleri (ortalama ± tohum std)
   - Tablo 2: Kat × tohum ayrıntısı
   - Tablo 3: Eşleştirilmiş bootstrap Δ ve %95 CI
   - Tablo 4: Ablasyonlar (tek tohum, ayrı tablo)
   - Tablo 5: Sızıntı tanısı (açık uyarı başlığıyla)
   - Şekil: tahmin–gerçek (log-log), hata vs grup büyüklüğü, attention vs token silme
8. Hata analizi
9. RAG: sistem kartı (model, nicemleme, prompt, ayarlar, donanım), metrik tablosu R0/R1/R2, hata sınıflandırması
10. Hipotezlerin akıbeti: doğrulandı / yanlışlandı / belirsiz
11. Tartışma ve sınırlılıklar: ~76 grup, dosya yılı anlamı, etap belirsizliği, gürültü tabanı, bootstrap'ın kapsamı, tek veri kaynağı
12. Yeniden üretim: `RUN_ALL.sh`, sürümler, sha256'lar

---

## 15. Zaman planı (≈35–50 saat, 4 hafta)

| Hafta | İş | Teslim |
|---|---|---|
| 1 | Ortam kurulumu, ingest/clean/match, manuel eşleştirme kararları, EDA, bölme manifesti, hipotezler, **30 soru + altın yanıtlar**. LLM erişimini test et (sorun varsa hocaya bildir) | **Dondurulmuş protokol + kilitli test soruları** |
| 2 | M1–M4 türetmeleri; `attention.py` + testler; B0/B1; D1/D2 | Matematik taslağı, testler yeşil |
| 3 | D3; iç içe CV koşuları; 3 tohum; ablasyonlar; sızıntı tanısı; bootstrap; hata analizi | Sonuç tabloları |
| 4 | KB + BM25 + araç + R0/R1/R2; dev setinde k/prompt; kilitli testte tek koşu; RAG değerlendirmesi; rapor | Rapor + kod + `RUN_ALL.sh` |

---

## 16. Tuzak kontrol listesi (teslimden önce tek tek işaretle)

- [ ] Hiçbir yerde “kurulum yılı” yok; `file_year` = kaynak sürümü.
- [ ] Ham Excel dosyaları değişmedi (sha256 aynı).
- [ ] 153 kayıt; atlanan her satır günlükte.
- [ ] `Atnavutköy` düzeltildi, ham değer korundu.
- [ ] Ömerli 2024 iki kayıt: silinmedi, ortalanmadı, regresyondan karantinada, KB'de çelişki etiketli.
- [ ] Eşleştirme kodunda güç/durum sütunu yok (assert).
- [ ] Etaplar ve Su Deposu (1)(2)(3) ayrı `unit_id`; birleşme yalnız gerekçeli `entity_group_id` düzeyinde.
- [ ] İlçe eşleştirme anahtarında (aynı adlı farklı ilçe tesisleri ayrık).
- [ ] Ad kalıp ekleri (GES / Güneş Enerji Santrali) temizlendi → yıl ablasyonu anlamlı.
- [ ] Her katta eğitim∩test grupları = ∅ (assert).
- [ ] Sözlük, TF-IDF, encoder'lar yalnız iç eğitimde fit; `[UNK]` eğitimde öğreniliyor (`min_freq=2`).
- [ ] Ölçek `√d_h` (=√8), `√d_model` değil; softmax `dim=-1`; head birleştirmede transpose önce.
- [ ] Maske softmax'tan önce, `finfo.min`; tamamen maskeli satır NaN üretmiyor.
- [ ] Hazır `MultiheadAttention`/`TransformerEncoder` yalnız testlerde.
- [ ] Dış test skoruna göre hiçbir seçim yok; epoch iç doğrulamadan taşındı.
- [ ] Tohumlar ayrı raporlandı, bağımsız tesis sayılmadı.
- [ ] Bootstrap grup düzeyinde, eşleştirilmiş, 1000 tekrar, kapsam notu yazıldı.
- [ ] Ablasyonlar tek tohum, ayrı tablo.
- [ ] Satır bölmesi “sızıntı tanısı” etiketli + ortak tesis oranı.
- [ ] Attention haritası nedensel açıklama olarak sunulmadı; token silmeyle karşılaştırıldı.
- [ ] RAG toplamları tüm filtrelenmiş tablodan; LLM kodu çalıştırılmadı.
- [ ] Test soruları prompt/k seçiminden önce kilitlendi; boş paydalar “uygulanamaz”.
- [ ] LLM model/sürüm/nicemleme/prompt/ayarlar/donanım kaydedildi.
- [ ] Kötü sonuçlar gizlenmedi; küçük örneklem sınırlılığı tartışıldı.
- [ ] Eşleştirme sürümü bölmeden önce kilitlendi; sonradan değişiklik sürümlü ve gerekçeli.
- [ ] Negatif kontrol tablosundaki her hata testle yakalandı.
- [ ] Grafik başlıkları/altyazıları bunun bir **envanter** verisi olduğunu, yeni kurulum veya üretim verisi olmadığını belirtiyor.
- [ ] Öğretim elemanı notlarındaki metin rapora kopyalanmadı; o notlar yalnız kontrol listesi olarak kullanıldı ve tüm sayılar kendi kodunla üretildi.

---

## 17. Savunma provası

Hoca önceden seçmediğin **rastgele bir kayıt** soracak. Şunu canlı gösterebilmelisin:
1. Excel dosyası + satır → `record_id` → `cleaning_log`'daki dönüşümler.
2. Hangi `unit_id`/`entity_group_id`, neden, hangi güvenle (`matching_log`).
3. Hangi dış kattaydı; aynı grubun başka kaydı eğitimde **yok** (manifestten göster).
4. Modelin gördüğü token dizisi (`[CLS] [ILCE] … [YIL] … [MEKAN] …`) ve tahmin.
5. RAG'de bu kayıtla ilgili bir soru → getirilen record_id'ler → yanıt ve atıf.

Prova için `src/trace_record.py GES-2025-016` gibi tek komutluk bir izleme betiği yaz. Bu, savunmanın en güçlü anı olur.
