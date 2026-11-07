# 07 — Girdi Temsili ve Tokenizasyon

> İlgili kod: [`src/tokenizer.py`](../src/tokenizer.py), girdi okuma noktası `train.model_inputs` ([`src/train.py`](../src/train.py))

## 1. Girdi sözleşmesi (kodda zorlanır)

- Her örnek **tek bir tesis kaydıdır.** Minibatch içindeki farklı kayıtlar birbirine attention uygulamaz: her satır kendi `[B, T]` dizisi, attention yalnız `T` ekseninde. Test `test_batch_equals_single`, aynı kaydın tek başına ve batch içinde aynı tahmini aldığını doğrular.
- Modele giren **yalnız üç alan:** `district_clean`, `file_year`, `name_clean`. Bu alanlar tek bir fonksiyondan (`train.model_inputs`) okunur. Fonksiyon, yasak sütun kümesiyle (`power_*`, `status_*`, `record_id`, `excel_row`, `source_no`, `unit_id`, `entity_group_id`, `quality_flag`) kesişim olmadığını `assert` eder.
- `file_year` kategorik **kaynak sürümü** olarak girer (`yil_2024` gibi bir token). Sayısal yıl değildir ve “kurulum yılı” anlamı taşımaz.

## 2. Dizi biçimi

```
[CLS] [ILCE] ilce_<ilçe> [YIL] yil_<dosya_yılı> [MEKAN] <ad tokenları ...> [PAD] ...
```
Örnek, `GES-2024-031` (Maltepe, 2024, “Orhangazi Şehir Parkı - Su Deposu Üzeri (2)”):
```
[CLS] [ILCE] ilce_maltepe [YIL] yil_2024 [MEKAN] orhangazi şehir parkı su deposu üzeri 2
```

| Tasarım kararı | Gerekçe |
|---|---|
| Alan işaretleri `[ILCE] [YIL] [MEKAN]` | Yönerge; D3'e alanların sınırını verir |
| **Önekli değer tokenları** `ilce_…`, `yil_…` | Özellik kimliği konuma değil token kimliğine gömülür ([06 M4.2](06_matematik.md)). Fatih'teki “**Silivri** Kapı Buz Pisti” adındaki `silivri` sözcüğü, `ilce_silivri` ile karışmaz. D1 gibi sırasız modeller de alanları ayırt eder |
| Türkçe-duyarlı küçük harf (`tr_lower`) | `İ→i`, `I→ı`; Türkçe karakterler korunur (`ş, ğ, ü, ö, ç, ı`) |
| Sayılar korunur | Etap ve alt birim numaraları (`1`, `2`, `3`), `75` (75. Yıl) |
| Yıl kalıpları yok | `name_clean` kullanılır ([02 R-NAME-02](02_veri_temizleme_on_isleme.md)) |
| Noktalama atılır | `(`, `)`, `-`, `.` token üretmez |

## 3. Sözlük — yalnız eğitimden, `min_freq = 2`

- Sözlük, **kullanıldığı eğitim parçasından** kurulur: her iç eğitim (15 × 4 aday) ve her dış eğitim (5 × 3 tohum) için yeniden. Doğrulama/test tokenları sözlüğü hiç etkilemez.
- Özel tokenlar: `[PAD]=0, [UNK]=1, [CLS]=2, [ILCE]=3, [YIL]=4, [MEKAN]=5`.
- **`[UNK]` tuzağı ve çözümü:** Eğitimde hiç `[UNK]` görülmezse embedding'i rastgele başlangıç değerinde kalır ve testteki her bilinmeyen token gürültü olur. `min_freq = 2` ile eğitimde **tek kez** geçen tokenlar da `[UNK]`'e eşlenir; böylece `[UNK]` embedding'i “nadir/bilinmeyen sözcük” anlamını gerçekten öğrenir.
- Dış eğitim sözlük boyutları: **145–157** (ortalama 151,6). Tokenların bir kısmı tek kez geçtiği için `[UNK]` oluyor; test kayıtlarında ortalama **2,07** token `[UNK]` (`attention_vs_deletion.csv`, `n_unk`).
- İlçesi eğitimde < 2 kez geçen test kayıtları (`ilce_…` → `[UNK]`): 151 kaydın **45**'i. Bu kayıtlarda hata belirgin şekilde büyük ([11](11_hata_analizi.md)).

## 4. Uzunluk ve kesme

- Dizi uzunluğu min 8, medyan 10, **maks 16** ([04 Şekil 13](04_kesifsel_veri_analizi.md)).
- `max_len` = eğitim parçasındaki en uzun dizi (14–16). Yönergedeki başlangıç üst sınırı 64'e hiç ulaşılmadı.
- **Kesilen kayıt sayısı: 0** (her dış kat ve tohum için `fold_info_*.csv` → `n_truncated_test = 0`).
- Padding sağdan `[PAD]` ile yapılır. Maskeler padding'i dışlar; padding eklemek tahmini değiştirmez (`test_padding_invariance`).

## 5. B1 için temsil (sinir ağı dışı)

B1 aynı üç alanı kullanır ama tokenları sayısal özelliklere çevirir: `TfidfVectorizer(tokenizer=tokenize_tr, ngram_range=(1,2), sublinear_tf=True)` ad üzerinde, `OneHotEncoder(handle_unknown="ignore")` ilçe ve yıl üzerinde. Hepsi `Pipeline` içinde, yalnız eğitim parçasında `fit` edilir. Bilinmeyen kategori testte sıfır vektör olur (`[UNK]` karşılığı).

## 6. Hedef dönüşümü

`z = ln(1 + kWp)` (D1–D3 ve B1 aynı). Tahmin `ŷ = max(0, exp(ẑ) − 1)`. Sinir ağlarında başlığın son bias'ı **eğitim parçasının** `z` ortalamasıyla başlatılır. Bu yalnız eğitimden hesaplanan bir istatistiktir ve optimizasyonun 0'dan başlayıp 4'e tırmanmasını önler.
