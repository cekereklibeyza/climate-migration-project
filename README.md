# Küresel Isınma, Aşırı Hava Olayları ve Göç İlişkisi — Streamlit Uygulaması

## Yerelde çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

Uygulama `data/birlesik_tablo.csv` dosyasını kullanır (1.190 satır, ekibin
doğrulanmış birleşik veri seti). Bu dosyayı `app.py` ile aynı klasörde
tutun.

## Streamlit Community Cloud'a yükleme

1. Bu klasörü (`app.py`, `requirements.txt`, `data/`) bir GitHub reposuna
   yükleyin.
2. [share.streamlit.io](https://share.streamlit.io) üzerinden repo'yu bağlayın,
   ana dosya olarak `app.py`'yi seçin.
3. Deploy edin — birkaç dakika içinde herkese açık bir link alırsınız.

## Sayfalar

- **Genel Bakış** — proje özeti, temel istatistikler.
- **Veri Gezgini** — ülke/yıl filtreleme, zaman serisi grafikleri.
- **EDA & Hipotezler** — dağılımlar, korelasyon matrisi, H1/H7/H5/H20 canlı
  hesaplanır.
- **Modelleme Sonuçları** — uygulama açıldığında modeller (Linear, Ridge,
  KNN, Decision Tree, Random Forest, Gradient Boosting, varsa CatBoost)
  2001-2020 eğitim / 2021-2025 test ayrımıyla canlı eğitilir; ekibin
  orijinal notebook sonuçlarıyla karşılaştırma tablosu da gösterilir.
- **Canlı Tahmin** — iklim/afet değerlerini girip seçilen modelle anlık
  göç tahmini üretme.

## Not

`catboost` paketi bazı ortamlarda kurulamayabilir; bu durumda uygulama
otomatik olarak CatBoost'u atlar, diğer modeller etkilenmez.
