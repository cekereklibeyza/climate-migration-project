# 🌍 İklim Kaynaklı Göç Tahmin Modeli (Climate-Induced Displacement Predictor)

**Bu proje; küresel sıcaklık kayıtları, aşırı hava olayları/afet verileri ve Birleşmiş Milletler (BM) göç istatistiklerini entegre ederek iklim değişikliği kaynaklı kitlesel göç hareketlerini tahmin eden bir makine öğrenmesi modelidir.**

İklim krizi yalnızca çevresel bir sorun değil, aynı zamanda küresel bir insani ve demografik krizdir. Bu model, hangi bölgelerin iklimsel riskler nedeniyle ne kadarlık bir göç dalgası üretebileceğini veya hangi bölgelerin hedef göç noktası haline geleceğini öngörmeyi amaçlar.

---

## 🚀 Öne Çıkan Özellikler

* **Çoklu Veri Entegrasyonu:** Üç farklı dikeydeki (Sıcaklık, Afet, Göç) devasa veri kümelerini zaman ve mekan bazlı eşleştirme.
* **Zaman Serisi ve Tahmin:** Geleceğe dönük 5, 10 ve 20 yıllık projeksiyon senaryoları.
* **Risk Haritalandırması:** Coğrafi bilgi sistemleri (GIS) entegrasyonu ile yüksek riskli göç koridorlarının görselleştirilmesi.
* **Senaryo Analizi (Anomaliler):** Ani sıcaklık artışları veya ardışık afet senaryolarında göç tetiklenme hızının ölçülmesi.

---

## 📊 Kullanılan Veri Setleri (Data Sources)

Modelin eğitimi ve doğrulanması için aşağıdaki açık kaynaklı ve güvenilir küresel veri tabanları normalize edilerek birleştirilmiştir:

| Veri Seti | Sağlayıcı / Kurum | İçerik |
| :--- | :--- | :--- |
| **Küresel Sıcaklık Kayıtları** | NASA GISS / NOAA | Tarihsel yüzey sıcaklıkları, anomali endeksleri, kuraklık verileri |
| **Küresel Afet Veritabanı** | EM-DAT (CRED) | Sel, kasırga, kuraklık ve orman yangınlarının sıklığı ve etki derecesi |
| **Göç İstatistikleri** | UNHCR / UN DESA | Ülkeler arası resmi göç akışları, sığınmacı ve mülteci hareketleri |

---

## 🛠️ Teknolojik Altyapı ve Kütüphaneler

* **Veri İşleme & Analiz:** `Python`, `Pandas`, `NumPy`, `SciPy`
* **Makine Öğrenmesi & Modelleme:** `Scikit-Learn`, `XGBoost`, `Prophet` (Zaman Serileri için)
* **Veri Görselleştirme:** `Matplotlib`, `Seaborn`, `Geopandas`, `Plotly` (İnteraktif Haritalar)

---

## 💻 Kurulum ve Kullanım

### 1. Depoyu Klonlayın
```bash
git clone https://github.com
cd iklim-goc-tahmin-modeli
```

### 2. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Modeli Çalıştırın ve Tahmin Üretin
Veri setlerini `data/` klasörüne yükledikten sonra ana scripti çalıştırarak analizleri başlatabilirsiniz:
```bash
python src/predict_migration.py --region "Sub-Saharan Africa" --year 2035
```

---

## 📈 Model Sonuçları ve Görselleştirme

Model çalıştırıldığında `outputs/` klasörü altında aşağıdaki analiz çıktılarını üretir:
1. **Risk Isı Haritası:** Önümüzdeki 10 yılda en çok göç vermesi beklenen "Sıcak Noktalar (Hotspots)".
2. **Göç Akış Diyagramı (Sankey):** İklim mültecilerinin ana çıkış ve olası varış ülkeleri arasındaki rotalar.
3. **Korelasyon Grafikleri:** Afet sıklığı ile ani göç dalgaları arasındaki zamansal gecikme (lag) süreleri.

*(Örnek görselleştirmelerinizi buraya ekleyebilirsiniz)*
`![Örnek Tahmin Grafiği](outputs/sample_prediction_map.png)`

---

## 🤝 Katkıda Bulunma (Contributing)

Projeyi geliştirmeye yönelik her türlü katkıya açığız! Yeni bir özellik eklemek, veri setlerini güncellemek veya hata bildiriminde bulunmak için lütfen bir **Issue** açın ya da **Pull Request** gönderin.

1. Bu depoyu çatallayın (Fork).
2. Özellik dalınızı oluşturun (`git checkout -b yeni-ozellik`).
3. Değişikliklerinizi kaydedin (`git commit -m 'Yeni özellik eklendi'`).
4. Dalınızı itin (`git push origin yeni-ozellik`).
5. Bir Pull Request oluşturun.

---

## 📄 Lisans

Bu proje lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına göz atabilirsiniz.

---

**İletişim:** Proje hakkında sorularınız veya iş birliği talepleriniz için [E-posta Adresiniz] üzerinden ulaşabilirsiniz.
