import pandas as pd

# sicaklik.csv dosyasını yükle
try:
    sicaklik_df = pd.read_csv('/content/sicaklik.csv')
    print("sicaklik.csv başarıyla yüklendi")
except FileNotFoundError:
    print("Hata: sicaklik.csv bulunamadı. Lütfen dosya yolunu kontrol edin.")
except Exception as e:
    print(f"sicaklik.csv yüklenirken bir hata oluştu: {e}")

### `sicaklik.csv` için İlk Veri İncelemesi

# DataFrame'in ilk 5 satırını görüntüle
display(sicaklik_df.head())

# DataFrame hakkında temel bilgileri görüntüle (veri tipleri, boş olmayan değerler)
display(sicaklik_df.info())

# Sayısal sütunlar için tanımlayıcı istatistikleri görüntüle
display(sicaklik_df.describe())

# goc.xlsx dosyasını yükle
try:
    goc_df = pd.read_excel('/content/goc.xlsx')
    print("goc.xlsx başarıyla yüklendi")
except FileNotFoundError:
    print("Hata: goc.xlsx bulunamadı. Lütfen dosya yolunu kontrol edin.")
except Exception as e:
    print(f"goc.xlsx yüklenirken bir hata oluştu: {e}")

### `goc.xlsx` için İlk Veri İncelemesi

# DataFrame'in ilk 5 satırını görüntüle
display(goc_df.head())

# DataFrame hakkında temel bilgileri görüntüle (veri tipleri, boş olmayan değerler)
display(goc_df.info())

# Sayısal sütunlar için tanımlayıcı istatistikleri görüntüle
display(goc_df.describe())

### `sicaklik_df` için Eksik Değer Analizi

# Eksik değerleri hesapla
missing_values = sicaklik_df.isnull().sum()
missing_percentage = (sicaklik_df.isnull().sum() / len(sicaklik_df)) * 100

# Eksik değerler için bir DataFrame oluştur
missing_df = pd.DataFrame({'Eksik Sayısı': missing_values, 'Eksik Yüzdesi': missing_percentage})

# Yalnızca eksik değerleri olan sütunları filtrele
missing_df = missing_df[missing_df['Eksik Sayısı'] > 0].sort_values(by='Eksik Yüzdesi', ascending=False)

display(missing_df)

### `sicaklik_df`'teki Eksik Değerleri Ele Alma

# 'catisma_degisim'deki eksik değerleri medyanı ile doldur
median_catisma_degisim = sicaklik_df['catisma_degisim'].median()
sicaklik_df['catisma_degisim'] = sicaklik_df['catisma_degisim'].fillna(median_catisma_degisim)

# Diğer belirlenen sütunlardaki eksik değerleri medyanları ile doldur
columns_with_few_missing = ['multeci', 'siginmaci', 'unhcr_idp', 'sinir_otesi', 'multeci_oran']
for col in columns_with_few_missing:
    median_val = sicaklik_df[col].median()
    sicaklik_df[col] = sicaklik_df[col].fillna(median_val)

print("sicaklik_df'deki belirtilen sütunlar için eksik değerler ele alındı.")

# Eksik değerlerin gitmiş olduğunu doğrula
print("\nDoldurma sonrası eksik değerler:")
display(sicaklik_df[columns_with_few_missing + ['catisma_degisim']].isnull().sum())

### Birleştirme için Ortak Sütunları Keşfetme

print("sicaklik_df'deki benzersiz yıllar:", sicaklik_df['yil'].unique())
print("sicaklik_df'deki benzersiz ülkeler (ulke) (ilk 10):", sicaklik_df['ulke'].unique()[:10])
print("\ngoc_df'deki benzersiz yıllar:", goc_df['Year'].unique())
print("goc_df'deki benzersiz ülkeler (Country of Origin) (ilk 10):", goc_df['Country of Origin'].unique()[:10])

Her iki veri çerçevesinde de 'Yıl' bilgisi bulunmaktadır. Ülke düzeyinde birleştirme için, `sicaklik_df`'de `ulke` (ülke adı) ve `iso3` (ISO3 kodu) bulunurken, `goc_df`'de `Country of Origin` (ülke adı) ve `Country of Origin ISO` (ISO3 kodu) bulunmaktadır. ISO3 kodları, ülke verilerini birleştirmek için en tutarlı yol gibi görünmektedir. Her iki veri çerçevesini 'Yıl' ve 'ISO3' kodlarını kullanarak birleştirelim.

# sicaklik_df'deki sütunları goc_df ile birleştirme için yeniden adlandır
sicaklik_df_renamed = sicaklik_df.rename(columns={
    'yil': 'Year',
    'iso3': 'Country of Origin ISO'
})

# İki veri çerçevesini 'Yıl' ve 'Country of Origin ISO' sütunlarında birleştir
merged_df = pd.merge(
    sicaklik_df_renamed,
    goc_df,
    on=['Year', 'Country of Origin ISO'],
    how='inner'
)

print(f"Birleştirilmiş DataFrame boyutu: {merged_df.shape}")
display(merged_df.head())

### Birleştirilmiş DataFrame'in İlk Veri Keşfi

display(merged_df.info())

display(merged_df.describe())

### `merged_df`'de Tekrarlanan Kayıtları Kontrol Etme

# Birleştirilmiş DataFrame'deki tekrarlanan satırları kontrol et
duplicate_rows = merged_df.duplicated().sum()
print(f"merged_df'deki tekrarlanan satır sayısı: {duplicate_rows}")

if duplicate_rows > 0:
    print("Tekrarlanan satırlar kaldırılıyor...")
    merged_df.drop_duplicates(inplace=True)
    print(f"Tekrarlanan kayıtlar kaldırıldıktan sonraki satır sayısı: {len(merged_df)}")
else:
    print("Tekrarlanan satır bulunamadı.")

### Ana İlişkileri Görselleştirme: İklim Olaylarının Göç Üzerindeki Etkisi

import matplotlib.pyplot as plt
import seaborn as sns

# Grafikler için stili ayarla
sns.set_style("whitegrid")

# 'sicaklik' (sıcaklık) ve 'Total' (göç) arasındaki ilişkiyi inceleyelim
plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_df, x='sicaklik', y='Total', alpha=0.6)
plt.title('Sıcaklık vs. Toplam Göç (sicaklik_df ile goc_df)')
plt.xlabel('Ortalama Sıcaklık (sicaklik)')
plt.ylabel('Toplam Göç')
plt.xscale('linear') # Sıcaklık için doğrusal ölçek olduğundan emin ol
plt.yscale('log') # Geniş bir aralığa sahip olduğu için Toplam göç için logaritmik ölçek kullan
plt.show()

# Ayrıca 'yagis' (yağış) ve 'Total' göçe de bakalım
plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_df, x='yagis', y='Total', alpha=0.6)
plt.title('Yağış vs. Toplam Göç')
plt.xlabel('Yağış (yagis)')
plt.ylabel('Toplam Göç')
plt.xscale('linear')
plt.yscale('log')
plt.show()

# Farklı afet türlerinin toplam göç üzerindeki etkisini inceleyelim
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Afet Sayıları vs. Toplam Göç', fontsize=16)

sns.scatterplot(data=merged_df, x='afet_sayisi', y='Total', alpha=0.6, ax=axes[0, 0])
axes[0, 0].set_title('Toplam Afet Sayısı vs. Toplam Göç')
axes[0, 0].set_xlabel('Afet Sayısı')
axes[0, 0].set_ylabel('Toplam Göç (logaritmik)')
axes[0, 0].set_yscale('log')

sns.scatterplot(data=merged_df, x='kuraklik_sayisi', y='Total', alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title('Kuraklık Sayısı vs. Toplam Göç')
axes[0, 1].set_xlabel('Kuraklık Sayısı')
axes[0, 1].set_ylabel('Toplam Göç (logaritmik)')
axes[0, 1].set_yscale('log')

sns.scatterplot(data=merged_df, x='sel_sayisi', y='Total', alpha=0.6, ax=axes[1, 0])
axes[1, 0].set_title('Sel Sayısı vs. Toplam Göç')
axes[1, 0].set_xlabel('Sel Sayısı')
axes[1, 0].set_ylabel('Toplam Göç (logaritmik)')
axes[1, 0].set_yscale('log')

sns.scatterplot(data=merged_df, x='asiri_sicaklik_sayisi', y='Total', alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title('Aşırı Sıcaklık Sayısı vs. Toplam Göç')
axes[1, 1].set_xlabel('Aşırı Sıcaklık Sayısı')
axes[1, 1].set_ylabel('Toplam Göç (logaritmik)')
axes[1, 1].set_yscale('log')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Supratitle için boşluk bırak
plt.show()

### Ana sayısal özelliklerin dağılımı

# İklim ve göçle ilgili ana sayısal özellikler için histogramlar çiz
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(merged_df['sicaklik'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('sicaklik (Sıcaklık) Dağılımı')

sns.histplot(merged_df['yagis'], bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('yagis (Yağış) Dağılımı')

sns.histplot(merged_df['Total'], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Toplam Göç Dağılımı')
axes[1, 0].set_xscale('log') # Geniş aralık nedeniyle logaritmik ölçek

sns.histplot(merged_df['nufus'], bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Nufus (Nüfus) Dağılımı')
axes[1, 1].set_xscale('log') # Geniş aralık nedeniyle logaritmik ölçek

plt.tight_layout()
plt.show()

### Korelasyon Analizi

import numpy as np

# Korelasyon analizi için ilgili sayısal sütunları seç
correlation_cols = [
    'sicaklik', 'yagis', 'sic_anomali', 'yag_anomali',
    'yer_edilme', 'afet_sayisi', 'etkilenen', 'olum',
    'kuraklik_sayisi', 'sel_sayisi', 'asiri_sicaklik_sayisi',
    'nufus', 'Total'
]

# merged_df'de gerçekten var olan sütunları filtrele
existing_correlation_cols = [col for col in correlation_cols if col in merged_df.columns]

# Korelasyon matrisini hesapla
correlation_matrix = merged_df[existing_correlation_cols].corr()

# Isı haritasını çiz
plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=.5,
    linecolor='black',
    cbar_kws={'label': 'Korelasyon Katsayısı'}
)
plt.title('İklim ve Göç Faktörlerinin Korelasyon Matrisi')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

### Zaman İçindeki Eğilimleri Keşfetme

# Yıla göre gruplandır ve ilgili değişkenlerin ortalamasını hesapla
year_trends = merged_df.groupby('Year')[['sicaklik', 'yagis', 'Total', 'yer_edilme', 'etkilenen']].mean().reset_index()

fig, axes = plt.subplots(3, 1, figsize=(15, 18))

# Yıllara göre ortalama sıcaklığı çiz
sns.lineplot(data=year_trends, x='Year', y='sicaklik', marker='o', ax=axes[0])
axes[0].set_title('Yıllara Göre Ortalama Sıcaklık (sicaklik)')
axes[0].set_ylabel('Ortalama Sıcaklık')

# Yıllara göre ortalama yağışı çiz
sns.lineplot(data=year_trends, x='Year', y='yagis', marker='o', ax=axes[1])
axes[1].set_title('Yıllara Göre Ortalama Yağış (yagis)')
axes[1].set_ylabel('Ortalama Yağış')

# Yıllara göre toplam göç ve iç göçü çiz
sns.lineplot(data=year_trends, x='Year', y='Total', marker='o', label='Toplam Göç', ax=axes[2])
sns.lineplot(data=year_trends, x='Year', y='yer_edilme', marker='o', label='İç Göç', ax=axes[2])
axes[2].set_title('Yıllara Göre Toplam Göç ve İç Göç')
axes[2].set_ylabel('Ortalama Sayı')
axes[2].legend()

plt.tight_layout()
plt.show()
