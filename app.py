# -*- coding: utf-8 -*-
"""
Küresel Isınma, Aşırı Hava Olayları ve Göç İlişkisi - Streamlit Uygulaması
Çalıştırmak için:  pip install -r requirements.txt  &&  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Küresel Isınma, Aşırı Hava Olayları ve Göç İlişkisi",
    page_icon="\U0001F30D",
    layout="wide",
)

# ------------------------------------------------------------
# RENK PALETİ - mavi/turkuaz gradyan temalı
# ------------------------------------------------------------
DEEP_BLUE = "#0D2C4F"
NAVY = "#123C63"
BLUE = "#1B5E85"
TEAL = "#1F8A8C"
MID_TEAL = "#3FA9A0"
LIGHT_TEAL = "#7FC8BE"
PALE_TEAL = "#D7EEEA"
INK = "#1C2B36"
MUTED = "#5B7280"

ORANGE = "#E8A33D"
SOFT_ORANGE_BG = "#FCEFDD"

BLUE_GRADIENT = [DEEP_BLUE, NAVY, BLUE, TEAL, MID_TEAL, LIGHT_TEAL]
TEAL_SCALE = "Teal"
BLUE_SCALE = "Blues"
# negatif -> turuncu, sifir -> beyaz, pozitif -> lacivert (diverging skala)
ORANGE_BLUE_DIVERGING = [
    [0.0, "#B5651D"],
    [0.25, "#E8935C"],
    [0.5, "#FFFFFF"],
    [0.75, "#5FA8D3"],
    [1.0, DEEP_BLUE],
]

TEAM = [
    ("Beyza Fatıma Çekerekli", "Takım Lideri",
     "Proje kapsamının ve veri/analiz kararlarının belirlenmesi, tüm ekip çıktılarının "
     "tutarlılık kontrolü, literatür taramasına katkı, nihai rapor ve uygulamanın koordinasyonu."),
    ("Dilan Sazan", "Modelleme (CatBoost & Ridge)",
     "Sınır ötesi göç tahmini için ağaç tabanlı ve doğrusal model karşılaştırması; "
     "veri taraması, birleştirme ve literatür taramasına katkı sağladı."),
    ("Esma Nur Beğbağa", "Modelleme (Random Forest / Gradient Boosting / Decision Tree / Linear)",
     "Çoklu model karşılaştırması ve zaman bazlı doğrulama; veri taramasına katkı sağladı."),
    ("Feyza Nur Demirbaş", "Veri Toplama, Veri Temizleme ve Birleştirme",
     "Ham göç, iklim ve afet verilerinin derlenmesi, temizlenmesi ve nihai veri setinin oluşturulması."),
    ("Hilal Üçüncü", "Hipotez Geliştirme",
     "Literatüre dayalı hipotezlerin belirlenmesi ve test edilecek hipotez setinin tasarımı; "
     "literatür taramasına katkı sağladı."),
    ("Şevval Ok", "Keşifsel Veri Analizi (EDA)",
     "Veri setinin dağılım, korelasyon ve örüntü analizleriyle keşfedilmesi; "
     "veri taramasına katkı sağladı."),
]

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #F4F9F8 0%, #FFFFFF 45%);
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(200deg, {DEEP_BLUE} 0%, {TEAL} 130%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #F2FBF9 !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.98rem;
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, {PALE_TEAL} 0%, #FFFFFF 100%);
        border: 1px solid {LIGHT_TEAL};
        border-radius: 14px;
        padding: 14px 16px 8px 16px;
    }}
    div[data-testid="stMetricValue"] {{
        color: {DEEP_BLUE};
    }}
    div[data-testid="stImage"] {{
        margin-bottom: 16px;
    }}
    h1, h2, h3 {{
        color: {DEEP_BLUE};
    }}
    .app-hero {{
        background: linear-gradient(120deg, {DEEP_BLUE} 0%, {TEAL} 100%);
        padding: 26px 30px;
        border-radius: 18px;
        color: #F5FCFB;
        margin-bottom: 18px;
    }}
    .app-hero h1 {{
        color: #FFFFFF !important;
        margin-bottom: 6px;
        font-size: 2.0rem;
    }}
    .app-hero p {{
        color: #E4F4F1;
        font-size: 1.02rem;
        margin: 0;
    }}
    .note-box {{
        background: {PALE_TEAL};
        border-left: 4px solid {TEAL};
        border-radius: 8px;
        padding: 12px 16px;
        color: {INK};
        font-size: 0.92rem;
    }}
    .warn-box {{
        background: #FDF3E7;
        border-left: 4px solid {ORANGE};
        border-radius: 8px;
        padding: 12px 16px;
        color: {INK};
        font-size: 0.92rem;
    }}
    .team-card {{
        background: linear-gradient(135deg, #FFFFFF 0%, {PALE_TEAL} 130%);
        border: 1px solid {LIGHT_TEAL};
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 10px;
        min-height: 96px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .team-card .name {{
        font-weight: 700;
        color: {DEEP_BLUE};
        font-size: 1.02rem;
    }}
    .team-card .role {{
        font-weight: 600;
        color: {TEAL};
        font-size: 0.88rem;
        margin-bottom: 4px;
    }}
    .team-card .desc {{
        color: {MUTED};
        font-size: 0.86rem;
        line-height: 1.35;
    }}
    table.soft-table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 0.86rem;
    }}
    table.soft-table thead th {{
        background: {SOFT_ORANGE_BG} !important;
        color: {INK} !important;
        border-bottom: 2px solid {ORANGE};
        padding: 8px 10px;
        text-align: left;
        position: sticky;
        top: 0;
    }}
    table.soft-table tbody td {{
        padding: 6px 10px;
        border-bottom: 1px solid #EEE;
    }}
    table.soft-table tbody tr:nth-child(even) {{
        background: #FAFCFC;
    }}
    .soft-table-wrap {{
        max-height: 340px;
        overflow-y: auto;
        border: 1px solid {LIGHT_TEAL};
        border-radius: 10px;
    }}
    .sidebar-title {{
        font-size: 1.18rem;
        font-weight: 700;
        line-height: 1.4;
        color: #F2FBF9;
        margin-bottom: 2px;
    }}
    .finding-box {{
        background: linear-gradient(135deg, {PALE_TEAL} 0%, #FFFFFF 100%);
        border: 1px solid {LIGHT_TEAL};
        border-left: 5px solid {TEAL};
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }}
    .finding-box .finding-title {{
        font-weight: 700;
        color: {DEEP_BLUE};
        font-size: 1.05rem;
        margin-bottom: 8px;
    }}
    .finding-box ul {{
        margin: 0;
        padding-left: 20px;
    }}
    .finding-box li {{
        color: {INK};
        font-size: 0.92rem;
        margin-bottom: 6px;
        line-height: 1.45;
    }}
    .intro-box {{
        background: linear-gradient(135deg, #FFFFFF 0%, {PALE_TEAL} 100%);
        border: 1px solid {LIGHT_TEAL};
        border-left: 5px solid {NAVY};
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }}
    .intro-box .intro-title {{
        font-weight: 700;
        color: {DEEP_BLUE};
        font-size: 1.05rem;
        margin-bottom: 8px;
    }}
    .intro-box p {{
        color: {INK};
        font-size: 0.94rem;
        line-height: 1.5;
        margin-bottom: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_soft_table(dataframe):
    html = dataframe.to_html(index=False, classes="soft-table", border=0)
    st.markdown(f'<div class="soft-table-wrap">{html}</div>', unsafe_allow_html=True)

FEATURES = [
    "Yil", "Avg_Temp", "Avg_Precip", "Afet_Sayisi", "Kuraklik_Sayisi",
    "Sel_Sayisi", "Firtina_Sayisi", "Asiri_Sicaklik_Sayisi",
    "log_Toplam_Olum", "log_Toplam_Etkilenen", "log_Toplam_Hasar",
]

FEATURE_LABELS = {
    "Yil": "Yıl",
    "Avg_Temp": "Ort. Sıcaklık (°C)",
    "Avg_Precip": "Ort. Yağış (mm)",
    "Afet_Sayisi": "Toplam Afet Sayısı",
    "Kuraklik_Sayisi": "Kuraklık Sayısı",
    "Sel_Sayisi": "Sel Sayısı",
    "Firtina_Sayisi": "Fırtına Sayısı",
    "Asiri_Sicaklik_Sayisi": "Aşırı Sıcaklık Sayısı",
    "log_Toplam_Olum": "log(1+Toplam Ölüm)",
    "log_Toplam_Etkilenen": "log(1+Toplam Etkilenen)",
    "log_Toplam_Hasar": "log(1+Toplam Hasar, 1000 USD)",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="Source Sans Pro, sans-serif", color=INK),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


def style_fig(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), **PLOTLY_LAYOUT)
    return fig


# ============================================================
# VERİ YÜKLEME
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/birlesik_tablo.csv")
    df["log_Goc"] = np.log1p(df["Goc"])
    df["log_Toplam_Olum"] = np.log1p(df["Toplam_Olum"])
    df["log_Toplam_Etkilenen"] = np.log1p(df["Toplam_Etkilenen"])
    df["log_Toplam_Hasar"] = np.log1p(df["Toplam_Hasar_1000USD"])
    return df


def spearman(a, b):
    return a.rank().corr(b.rank())


# ============================================================
# MODEL EĞİTİMİ (zaman bazlı split: <=2020 eğitim, >2020 test)
# ============================================================
@st.cache_resource
def train_models(df_hash):
    df = load_data()
    train = df[df["Yil"] <= 2020].copy()
    test = df[df["Yil"] > 2020].copy()

    Xtr, ytr = train[FEATURES], train["log_Goc"]
    Xte, yte = test[FEATURES], test["log_Goc"]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    fitted = {}
    rows = []

    specs = [
        ("Linear Regression", LinearRegression(), True),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=42), True),
        ("KNN Regressor", KNeighborsRegressor(n_neighbors=7), True),
        ("Decision Tree", DecisionTreeRegressor(max_depth=6, random_state=42), False),
        ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), False),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42), False),
    ]

    for name, model, needs_scale in specs:
        Xtr_use = Xtr_s if needs_scale else Xtr
        Xte_use = Xte_s if needs_scale else Xte
        model.fit(Xtr_use, ytr)
        pred = model.predict(Xte_use)
        r2 = r2_score(yte, pred)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae = float(mean_absolute_error(yte, pred))
        rows.append({"Model": name, "R2": round(r2, 4), "RMSE_log": round(rmse, 4), "MAE_log": round(mae, 4)})
        fitted[name] = (model, needs_scale)

    if CATBOOST_AVAILABLE:
        cb = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, verbose=False, random_state=42)
        cb.fit(Xtr, ytr)
        pred = cb.predict(Xte)
        r2 = r2_score(yte, pred)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae = float(mean_absolute_error(yte, pred))
        rows.append({"Model": "CatBoost", "R2": round(r2, 4), "RMSE_log": round(rmse, 4), "MAE_log": round(mae, 4)})
        fitted["CatBoost"] = (cb, False)

    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return fitted, scaler, results, Xte, yte, test


df = load_data()
with st.spinner("Modeller eğitiliyor..."):
    fitted_models, scaler, results_df, Xte, yte, test_df = train_models(len(df))

# ============================================================
# SIDEBAR - NAVİGASYON
# ============================================================
st.sidebar.markdown(
    '<div class="sidebar-title">\U0001F30D Küresel Isınma, Aşırı Hava Olayları ve Göç İlişkisi</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Doğu-Güney Afrika ve Batı-Orta Afrika (2001-2025)")
page = st.sidebar.radio(
    "Sayfa seçin",
    [
        "Genel Bakış",
        "Veri Gezgini",
        "EDA & Hipotezler",
        "Modelleme Sonuçları",
        "Canlı Tahmin",
        "Sonuç ve Öneriler",
        "Proje Ekibi",
    ],
)
st.sidebar.divider()
st.sidebar.caption(
    "Bu uygulama, ekibin veri temizleme + birleştirme + hipotez + modelleme "
    "çalışmalarının interaktif özetidir."
)

# ============================================================
# SAYFA 1: GENEL BAKIŞ
# ============================================================
if page == "Genel Bakış":
    _logo_l, _logo_r = st.columns([6, 1])
    with _logo_r:
        st.image("assets/logo_softito.png", width=110)
    st.markdown(
        """
        <div class="app-hero">
        <h1>Küresel Isınma, Aşırı Hava Olayları ve Göç İlişkisi</h1>
        <p>Doğu-Güney Afrika ve Batı-Orta Afrika Bölgeleri (2001-2025) — Veri Analitiği
        Dönem Projesi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Göç, UNHCR'ın ülkeler arası göç verisiyle tanımlanmıştır "
        "(ülke içi yerinden edilme dahil değildir)."
    )

    st.markdown(
        """
        <div class="intro-box">
        <div class="intro-title">Proje Kapsamı</div>
        <p>Bu proje, iklim değişkenlerinin (sıcaklık, yağış) ve aşırı hava olaylarının
        (kuraklık, sel, fırtına, aşırı sıcaklık) Doğu-Güney Afrika ve Batı-Orta Afrika
        bölgelerindeki sınır ötesi göç ile ilişkisini incelemektedir. İklim ve afet verileri,
        UNHCR'ın resmi göç istatistikleriyle birleştirilerek hem istatistiksel hipotez
        testleriyle hem de makine öğrenmesi modelleriyle analiz edilmiştir.</p>
        <p>Literatür taramasında incelenen çalışmalarda, aşırı hava olayı sıklığının
        (yalnızca ortalama sıcaklık/yağış değil) çoğu ekonometrik modelde ayrı ayrı
        temsil edilmediği ve nicel/makine öğrenmesi tabanlı çalışmaların sınırlı sayıda
        olduğu görülmüştür.</p>
        <p><b>Araştırma Sorumuz:</b> İklim değişkenleri ve aşırı hava olayları, bu iki
        bölgedeki sınır ötesi göçü ne ölçüde açıklıyor ve tahmin edilebilir kılıyor?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-box">
        <div class="intro-title">Küresel Isınma, Aşırı Hava Olayları ve Göç İlişkisi</div>
        <p>Küresel ısınma tek bir sonuç değil; kuraklık, sel, fırtına ve aşırı sıcaklık gibi
        birden fazla aşırı hava olayını tetikleyen bir şemsiye kavramdır. Bu olayların her
        biri, insanların yaşam koşullarını farklı şekillerde etkileyerek sınır ötesi göçü
        tetikleyebilir. Projemizde küresel ısınmayı doğrudan değil, onun somut göstergeleri
        olan ortalama sıcaklık, ortalama yağış ve dört afet türünün sıklığı üzerinden
        inceledik; bu değişkenlerin göçle ilişkisini hem istatistiksel hem de makine
        öğrenmesi yöntemleriyle test ettik.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-box">
        <div class="intro-title">Bölge Seçimi</div>
        <p>UNHCR verisinde 6 farklı bölge bulunuyor: Asya-Pasifik, Amerika, Avrupa,
        Ortadoğu-Kuzey Afrika, Doğu-Güney Afrika ve Batı-Orta Afrika. Biz analizimizi
        <b>Doğu-Güney Afrika</b> ve <b>Batı-Orta Afrika</b> ile sınırlandırdık; çünkü bu iki
        bölge iklim değişikliğine coğrafi ve ekonomik açıdan yüksek kırılganlık gösteriyor,
        göç verisinde yeterli çeşitlilik sunuyor, veri tek bir kriz ülkesine yığılmamış
        durumda ve 2001-2025 arası verisi eksiksiz. Diğer dört bölge, zaman ve kapsam
        kısıtları nedeniyle bu çalışmanın dışında bırakıldı.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-box">
        <div class="intro-title">Veri Kaynakları</div>
        <p>Göç verisi UNHCR'ın yayınladığı ülkeler arası (bilateral) iltica/göç
        istatistiklerinden alındı; çünkü projemiz göçü sınır ötesi hareket olarak
        tanımlıyor, ülke içi yerinden edilmeyi (IDP) kapsam dışında bırakıyor. Afet verisi
        EM-DAT (The International Disaster Database) kaynağından geldi; iklim verisi ise
        ülke-yıl bazında ortalama sıcaklık ve yağış kayıtlarını içeriyor. Bu üç kaynağı
        birleştirerek 1.190 satırlık ortak bir veri seti oluşturduk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-box">
        <div class="intro-title">Modelleme Yaklaşımı</div>
        <p>Ekibimiz iki bağımsız modelleme çalışması yürüttü: birincisinde <b>CatBoost</b> ve
        <b>Ridge Regression</b>, ikincisinde <b>Random Forest</b>, <b>Gradient Boosting</b>,
        <b>Decision Tree</b>, <b>Ridge</b> ve <b>Linear Regression</b>
        karşılaştırıldı. Amacımız, farklı model ailelerinin aynı veride tutarlı sonuç verip
        vermediğini görmekti. Her iki çalışmada da ağaç tabanlı modeller doğrusal modelleri
        açık farkla geçti; bu, iklim/afet değişkenleri ile göç arasındaki ilişkinin doğrusal
        olmadığını, eşik değerler ve değişken etkileşimleri içerdiğini gösteriyor. Ayrıca
        ekipten bir arkadaşımız IDMC (iç yerinden edilme) verisiyle ayrı bir keşif çalışması
        yaptı; bu, projenin sınır ötesi göç tanımı dışında kaldığı için ana bulgulara değil,
        tamamlayıcı bir çalışma olarak rapora dahil edildi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-box">
        <div class="intro-title">Projenin Gelişim Süreci</div>
        <p>Proje; veri temizleme ve birleştirmeyle başladı, keşifsel veri analizi ve hipotez
        testleriyle devam etti, iki bağımsız modelleme çalışmasıyla derinleşti ve
        interaktif bir uygulamayla tamamlandı. Süreç boyunca önemli bir metodolojik
        düzeltme de yaptık: ilk modelleme denemesinde rastgele train/test ayrımının veri
        sızıntısına yol açtığını fark edip, zaman bazlı bir ayrıma geçtik — bu, sonuçlarımızın
        güvenilirliğini önemli ölçüde artırdı.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ülke Sayısı", df["ISO3"].nunique())
    c2.metric("Yıl Aralığı", f"{df['Yil'].min()}-{df['Yil'].max()}")
    c3.metric("Toplam Kayıt", f"{len(df):,}".replace(",", "."))
    c4.metric("Toplam Göç (Kümülatif)", f"{int(df['Goc'].sum()):,}".replace(",", "."))

    st.divider()
    st.markdown(
        """
        <div class="finding-box">
        <div class="finding-title">📌 Öne Çıkan Bulgular</div>
        <ul>
            <li><b>Sel, Kuraklıktan Daha Güçlü Bir Göç Belirleyicisi:</b> Sel sayısı ile göç
            arasındaki ilişki (Spearman r ≈ 0.33), kuraklığın ilişkisinden (r ≈ 0.01) belirgin
            şekilde daha güçlü.</li>
            <li><b>Kuraklığın Etkisi Dolaylı:</b> Kuraklık göçü doğrudan artırmıyor, etkilenen
            nüfusu artırıyor; bu da sınır ötesi göçe ancak zayıf biçimde yansıyor
            ("hapsolmuş nüfus" etkisi).</li>
            <li><b>Çok Değişkenli Model, Tek Değişkenliden Çok Daha Güçlü:</b> Sadece kuraklıkla
            açıklanan varyans ~%0 iken, 6 iklim/afet değişkeninin birlikte kullanılması açıklama
            gücünü ~%18'e çıkarıyor.</li>
            <li><b>En İyi Modeller Göçü Yüksek Doğrulukla Tahmin Ediyor:</b> Random Forest ve
            CatBoost gibi ağaç tabanlı modeller, test setinde R² = 0.56-0.72 arası performans
            gösteriyor.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Yıllara Göre Toplam Göç")
        yearly = df.groupby("Yil", as_index=False)["Goc"].sum()
        fig = px.area(yearly, x="Yil", y="Goc", color_discrete_sequence=[TEAL])
        fig.update_traces(line=dict(color=DEEP_BLUE, width=2), fillcolor="rgba(31,138,140,0.25)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col2:
        st.subheader("En Çok Göç Veren 10 Ülke (Toplam)")
        top10 = df.groupby("Ulke", as_index=False)["Goc"].sum().sort_values("Goc", ascending=False).head(10)
        top10 = top10.sort_values("Goc")
        fig = px.bar(top10, x="Goc", y="Ulke", orientation="h", color="Goc",
                     color_continuous_scale=TEAL_SCALE)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.divider()
    st.markdown(
        "**Metodoloji Özeti:** Bilateral göç verisi (43.162 satır) ülke-yıl bazında "
        "toplanıp iklim/afet tablosuyla birleştirilerek 1.190 satırlık nihai veri seti "
        "oluşturulmuştur. Hipotez testleri ve iki bağımsız modelleme çalışması "
        "(CatBoost/Ridge ve Random Forest/Gradient Boosting/Decision Tree/KNN/Linear) "
        "bu veri seti üzerinde yürütülmüştür."
    )

    st.markdown(
        """
        <div class="intro-box" style="border-left-color:#E8A33D;">
        <div class="intro-title">🌍 Afrika Bağlamı — Literatürden Bir Kanıt</div>
        <p>Bu konuyu özellikle Afrika bağlamında seçmemizin bir nedeni de literatür:
        Marchiori, Maystadt ve Schumacher'in 2012 tarihli çalışması, Sahra-altı Afrika'da
        1960-2000 arasında hava anomalilerinin toplam 5 milyon net yerinden edilmeye yol
        açtığını gösteriyor. Yani bu bölge, iklim-göç ilişkisinin gerçek hayatta en somut
        şekilde gözlemlendiği yerlerden biri.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# SAYFA 2: VERİ GEZGİNİ
# ============================================================
elif page == "Veri Gezgini":
    _logo_l, _logo_r = st.columns([6, 1])
    with _logo_r:
        st.image("assets/logo_softito.png", width=110)
    st.title("🔍 Veri Gezgini")
    st.caption("Ülke ve yıl seçerek birleşik veri setini filtreleyin.")

    ulkeler = sorted(df["Ulke"].unique())
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        secili_ulkeler = st.multiselect("Ülke(ler)", ulkeler, default=ulkeler[:5])
    with col_f2:
        yil_araligi = st.slider("Yıl aralığı", int(df["Yil"].min()), int(df["Yil"].max()),
                                 (int(df["Yil"].min()), int(df["Yil"].max())))

    filtered = df[
        (df["Ulke"].isin(secili_ulkeler) if secili_ulkeler else True)
        & (df["Yil"].between(*yil_araligi))
    ]

    st.write(f"**{len(filtered)}** satır görüntüleniyor.")
    render_soft_table(
        filtered[["Yil", "Ulke", "ISO3", "Avg_Temp", "Avg_Precip", "Kuraklik_Sayisi",
                  "Sel_Sayisi", "Firtina_Sayisi", "Toplam_Etkilenen", "Goc"]]
    )

    if len(filtered) > 0 and secili_ulkeler:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Göç - zaman serisi")
            fig = px.line(filtered.sort_values("Yil"), x="Yil", y="Goc", color="Ulke",
                          markers=True, color_discrete_sequence=BLUE_GRADIENT)
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        with col2:
            st.subheader("Afet sayıları (toplam)")
            afet_toplam = filtered.groupby("Ulke")[["Kuraklik_Sayisi", "Sel_Sayisi", "Firtina_Sayisi", "Asiri_Sicaklik_Sayisi"]].sum()
            fig2 = px.bar(afet_toplam, barmode="group", color_discrete_sequence=BLUE_GRADIENT)
            st.plotly_chart(style_fig(fig2, 380), use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Filtrelenmiş veriyi CSV indir", csv, "filtrelenmis_veri.csv", "text/csv")

# ============================================================
# SAYFA 3: EDA & HİPOTEZLER
# ============================================================
elif page == "EDA & Hipotezler":
    _logo_l, _logo_r = st.columns([6, 1])
    with _logo_r:
        st.image("assets/logo_softito.png", width=110)
    st.title("📊 Keşifsel Veri Analizi ve Hipotezler")

    tab1, tab2 = st.tabs(["EDA", "Hipotez Testleri"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Göç dağılımı (ham)")
            fig = px.histogram(df, x="Goc", nbins=40, color_discrete_sequence=[BLUE])
            skew = df["Goc"].skew()
            fig.update_layout(title=f"Çarpıklık (skewness) = {skew:.2f}")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        with col2:
            st.subheader("log(1+Göç) dağılımı")
            fig = px.histogram(df, x="log_Goc", nbins=40, color_discrete_sequence=[TEAL])
            fig.update_layout(title="Log dönüşümü sonrası dağılım dengelenir")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)

        st.subheader("Korelasyon matrisi")
        num_cols = ["Avg_Temp", "Avg_Precip", "Afet_Sayisi", "Kuraklik_Sayisi", "Sel_Sayisi",
                    "Firtina_Sayisi", "Asiri_Sicaklik_Sayisi", "Toplam_Olum", "Toplam_Etkilenen",
                    "Toplam_Hasar_1000USD", "Goc"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=ORANGE_BLUE_DIVERGING, zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)

    with tab2:
        r_h1 = df["Kuraklik_Sayisi"].corr(df["Goc"])
        r_sel = spearman(df["Sel_Sayisi"], df["Goc"])
        r_kuraklik = spearman(df["Kuraklik_Sayisi"], df["Goc"])
        r_h5a = df["Kuraklik_Sayisi"].corr(df["Toplam_Etkilenen"])
        r_h5b = df["Toplam_Etkilenen"].corr(df["Goc"])

        def ols_r2(X, y):
            X = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot

        r2_simple = ols_r2(df[["Kuraklik_Sayisi"]].values, df["log_Goc"].values)
        clim_cols = ["Avg_Temp", "Avg_Precip", "Kuraklik_Sayisi", "Sel_Sayisi", "Firtina_Sayisi", "Asiri_Sicaklik_Sayisi"]
        r2_multi = ols_r2(df[clim_cols].values, df["log_Goc"].values)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### H1 - Kuraklık arttıkça göç artar")
            st.metric("Pearson r", f"{r_h1:.3f}")
            st.markdown('<div class="note-box">İlişki çok zayıf → desteklenmedi.</div>', unsafe_allow_html=True)

            st.markdown("#### H5 - Kuraklık etkisi dolaylı (mediation)")
            st.metric("Kuraklık → Etkilenen (r)", f"{r_h5a:.3f}")
            st.metric("Etkilenen → Göç (r)", f"{r_h5b:.3f}")
            st.markdown(
                '<div class="note-box">Kuraklık etkilenen nüfusu artırıyor ama bu sınır ötesi '
                'göçe zayıf yansıyor → kısmi destek ("hapsolmuş nüfus").</div>',
                unsafe_allow_html=True,
            )
        with colB:
            st.markdown("#### H7 - Sel, kuraklıktan daha güçlü etkiliyor")
            st.metric("Sel Spearman r", f"{r_sel:.3f}", delta=f"kuraklık: {r_kuraklik:.3f}")
            st.markdown(
                '<div class="note-box">Sel, kuraklıktan belirgin şekilde daha güçlü ilişkili → destekli.</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### H20 - Çoklu model tekliden iyi mi?")
            fig = go.Figure(go.Bar(
                x=["Basit (kuraklık)", "Çoklu (6 değişken)"], y=[r2_simple, r2_multi],
                marker=dict(color=[LIGHT_TEAL, DEEP_BLUE]),
            ))
            fig.update_layout(yaxis_title="R²")
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
            st.caption(f"Basit R²={r2_simple:.4f} vs Çoklu R²={r2_multi:.4f} → destekli.")

# ============================================================
# SAYFA 4: MODELLEME SONUÇLARI
# ============================================================
elif page == "Modelleme Sonuçları":
    _logo_l, _logo_r = st.columns([6, 1])
    with _logo_r:
        st.image("assets/logo_softito.png", width=110)
    st.title("🤖 Modelleme Sonuçları")
    st.caption("Modeller bu sayfa açıldığında 2001-2020 eğitim / 2021-2025 test ayrımıyla canlı olarak eğitilir.")

    st.subheader("Model karşılaştırması (canlı eğitim, bu veri seti üzerinde)")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    fig = px.bar(results_df.sort_values("R2"), x="R2", y="Model", orientation="h",
                 color="R2", color_continuous_scale=BLUE_SCALE)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    st.divider()
    st.subheader("Özellik önemleri (en iyi ağaç tabanlı model)")
    tree_models = {n: m for n, (m, s) in fitted_models.items() if hasattr(m, "feature_importances_")}
    if tree_models:
        secim = st.selectbox("Model seçin", list(tree_models.keys()))
        model = tree_models[secim]
        importances = pd.Series(model.feature_importances_, index=[FEATURE_LABELS[f] for f in FEATURES]).sort_values()
        fig = px.bar(importances, orientation="h", color=importances.values, color_continuous_scale=TEAL_SCALE)
        fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="Göreli önem", yaxis_title="")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.divider()
    st.subheader("Referans: ekibin orijinal çalışma sonuçları (rapor)")
    ref = pd.DataFrame({
        "Model": ["CatBoost (Çalışma 1)", "Ridge (Çalışma 1)", "Random Forest (Çalışma 2)",
                  "Gradient Boosting (Çalışma 2)", "Decision Tree (Çalışma 2)", "KNN (Çalışma 2)",
                  "Ridge (Çalışma 2)", "Linear (Çalışma 2)"],
        "R2": [0.560, 0.179, 0.7151, 0.5903, 0.4340, 0.3406, 0.2634, 0.2634],
    })
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.caption(
        "Not: Bu sayfadaki 'canlı eğitim' sonuçları, aynı veri seti ve yöntemle "
        "(zaman bazlı split) yeniden eğitilen modellerden gelir; referans tablo ise "
        "ekibin orijinal notebook'larındaki sonuçlardır. Küçük farklar hiperparametre "
        "ve rastgelelik kaynaklıdır."
    )

# ============================================================
# SAYFA 5: CANLI TAHMİN
# ============================================================
elif page == "Canlı Tahmin":
    _logo_l, _logo_r = st.columns([6, 1])
    with _logo_r:
        st.image("assets/logo_softito.png", width=110)
    st.title("🎯 Canlı Göç Tahmini")
    st.caption("İklim/afet değerlerini girin, model o ülke-yıl için tahmini göç sayısını hesaplasın.")

    model_secenekleri = [n for n in fitted_models.keys()]
    secili_model = st.selectbox(
        "Model", model_secenekleri,
        index=model_secenekleri.index("Random Forest") if "Random Forest" in model_secenekleri else 0,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        yil = st.number_input("Yıl", min_value=2001, max_value=2035, value=2025)
        avg_temp = st.slider("Ortalama sıcaklık (°C)", float(df["Avg_Temp"].min()), float(df["Avg_Temp"].max()), float(df["Avg_Temp"].median()))
        avg_precip = st.slider("Ortalama yağış (mm)", float(df["Avg_Precip"].min()), float(df["Avg_Precip"].max()), float(df["Avg_Precip"].median()))
    with col2:
        kuraklik = st.slider("Kuraklık sayısı", 0, int(df["Kuraklik_Sayisi"].max()), 0)
        sel = st.slider("Sel sayısı", 0, int(df["Sel_Sayisi"].max()), 0)
        firtina = st.slider("Fırtına sayısı", 0, int(df["Firtina_Sayisi"].max()), 0)
        asiri_sicaklik = st.slider("Aşırı sıcaklık sayısı", 0, int(df["Asiri_Sicaklik_Sayisi"].max()), 0)
    with col3:
        olum = st.number_input("Toplam ölüm", min_value=0, value=0, step=10)
        etkilenen = st.number_input("Toplam etkilenen", min_value=0, value=0, step=1000)
        hasar = st.number_input("Toplam hasar (1000 USD)", min_value=0, value=0, step=1000)

    afet_sayisi = kuraklik + sel + firtina + asiri_sicaklik

    girdi = pd.DataFrame([{
        "Yil": yil, "Avg_Temp": avg_temp, "Avg_Precip": avg_precip, "Afet_Sayisi": afet_sayisi,
        "Kuraklik_Sayisi": kuraklik, "Sel_Sayisi": sel, "Firtina_Sayisi": firtina,
        "Asiri_Sicaklik_Sayisi": asiri_sicaklik, "log_Toplam_Olum": np.log1p(olum),
        "log_Toplam_Etkilenen": np.log1p(etkilenen), "log_Toplam_Hasar": np.log1p(hasar),
    }])[FEATURES]

    if st.button("Tahmin et", type="primary"):
        model, needs_scale = fitted_models[secili_model]
        X_use = scaler.transform(girdi) if needs_scale else girdi
        pred_log = model.predict(X_use)[0]
        pred = np.expm1(pred_log)
        pred = max(pred, 0)

        st.divider()
        st.metric(f"{secili_model} - Tahmini sınır ötesi göç", f"{pred:,.0f} kişi".replace(",", "."))

        ortalama = df["Goc"].mean()
        oran = pred / ortalama if ortalama else 0
        st.caption(
            f"Bu tahmin, veri setindeki ortalama göç değerinin ({ortalama:,.0f} kişi) "
            f"yaklaşık {oran:.1f} katıdır.".replace(",", ".")
        )

        st.divider()
        st.subheader("📏 Model Kıyası")
        st.caption("Aynı girdi değerleri için tüm modellerin tahminleri.")

        kiyas_satirlari = []
        for isim, (m, needs_scale_i) in fitted_models.items():
            Xi = scaler.transform(girdi) if needs_scale_i else girdi
            p_log = m.predict(Xi)[0]
            p = max(np.expm1(p_log), 0)
            kiyas_satirlari.append({"Model": isim, "Tahmini Göç": p})

        kiyas_df = pd.DataFrame(kiyas_satirlari).sort_values("Tahmini Göç", ascending=True)
        renkler = [ORANGE if m == secili_model else BLUE for m in kiyas_df["Model"]]

        fig = go.Figure(go.Bar(
            x=kiyas_df["Tahmini Göç"], y=kiyas_df["Model"], orientation="h",
            marker=dict(color=renkler),
            text=[f"{v:,.0f}".replace(",", ".") for v in kiyas_df["Tahmini Göç"]],
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="Tahmini Göç (kişi)", yaxis_title="")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        st.caption(
            f"Turuncu çubuk seçtiğin modeli ({secili_model}) gösterir. Modeller arası "
            "farklar, veri ilişkisinin doğrusal olmayan yapısından ve modellerin farklı "
            "öğrenme mantığından kaynaklanır (bkz. Modelleme Sonuçları sayfası)."
        )

    st.divider()
    st.markdown(
        '<div class="warn-box">Bu araç eğitim/gösterim amaçlıdır. Model, sadece 48 ülke ve '
        '2001-2025 verisiyle eğitilmiştir; sosyoekonomik değişkenler (nüfus, GSYİH vb.) '
        'içermez ve savaş/kriz gibi göç üzerinde çok büyük etkisi olan faktörleri '
        'yakalayamaz. Tahminler kesin değil, referans amaçlıdır.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# SAYFA 6: SONUÇ VE ÖNERİLER
# ============================================================
elif page == "Sonuç ve Öneriler":
    _logo_l, _logo_r = st.columns([6, 1])
    with _logo_r:
        st.image("assets/logo_softito.png", width=110)
    st.markdown(
        """
        <div class="app-hero">
        <h1>📋 Sonuç ve Öneriler</h1>
        <p>Hipotez ve modelleme bulgularının özeti, pratik öneriler ve veri kaynakları.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Hipotez Sonuçları Özeti")
    hipotez_ozet = pd.DataFrame({
        "Hipotez": [
            "H1 - Kuraklık arttıkça göç artar",
            "H5 - Kuraklığın etkisi dolaylı (mediation)",
            "H7 - Sel, kuraklıktan daha güçlü etkiliyor",
            "H20 - Çok değişkenli model tekliden iyi tahmin eder",
        ],
        "Sonuç": ["Desteklenmedi", "Kısmi destek", "Destekli", "Destekli"],
    })
    render_soft_table(hipotez_ozet)

    st.divider()
    st.subheader("Modelleme Özeti")
    st.markdown(
        "İki bağımsız modelleme çalışması, göç değişkeninin iklim/afet değişkenleriyle "
        "**kısmen** açıklanabildiğini, ama tek başına yeterli olmadığını gösteriyor. En iyi "
        "performansı ağaç tabanlı modeller veriyor (Random Forest R² = 0.72, CatBoost R² = "
        "0.56); doğrusal modeller (Linear/Ridge, R² ≈ 0.18-0.26) ilişkinin doğrusal olmayan "
        "yapısını yeterince yakalayamıyor."
    )

    st.divider()
    st.subheader("Öneriler")
    st.markdown(
        """
        <div class="note-box">
        <b>1. Sosyoekonomik değişkenler eklenmeli:</b> Nüfus, GSYİH, yönetişim endeksleri gibi
        değişkenlerin dahil edilmesi model açıklama gücünü artırabilir.<br><br>
        <b>2. Sel odaklı erken uyarı sistemlerine öncelik verilmeli:</b> Sel, kuraklıktan daha
        güçlü bir göç tetikleyicisi olarak öne çıkıyor.<br><br>
        <b>3. "Hapsolmuş nüfus" (trapped population) etkisi izlenmeli:</b> Kuraklığın göçü
        azaltıcı bir etkisi olabileceğinden, sadece göç sayılarına bakmak yetersiz kalabilir;
        yerinden edilme (IDP) verisiyle birlikte değerlendirilmeli.<br><br>
        <b>4. İç göç (IDP) ayrı bir çalışma konusu olarak ele alınmalı:</b> Bu proje yalnızca
        sınır ötesi göçe odaklandı; iç yerinden edilme dinamikleri farklı sonuçlar verebilir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("📚 Kaynakça / Veri Kaynakları")
    st.markdown(
        "- **Göç verisi:** UNHCR (United Nations High Commissioner for Refugees) - ülkeler "
        "arası (bilateral) iltica/göç istatistikleri, 2001-2025.\n"
        "- **Afet verisi:** EM-DAT - The International Disaster Database (CRED) - kuraklık, "
        "sel, fırtına ve aşırı sıcaklık olayları.\n"
        "- **İklim verisi:** Ülke-yıl bazında ortalama sıcaklık ve yağış kayıtları.\n"
        "- Literatür taramasında kullanılan akademik/kurumsal kaynakların tam listesi proje "
        "raporunun Kaynakça bölümünde yer almaktadır."
    )

# ============================================================
# SAYFA 7: PROJE EKİBİ
# ============================================================
elif page == "Proje Ekibi":
    _logo_l, _logo_r = st.columns([6, 1])
    with _logo_r:
        st.image("assets/logo_softito.png", width=110)
    st.markdown(
        """
        <div class="app-hero">
        <h1>👥 Proje Ekibi</h1>
        <p>Bu projeyi birlikte hazırlayan 6 kişilik ekip ve görev dağılımı.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for isim, gorev, aciklama in TEAM:
        st.markdown(
            f"""
            <div class="team-card">
                <div class="name">{isim}</div>
                <div class="role">{gorev}</div>
                <div class="desc">{aciklama}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
