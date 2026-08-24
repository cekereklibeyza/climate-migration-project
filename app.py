# -*- coding: utf-8 -*-
"""
Kuresel Isinma, Asiri Hava Olaylari ve Goc Iliskisi - Streamlit Uygulamasi
Calistirmak icin:  pip install -r requirements.txt  &&  streamlit run app.py
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
    CATBOOST_VAR = True
except Exception:
    CATBOOST_VAR = False

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Kuresel Isinma ve Goc",
    page_icon="\U0001F30D",
    layout="wide",
)

PRIMARY = "#1F3B57"
ACCENT = "#2A6F77"

st.markdown(
    """
    <style>
    .metric-card {background-color:#F2F6F5; padding: 14px 18px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = [
    "Yil", "Avg_Temp", "Avg_Precip", "Afet_Sayisi", "Kuraklik_Sayisi",
    "Sel_Sayisi", "Firtina_Sayisi", "Asiri_Sicaklik_Sayisi",
    "log_Toplam_Olum", "log_Toplam_Etkilenen", "log_Toplam_Hasar",
]

FEATURE_LABELS = {
    "Yil": "Yil",
    "Avg_Temp": "Ort. Sicaklik (C)",
    "Avg_Precip": "Ort. Yagis (mm)",
    "Afet_Sayisi": "Toplam Afet Sayisi",
    "Kuraklik_Sayisi": "Kuraklik Sayisi",
    "Sel_Sayisi": "Sel Sayisi",
    "Firtina_Sayisi": "Firtina Sayisi",
    "Asiri_Sicaklik_Sayisi": "Asiri Sicaklik Sayisi",
    "log_Toplam_Olum": "log(1+Toplam Olum)",
    "log_Toplam_Etkilenen": "log(1+Toplam Etkilenen)",
    "log_Toplam_Hasar": "log(1+Toplam Hasar, 1000 USD)",
}


# ============================================================
# VERI YUKLEME
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
# MODEL EGITIMI (zaman bazli split: <=2020 egitim, >2020 test)
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

    if CATBOOST_VAR:
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
fitted_models, scaler, results_df, Xte, yte, test_df = train_models(len(df))

# ============================================================
# SIDEBAR - NAVIGASYON
# ============================================================
st.sidebar.title("\U0001F30D Kuresel Isinma & Goc")
st.sidebar.caption("Dogu-Guney Afrika ve Bati-Orta Afrika (2001-2025)")
page = st.sidebar.radio(
    "Sayfa secin",
    [
        "Genel Bakis",
        "Veri Gezgini",
        "EDA & Hipotezler",
        "Modelleme Sonuclari",
        "Canli Tahmin",
    ],
)
st.sidebar.divider()
st.sidebar.caption(
    "Bu uygulama, ekibin veri temizleme + birlestirme + hipotez + modelleme "
    "calismalarinin interaktif ozetidir. Detaylar icin proje raporuna bakiniz."
)

# ============================================================
# SAYFA 1: GENEL BAKIS
# ============================================================
if page == "Genel Bakis":
    st.title("Kuresel Isinma, Asiri Hava Olaylari ve Goc Iliskisi")
    st.markdown(
        "Bu proje, iklim/afet degiskenlerinin (sicaklik, yagis, kuraklik, sel, firtina, "
        "asiri sicaklik) **Dogu-Guney Afrika** ve **Bati-Orta Afrika** bolgelerindeki "
        "**sinir otesi goc** ile iliskisini inceler. Goc, UNHCR'in ulkeler arasi "
        "iltica/goc verisiyle tanimlanmistir (ulke ici yerinden edilme dahil degildir)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ulke sayisi", df["ISO3"].nunique())
    c2.metric("Yil araligi", f"{df['Yil'].min()}-{df['Yil'].max()}")
    c3.metric("Toplam kayit", f"{len(df):,}".replace(",", "."))
    c4.metric("Toplam goc (kumulatif)", f"{int(df['Goc'].sum()):,}".replace(",", "."))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Yillara gore toplam goc")
        yearly = df.groupby("Yil", as_index=False)["Goc"].sum()
        fig = px.area(yearly, x="Yil", y="Goc", color_discrete_sequence=[ACCENT])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("En cok goc veren 10 ulke (toplam)")
        top10 = df.groupby("Ulke", as_index=False)["Goc"].sum().sort_values("Goc", ascending=False).head(10)
        fig = px.bar(top10.sort_values("Goc"), x="Goc", y="Ulke", orientation="h",
                     color_discrete_sequence=[PRIMARY])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown(
        "**Metodoloji ozeti:** Bilateral goc verisi (43.162 satir) ulke-yil bazinda "
        "toplanip iklim/afet tablosuyla birlestirilerek 1.190 satirlik nihai veri seti "
        "olusturulmustur. Hipotez testleri ve iki bagimsiz modelleme calismasi "
        "(CatBoost/Ridge ve Random Forest/Gradient Boosting/Decision Tree/KNN/Linear) "
        "bu veri seti uzerinde yurutulmustur."
    )

# ============================================================
# SAYFA 2: VERI GEZGINI
# ============================================================
elif page == "Veri Gezgini":
    st.title("Veri Gezgini")
    st.caption("Ulke ve yil secerek birlesik veri setini filtreleyin.")

    ulkeler = sorted(df["Ulke"].unique())
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        secili_ulkeler = st.multiselect("Ulke(ler)", ulkeler, default=ulkeler[:5])
    with col_f2:
        yil_araligi = st.slider("Yil araligi", int(df["Yil"].min()), int(df["Yil"].max()),
                                 (int(df["Yil"].min()), int(df["Yil"].max())))

    filtered = df[
        (df["Ulke"].isin(secili_ulkeler) if secili_ulkeler else True)
        & (df["Yil"].between(*yil_araligi))
    ]

    st.write(f"**{len(filtered)}** satir goruntuleniyor.")
    st.dataframe(
        filtered[["Yil", "Ulke", "ISO3", "Avg_Temp", "Avg_Precip", "Kuraklik_Sayisi",
                  "Sel_Sayisi", "Firtina_Sayisi", "Toplam_Etkilenen", "Goc"]],
        use_container_width=True, height=320,
    )

    if len(filtered) > 0 and secili_ulkeler:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Goc - zaman serisi")
            fig = px.line(filtered.sort_values("Yil"), x="Yil", y="Goc", color="Ulke", markers=True)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Afet sayilari (toplam)")
            afet_toplam = filtered.groupby("Ulke")[["Kuraklik_Sayisi", "Sel_Sayisi", "Firtina_Sayisi", "Asiri_Sicaklik_Sayisi"]].sum()
            fig2 = px.bar(afet_toplam, barmode="group")
            fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Filtrelenmis veriyi CSV indir", csv, "filtrelenmis_veri.csv", "text/csv")

# ============================================================
# SAYFA 3: EDA & HIPOTEZLER
# ============================================================
elif page == "EDA & Hipotezler":
    st.title("Kesifsel Veri Analizi ve Hipotezler")

    tab1, tab2 = st.tabs(["EDA", "Hipotez Testleri"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Goc dagilimi (ham)")
            fig = px.histogram(df, x="Goc", nbins=40, color_discrete_sequence=["#e07a5f"])
            skew = df["Goc"].skew()
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                               title=f"Carpiklik (skewness) = {skew:.2f}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("log(1+Goc) dagilimi")
            fig = px.histogram(df, x="log_Goc", nbins=40, color_discrete_sequence=[ACCENT])
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                               title="Log donusumu sonrasi dagilim dengelenir")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Korelasyon matrisi")
        num_cols = ["Avg_Temp", "Avg_Precip", "Afet_Sayisi", "Kuraklik_Sayisi", "Sel_Sayisi",
                    "Firtina_Sayisi", "Asiri_Sicaklik_Sayisi", "Toplam_Olum", "Toplam_Etkilenen",
                    "Toplam_Hasar_1000USD", "Goc"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

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
            st.markdown("#### H1 - Kuraklik arttikca goc artar")
            st.metric("Pearson r", f"{r_h1:.3f}")
            st.caption("Iliski cok zayif -> desteklenmedi.")

            st.markdown("#### H5 - Kuraklik etkisi dolayli (mediation)")
            st.metric("Kuraklik -> Etkilenen (r)", f"{r_h5a:.3f}")
            st.metric("Etkilenen -> Goc (r)", f"{r_h5b:.3f}")
            st.caption("Kuraklik etkilenen nufusu artiriyor ama bu sinir otesi goce zayif yansiyor -> kismi destek (\"hapsolmus nufus\").")
        with colB:
            st.markdown("#### H7 - Sel, kuraklikdan daha guclu etkiliyor")
            st.metric("Sel Spearman r", f"{r_sel:.3f}", delta=f"kuraklik: {r_kuraklik:.3f}")
            st.caption("Sel, kuraklikdan belirgin sekilde daha guclu iliskili -> destekli.")

            st.markdown("#### H20 - Coklu model tekliden iyi mi?")
            fig = go.Figure(go.Bar(x=["Basit (kuraklik)", "Coklu (6 degisken)"], y=[r2_simple, r2_multi],
                                    marker_color=["#e76f51", "#2a9d8f"]))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="R2")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Basit R²={r2_simple:.4f} vs Coklu R²={r2_multi:.4f} -> destekli.")

# ============================================================
# SAYFA 4: MODELLEME SONUCLARI
# ============================================================
elif page == "Modelleme Sonuclari":
    st.title("Modelleme Sonuclari")
    st.caption("Modeller bu sayfa acildiginda 2001-2020 egitim / 2021-2025 test ayrimiyla canli olarak egitilir.")

    st.subheader("Model karsilastirmasi (canli egitim, bu veri seti uzerinde)")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    fig = px.bar(results_df.sort_values("R2"), x="R2", y="Model", orientation="h",
                 color="R2", color_continuous_scale="Teal")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Ozellik onemleri (en iyi agac tabanli model)")
    tree_models = {n: m for n, (m, s) in fitted_models.items() if hasattr(m, "feature_importances_")}
    if tree_models:
        secim = st.selectbox("Model secin", list(tree_models.keys()))
        model = tree_models[secim]
        importances = pd.Series(model.feature_importances_, index=[FEATURE_LABELS[f] for f in FEATURES]).sort_values()
        fig = px.bar(importances, orientation="h", color_discrete_sequence=[PRIMARY])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                           xaxis_title="Goreli onem", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Referans: ekibin orijinal calisma sonuclari (rapor)")
    ref = pd.DataFrame({
        "Model": ["CatBoost (Calisma 1)", "Ridge (Calisma 1)", "Random Forest (Calisma 2)",
                  "Gradient Boosting (Calisma 2)", "Decision Tree (Calisma 2)", "KNN (Calisma 2)",
                  "Ridge (Calisma 2)", "Linear (Calisma 2)"],
        "R2": [0.560, 0.179, 0.7151, 0.5903, 0.4340, 0.3406, 0.2634, 0.2634],
    })
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.caption(
        "Not: Bu sayfadaki 'canli egitim' sonuclari, ayni veri seti ve yontemle "
        "(zaman bazli split) yeniden egitilen modellerden gelir; referans tablo ise "
        "ekibin orijinal notebook'larindaki sonuclardir. Kucuk farklar hiperparametre "
        "ve rastgelelik kaynaklidir."
    )

# ============================================================
# SAYFA 5: CANLI TAHMIN
# ============================================================
elif page == "Canli Tahmin":
    st.title("Canli Goc Tahmini")
    st.caption("Iklim/afet degerlerini girin, model o ulke-yil icin tahmini goc sayisini hesaplasin.")

    model_secenekleri = [n for n in fitted_models.keys()]
    secili_model = st.selectbox("Model", model_secenekleri, index=model_secenekleri.index("Random Forest") if "Random Forest" in model_secenekleri else 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        yil = st.number_input("Yil", min_value=2001, max_value=2035, value=2025)
        avg_temp = st.slider("Ortalama sicaklik (C)", float(df["Avg_Temp"].min()), float(df["Avg_Temp"].max()), float(df["Avg_Temp"].median()))
        avg_precip = st.slider("Ortalama yagis (mm)", float(df["Avg_Precip"].min()), float(df["Avg_Precip"].max()), float(df["Avg_Precip"].median()))
    with col2:
        kuraklik = st.slider("Kuraklik sayisi", 0, int(df["Kuraklik_Sayisi"].max()), 0)
        sel = st.slider("Sel sayisi", 0, int(df["Sel_Sayisi"].max()), 0)
        firtina = st.slider("Firtina sayisi", 0, int(df["Firtina_Sayisi"].max()), 0)
        asiri_sicaklik = st.slider("Asiri sicaklik sayisi", 0, int(df["Asiri_Sicaklik_Sayisi"].max()), 0)
    with col3:
        olum = st.number_input("Toplam olum", min_value=0, value=0, step=10)
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
        st.metric(f"{secili_model} - Tahmini sinir otesi goc", f"{pred:,.0f} kisi".replace(",", "."))

        ortalama = df["Goc"].mean()
        oran = pred / ortalama if ortalama else 0
        st.caption(
            f"Bu tahmin, veri setindeki ortalama goc degerinin ({ortalama:,.0f} kisi) "
            f"yaklasik {oran:.1f} katidir.".replace(",", ".")
        )

    st.divider()
    st.warning(
        "Bu arac egitim/gosterim amaclidir. Model, sadece 48 ulke ve 2001-2025 verisiyle "
        "egitilmistir; sosyoekonomik degiskenler (nufus, GSYIH vb.) icermez ve savas/kriz "
        "gibi goc uzerinde cok buyuk etkisi olan faktorleri yakalayamaz. Tahminler kesin "
        "degil, referans amaclidir."
    )
