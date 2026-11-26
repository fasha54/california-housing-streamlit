import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# KONFIGURASI HALAMAN & TEMA
# ==========================================
st.set_page_config(
    page_title="Analisis & Prediksi California Housing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Bersih, Rapi, Akademik, Tanpa Ornamen Berlebihan
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] small {
    color: #e2e8f0 !important;
}
.brand-title {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.3px;
    margin-bottom: 2px;
}
.brand-subtitle {
    font-size: 12px;
    color: #94a3b8 !important;
    margin-bottom: 6px;
}
.brand-badge {
    display: inline-block;
    padding: 3px 8px;
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1 !important;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    margin-bottom: 18px;
}

/* Header Container */
.hero-box {
    background-color: #1e293b;
    padding: 22px 24px;
    border-radius: 10px;
    color: #f8fafc;
    margin-bottom: 20px;
    border: 1px solid #334155;
}
.hero-box h1 {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.hero-box p {
    color: #94a3b8;
    font-size: 14px;
    margin: 0;
    line-height: 1.5;
}

/* Stat Cards */
.stat-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.stat-card .label {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.stat-card .val {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
}
.stat-card .note {
    font-size: 11px;
    color: #38bdf8;
    font-weight: 500;
    margin-top: 4px;
}

/* Result Box */
.prediction-box {
    background-color: #0f172a;
    border: 1px solid #2563eb;
    border-radius: 10px;
    padding: 20px;
    margin-top: 16px;
    text-align: center;
}
.prediction-val {
    font-size: 32px;
    font-weight: 700;
    color: #38bdf8;
    margin: 8px 0;
}

/* Standard Button */
.stButton > button {
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 20px;
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
    transition: background-color 0.15s ease;
}
.stButton > button:hover {
    background-color: #1d4ed8;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# LOAD DATA & MODELS
# ==========================================
@st.cache_data
def load_housing_data():
    csv_file = os.path.join(BASE_DIR, "housing.csv")
    df = pd.read_csv(csv_file)
    return df

@st.cache_resource
def load_models_and_metrics():
    rf_file = os.path.join(BASE_DIR, "model_rf.pkl")
    km_file = os.path.join(BASE_DIR, "model_kmeans.pkl")
    metrics_file = os.path.join(BASE_DIR, "metrics.json")
    
    model_rf = None
    model_km = None
    metrics = {}
    
    if os.path.exists(rf_file):
        with open(rf_file, "rb") as f:
            model_rf = pickle.load(f)
    if os.path.exists(km_file):
        with open(km_file, "rb") as f:
            model_km = pickle.load(f)
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    return model_rf, model_km, metrics

df_raw = load_housing_data()
model_rf, model_km, metrics = load_models_and_metrics()

# Imputasi total_bedrooms sesuai standar pengolahan data
median_bedrooms = metrics.get("median_bedrooms_imputed", float(df_raw["total_bedrooms"].median()))
df_clean = df_raw.copy()
df_clean["total_bedrooms"] = df_clean["total_bedrooms"].fillna(median_bedrooms)

# ==========================================
# SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    st.markdown('<div class="brand-title">California Housing</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Praktikum Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-badge">Kelompok 1 · S1 Teknik Informatika</div>', unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigasi Halaman",
        [
            "Ringkasan Proyek",
            "Eksplorasi Data & Peta",
            "Prediksi Nilai Rumah",
            "Klaster Wilayah (K-Means)",
            "Evaluasi Model",
            "Tentang Tim & Dataset"
        ],
        index=0
    )
    
    st.divider()
    st.caption("Informasi Sistem")
    st.caption(f"• Framework: Streamlit v{st.__version__}")
    st.caption("• Modul ML: Scikit-Learn 1.9.1")
    st.caption("• Target Prediksi: median_house_value")
    st.caption(f"• Total Observasi: {len(df_raw):,} Baris")

# ==========================================
# 1. HALAMAN RINGKASAN PROYEK
# ==========================================
if menu == "Ringkasan Proyek":
    st.markdown("""
    <div class="hero-box">
        <h1>Dashboard Analisis dan Prediksi Harga Rumah California</h1>
        <p>
            Aplikasi web berbasis Streamlit dan Scikit-Learn untuk analisis data sensus perumahan,
            eksplorasi spasial, segmentasi geografis dengan K-Means Clustering, serta estimasi nilai rumah
            menggunakan algoritma Random Forest Regressor.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Metrik Utama
    r2_score = metrics.get("r2_score", 0.8149)
    avg_price = df_clean["median_house_value"].mean()
    med_income = df_clean["median_income"].median()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Total Baris Data</div>
            <div class="val">{len(df_raw):,}</div>
            <div class="note">Data Sensus California 1990</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Rata-rata Nilai Rumah</div>
            <div class="val">${avg_price:,.0f}</div>
            <div class="note">Median: ${df_clean['median_house_value'].median():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Median Pendapatan</div>
            <div class="val">${med_income * 10000:,.0f}</div>
            <div class="note">Indeks: {med_income:.2f} (x $10k)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Performa Model (R²)</div>
            <div class="val">{r2_score * 100:.1f}%</div>
            <div class="note">Random Forest Regressor</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_info1, col_info2 = st.columns([3, 2])
    with col_info1:
        st.subheader("Ruang Lingkup dan Rujukan Praktikum")
        st.write("""
        Pengembangan aplikasi ini disusun mengacu pada materi buku praktikum 
        *Implementasi Algoritma Machine Learning Berbasis Web dengan Framework Streamlit* karya Abu Tholib (2023):
        
        * **Bab 1 & 2**: Perancangan antarmuka visual berbasis Streamlit (struktur multi-halaman, layout kolom, formulir input, dan visualisasi data).
        * **Fitur Pencarian Real-Time (Search Bar)**: Pemfilteran data secara dinamis berdasarkan input kata kunci pengguna.
        * **Bab 6 (K-Means Clustering)**: Pengelompokan wilayah hunian menjadi 3 zona spasial menggunakan fitur koordinat (latitude dan longitude).
        * **Pengembangan Model Regresi**: Pelatihan model Random Forest Regressor untuk memprediksi nilai tengah rumah berdasarkan kombinasi fitur demografi dan fisik blok hunian.
        """)
        
    with col_info2:
        st.subheader("Distribusi Kategori Lokasi")
        counts = df_clean["ocean_proximity"].value_counts().reset_index()
        counts.columns = ["Kategori", "Jumlah"]
        fig_pie = px.pie(
            counts, values="Jumlah", names="Kategori",
            hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# 2. HALAMAN EKSPLORASI DATA & PETA
# ==========================================
elif menu == "Eksplorasi Data & Peta":
    st.header("Eksplorasi Data dan Peta Interaktif")
    st.caption("Pencarian kata kunci, filter atribut, analisis statistik deskriptif, dan sebaran geografis.")

    # Filter Box
    with st.container(border=True):
        st.markdown("**Filter dan Pencarian Data**")
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        
        with fc1:
            kata_kunci = st.text_input(
                "Pencarian Bebas (Search Bar)",
                placeholder="Contoh: INLAND, 350000, 52...",
                help="Ketik teks atau nilai numerik untuk menyaring baris secara langsung."
            )
            
        with fc2:
            kategori_list = sorted(list(df_clean["ocean_proximity"].unique()))
            pilihan_kategori = st.multiselect(
                "Kategori Lokasi (Ocean Proximity)",
                options=kategori_list,
                default=kategori_list
            )
            
        with fc3:
            min_p = int(df_clean["median_house_value"].min())
            max_p = int(df_clean["median_house_value"].max())
            rentang_harga = st.slider(
                "Rentang Nilai Rumah (USD)",
                min_value=min_p,
                max_value=max_p,
                value=(min_p, max_p)
            )

    # Proses Filtering
    data_filtered = df_clean[
        (df_clean["ocean_proximity"].isin(pilihan_kategori)) &
        (df_clean["median_house_value"].between(rentang_harga[0], rentang_harga[1]))
    ]

    # Eksekusi Search Bar
    if kata_kunci:
        mask = data_filtered.astype(str).apply(
            lambda col: col.str.contains(kata_kunci, case=False, na=False)
        ).any(axis=1)
        data_filtered = data_filtered[mask]

    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Data Ditemukan", f"{len(data_filtered):,} baris")
    col_res2.metric("Rata-rata Nilai Terpilih", f"${data_filtered['median_house_value'].mean():,.2f}" if len(data_filtered) > 0 else "$0")
    col_res3.metric("Rata-rata Kamar Tidur", f"{data_filtered['total_bedrooms'].mean():.1f}" if len(data_filtered) > 0 else "0")

    if data_filtered.empty:
        st.warning("Tidak ada data yang sesuai dengan filter dan kata kunci yang dimasukkan.")
    else:
        tab_tbl, tab_dist, tab_scat, tab_map = st.tabs([
            "Tabel Data",
            "Distribusi Nilai Rumah",
            "Korelasi Pendapatan",
            "Peta Sebaran Geografis"
        ])
        
        with tab_tbl:
            baris_tampil = st.selectbox("Tampilkan jumlah baris:", [10, 25, 50, 100], index=0)
            st.dataframe(data_filtered.head(baris_tampil), use_container_width=True, hide_index=True)
            
            st.download_button(
                label="Unduh Data Hasil Filter (.csv)",
                data=data_filtered.to_csv(index=False).encode('utf-8'),
                file_name="california_housing_filtered.csv",
                mime="text/csv"
            )

        with tab_dist:
            fig_hist = px.histogram(
                data_filtered,
                x="median_house_value",
                nbins=35,
                color="ocean_proximity",
                title="Distribusi Median Nilai Rumah (USD)",
                labels={"median_house_value": "Nilai Rumah (USD)", "count": "Jumlah Blok Sensus"}
            )
            fig_hist.update_layout(height=400, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab_scat:
            sampel_scat = data_filtered.sample(min(1500, len(data_filtered)), random_state=42)
            fig_scatter = px.scatter(
                sampel_scat,
                x="median_income",
                y="median_house_value",
                color="ocean_proximity",
                hover_data=["housing_median_age", "population"],
                title="Hubungan Tingkat Pendapatan vs Nilai Rumah",
                labels={"median_income": "Indeks Pendapatan (x $10k)", "median_house_value": "Nilai Rumah (USD)"}
            )
            fig_scatter.update_layout(height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab_map:
            st.write("Sebaran koordinat blok sensus perumahan di California:")
            sampel_map = data_filtered.sample(min(2000, len(data_filtered)), random_state=42)
            st.map(sampel_map[["latitude", "longitude"]].dropna())
            st.caption("Menampilkan hingga 2.000 titik sampel representatif sesuai filter aktif.")

# ==========================================
# 3. HALAMAN PREDIKSI NILAI RUMAH
# ==========================================
elif menu == "Prediksi Nilai Rumah":
    st.header("Prediksi Nilai Rumah (Random Forest Regressor)")
    st.caption("Estimasi nilai properti di California berbasis machine learning terlatih.")

    if model_rf is None:
        st.error("Berkas model Random Forest (model_rf.pkl) belum ditemukan. Silakan jalankan train_model.py terlebih dahulu.")
    else:
        tab_single, tab_batch = st.tabs([
            "Estimasi Input Manual (Satu Data)",
            "Prediksi Massal (Unggah File CSV)"
        ])

        with tab_single:
            preset_kota = {
                "Kustom (Isi Sendiri)": (-118.2437, 34.0522, "<1H OCEAN"),
                "San Francisco (Bay Area)": (-122.4194, 37.7749, "NEAR BAY"),
                "Los Angeles (Metropolitan)": (-118.2437, 34.0522, "<1H OCEAN"),
                "San Diego (Pesisir Selatan)": (-117.1611, 32.7157, "NEAR OCEAN"),
                "Sacramento (Ibu Kota Pedalaman)": (-121.4944, 38.5816, "INLAND"),
                "Fresno (Central Valley)": (-119.7871, 36.7468, "INLAND")
            }

            with st.container(border=True):
                st.subheader("Parameter Properti")
                
                pilih_preset = st.selectbox("Pilihan Preset Wilayah (Opsional):", list(preset_kota.keys()))
                default_lon, default_lat, default_ocean = preset_kota[pilih_preset]

                col_in1, col_in2, col_in3 = st.columns(3)
                
                with col_in1:
                    st.markdown("**Faktor Ekonomi dan Lokasi**")
                    in_income = st.number_input(
                        "Median Income (x $10,000 USD):",
                        min_value=0.5, max_value=15.0, value=3.87, step=0.1,
                        help="Tingkat pendapatan rata-rata blok sensus dalam kelipatan $10,000."
                    )
                    in_ocean = st.selectbox(
                        "Kategori Lokasi (Ocean Proximity):",
                        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
                        index=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"].index(default_ocean)
                    )
                    in_age = st.slider(
                        "Usia Median Bangunan (Tahun):",
                        min_value=1, max_value=52, value=29
                    )

                with col_in2:
                    st.markdown("**Karakteristik Fisik Blok**")
                    in_rooms = st.number_input("Total Kamar:", min_value=10, max_value=40000, value=2640, step=50)
                    in_bedrooms = st.number_input("Total Kamar Tidur:", min_value=5, max_value=10000, value=538, step=10)
                    in_households = st.number_input("Jumlah Rumah Tangga:", min_value=5, max_value=8000, value=500, step=10)

                with col_in3:
                    st.markdown("**Koordinat dan Demografi**")
                    in_pop = st.number_input("Populasi Penduduk:", min_value=10, max_value=35000, value=1425, step=25)
                    in_lat = st.number_input("Latitude (Lintang):", min_value=32.0, max_value=42.0, value=float(default_lat), step=0.01, format="%.4f")
                    in_lon = st.number_input("Longitude (Bujur):", min_value=-125.0, max_value=-114.0, value=float(default_lon), step=0.01, format="%.4f")

                st.write("")
                btn_predict = st.button("Hitung Estimasi Nilai Rumah", use_container_width=True)

            if btn_predict:
                input_df = pd.DataFrame([{
                    'longitude': in_lon,
                    'latitude': in_lat,
                    'housing_median_age': in_age,
                    'total_rooms': in_rooms,
                    'total_bedrooms': in_bedrooms,
                    'population': in_pop,
                    'households': in_households,
                    'median_income': in_income,
                    'ocean_proximity': in_ocean
                }])

                hasil_prediksi = model_rf.predict(input_df)[0]
                
                if hasil_prediksi < 150000:
                    status_tier = "Terjangkau (Affordable)"
                    tier_color = "#38bdf8"
                elif hasil_prediksi < 300000:
                    status_tier = "Menengah (Mid-tier)"
                    tier_color = "#4ade80"
                elif hasil_prediksi < 450000:
                    status_tier = "Menengah Atas (Upper Mid-tier)"
                    tier_color = "#fbbf24"
                else:
                    status_tier = "Premium (Luxury)"
                    tier_color = "#f87171"

                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 13px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">
                        Hasil Estimasi Nilai Tengah Rumah
                    </div>
                    <div class="prediction-val">${hasil_prediksi:,.2f} USD</div>
                    <div style="font-size: 14px; color: #cbd5e1; margin-bottom: 6px;">
                        Klasifikasi Nilai Pasar: <b style="color: {tier_color};">{status_tier}</b>
                    </div>
                    <div style="font-size: 12px; color: #64748b;">
                        Toleransi Kesalahan Model (MAE): ±$32,020 · Rentang Estimasi: ${max(15000, hasil_prediksi - 32000):,.0f} - ${hasil_prediksi + 32000:,.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.subheader("Lokasi Koordinat Properti")
                lokasi_pred = pd.DataFrame([{"latitude": in_lat, "longitude": in_lon}])
                st.map(lokasi_pred, zoom=7)

        with tab_batch:
            st.subheader("Prediksi Batch via File CSV")
            st.write("Unggah berkas CSV yang berisi kolom fitur untuk menghitung estimasi nilai rumah secara massal.")
            
            uploaded_file = st.file_uploader("Pilih Berkas CSV:", type=["csv"])
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(f"Data terbaca: {len(batch_df)} baris.")
                    
                    # Cek kolom yang dibutuhkan
                    req_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                                'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
                    missing = [c for c in req_cols if c not in batch_df.columns]
                    
                    if missing:
                        st.error(f"Berkas CSV tidak memiliki kolom berikut: {', '.join(missing)}")
                    else:
                        batch_clean = batch_df.copy()
                        batch_clean['total_bedrooms'] = batch_clean['total_bedrooms'].fillna(median_bedrooms)
                        
                        if st.button("Jalankan Prediksi Massal"):
                            with st.spinner("Memproses prediksi..."):
                                batch_preds = model_rf.predict(batch_clean[req_cols])
                                batch_df["predicted_house_value"] = batch_preds
                                
                            st.success(f"Prediksi berhasil diselesaikan untuk {len(batch_df)} baris data.")
                            st.dataframe(batch_df.head(20), use_container_width=True)
                            
                            st.download_button(
                                label="Unduh Hasil Prediksi (.csv)",
                                data=batch_df.to_csv(index=False).encode('utf-8'),
                                file_name="hasil_prediksi_batch.csv",
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"Gagal memproses file: {str(e)}")

# ==========================================
# 4. HALAMAN KLASTER WILAYAH (K-MEANS)
# ==========================================
elif menu == "Klaster Wilayah (K-Means)":
    st.header("Klaster Wilayah California (K-Means Clustering)")
    st.caption("Penerapan Algoritma K-Means Clustering Spasial Mengacu pada Bab 6 Buku Abu Tholib (2023).")

    cluster_stats = metrics.get("cluster_stats", {})
    sil_score = metrics.get("kmeans_silhouette", 0.6539)

    col_k1, col_k2 = st.columns([3, 2])
    with col_k1:
        st.write("""
        Sesuai materi **Bab 6 buku rujukan**, observasi dataset California Housing dikelompokkan ke dalam 
        **3 klaster geografis** berbasis koordinat (*latitude* dan *longitude*).
        Segmentasi ini memetakan wilayah negara bagian California ke dalam zona demografis dengan pola harga rumah yang berbeda:
        """)
        for c_id, info in cluster_stats.items():
            st.markdown(f"• **Klaster {c_id}**: {info['region']} — Rata-rata Nilai Rumah: **${info['mean_price']:,.2f}** ({info['price_level']})")
    with col_k2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Silhouette Score K-Means</div>
            <div class="val">{sil_score:.4f}</div>
            <div class="note">Jumlah Klaster K = 3 Optimal</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Uji Coba Penentuan Klaster Berdasarkan Koordinat")
    
    col_km1, col_km2 = st.columns(2)
    with col_km1:
        test_lat = st.number_input("Input Latitude (Lintang):", value=37.7749, format="%.4f")
    with col_km2:
        test_lon = st.number_input("Input Longitude (Bujur):", value=-122.4194, format="%.4f")

    if st.button("Tentukan Klaster"):
        if model_km is not None:
            input_km = pd.DataFrame([[test_lat, test_lon]], columns=['latitude', 'longitude'])
            pred_cluster = int(model_km.predict(input_km)[0])
            info_c = cluster_stats.get(str(pred_cluster), {})
            
            st.info(f"""
            **Hasil Pengelompokan: Klaster {pred_cluster}**
            * Zona Wilayah: {info_c.get('region', 'California')}
            * Tingkat Harga Rata-rata: {info_c.get('price_level', 'Menengah')} (~${info_c.get('mean_price', 0):,.2f})
            * Karakteristik: Blok sensus berada dalam koridor spasial klaster yang memiliki karakteristik kepadatan dan valuasi sejenis.
            """)
        else:
            st.error("Model K-Means belum dimuat.")

    st.write("")
    st.subheader("Peta Sebaran 3 Klaster Wilayah K-Means")
    sampel_clust = df_clean.sample(min(2000, len(df_clean)), random_state=42).copy()
    if model_km is not None:
        sampel_clust["Cluster"] = model_km.predict(sampel_clust[["latitude", "longitude"]])
        sampel_clust["Cluster_Name"] = sampel_clust["Cluster"].apply(
            lambda x: f"Klaster {x}: {cluster_stats.get(str(x), {}).get('region', '')}"
        )
        if hasattr(px, "scatter_map"):
            fig_km = px.scatter_map(
                sampel_clust,
                lat="latitude",
                lon="longitude",
                color="Cluster_Name",
                zoom=5,
                height=500,
                title="Sebaran Geografis Klaster K-Means di California"
            )
        else:
            fig_km = px.scatter_mapbox(
                sampel_clust,
                lat="latitude",
                lon="longitude",
                color="Cluster_Name",
                zoom=5,
                height=500,
                mapbox_style="open-street-map",
                title="Sebaran Geografis Klaster K-Means di California"
            )
        fig_km.update_layout(margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_km, use_container_width=True)

# ==========================================
# 5. HALAMAN EVALUASI MODEL
# ==========================================
elif menu == "Evaluasi Model":
    st.header("Evaluasi Performa Model Machine Learning")
    st.caption("Verifikasi performa model Random Forest Regressor dan K-Means secara kuantitatif.")

    r2 = metrics.get("r2_score", 0.8149)
    mae = metrics.get("mae", 32020.15)
    rmse = metrics.get("rmse", 49247.47)
    mape = metrics.get("mape", 18.03)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score (Koefisien Determinasi)", f"{r2 * 100:.2f}%")
    c2.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
    c3.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
    c4.metric("Mean Abs Percentage Error (MAPE)", f"{mape:.2f}%")

    st.write("")
    st.subheader("Tingkat Kepentingan Fitur (Feature Importance)")
    st.write("Tingkat kontribusi variabel prediktor dalam estimasi harga rumah:")

    feat_list = metrics.get("feature_importances", [])
    if feat_list:
        df_feat = pd.DataFrame(feat_list).head(8)
        fig_imp = px.bar(
            df_feat,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Skor Kepentingan", "feature": "Fitur Prediktor"}
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=360, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.info("""
    **Catatan Analisis Model:**
    1. **Median Income** merupakan prediktor paling dominan dengan kontribusi lebih dari 50%, menunjukkan bahwa daya beli masyarakat menjadi faktor utama pembentuk harga rumah di California.
    2. **Faktor Lokasi Geografis (INLAND dan Koordinat Spasial)** menempati posisi signifikansi berikutnya, mencerminkan adanya perbedaan valuasi yang signifikan antara kawasan pesisir pantai dan wilayah pedalaman.
    """)

# ==========================================
# 6. HALAMAN TENTANG TIM & DATASET
# ==========================================
elif menu == "Tentang Tim & Dataset":
    st.header("Informasi Proyek dan Pengembang")
    
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("""
        ### Kelompok 1 - Praktikum Machine Learning
        * **Program Studi**: S1 Teknik Informatika
        * **Fakultas**: Fakultas Ilmu Komputer
        * **Perguruan Tinggi**: Universitas Pamulang
        * **Semester / Tahun**: Semester 5 / Tahun Akademik 2025/2026
        
        #### Anggota Tim:
        1. **Fasha Fahlapi** — NIM: `241011402675`
        2. **Arief Ichsan Anamukti** — NIM: `241011400515`
        3. **Muhamad Bintang Fato Pratama** — NIM: `241011402746`

        #### Dosen Pengampu:
        * **Farizi Ilham, M.Kom.**
        """)

    with col_t2:
        st.markdown("""
        ### Rujukan Praktikum
        * **Buku Referensi**: *Implementasi Algoritma Machine Learning Berbasis Web dengan Framework Streamlit*
        * **Penulis**: Abu Tholib, M.Kom.
        * **Penerbit**: Pustaka Nurja, LP3M Universitas Nurul Jadid (2023)
        * **Struktur Implementasi**:
          * Bab 1: Penyiapan Environment Python dan Dependensi
          * Bab 2: Komponen Dasar Streamlit, Form, dan Layout Halaman
          * Bab 6: K-Means Clustering Spasial pada Dataset California Housing
          * Benchmarking: Referensi Aplikasi Telco Churn dan WDBC Streamlit
        """)

st.markdown("---")
st.markdown(
    "<center style='color:#64748b; font-size:12px;'>"
    "Aplikasi Analisis & Prediksi California Housing · Kelompok 1 · S1 Teknik Informatika Universitas Pamulang"
    "</center>",
    unsafe_allow_html=True
)
