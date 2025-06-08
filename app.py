import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Konfigurasi Halaman
# ---------------------------
st.set_page_config(page_title="Dashboard Prediksi Kepadatan Penduduk")

st.title("📊 Prediksi dan Analisis Kepadatan Penduduk Jawa Barat")
st.write("Aplikasi ini memprediksi dan menganalisis kepadatan penduduk berdasarkan data tahun sebelumnya dan luas wilayah.")

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("Jumlah_penduduk_menurut_kabupaten_kota_2018_2020.csv")
dd_2 = pd.read_csv("Luas_daerah_menurut_kabupaten_kota_jawabarat_2020.csv")
dd_2.drop(["Jumlah Pulau"], axis=1, inplace=True)

# Merge dan Preprocessing
df_final = df.merge(dd_2, left_on="Wilayah Jawa Barat", right_on="Kabupaten/Kota")
df_final.drop(columns=["Kabupaten/Kota"], inplace=True)
df_final["Kepadatan_2018"] = df_final["2018"] / df_final["Luas Wilayah (Km2)"]
df_final["Kepadatan_2019"] = df_final["2019"] / df_final["Luas Wilayah (Km2)"]
df_final["Kepadatan_2020"] = df_final["2020"] / df_final["Luas Wilayah (Km2)"]

def klasifikasi_kepadatan(nilai):
    if nilai < 1000:
        return "Sangat Rendah"
    elif nilai < 5000:
        return "Rendah"
    elif nilai < 10000:
        return "Sedang"
    elif nilai < 15000:
        return "Tinggi"
    else:
        return "Sangat Tinggi"

# ---------------------------
# Train Model Linear Regression
# ---------------------------
X = df_final[["Kepadatan_2018", "Kepadatan_2019"]]
y = df_final["Kepadatan_2020"]
model = LinearRegression()
model.fit(X, y)

# ---------------------------
# Visualisasi Bar Chart Tiap Tahun
# ---------------------------
st.subheader("📌 Visualisasi Kepadatan Penduduk per Tahun")
tahun_pilihan = st.selectbox("Pilih Tahun", [2018, 2019], key="tahun_bar_chart")

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(df_final["Wilayah Jawa Barat"], df_final[f"Kepadatan_{tahun_pilihan}"])
ax1.set_title(f"Kepadatan Penduduk Tahun {tahun_pilihan}")
ax1.set_ylabel("Kepadatan (jiwa/km²)")
ax1.set_xticks(np.arange(len(df_final)))
ax1.set_xticklabels(df_final["Wilayah Jawa Barat"], rotation=90)
ax1.grid(True)
st.pyplot(fig1)

# ---------------------------
# Kategori Kepadatan
# ---------------------------
st.subheader("📌 Kategori Kepadatan Penduduk")
data_kategori = {
    "Rentang Kepadatan (jiwa/km²)": [
        "< 1.000",
        "1.000 – < 5.000",
        "5.000 – < 10.000",
        "10.000 – < 15.000",
        "≥ 15.000"
    ],
    "Kategori": [
        "Sangat Rendah",
        "Rendah",
        "Sedang",
        "Tinggi",
        "Sangat Tinggi"
    ],
    "Contoh Nilai": [
        "800",
        "3.200",
        "7.500",
        "12.000",
        "15.700"
    ]
}
st.table(pd.DataFrame(data_kategori))


# ---------------------------
# Input dan Prediksi
# ---------------------------
st.subheader("📌 Prediksi Kepadatan Tahun 2020")
kepadatan_2018 = st.number_input("Masukkan Kepadatan 2018", min_value=0.0, step=10.0)
kepadatan_2019 = st.number_input("Masukkan Kepadatan 2019", min_value=0.0, step=10.0)

if st.button("Prediksi Sekarang"):
    prediksi = model.predict([[kepadatan_2018, kepadatan_2019]])[0]
    
    if prediksi < 1000:
        kategori = "Sangat Rendah"
    elif prediksi < 5000:
        kategori = "Rendah"
    elif prediksi < 10000:
        kategori = "Sedang"
    elif prediksi < 15000:
        kategori = "Tinggi"
    else:
        kategori = "Sangat Tinggi"

    st.success(f"Prediksi Kepadatan Tahun 2020: {prediksi:,.2f} jiwa/km²")
    st.info(f"Kategori Kepadatan: **{kategori}**")

# ---------------------------
# Tabel Prediksi vs Aktual
# ---------------------------
df_final["Prediksi_2020"] = model.predict(df_final[["Kepadatan_2018", "Kepadatan_2019"]])
df_final["Selisih_Error"] = df_final["Prediksi_2020"] - df_final["Kepadatan_2020"]

st.subheader("📌 Tabel Prediksi vs Aktual")
st.dataframe(df_final[["Wilayah Jawa Barat", "Kepadatan_2020", "Prediksi_2020", "Selisih_Error"]].round(2))

# ---------------------------
# Visualisasi Prediksi vs Aktual
# ---------------------------
st.subheader("📌 Visualisasi Perbandingan Prediksi vs Aktual")
fig2, ax2 = plt.subplots(figsize=(12, 5))
x = np.arange(len(df_final))
width = 0.4
ax2.bar(x - width/2, df_final["Kepadatan_2020"], width=width, label="Aktual")
ax2.bar(x + width/2, df_final["Prediksi_2020"], width=width, label="Prediksi")
ax2.set_xticks(x)
ax2.set_xticklabels(df_final["Wilayah Jawa Barat"], rotation=90)
ax2.set_ylabel("Kepadatan (jiwa/km²)")
ax2.legend()
st.pyplot(fig2)

# ---------------------------
# Clustering Berdasarkan Tahun
# ---------------------------
st.subheader("📌 Clustering Wilayah Berdasarkan Kepadatan")

tahun_klaster = st.selectbox("Pilih Tahun untuk Clustering", [2018, 2019, 2020], key="tahun_clustering")
X_cluster = df_final[[f"Kepadatan_{tahun_klaster}"]]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

kmeans = KMeans(n_clusters=5, random_state=42)
df_final["Cluster_Tahun"] = kmeans.fit_predict(X_cluster_scaled)

# Terapkan kategori berdasarkan nilai kepadatan, bukan cluster label
df_final[f"Kategori_{tahun_klaster}"] = df_final[f"Kepadatan_{tahun_klaster}"].apply(klasifikasi_kepadatan)

# Tampilkan hasil
st.write("📋 Hasil Cluster dan Kategori per Wilayah:")
st.dataframe(
    df_final[["Wilayah Jawa Barat", f"Kepadatan_{tahun_klaster}", f"Kategori_{tahun_klaster}"]]
    .sort_values(f"Kepadatan_{tahun_klaster}")
    .reset_index(drop=True)
)

# Visualisasi Cluster
fig3, ax3 = plt.subplots()
sns.countplot(x=f"Kategori_{tahun_klaster}", data=df_final, ax=ax3,
              order=["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"])
ax3.set_title(f"Distribusi Kategori Kepadatan - Tahun {tahun_klaster}")
ax3.set_xlabel("Kategori Kepadatan")
st.pyplot(fig3)
