import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard Prediksi Kepadatan Penduduk",
)

# Judul
st.title("Prediksi Kepadatan Penduduk Jawa Barat")
st.write("Model ini memprediksi kepadatan penduduk untuk tahun berikutnya berdasarkan data sebelumnya dan luas wilayah.")

model = joblib.load('modelprediksi.pkl')

# Load Dataset
df = pd.read_csv("Jumlah_penduduk_menurut_kabupaten_kota_2018_2020.csv")
dd_2 = pd.read_csv("Luas_daerah_menurut_kabupaten_kota_jawabarat_2020.csv")

# Preprocessing
dd_2.drop(["Jumlah Pulau"], axis=1, inplace=True)
df_final = df.merge(dd_2, left_on="Wilayah Jawa Barat", right_on="Kabupaten/Kota")
df_final.drop(columns=["Kabupaten/Kota"], inplace=True)

df_final["Kepadatan_2018"] = df_final["2018"] / df_final["Luas Wilayah (Km2)"]
df_final["Kepadatan_2019"] = df_final["2019"] / df_final["Luas Wilayah (Km2)"]
df_final["Kepadatan_2020"] = df_final["2020"] / df_final["Luas Wilayah (Km2)"]

# Fitting model untuk demo
X = df_final[["Kepadatan_2018", "Kepadatan_2019"]]
y = df_final["Kepadatan_2020"]
model = LinearRegression()
model.fit(X, y)

st.subheader("Visualisasi Kepadatan Penduduk per Tahun (Bar Chart)")

# Pilih tahun (2018 atau 2019)
tahun_pilihan = st.selectbox("Pilih Tahun", [2018, 2019], key="tahun_bar_chart")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_final["Wilayah Jawa Barat"], df_final[f"Kepadatan_{tahun_pilihan}"])
ax.set_title(f"Kepadatan Penduduk Tahun {tahun_pilihan}")
ax.set_ylabel("Kepadatan (jiwa/km²)")
ax.set_xticks(np.arange(len(df_final["Wilayah Jawa Barat"])))
ax.set_xticklabels(df_final["Wilayah Jawa Barat"], rotation=90)
ax.grid(True)
st.pyplot(fig)

st.subheader("Tabel Kategori Kepadatan Penduduk")

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
    ]
}

df_kategori = pd.DataFrame(data_kategori)
st.table(df_kategori)


# Input dari user
st.subheader("Input Data Tahun Sebelumnya")
kepadatan_2018 = st.number_input("Kepadatan 2018", min_value=0.0, step=10.0)
kepadatan_2019 = st.number_input("Kepadatan 2019", min_value=0.0, step=10.0)

if st.button("Prediksi Kepadatan 2020"):
    pred = model.predict([[kepadatan_2018, kepadatan_2019]])
    nilai_prediksi = pred[0]

    # Tentukan label kategori
    if nilai_prediksi < 1000:
        kategori = "Sangat Rendah"
    elif nilai_prediksi < 5000:
        kategori = "Rendah"
    elif nilai_prediksi < 10000:
        kategori = "Sedang"
    elif nilai_prediksi < 15000:
        kategori = "Tinggi"
    else:
        kategori = "Sangat Tinggi"

    st.success(f"Prediksi Kepadatan Tahun 2020: {nilai_prediksi:,.2f} jiwa/km²")
    st.info(f"Kategori Kepadatan: **{kategori}**")

# Tabel Prediksi vs Aktual
df_final["Prediksi_2020"] = model.predict(df_final[["Kepadatan_2018", "Kepadatan_2019"]])
df_final["Selisih_Error"] = df_final["Prediksi_2020"] - df_final["Kepadatan_2020"]

st.subheader("Tabel Prediksi vs Aktual")
st.dataframe(df_final[["Wilayah Jawa Barat", "Kepadatan_2020", "Prediksi_2020", "Selisih_Error"]].round(2))

# Bar Chart Perbandingan
st.subheader("Visualisasi Prediksi vs Aktual per Wilayah")
fig, ax = plt.subplots(figsize=(12, 5))
width = 0.4
x = np.arange(len(df_final["Wilayah Jawa Barat"]))
ax.bar(x - width/2, df_final["Kepadatan_2020"], width=width, label="Aktual")
ax.bar(x + width/2, df_final["Prediksi_2020"], width=width, label="Prediksi")
ax.set_xticks(x)
ax.set_xticklabels(df_final["Wilayah Jawa Barat"], rotation=90)
ax.set_ylabel("Kepadatan (jiwa/km²)")
ax.legend()
st.pyplot(fig)

# Clustering Visualisasi
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

st.subheader("Clustering Wilayah Berdasarkan Kepadatan per Tahun")

# Dropdown untuk memilih tahun
tahun_klaster = st.selectbox("Pilih Tahun untuk Clustering", [2018, 2019, 2020], key="tahun_clustering")

# Lakukan clustering berdasarkan tahun yang dipilih
X_cluster_single = df_final[[f"Kepadatan_{tahun_klaster}"]]
X_scaled_single = StandardScaler().fit_transform(X_cluster_single)

kmeans_single = KMeans(n_clusters=5, random_state=42)
df_final["Cluster_Tahun"] = kmeans_single.fit_predict(X_scaled_single)

# Tampilkan hasil cluster per wilayah
st.write("Hasil Cluster per Wilayah:")
st.dataframe(df_final[["Wilayah Jawa Barat", f"Kepadatan_{tahun_klaster}", "Cluster_Tahun"]].sort_values("Cluster_Tahun").reset_index(drop=True))

# Visualisasi distribusi cluster
fig, ax = plt.subplots()
sns.countplot(x="Cluster_Tahun", data=df_final, ax=ax)
ax.set_title(f"Distribusi Cluster KMeans berdasarkan Kepadatan Tahun {tahun_klaster}")
st.pyplot(fig)
