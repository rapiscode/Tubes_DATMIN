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

st.subheader("Visualisasi Prediksi vs Aktual (Training Set)")

y_pred_train = model.predict(X)

fig, ax = plt.subplots()
ax.scatter(y, y_pred_train, alpha=0.7, edgecolors='k')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Aktual")
ax.set_ylabel("Prediksi")
ax.set_title("Scatter Plot: Aktual vs Prediksi (Kepadatan 2020)")
ax.grid(True)

st.pyplot(fig)


# Input dari user
st.subheader("Input Data Tahun Sebelumnya")
kepadatan_2018 = st.number_input("Kepadatan 2018", min_value=0.0, step=10.0)
kepadatan_2019 = st.number_input("Kepadatan 2019", min_value=0.0, step=10.0)

if st.button("Prediksi Kepadatan 2020"):
    pred = model.predict([[kepadatan_2018, kepadatan_2019]])
    st.success(f"Prediksi Kepadatan Tahun 2020: {pred[0]:,.2f} jiwa/km²")
