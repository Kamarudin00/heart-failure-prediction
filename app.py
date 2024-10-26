import streamlit as st
import pandas as pd
import joblib
import os

# Pengaturan konfigurasi halaman
st.set_page_config(page_title="Heart Failure Prediction", page_icon="❤️", layout="centered")

# Fungsi untuk memuat model dengan pengecekan direktori dan file
@st.cache_resource
def load_model():
    model_path = 'models/heart_failure_model.pkl'
    
    # Verifikasi apakah direktori dan file model ada
    if not os.path.exists('models'):
        st.error("Directory 'models' tidak ditemukan. Pastikan direktori yang berisi file model sudah tersedia.")
        st.stop()
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di: {model_path}. Unggah file model yang diperlukan.")
        st.stop()
    
    # Memuat model
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# Memuat model
model = load_model()

# Judul dan Deskripsi
st.title("Heart Failure Prediction")
st.write("Aplikasi ini menggunakan model prediksi untuk memeriksa kemungkinan risiko gagal jantung berdasarkan informasi medis yang diberikan.")

# Input data dari pengguna dengan penjelasan tiap input
age = st.number_input("Age", min_value=1, step=1, help="Masukkan usia dalam tahun.")
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="0: Female, 1: Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="Jenis nyeri dada (0-3)")
trestbps = st.number_input("Resting Blood Pressure", min_value=1, help="Tekanan darah istirahat dalam mmHg.")
chol = st.number_input("Cholesterol", min_value=1, help="Kadar kolesterol dalam mg/dl.")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="0: No, 1: Yes")
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], help="Hasil EKG istirahat (0-2)")
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=1, help="Detak jantung maksimal yang dicapai.")
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="0: No, 1: Yes")
oldpeak = st.number_input("ST Depression Induced by Exercise", help="Depresi ST setelah latihan.")
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2], help="Kemiringan segmen ST puncak (0-2)")
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, step=1, help="Jumlah pembuluh besar utama (0-3).")
thal = st.selectbox("Thalassemia", [0, 1, 2, 3], help="Tipe talasemia (0-3)")

# Buat DataFrame untuk input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Prediksi
if st.button("Predict"):
    if model:
        try:
            prediction = model.predict(input_data)
            result = "Heart Failure" if prediction[0] == 1 else "No Heart Failure"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
    else:
        st.error("Model belum dimuat dengan benar. Harap periksa kembali file model Anda.")
