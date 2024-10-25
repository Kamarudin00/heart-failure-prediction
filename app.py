import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Heart Failure Prediction", page_icon="❤️")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model_path = 'models/heart_failure_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    model = joblib.load(model_path)
    return model

# Memuat model
model = load_model()

# Cek apakah model berhasil dimuat
if model is None:
    st.stop()

# Input data dari pengguna
st.title("Heart Failure Prediction")
age = st.number_input("Age")
sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # 0-3
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])  # 0: No, 1: Yes
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])  # 0-2
thalach = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", [0, 1])  # 0: No, 1: Yes
oldpeak = st.number_input("ST Depression Induced by Exercise")
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])  # 0-2
ca = st.number_input("Number of Major Vessels (0-3)")
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])  # 0-3

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
    prediction = model.predict(input_data)
    st.success(f"Prediction: {'Heart Failure' if prediction[0] == 1 else 'No Heart Failure'}")
