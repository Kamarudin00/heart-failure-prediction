import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Heart Failure Prediction", page_icon="❤️")

# Fungsi untuk memuat model dengan verifikasi direktori
@st.cache_resource
def load_model():
    model_path = 'models/heart_failure_model.pkl'
    
    # Verifikasi apakah direktori dan file model ada
    if not os.path.exists('models'):
        st.error("Directory 'models' tidak ditemukan.")
        st.stop()
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    
    # Memuat model
    model = joblib.load(model_path)
    return model

# Memuat model
model = load_model()

# Input data dari pengguna
st.title("Heart Failure Prediction")
age = st.number_input("Age", min_value=1, step=1)
sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # 0-3
trestbps = st.number_input("Resting Blood Pressure", min_value=1)
chol = st.number_input("Cholesterol", min_value=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])  # 0: No, 1: Yes
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])  # 0-2
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=1)
exang = st.selectbox("Exercise Induced Angina", [0, 1])  # 0: No, 1: Yes
oldpeak = st.number_input("ST Depression Induced by Exercise")
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])  # 0-2
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, step=1)
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
