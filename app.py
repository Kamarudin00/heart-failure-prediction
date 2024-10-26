import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Get absolute path
BASE_DIR = Path(__file__).resolve().parent

# Set page configuration
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="centered"
)

# Function to load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model_path = BASE_DIR / 'models' / 'heart_failure_model.pkl'
        scaler_path = BASE_DIR / 'models' / 'scaler.pkl'
        
        # Debug info
        st.sidebar.write("Current directory:", BASE_DIR)
        st.sidebar.write("Model path:", model_path)
        st.sidebar.write("Files in current directory:", os.listdir(BASE_DIR))
        
        if os.path.exists(BASE_DIR / 'models'):
            st.sidebar.write("Files in models directory:", 
                           os.listdir(BASE_DIR / 'models'))
        
        # Check if files exist
        if not model_path.exists():
            st.error(f"Model file not found at: {model_path}")
            return None, None
        
        if not scaler_path.exists():
            st.error(f"Scaler file not found at: {scaler_path}")
            return None, None
            
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
        
    except Exception as e:
        st.error(f"Error loading model and scaler: {str(e)}")
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

# Check if model and scaler loaded successfully
if model is None or scaler is None:
    st.error("Failed to load the model or scaler. Please check if the files exist.")
    st.stop()

# UI elements
st.title("Heart Failure Prediction")
st.write("Enter patient information to predict heart failure risk")

# Create input form
with st.form("prediction_form"):
    # Input fields
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate", min_value=0, max_value=300, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0)
    slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
    
    submit_button = st.form_submit_button("Predict")

if submit_button:
    try:
        # Prepare input data
        sex_encoded = 1 if sex == "Male" else 0
        cp_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        fbs_encoded = 1 if fbs == "Yes" else 0
        restecg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
        exang_encoded = 1 if exang == "Yes" else 0
        slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope)
        
        # Create input array
        input_data = np.array([[
            age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded,
            restecg_encoded, thalach, exang_encoded, oldpeak, slope_encoded
        ]])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        # Show results
        if prediction[0] == 1:
            st.error(f"High risk of heart failure (Probability: {probability[0][1]:.2%})")
        else:
            st.success(f"Low risk of heart failure (Probability: {probability[0][0]:.2%})")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add explanatory information
st.markdown("""
### About this predictor
This tool uses machine learning to predict the risk of heart failure based on various health indicators.
Please note that this is not a substitute for professional medical advice.
""")
