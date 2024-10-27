import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from init_model import initialize_model

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# Define paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'heart_model.pkl'
SCALER_PATH = BASE_DIR / 'models' / 'scaler.pkl'

# Initialize model if not exists
if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.info("Initializing model for first use...")
    success = initialize_model()
    if not success:
        st.error("Failed to initialize model!")
        st.stop()
    st.success("Model initialized successfully!")
    st.experimental_rerun()

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("Failed to load model or scaler")
    st.stop()

# Create the UI
st.title("Heart Disease Prediction")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", 
                          "Atypical Angina",
                          "Non-anginal Pain",
                          "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure", 90, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", 
                             ["Normal",
                              "ST-T Wave Abnormality",
                              "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0)
        slope = st.selectbox("Slope of Peak Exercise ST",
                           ["Upsloping", "Flat", "Downsloping"])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Prepare input data
        input_data = {
            'age': age,
            'sex': 1 if sex == "Male" else 0,
            'cp': ["Typical Angina", "Atypical Angina", 
                   "Non-anginal Pain", "Asymptomatic"].index(cp),
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if fbs == "Yes" else 0,
            'restecg': ["Normal", "ST-T Wave Abnormality",
                       "Left Ventricular Hypertrophy"].index(restecg),
            'thalach': thalach,
            'exang': 1 if exang == "Yes" else 0,
            'oldpeak': oldpeak,
            'slope': ["Upsloping", "Flat", "Downsloping"].index(slope)
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        # Show result
        if prediction[0] == 1:
            st.error(f"High risk of heart disease (Probability: {probability[0][1]:.2%})")
        else:
            st.success(f"Low risk of heart disease (Probability: {probability[0][0]:.2%})")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add information
st.markdown("""
### About this predictor This tool uses machine learning to predict the risk of heart disease based on various health indicators.
Please note that this is not a substitute for professional medical advice.
""")
