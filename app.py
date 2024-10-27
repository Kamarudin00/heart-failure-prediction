import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'heart_model.pkl'
SCALER_PATH = BASE_DIR / 'models' / 'scaler.pkl'

# Configure page
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Function to safely load pickle files
def safe_load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {str(e)}")
        return None

# Load model and scaler with proper error handling
@st.cache_resource
def load_model_and_scaler():
    # Check if files exist
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None, None
        
    if not SCALER_PATH.exists():
        st.error(f"Scaler file not found at: {SCALER_PATH}")
        return None, None
    
    # Load files
    model = safe_load_pickle(MODEL_PATH)
    scaler = safe_load_pickle(SCALER_PATH)
    
    if model is None or scaler is None:
        return None, None
        
    return model, scaler

# Add debug information to sidebar
def show_debug_info():
    st.sidebar.write("### Debug Information")
    st.sidebar.write("Current directory:", BASE_DIR)
    st.sidebar.write("Model path:", MODEL_PATH)
    st.sidebar.write("Scaler path:", SCALER_PATH)
    
    if MODEL_PATH.exists():
        st.sidebar.write("Model file exists")
    else:
        st.sidebar.write("Model file missing!")
        
    if SCALER_PATH.exists():
        st.sidebar.write("Scaler file exists")
    else:
        st.sidebar.write("Scaler file missing!")

# Show debug info
show_debug_info()

# Load model and scaler
model, scaler = load_model_and_scaler()

# Check if model and scaler loaded successfully
if model is None or scaler is None:
    st.error("Failed to load model or scaler. Please check the files and try again.")
    st.stop()

# UI elements
st.title("Heart Disease Prediction")
st.write("Enter patient information to predict heart disease risk")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", 
                          "Atypical Angina",
                          "Non-anginal Pain",
                          "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                 min_value=90, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)",
                             min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        
    with col2:
        restecg = st.selectbox("Resting ECG Results",
                             ["Normal",
                              "ST-T Wave Abnormality",
                              "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate",
                                min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression",
                                min_value=0.0, max_value=6.0, value=0.0)
        slope = st.selectbox("Slope of Peak Exercise ST",
                           ["Upsloping", "Flat", "Downsloping"])
    
    submit = st.form_submit_button("Predict")

if submit:
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
                       "Left Ventricular Hypert rophy"].index(restecg),
            'thalach': thalach,
            'exang': 1 if exang == "Yes" else 0,
            'oldpeak': oldpeak,
            'slope': ["Upsloping", "Flat", "Downsloping"].index(slope)
        }
        
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Scale input data
        scaled_input = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)
        
        # Show results
        if prediction[0] == 1:
            st.error(f"High risk of heart disease (Probability: {probability[0][1]:.2%})")
        else:
            st.success(f"Low risk of heart disease (Probability: {probability[0][0]:.2%})")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add explanatory information
st.markdown("""
### About this predictor
This tool uses machine learning to predict the risk of heart disease based on various health indicators.
Please note that this is not a substitute for professional medical advice.
""")
