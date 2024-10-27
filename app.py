import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

def load_pickle(file_path: Path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {str(e)}")
        return None

# Load model dan scaler
MODEL_PATH = Path('models/model.pkl')
SCALER_PATH = Path('models/scaler.pkl')

model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH)

if model is None or scaler is None:
    st.error("Failed to load model or scaler")
    st.stop()

# UI untuk input
st.title('Heart Disease Prediction')
st.write('Enter patient information:')

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=20, max_value=100, value=50)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 
                          'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.number_input('Resting Blood Pressure', 90, 200, 120)
        chol = st.number_input('Cholesterol', 100, 600, 200)
        
    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        restecg = st.selectbox('Resting ECG Results', 
                              ['Normal', 'ST-T Wave Abnormality',
                               'Left Ventricular Hypertrophy'])
        thalach = st.number_input('Maximum Heart Rate', 60, 220, 150)
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.number_input('ST Depression', 0.0, 6.0, 1.0)
        slope = st.selectbox('Slope of Peak Exercise ST', 
                           ['Upsloping', 'Flat', 'Downsloping'])
        
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input data
    data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'cp': ['Typical Angina', 'Atypical Angina', 
               'Non-anginal Pain', 'Asymptomatic'].index(cp),
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == 'Yes' else 0,
        'restecg': ['Normal', 'ST-T Wave Abnormality',
                    'Left Ventricular Hypertrophy'].index(restecg),
        'thalach': thalach,
        'exang': 1 if exang == 'Yes' else 0,
        'oldpeak': oldpeak,
        'slope': ['Upsloping', 'Flat', 'Downsloping'].index(slope)
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([data])
    
    # Scale numeric features
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    # Display result
    if prediction == 1:
        st.error(f'High risk of heart disease (Probability: {proba[1]:.2%})')
    else:
        st.success(f'Low risk of heart disease (Probability: {proba[0]:.2%})')

# Add information about the model
st.markdown("""
### About this predictor
This tool uses a Random Forest model to predict heart disease risk based on various health indicators.
Please note that this is not a substitute for professional medical advice.
""")
