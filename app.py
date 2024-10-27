import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from setup import create_model

# Initialize page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        # Create model if it doesn't exist
        if not os.path.exists('models/model.pkl'):
            result = create_model()
            st.info(result)
            if "Failed" in result:
                return None, None, None
        
        # Load model, scaler, and encoders
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, scaler, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load model
model, scaler, encoders = load_model()

if model is None or scaler is None or encoders is None:
    st.error("Failed to load model")
    st.stop()

# UI Components
st.title('Heart Disease Prediction')
st.write('Enter patient information:')

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', 20, 100, 50)
        sex = st.selectbox('Sex', ['M', 'F'])
        chest_pain = st.selectbox('Chest Pain Type', 
                                ['ATA', 'NAP', 'ASY', 'TA'])
        resting_bp = st.number_input('Resting Blood Pressure', 90, 200, 120)
        cholesterol = st.number_input('Cholesterol', 100, 600, 200)
        
    with col2:
        fasting_bs = st.number_input('Fasting Blood Sugar', 0, 1, 0)
        resting_ecg = st.selectbox('Resting ECG', 
                                 ['Normal', 'ST', 'LVH'])
        max_hr = st.number_input('Maximum Heart Rate', 60, 220, 150)
        exercise_angina = st.selectbox('Exercise Angina', ['Y', 'N'])
        oldpeak = st.number_input('ST Depression (Oldpeak)', -3.0, 6.0, 0.0)
        st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input data
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for column in categorical_columns:
        input_df[column] = encoders[column].transform(input_df[column])
    
    # Scale numeric features
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    # Show results
    if prediction == 1:
        st.error(f'High risk of heart disease (Probability: {proba[1]:.2%})')
    else:
        st.success(f'Low risk of heart disease (Probability: {proba[0]:.2%})')
