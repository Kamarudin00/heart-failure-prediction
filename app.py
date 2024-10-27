import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join('models', 'heart_failure_model.pkl')
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

if model is None:
    st.error("Failed to load model")
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
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Show result
        if prediction[0] == 1:
            st.error(f"High risk of heart disease (Probability: {probability[0][1]:.2%})")
        else:
            st.success(f"Low risk of heart disease (Probability: {probability[0][0]:.2%})")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add information
st.markdown("""
### About this predictor
This tool uses machine learning to predict the risk of heart disease based on various health indicators.
Please note that this is not a substitute for professional medical advice.
""")
