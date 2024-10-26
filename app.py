import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Debug information in sidebar
with st.sidebar:
    st.write("### Debug Information")
    st.write("Python version:", sys.version)
    st.write("Current working directory:", os.getcwd())
    st.write("Directory contents:", os.listdir())

@st.cache_resource
def load_model():
    try:
        # Get the absolute path to the model file
        model_path = os.path.join('models', 'heart_failure_model.pkl')
        logger.info(f"Attempting to load model from: {model_path}")
        
        # Debug information
        st.sidebar.write(f"Looking for model at: {model_path}")
        st.sidebar.write(f"File exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            st.error(f"Model file not found at: {model_path}")
            return None
            
        # Load model using joblib
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        st.write("Error details:", str(e))
        return None

# Load model
model = load_model()

# Check if model loaded successfully
if model is None:
    st.error("Failed to load the model. Please check if the model file exists.")
    st.stop()

# Create title and subtitle
st.title("Heart Failure Prediction System ❤️")
st.markdown("""
This application uses machine learning to predict the risk of heart failure based on various medical indicators.
Please fill in the patient's information below.
""")

# Create input form
with st.form("prediction_form"):
    st.write("### Patient Information")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 
                            min_value=0, 
                            max_value=100, 
                            value=40,
                            help="Patient's age in years")
        
        sex = st.selectbox("Sex", 
                          ["Female", "Male"],
                          help="Patient's biological sex")
        
        chest_pain_type = st.selectbox("Chest Pain Type", 
                                     ["Typical Angina", 
                                      "Atypical Angina", 
                                      "Non-Anginal Pain", 
                                      "Asymptomatic"],
                                     help="Type of chest pain experienced")
        
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 
                                   min_value=0, 
                                   max_value=300, 
                                   value=120,
                                   help="Resting blood pressure in mm Hg")
    
    with col2:
        cholesterol = st.number_input("Cholesterol (mg/dl)", 
                                    min_value=0, 
                                    max_value=600, 
                                    value=200,
                                    help="Serum cholesterol in mg/dl")
        
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                                ["No", "Yes"],
                                help="Whether fasting blood sugar > 120 mg/dl")
        
        resting_ecg = st.selectbox("Resting ECG", 
                                 ["Normal", 
                                  "ST-T Wave Abnormality", 
                                  "Left Ventricular Hypertrophy"],
                                 help="Resting electrocardiographic results")
        
        max_hr = st.number_input("Maximum Heart Rate", 
                               min_value=0, 
                               max_value=300, 
                               value=150,
                               help="Maximum heart rate achieved")
    
    with col3:
        exercise_angina = st.selectbox("Exercise-Induced Angina", 
                                     ["No", "Yes"],
                                     help="Exercise-induced angina")
        
        oldpeak = st.number_input("ST Depression (Oldpeak)", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=0.0,
                                help="ST depression induced by exercise relative to rest")
        
        st_slope = st.selectbox("ST Slope", 
                              ["Up", "Flat", "Down"],
                              help="Slope of the peak exercise ST segment")

    # Create submit button
    submit_button = st.form_submit_button("Predict Heart Failure Risk")

# When form is submitted
if submit_button:
    try:
        # Convert categorical inputs to numerical
        sex_encoded = 1 if sex == "Male" else 0
        
        chest_pain_map = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-Anginal Pain": 2,
            "Asymptomatic": 3
        }
        chest_pain_encoded = chest_pain_map[chest_pain_type]
        
        fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0
        
        resting_ecg_map = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }
        resting_ecg_encoded = resting_ecg_map[resting_ecg]
        
        exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
        
        st_slope_map = {
            "Up": 0,
            "Flat": 1,
            "Down": 2
        }
        st_slope_encoded = st_slope_map[st_slope]
        
        # Create input array for prediction
        input_data = np.array([
            age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol,
            fasting_bs_encoded, resting_ecg_encoded, max_hr,
            exercise_angina_encoded, oldpeak, st_slope_encoded
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Show prediction result
        st.write("---")
        st.write("### Prediction Result")
        
        # Create a container for the prediction
        with st.container():
            if prediction == 1:
                st.error("⚠️ High Risk of Heart Failure")
                st.write(f"Probability of heart failure: {prediction_proba[1]:.2%}")
                st.markdown("""
                    **Recommendation:**
                     - Consult a doctor immediately.
                     - Follow a healthy lifestyle, including a balanced diet and regular exercise.
                     - Manage stress through relaxation techniques like meditation or yoga.
                """)
            else:
                st.success("👍 Low Risk of Heart Failure")
                st.write(f"Probability of heart failure: {prediction_proba[1]:.2%}")
                st.markdown("""
                    **Recommendation:**
                     - Continue following a healthy lifestyle.
                     - Schedule regular check-ups with your doctor.
                     - Consider getting screened for heart disease.
                """)
        
    except Exception as e:
        st.error("Error making prediction")
        st.write("Error details:", str(e))
