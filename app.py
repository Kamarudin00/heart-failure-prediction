import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the saved model
@st.cache_resource
def load_model():
    with open('heart_failure_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Set page configuration
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="centered"
)

# Create title and subtitle
st.title("Heart Failure Prediction System ❤️")
st.write("Enter patient information to predict heart failure risk")

# Create input form
with st.form("prediction_form"):
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=40)
        resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        max_hr = st.number_input("Maximum Heart Rate", min_value=0, max_value=300, value=150)
        
    with col2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        chest_pain_type = st.selectbox("Chest Pain Type", 
                                     ["Typical Angina", "Atypical Angina", 
                                      "Non-Anginal Pain", "Asymptomatic"])
        resting_ecg = st.selectbox("Resting ECG", 
                                 ["Normal", "ST-T Wave Abnormality", 
                                  "Left Ventricular Hypertrophy"])
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Create submit button
    submit_button = st.form_submit_button("Predict")

# When form is submitted
if submit_button:
    # Convert categorical inputs to numerical
    sex_encoded = 1 if sex == "Male" else 0
    fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0
    exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
    
    # Create one-hot encoding for categorical variables
    chest_pain_map = {
        "Typical Angina": [1,0,0,0],
        "Atypical Angina": [0,1,0,0],
        "Non-Anginal Pain": [0,0,1,0],
        "Asymptomatic": [0,0,0,1]
    }
    
    resting_ecg_map = {
        "Normal": [1,0,0],
        "ST-T Wave Abnormality": [0,1,0],
        "Left Ventricular Hypertrophy": [0,0,1]
    }
    
    st_slope_map = {
        "Up": [1,0,0],
        "Flat": [0,1,0],
        "Down": [0,0,1]
    }
    
    # Get encoded values
    chest_pain_encoded = chest_pain_map[chest_pain_type]
    resting_ecg_encoded = resting_ecg_map[resting_ecg]
    st_slope_encoded = st_slope_map[st_slope]
    
    # Create input array for prediction
    input_data = np.array([
        age, sex_encoded, *chest_pain_encoded, resting_bp, cholesterol,
        fasting_bs_encoded, *resting_ecg_encoded, max_hr,
        exercise_angina_encoded, *st_slope_encoded
    ]).reshape(1, -1)
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Show prediction result
        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease (Confidence: {prediction_proba[1]:.2%})")
            st.write("The model predicts that this patient has a high risk of heart disease. Please consult with a healthcare professional for proper medical evaluation.")
        else:
            st.success(f"✅ Low Risk of Heart Disease (Confidence: {prediction_proba[0]:.2%})")
            st.write("The model predicts that this patient has a low risk of heart disease. However, regular health check-ups are still recommended.")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please check if all input values are within expected ranges.")

# Add information about the model
st.write("---")
st.write("### About This Model")
st.write("""
This heart disease prediction model uses machine learning (Random Forest) to assess the risk of heart disease based on various medical indicators. 
The model has been trained on clinical data and achieves an accuracy of over 90%.

**Note:** This prediction should not be used as a substitute for professional medical advice, diagnosis, or treatment.
""")

# Add feature importance plot if available
try:
    feature_importance = pd.DataFrame({
        'feature': ['Age', 'Sex', 'ChestPainType_TA', 'ChestPainType_ATA', 
                   'ChestPainType_NAP', 'ChestPainType_ASY', 'RestingBP', 
                   'Cholesterol', 'FastingBS', 'RestingECG_Normal', 
                   'RestingECG_ST', 'RestingECG_LVH', 'MaxHR', 
                   'ExerciseAngina', 'ST_Slope_Up', 'ST_Slope_Flat', 
                   'ST_Slope_Down'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    st.write("### Feature Importance")
    st.bar_chart(data=feature_importance.set_index('feature'))
except:
    pass
