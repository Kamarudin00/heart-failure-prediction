import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the saved model
with open('heart_failure_model.pkl', 'rb') as file:
    model = pickle.load(file)

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
        anaemia = st.selectbox("Anaemia", ["No", "Yes"])
        creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (CPK)", min_value=0, value=100)
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=60)
        high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])
        
    with col2:
        platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, value=250000.0)
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.0)
        serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0, value=140)
        sex = st.selectbox("Sex", ["Female", "Male"])
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        time = st.number_input("Follow-up Period (days)", min_value=0, value=30)

    # Create submit button
    submit_button = st.form_submit_button("Predict")

# When form is submitted
if submit_button:
    # Convert categorical inputs to numerical
    anaemia = 1 if anaemia == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
    sex = 1 if sex == "Male" else 0
    smoking = 1 if smoking == "Yes" else 0
    
    # Create input array for prediction
    input_data = np.array([
        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        high_blood_pressure, platelets, serum_creatinine, serum_sodium,
        sex, smoking, time
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Show prediction result
    st.write("---")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Failure")
        st.write("The model predicts that this patient has a high risk of heart failure. Please consult with a healthcare professional for proper medical evaluation.")
    else:
        st.success("✅ Low Risk of Heart Failure")
        st.write("The model predicts that this patient has a low risk of heart failure. However, regular health check-ups are still recommended.")
    
    # Show input data as table
    st.write("### Patient Information")
    input_df = pd.DataFrame({
        'Feature': [
            'Age', 'Anaemia', 'Creatinine Phosphokinase', 'Diabetes',
            'Ejection Fraction', 'High Blood Pressure', 'Platelets',
            'Serum Creatinine', 'Serum Sodium', 'Sex', 'Smoking',
            'Follow-up Period'
        ],
        'Value': [
            age, "Yes" if anaemia else "No", creatinine_phosphokinase,
            "Yes" if diabetes else "No", ejection_fraction,
            "Yes" if high_blood_pressure else "No", platelets,
            serum_creatinine, serum_sodium,
            "Male" if sex else "Female", "Yes" if smoking else "No", time
        ]
    })
    st.table(input_df)

# Add information about the model
st.write("---")
st.write("### About This Model")
st.write("""
This heart failure prediction model uses machine learning to assess the risk of heart failure based on various medical indicators. 
The prediction should not be used as a substitute for professional medical advice, diagnosis, or treatment.
""")
