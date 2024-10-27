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

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        # Check if dataset exists
        if not os.path.exists('heart.csv'):
            st.error("Dataset 'heart.csv' not found!")
            return None, None
        
        # Debug: Show dataset contents
        data = pd.read_csv('heart.csv')
        st.write("Dataset columns:", data.columns.tolist())
        
        # Create model if it doesn't exist
        if not os.path.exists('models/model.pkl'):
            result = create_model()
            st.info(result)
            if "Failed" in result:
                return None, None
        
        # Load model and scaler
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
model, scaler = load_model()

if model is None or scaler is None:
    st.error("Failed to load model")
    st.stop()

# Rest of your app.py code remains the same...
