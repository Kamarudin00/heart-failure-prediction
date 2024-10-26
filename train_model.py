import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pickle

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load and prepare the data
def load_data():
    try:
        df = pd.read_csv('data/heart.csv')
        return df
    except FileNotFoundError:
        print("Error: heart.csv not found in data directory!")
        return None

def train_model():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    try:
        # Save with joblib
        joblib.dump(model, 'models/heart_failure_model.pkl', compress=9)
        joblib.dump(scaler, 'models/scaler.pkl', compress=9)
        
        # Also save with pickle as backup
        with open('models/heart_failure_model_pickle.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/scaler_pickle.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print("Model and scaler saved successfully!")
        
        # Print model performance
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    train_model()
