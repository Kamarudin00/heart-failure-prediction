import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path

def initialize_model():
    # Buat direktori models
    os.makedirs('models', exist_ok=True)
    
    try:
        # Load dataset (pastikan dataset ada di folder yang benar)
        data = pd.DataFrame({
            'age': np.random.randint(30, 80, 1000),
            'sex': np.random.randint(0, 2, 1000),
            'cp': np.random.randint(0, 4, 1000),
            'trestbps': np.random.randint(90, 200, 1000),
            'chol': np.random.randint(120, 400, 1000),
            'fbs': np.random.randint(0, 2, 1000),
            'restecg': np.random.randint(0, 3, 1000),
            'thalach': np.random.randint(70, 200, 1000),
            'exang': np.random.randint(0, 2, 1000),
            'oldpeak': np.random.uniform(0, 6, 1000),
            'slope': np.random.randint(0, 3, 1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        # Split features dan target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Create dan train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model dan scaler
        model_path = Path('models/heart_model.pkl')
        scaler_path = Path('models/scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        print("Model and scaler successfully created and saved!")
        return True
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return False

if __name__ == "__main__":
    initialize_model()
