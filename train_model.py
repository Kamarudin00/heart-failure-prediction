import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path

def create_and_save_model():
    # Buat direktori models jika belum ada
    os.makedirs('models', exist_ok=True)
    
    try:
        # Load dataset
        data = pd.read_csv('data/heart.csv')
        
        # Prepare features dan target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create dan train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Simpan model menggunakan pickle
        model_path = Path('models/heart_model.pkl')
        scaler_path = Path('models/scaler.pkl')
        
        # Simpan model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Simpan scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return False

if __name__ == "__main__":
    success = create_and_save_model()
    if success:
        print("Model creation and saving completed successfully!")
    else:
        print("Failed to create and save model!")
