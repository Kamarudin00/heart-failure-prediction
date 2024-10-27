import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def create_model():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load and prepare data
    data = pd.read_csv('heart_cleveland_upload.csv')  # Updated dataset name
    
    # Check column names
    print("Available columns:", data.columns.tolist())
    
    # Assuming 'condition' is the target column in this dataset
    X = data.drop('condition', axis=1)  # Update column name based on actual target
    y = data['condition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model and scaler
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=4)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f, protocol=4)
    
    return "Model and scaler successfully created and saved!"

if __name__ == "__main__":
    print(create_model())
