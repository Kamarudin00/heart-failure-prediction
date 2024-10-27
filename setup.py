import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def create_model():
    """Create and save the model"""
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Load and prepare data
        data = pd.read_csv('heart.csv')
        
        # Debugging: Print column names and first few rows
        print("Available columns:", data.columns.tolist())
        print("\nFirst few rows of data:")
        print(data.head())
        
        # Check if 'condition' exists instead of 'target'
        target_column = 'condition' if 'condition' in data.columns else 'HeartDisease' if 'HeartDisease' in data.columns else 'target'
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {data.columns.tolist()}")
        
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model and scaler
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f, protocol=4)
        
        # Print success message with model accuracy
        accuracy = model.score(X_test, y_test)
        return f"Model and scaler successfully created and saved! Model accuracy: {accuracy:.2%}"
    
    except Exception as e:
        print(f"Error creating model: {e}")
        return f"Failed to create model: {str(e)}"

if __name__ == "__main__":
    print(create_model())
