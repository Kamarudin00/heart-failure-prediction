import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def create_model():
    """Create and save the model"""
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Load data
        data = pd.read_csv('heart.csv')
        
        # Create label encoders for categorical columns
        categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        label_encoders = {}
        
        # Encode categorical variables
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])
        
        # Separate features and target
        X = data.drop('HeartDisease', axis=1)
        y = data['HeartDisease']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale numeric features
        numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model, scaler, and label encoders
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        with open('models/encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        
        # Calculate and return accuracy
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
        accuracy = model.score(X_test, y_test)
        return f"Model created successfully! Accuracy: {accuracy:.2%}"
    
    except Exception as e:
        print(f"Error creating model: {e}")
        return f"Failed to create model: {str(e)}"

if __name__ == "__main__":
    print(create_model())
