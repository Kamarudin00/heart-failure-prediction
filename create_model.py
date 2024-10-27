import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Buat direktori models jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')

# Load dataset
data = pd.read_csv('heart.csv')

# Pisahkan features dan target
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisasi fitur numerik
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model dan scaler menggunakan pickle dengan protocol=3
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=3)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=3)

print("Model and scaler successfully created and saved!")

# Evaluasi model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nTraining accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")
