# create_scaler.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Pastikan folder models ada
if not os.path.exists('models'):
    os.makedirs('models')

# Baca dataset
df = pd.read_csv('heart.csv')

# Pilih kolom numerik yang perlu di-scale
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Buat dan fit scaler
scaler = StandardScaler()
scaler.fit(df[numeric_features])

# Simpan scaler
joblib.dump(scaler, 'models/scaler.pkl')

print("Scaler has been created and saved to models/scaler.pkl")
