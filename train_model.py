import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Buat direktori models jika belum ada
os.makedirs('models', exist_ok=True)

# Load dataset
df = pd.read_csv('data/heart.csv')  # Pastikan file dataset ada

# Pisahkan fitur dan target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, 'models/heart_failure_model.pkl', compress=9)

print("Model berhasil disimpan ke models/heart_failure_model.pkl")
