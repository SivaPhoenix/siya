"""
Retrain the DecisionTree model and save it with the current scikit-learn version.
Run this whenever scikit-learn is upgraded.
"""
import sys
import os

basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, basedir)

import joblib
from ml import train_model

print("Training model with current scikit-learn version...")
model, X = train_model()
print("Training complete.")

out_path = os.path.join(basedir, "model_save.pkl")
joblib.dump(model, out_path)
print(f"Model saved to: {out_path}")
