import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'asl_model.pkl')

def normalize_landmarks_batch(X):
    """Center landmarks on the wrist (x0, y0, z0) and scale by hand size."""
    N = X.shape[0]
    coords = X.reshape(N, 21, 3).astype(np.float64)
    wrist = coords[:, 0:1, :]
    centered = coords - wrist
    # Scale by max Euclidean distance from wrist so camera distance doesn't matter
    dists = np.linalg.norm(centered, axis=2)
    max_dist = np.max(dists, axis=1, keepdims=True)
    max_dist[max_dist == 0] = 1.0
    normalized = centered / max_dist[:, :, np.newaxis]
    return normalized.reshape(N, 63)

print("Loading dataset.csv...")
try:
    # Load the Kaggle dataset
    df = pd.read_csv(DATASET_PATH)
    
    # Assume the last column is the word (label) and everything else is coordinates
    X_raw = df.iloc[:, :-1].values  
    y_raw = df.iloc[:, -1].values   

    print(f"Dataset loaded successfully! Found {len(df)} samples.")

    # Normalize coordinates so hand position and distance from camera don't matter
    X_norm = normalize_landmarks_batch(X_raw)

    # Augment with horizontally mirrored hands so BOTH left and right hands work
    X_mirrored = X_norm.copy().reshape(-1, 21, 3)
    X_mirrored[:, :, 0] *= -1
    X_mirrored = X_mirrored.reshape(-1, 63)

    X = np.vstack([X_norm, X_mirrored])
    y = np.concatenate([y_raw, y_raw])
    
    # Convert text labels into numbers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print("Training Random Forest ML Model (This takes just a few seconds)...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Check accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model Training Complete! Accuracy on test data: {accuracy * 100:.2f}%")

    # Save the model and the label encoder to a file
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'encoder': encoder, 'normalized': True}, f)
        
    print(f"SUCCESS: '{MODEL_PATH}' has been saved. You can now run app.py!")

except FileNotFoundError:
    print("ERROR: dataset.csv not found. Please download it from Kaggle and put it in this folder.")
except Exception as e:
    print(f"ERROR: Something went wrong reading the CSV. Details: {e}")
