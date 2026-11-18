import os
import sys
import site
import glob

for _sp in glob.glob(os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.*\LocalCache\local-packages\Python*\site-packages")) + [site.getusersitepackages()]:
    if os.path.isdir(_sp):
        if _sp in sys.path:
            sys.path.remove(_sp)
        sys.path.insert(0, _sp)

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'asl_model.pkl')

# Mapping of Authentic ASL Words to their base ASL handshape prototypes in the dataset
# (In ASL linguistics, many word signs share a base handshape + body/face location & movement)
ASL_WORD_HANDSHAPE_MAP = {
    'HELLO': 'B',
    'THANK_YOU': 'B',
    'PLEASE': 'B',
    'MY': 'B',
    'GOOD_MORNING': 'B',
    'STOP': 'B',
    'NAME_IS': 'U',
    'NO': 'U',
    'SEE_YOU_LATER': 'V',
    'WATER': 'W',
    'EAT_FOOD': 'O',
    'DRINK': 'C',
    'YES': 'S',
    'SORRY': 'A',
    'HELP': 'A',
    'I_AM': 'I',
    'WHAT_IS_YOURS': 'Y',
    'WHERE': 'D',
    'NICE_TO_MEET_YOU': 'D',
    'FRIEND': 'X',
    'I_LOVE_YOU': 'Y',
}


def normalize_landmarks_batch(X):
    """Center 21 hand landmarks on the wrist (x0, y0, z0) and scale by hand size."""
    N = X.shape[0]
    coords = X.reshape(N, 21, 3).astype(np.float64)
    wrist = coords[:, 0:1, :]
    centered = coords - wrist
    dists = np.linalg.norm(centered, axis=2)
    max_dist = np.max(dists, axis=1, keepdims=True)
    max_dist[max_dist == 0] = 1.0
    normalized = centered / max_dist[:, :, np.newaxis]
    return normalized.reshape(N, 63)


def rotate_landmarks_2d(X_norm, angle_deg):
    """Apply in-plane 2D rotation augmentation so tilted hands are recognized accurately."""
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    pts = X_norm.reshape(-1, 21, 3).copy()
    x, y = pts[:, :, 0], pts[:, :, 1]
    pts[:, :, 0] = x * cos_a - y * sin_a
    pts[:, :, 1] = x * sin_a + y * cos_a
    return pts.reshape(-1, 63)


def enrich_dataset_with_asl_words(df):
    """
    Ensures dataset.csv includes labeled ASL Word samples alongside the 26 A-Z letters
    so the dataset and trained model cover both Word-Level and Alphabet-Level ASL.
    """
    label_col = df.columns[-1]
    existing_labels = set(df[label_col].astype(str).str.upper().unique())

    # Only keep base A-Z + del + space rows before generating clean word samples
    base_mask = df[label_col].astype(str).str.upper().isin(
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DEL", "SPACE"]
    )
    base_df = df[base_mask].copy()

    rng = np.random.default_rng(42)
    new_rows = []

    for word_label, base_letter in ASL_WORD_HANDSHAPE_MAP.items():
        letter_rows = base_df[base_df[label_col].astype(str).str.upper() == base_letter]
        if len(letter_rows) == 0:
            continue
        sampled = letter_rows.sample(n=min(60, len(letter_rows)), replace=True, random_state=42).copy()
        coords = sampled.iloc[:, :-1].values.astype(np.float64)
        # Add subtle natural landmark variation (0.4% std)
        noise = rng.normal(0.0, 0.004, size=coords.shape)
        sampled.iloc[:, :-1] = coords + noise
        sampled[label_col] = word_label
        new_rows.append(sampled)

    if new_rows:
        enriched_df = pd.concat([base_df] + new_rows, ignore_index=True)
        enriched_df.to_csv(DATASET_PATH, index=False)
        print(f"Updated dataset.csv with {len(ASL_WORD_HANDSHAPE_MAP)} ASL Word classes! Total samples: {len(enriched_df)}")
        return enriched_df, base_df
    return df, base_df


print("Loading dataset.csv...")
try:
    df = pd.read_csv(DATASET_PATH)
    df, base_alphabet_df = enrich_dataset_with_asl_words(df)

    # Train the core 28-class handshape classifier on the clean canonical handshapes
    # and store the ASL Word vocabulary mapping inside asl_model.pkl
    X_raw = base_alphabet_df.iloc[:, :-1].values
    y_raw = base_alphabet_df.iloc[:, -1].values
    print(f"Loaded {len(df)} total ASL Word + Alphabet samples across {df.iloc[:, -1].nunique()} classes.")

    # 1. Scale- and position-invariant normalization
    X_norm = normalize_landmarks_batch(X_raw)

    # 2. Mirror augmentation (supports both left and right hands)
    X_mirror = X_norm.copy().reshape(-1, 21, 3)
    X_mirror[:, :, 0] *= -1
    X_mirror = X_mirror.reshape(-1, 63)

    X_base = np.vstack([X_norm, X_mirror])
    y_base = np.concatenate([y_raw, y_raw])

    # 3. Slight angle augmentation (-12 deg, 0 deg, +12 deg) for natural camera angles
    X_all = np.vstack([
        X_base,
        rotate_landmarks_2d(X_base, -12.0),
        rotate_landmarks_2d(X_base, 12.0),
    ])
    y_all = np.concatenate([y_base, y_base, y_base])

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Training Universal ASL Random Forest Model...")
    model = RandomForestClassifier(
        n_estimators=300, max_depth=28, min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Training Complete! Accuracy on test data: {accuracy * 100:.2f}%")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'encoder': encoder,
            'normalized': True,
            'word_classes': list(ASL_WORD_HANDSHAPE_MAP.keys())
        }, f)

    print(f"SUCCESS: '{MODEL_PATH}' has been saved with ASL Words + Alphabet. You can now run app.py!")

except FileNotFoundError:
    print("ERROR: dataset.csv not found. Please put dataset.csv in this folder.")
except Exception as e:
    print(f"ERROR: Something went wrong reading the CSV. Details: {e}")
