import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tsseg.algorithms.vqtss import VQTSSDetector
from tsseg.metrics import Covering, AdjustedRandIndex, StateMatchingScore
from tsseg.algorithms.utils import extract_cps

def load_mocap_sample():
    # Load a sample from the local mocap folder
    mocap_dir = Path(__file__).resolve().parent.parent.parent / "data" / "mocap"
    files = sorted(list(mocap_dir.glob("*.csv")))
    if not files:
        raise FileNotFoundError("No Mocap CSV files found.")
    
    # Pick the first one, e.g., 86_01.csv
    file_path = files[0]
    print(f"Loading {file_path}...")
    
    df = pd.read_csv(file_path)
    
    print("Columns:", df.columns)
    
    if 'label' in df.columns:
        y = df['label'].values
        X = df.drop(columns=['label']).values
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
        
    return X, y

def evaluate_vqtss():
    X, y = load_mocap_sample()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Number of true segments: {len(extract_cps(y)) + 1}")
    print(f"Number of true states: {len(np.unique(y))}")
    
    # Configure VQTSS
    # We use a large codebook (64) as per our "Over-provisioning" strategy
    detector = VQTSSDetector(
        window_size=128,
        stride=1,
        hidden_dim=64,
        num_embeddings=16,
        commitment_cost=1.0,
        decay=0.85,
        smoothness_weight=1.0,  # Stronger smoothness
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        random_state=42
    )
    
    print("\nTraining VQTSS...")
    detector.fit(X, axis=0)
    
    print("Predicting...")
    y_pred = detector.predict(X, axis=0)
    
    print(f"Predicted unique states: {len(np.unique(y_pred))}")
    print(f"Predicted states: {np.unique(y_pred)}")
    
    # Evaluation
    print("\n--- Evaluation ---")
    
    # 1. State Detection Metrics (Clustering quality)
    ari_res = AdjustedRandIndex().compute(y, y_pred)
    ari = ari_res["score"] if "score" in ari_res else list(ari_res.values())[0]
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    
    # 2. Segmentation Metrics (Temporal coherence)
    cov_res = Covering().compute(y, y_pred)
    cov = cov_res["score"] if "score" in cov_res else list(cov_res.values())[0]
    print(f"Covering Score: {cov:.4f}")
    
    # 3. State Matching (if available and applicable)
    try:
        sms_res = StateMatchingScore().compute(y, y_pred)
        sms = sms_res["score"] if "score" in sms_res else list(sms_res.values())[0]
        print(f"State Matching Score: {sms:.4f}")
    except Exception as e:
        print(f"State Matching Score failed: {e}")

    # Visualization (Optional, text-based)
    print("\n--- Visual Check (First 100 points) ---")
    print(f"True: {y[:100]}")
    print(f"Pred: {y_pred[:100]}")

if __name__ == "__main__":
    evaluate_vqtss()
