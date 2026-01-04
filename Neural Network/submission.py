import pandas as pd
import numpy as np
import os
from model.model import Model 

# =========================================================
# FILES CONFIGURATION
# =========================================================
RAW_TEST_FILE = "x_test.csv"             # Must contain the 'id' column
PROCESSED_TEST_FILE = "x_test_processed.csv" # Must contain the 41 features
CHECKPOINT_FILE = "global_model.npz"     # Your trained weights
OUTPUT_FILE = "submission.csv"           # The file to upload to Kaggle
# =========================================================

def main():
    print("="*40)
    print("   GENERATING KAGGLE SUBMISSION")
    print("="*40)

    # 1. GET IDs
    print(f"\n1. Reading IDs from '{RAW_TEST_FILE}'...")
    try:
        df_raw = pd.read_csv(RAW_TEST_FILE, sep=None, engine='python')
        if 'id' in df_raw.columns:
            ids = df_raw['id'].values
            print(f"   -> Found {len(ids)} IDs.")
        else:
            print("   [!] 'id' column NOT found. Generating dummy IDs (0..N).")
            ids = np.arange(len(df_raw))
    except Exception as e:
        print(f"   [!] Error reading raw file: {e}")
        return

    # 2. GET FEATURES
    print(f"\n2. Reading Features from '{PROCESSED_TEST_FILE}'...")
    try:
        # Smart load to handle ; or , separators
        df_proc = pd.read_csv(PROCESSED_TEST_FILE, sep=None, engine='python')
        if df_proc.shape[1] < 2: 
            df_proc = pd.read_csv(PROCESSED_TEST_FILE, sep=';', engine='python')
        
        # Take first 41 columns (Features)
        if df_proc.shape[1] >= 41:
            X_test = df_proc.iloc[:, :41].values
            print(f"   -> Loaded features. Shape: {X_test.shape}")
        else:
            print(f"   [!] Error: Processed file has only {df_proc.shape[1]} columns. Need 41.")
            return
            
        # Safety check: Length mismatch?
        if len(ids) != len(X_test):
            print(f"   [!] CRITICAL WARNING: Row count mismatch!")
            print(f"       Raw IDs: {len(ids)}")
            print(f"       Features: {len(X_test)}")
            # Truncate to the smaller size to prevent crash
            min_len = min(len(ids), len(X_test))
            ids = ids[:min_len]
            X_test = X_test[:min_len]
            print(f"       -> Truncated both to {min_len} rows.")

    except Exception as e:
        print(f"   [!] Error reading processed file: {e}")
        return

    # 3. LOAD MODEL
    print(f"\n3. Loading Model & Weights ('{CHECKPOINT_FILE}')...")
    if not os.path.exists(CHECKPOINT_FILE):
        print("   [!] Model file not found!")
        return

    model_container = Model(input_size=41) 
    model = model_container.model
    
    try:
        data = np.load(CHECKPOINT_FILE)
        weights = [data[f"arr_{i}"] for i in range(len(data.files))]
        model.set_weights(weights)
        print("   -> Weights loaded successfully.")
    except Exception as e:
        print(f"   [!] Error loading weights: {e}")
        return

    # 4. PREDICT
    print("\n4. Generating Predictions...")
    try:
        # Apply Scaling (Critical for your model)
        model_container._ensure_scaler(X_test)
        X_scaled = model_container.scaler.transform(X_test)
        
        # Predict & Scale back (0-1 -> 0-100)
        preds = model.predict(X_scaled, verbose=0) * 100.0
        preds_flat = preds.flatten()
        
        print(f"   -> Prediction Stats: Mean={np.mean(preds_flat):.2f}, Min={np.min(preds_flat):.2f}, Max={np.max(preds_flat):.2f}")
        
    except Exception as e:
        print(f"   [!] Prediction failed: {e}")
        return

    # 5. SAVE SUBMISSION
    print(f"\n5. Saving to '{OUTPUT_FILE}'...")
    submission = pd.DataFrame({
        'id': ids,
        'label': preds_flat
    })
    
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"   -> SUCCESS! File saved.")
    print("\n   First 5 rows:")
    print(submission.head())
    print("="*40)

if __name__ == "__main__":
    main()
