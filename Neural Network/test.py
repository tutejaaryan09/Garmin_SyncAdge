import numpy as np
import pandas as pd
import tensorflow as tf
import os
from model.model import Model 

CHECKPOINT_FILE = "global_model.npz"
TEST_FILE = "x_test_processed"

def load_test_data(file_path):
    print(f"Loading {file_path}...")
    try:
        # Try reading with Python engine 
        df = pd.read_csv(file_path, sep=None, engine='python')
        
        # If that failed try ;
        if df.shape[1] < 2:
            print("  -> Auto-detect failed (found 1 column). Retrying with sep=';'...")
            df = pd.read_csv(file_path, sep=';', engine='python')

        # If Failed agin try ,
        if df.shape[1] < 2:
            print("  -> Retrying with sep=','...")
            df = pd.read_csv(file_path, sep=',', engine='python')

        print(f"  -> Data Loaded. Shape: {df.shape}")

        # Now check columns
        if df.shape[1] == 41:
            print("  -> 41 columns detected (Features only).")
            X = df.values
            y = np.zeros(len(X)) 
        elif df.shape[1] == 42:
            print("  -> 42 columns detected (Features + Label).")
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        else:
            # Fallback for weird sizes (e.g. index column included)
            print(f"  [!] Warning: Unexpected column count: {df.shape[1]}. Trying to auto-fix...")
            if df.shape[1] > 41:
                # Assume last columns are extra/labels, take first 41
                X = df.iloc[:, :41].values
                y = np.zeros(len(X))
            else:
                print("  [!] Error: Not enough columns.")
                return None, None

        return X, y

    except Exception as e:
        print(f"[!] Critical Error loading data: {e}")
        return None, None

def main():
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"[!] No model file found at {CHECKPOINT_FILE}")
        return

    # Load Data
    X_test, y_test = load_test_data(TEST_FILE)
    if X_test is None: return

    # Initialize Model (Use input_size)
    
    model_container = Model(input_size=41) 
    model = model_container.model
    
    # Load Weights
    print(f"Loading weights from {CHECKPOINT_FILE}...")
    try:
        data = np.load(CHECKPOINT_FILE)
        weights = [data[f"arr_{i}"] for i in range(len(data.files))]
        model.set_weights(weights)
        print("  -> Weights loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading weights: {e}")
        return

    # PREDICTION CHECK
    print("\n" + "="*30)
    print("DEBUG: Checking Variance in Predictions...")
    
    # Predict with Scaler (Standard Behavior)
    model_container._ensure_scaler(X_test) # Fits scaler on X_test (might be bad if dist is different)
    X_scaled = model_container.scaler.transform(X_test)
    preds_scaled = model.predict(X_scaled, verbose=0) * 100.0
    
    # Predict RAW (In case model expects raw inputs or specific range)
    preds_raw = model.predict(X_test, verbose=0) * 100.0

    print(f"Stats with Scaler -> Mean: {np.mean(preds_scaled):.2f}, Std: {np.std(preds_scaled):.2f}")
    print(f"Stats Raw Input   -> Mean: {np.mean(preds_raw):.2f}, Std: {np.std(preds_raw):.2f}")
    
    # Logic to choose best
    if np.std(preds_scaled) > 0.01:
        print("  -> Using Scaled Predictions (Variance detected).")
        final_preds = preds_scaled
    elif np.std(preds_raw) > 0.01:
        print("  -> Using Raw Predictions (Variance detected).")
        final_preds = preds_raw
    else:
        print("  [!] WARNING: Both approaches yield constant predictions.")
        print("      Possible reasons: 1. Model collapsed. 2. Input data is constant. 3. Normalization failed.")
        final_preds = preds_scaled

    print("="*30 + "\n")
    
    # Print & Save
    for i in range(5):
        print(f" Sample {i}: Predicted Sleep Quality = {final_preds[i][0]:.2f}")

    pd.DataFrame(final_preds, columns=["predicted_sleep_quality"]).to_csv("predictions.csv", index=False)
    print("\nAll predictions saved to 'predictions.csv'.")

if __name__ == "__main__":
    main()
