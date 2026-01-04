"""
Fast Data Loader.
Expects that you have run 'preprocess_once.py' to generate _processed.csv files.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_dataset(csv_path: str, random_state: int = 42):
    # 1. AUTO-SWITCH to Processed File
    # If client asks for "dataset_user_0_train.csv", we try to open "dataset_user_0_train_processed.csv"
    processed_path = csv_path.replace(".csv", "_processed.csv")
    
    if os.path.exists(processed_path):
        # FAST PATH: Read clean numbers directly
        df = pd.read_csv(processed_path, sep=';')
    else:
        # Fallback (Slow) if you forgot to run preprocess_once.py
        # raise FileNotFoundError(f"Processed file not found: {processed_path}. Run preprocess_once.py first!")
        print(f"Warning: Processed file not found for {csv_path}. Reading raw file (Slow)...")
        df = pd.read_csv(csv_path, sep=';')
        # (Add parsing logic here if you really want fallback, but better to force preprocessing)

    # 2. Prepare X and y
    target_col = "label"
    drop_cols = ["day", "date", target_col]
    
    # Select only numeric feature columns
    # (The processed file should already be all numeric, but this is safe)
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].values
    y = df[target_col].values

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    return (X_train.astype(np.float32), X_test.astype(np.float32), 
            y_train.astype(np.float32), y_test.astype(np.float32))
