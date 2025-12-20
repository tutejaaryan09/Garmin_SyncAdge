import pandas as pd
import numpy as np
import ast
import os
import glob

def parse_list(val):
    """Parses string '[1, 2]' -> numpy array. Returns empty array if error."""
    try:
        if isinstance(val, (int, float)): return np.array([val])
        # Handle sensor errors immediately
        if pd.isna(val) or val == -1 or val == "-1": return np.array([])
        
        lst = ast.literal_eval(val)
        # Filter negative values inside the list too
        arr = np.array([x for x in lst if x >= 0])
        return arr
    except:
        return np.array([])

def process_file(filepath):
    print(f"Processing {filepath}...", end=" ")
    
    # Read raw file
    df = pd.read_csv(filepath, sep=';')
    
    # 1. Expand List Columns (The Slow Part)
    ts_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in ts_cols:
        if col in df.columns:
            # Parse the text column
            parsed = df[col].apply(parse_list)
            
            # Extract features (Vectorized operations would be faster, but apply is safe)
            df[f'{col}_mean'] = parsed.apply(lambda x: np.mean(x) if x.size > 0 else 0)
            df[f'{col}_std']  = parsed.apply(lambda x: np.std(x)  if x.size > 0 else 0)
            df[f'{col}_min']  = parsed.apply(lambda x: np.min(x)  if x.size > 0 else 0)
            df[f'{col}_max']  = parsed.apply(lambda x: np.max(x)  if x.size > 0 else 0)
            
            # DROP the slow text column
            df = df.drop(columns=[col])

    # 2. General Cleaning
    df = df.replace([-1, -2], np.nan)
    df = df.fillna(0.0)
    
    # 3. Save as "_processed.csv"
    # This file will contain ONLY numbers, no complex parsing needed later
    new_path = filepath.replace(".csv", "_processed.csv")
    df.to_csv(new_path, index=False, sep=';')
    print(f"-> Done. Saved to {new_path}")

if __name__ == "__main__":
    # Find all original CSVs
    files = glob.glob(os.path.join("dataset", "*_train.csv"))
    
    # Filter out files that are already processed
    files = [f for f in files if "_processed" not in f]
    
    print(f"Found {len(files)} files to process.")
    for f in files:
        process_file(f)
