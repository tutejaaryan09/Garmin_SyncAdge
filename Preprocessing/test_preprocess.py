import pandas as pd
import numpy as np
import ast
import os

TARGET_FILE = "x_test.csv" 

def parse_list(val):
    """Parses string '[1, 2]' -> numpy array. Returns empty array if error."""
    try:
        if isinstance(val, (int, float)): return np.array([val])
        # Handle sensor errors immediately
        if pd.isna(val) or val == -1 or val == "-1": return np.array([])
        
        lst = ast.literal_eval(val)
        # Filter negative values 
        arr = np.array([x for x in lst if x >= 0])
        return arr
    except:
        return np.array([])


def process_file(filepath):
    print(f"Processing {filepath}...", end=" ")
    
    # Read raw file 
    df = pd.read_csv(filepath, sep=';')
    
    # Expand List Columns
    ts_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in ts_cols:
        if col in df.columns:
            # Parse the text column
            parsed = df[col].apply(parse_list)
            
            # Extract features
            df[f'{col}_mean'] = parsed.apply(lambda x: np.mean(x) if x.size > 0 else 0)
            df[f'{col}_std']  = parsed.apply(lambda x: np.std(x)  if x.size > 0 else 0)
            df[f'{col}_min']  = parsed.apply(lambda x: np.min(x)  if x.size > 0 else 0)
            df[f'{col}_max']  = parsed.apply(lambda x: np.max(x)  if x.size > 0 else 0)
            
            # DROP
            df = df.drop(columns=[col])


    # General Cleaning
    df = df.replace([-1, -2], np.nan)
    df = df.fillna(0.0)
    
    # Save as "_processed.csv"
    new_path = filepath.replace(".csv", "_processed.csv")
    df.to_csv(new_path, index=False, sep=';')
    print(f"-> Done. Saved to {new_path}")


if __name__ == "__main__":
    if os.path.exists(TARGET_FILE):
        process_file(TARGET_FILE)
    else:
        print(f"File not found: {TARGET_FILE}")
