# utility script for dataset inspection

import pandas as pd

DATASET_PATH = "datasets/healthcare-dataset-stroke-data.csv"

def inspect_columns(file_path: str):
    df = pd.read_csv(file_path)
    
    print(f"DATASET: ({df.shape[0]} row, {df.shape[1]} col)\n")
    
    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)
        n_missing = df[col].isna().sum()
        dtype = df[col].dtype
        
        print(f"Column: '{col}' | Type: {dtype} | Missing: {n_missing} ({n_missing/len(df)*100:.1f}%)")
        
        if n_unique <= 10:
            print("   Value Distribution:")
            counts = df[col].value_counts(dropna=False)
            for val, count in counts.items():
                print(f"     - {val}: {count} ({count/len(df)*100:.1f}%)")
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"   Value Range: Min = {df[col].min()} | Max = {df[col].max()} | Mean = {df[col].mean():.2f}")
            sample_vals = list(df[col].dropna().unique()[:5])
            print(f"   Value Example: {sample_vals} ... (Unique Total: {n_unique})")
            
        print("-" * 65)

if __name__ == "__main__":
    inspect_columns(DATASET_PATH)
