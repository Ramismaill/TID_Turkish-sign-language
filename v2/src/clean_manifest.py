import numpy as np
import pandas as pd
from pathlib import Path

def clean_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    bad = []

    for i, row in df.iterrows():
        path = row["npy_path"]  # or "npy_path" depending on column name
        if not Path(path).exists():
            bad.append(i)
            print(f"[MISSING] {path}")
            continue
        try:
            x = np.load(path)
            if np.isnan(x).all() or x.shape != (64, 225):
                bad.append(i)
                print(f"[CORRUPT] {path} shape={x.shape}")
        except Exception as e:
            bad.append(i)
            print(f"[ERROR] {path}: {e}")

    print(f"\nFound {len(bad)} bad files out of {len(df)}")
    df_clean = df.drop(index=bad).reset_index(drop=True)
    df_clean.to_csv(manifest_path, index=False)
    print(f"Cleaned manifest saved → {manifest_path}")

clean_manifest("C:/AUTSL_project/landmarks/train_manifest.csv")
clean_manifest("C:/AUTSL_project/landmarks/val_manifest.csv")