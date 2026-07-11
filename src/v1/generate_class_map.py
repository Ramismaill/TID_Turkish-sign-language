"""
generate_class_map.py — Creates class_map.json from train_manifest.csv
Maps label index → sign name (using AUTSL class indices)

Usage:
  python src/generate_class_map.py ^
    --manifest "C:/AUTSL_project/landmarks/train_manifest.csv" ^
    --out      "C:/AUTSL_project/src/class_map.json"
"""

import pandas as pd
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out",      required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    labels = sorted(df["label"].unique())

    # AUTSL uses numeric class IDs 0-225
    # Map each to "SIGN_XXX" — replace with real Turkish names if available
    class_map = {str(lbl): f"SIGN_{lbl:03d}" for lbl in labels}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    print(f"class_map.json saved → {args.out}")
    print(f"Total classes: {len(class_map)}")

if __name__ == "__main__":
    main()
