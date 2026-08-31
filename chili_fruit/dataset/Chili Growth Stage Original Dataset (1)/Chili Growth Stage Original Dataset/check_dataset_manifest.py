#!/usr/bin/env python3
"""Check the generated five-class dataset manifest without training."""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST = SCRIPT_DIR / "data/chili_growth_stage/dataset_manifest.csv"
EXPECTED = {
    (0, "Dry Chili"): 410,
    (1, "Flower"): 397,
    (2, "Green Chili"): 328,
    (3, "Red Chili"): 200,
    (4, "Rotten Chili"): 379,
}

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)
counts = df.groupby(["class_id", "class_name"]).size().to_dict()
print("Manifest:", MANIFEST)
for key, expected in EXPECTED.items():
    actual = int(counts.get(key, 0))
    status = "OK" if actual == expected else f"CHECK, expected {expected}"
    print(f"{key[0]}  {key[1]}: {actual} [{status}]")
print("Total:", len(df))

missing_paths = [p for p in df["absolute_path"] if not Path(str(p)).exists()]
print("Missing image paths:", len(missing_paths))
if missing_paths:
    print("Example:", missing_paths[0])
