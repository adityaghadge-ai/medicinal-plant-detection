# quick_inspect.py
import os, glob
from collections import Counter

TRAIN_DIR = "disease_dataset_split/train"

classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
counts = {c: len(glob.glob(os.path.join(TRAIN_DIR, c, "*.*"))) for c in classes}

import pandas as pd
pd.Series(counts).sort_values().plot(kind="barh", figsize=(10,10))
print("min/max/median:", min(counts.values()), max(counts.values()), int(pd.Series(counts).median()))
print("Top 10 smallest classes:\n", pd.Series(counts).sort_values().head(10))
