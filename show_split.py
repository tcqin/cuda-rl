"""
Show the train/test split for the current dataset configuration.

Usage:
    python show_split.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "train"))

from dataset import load_kernelbench_dataset

train_ds, test_ds = load_kernelbench_dataset(level=1, difficulty_sort=True, seed=42, test_fraction=0.1)

print(f"Train: {len(train_ds)} problems")
print(f"Test:  {len(test_ds)} problems\n")

print(f"{'#':<5} {'problem':<50} set")
print("-" * 60)
for i, p in enumerate(train_ds):
    print(f"{i+1:<5} {p['problem_name']:<50} train")

print()
for p in test_ds:
    print(f"{'—':<5} {p['problem_name']:<50} test")
