"""
Verify LEVEL1_DIFFICULTY_ORDER: all problems present in HF dataset,
no duplicates, and print the full training order after the train/test split.

Usage: python train/check_difficulty_order.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_kernelbench_dataset, LEVEL1_DIFFICULTY_ORDER

# Check for duplicates
duplicates = [name for name in LEVEL1_DIFFICULTY_ORDER if LEVEL1_DIFFICULTY_ORDER.count(name) > 1]
if duplicates:
    print(f"DUPLICATES FOUND: {set(duplicates)}")
else:
    print(f"No duplicates. {len(LEVEL1_DIFFICULTY_ORDER)} problems in order.\n")

# Load the actual train/test split and print order
train_ds, test_ds = load_kernelbench_dataset(level=1, difficulty_sort=True)

print(f"Train: {len(train_ds)} problems")
print(f"Test:  {len(test_ds)} problems\n")

print("Training order (easiest to hardest):")
for i, p in enumerate(train_ds):
    print(f"  {i+1:3d}. {p['problem_name']}")

print("\nTest set:")
for p in test_ds:
    print(f"       {p['problem_name']}")
