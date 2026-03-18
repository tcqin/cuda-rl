"""
Check which problems in LEVEL1_DIFFICULTY_ORDER are missing from the HuggingFace dataset.
Usage: python train/check_dataset_names.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from dataset import LEVEL1_DIFFICULTY_ORDER

hf = load_dataset("ScalingIntelligence/KernelBench")
hf_names = {p["name"] for p in hf["level_1"]}

print(f"HF dataset has {len(hf_names)} level_1 problems")
print(f"LEVEL1_DIFFICULTY_ORDER has {len(LEVEL1_DIFFICULTY_ORDER)} problems\n")

missing = [name for name in LEVEL1_DIFFICULTY_ORDER if name not in hf_names]
present = [name for name in LEVEL1_DIFFICULTY_ORDER if name in hf_names]

print(f"Missing from HF ({len(missing)}):")
for name in missing:
    print(f"  {name}")

print(f"\nPresent in HF ({len(present)}):")
for name in present:
    print(f"  {name}")
