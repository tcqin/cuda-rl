"""
Check if GRPOConfig supports disabling DataLoader shuffling.
Usage: python train/check_grpo_shuffle.py
"""

import inspect
from trl import GRPOConfig

params = inspect.signature(GRPOConfig).parameters
shuffle_params = [p for p in params if any(k in p.lower() for k in ("shuffle", "sampler", "dataloader"))]

print("GRPOConfig params related to shuffling/dataloader:")
for p in shuffle_params:
    print(f"  {p} = {params[p].default}")
