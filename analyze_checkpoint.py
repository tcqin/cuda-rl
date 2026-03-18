"""
Analyze how much a LoRA checkpoint differs from the base Qwen3-8B weights.

Usage:
    python analyze_checkpoint.py --checkpoint checkpoints/reset_2
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/kbl1-multiturn-v1-s24")
parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
args = parser.parse_args()

print(f"Loading base model: {args.base_model}")
base = AutoModelForCausalLM.from_pretrained(
    args.base_model, dtype=torch.bfloat16, device_map="cpu"
)

print(f"Loading LoRA checkpoint: {args.checkpoint}")
model = PeftModel.from_pretrained(base, args.checkpoint)

print(f"\n{'Layer':<60} {'delta_norm':>12} {'base_norm':>12} {'relative':>10}")
print("-" * 96)

rows = []
for name, module in model.named_modules():
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        A = module.lora_A["default"].weight.float()
        B = module.lora_B["default"].weight.float()
        alpha = (
            module.lora_alpha["default"]
            if isinstance(module.lora_alpha, dict)
            else module.lora_alpha
        )
        r = module.r["default"] if isinstance(module.r, dict) else module.r
        scale = alpha / r
        delta = scale * (B @ A)
        base_w = module.base_layer.weight.float()
        delta_norm = delta.norm().item()
        base_norm = base_w.norm().item()
        relative = delta_norm / base_norm
        rows.append((name, delta_norm, base_norm, relative))

rows.sort(key=lambda x: x[3], reverse=True)

for name, delta_norm, base_norm, relative in rows:
    print(f"{name:<60} {delta_norm:>12.4f} {base_norm:>12.4f} {relative:>10.4f}")

total_delta = sum(r[1] for r in rows)
total_base = sum(r[2] for r in rows)
print("-" * 96)
print(
    f"{'TOTAL':<60} {total_delta:>12.4f} {total_base:>12.4f} {total_delta/total_base:>10.4f}"
)
