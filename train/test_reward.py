"""
Quick test of the reward function end-to-end.
Loads a KernelBench problem, sends a completion to Modal, prints the reward.

Usage:
    python train/test_reward.py
"""

import time
from dataset import load_kernelbench_dataset
from reward import kernelbench_reward_fn

train_ds, test_ds = load_kernelbench_dataset(level=1)
all_problems = list(train_ds) + list(test_ds)

# Find the 4D tensor matrix multiplication problem
problem = next(p for p in all_problems if p["problem_name"] == "11_4D_tensor_matrix_multiplication")
print(f"Problem: {problem['problem_name']}")

completion = """
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

custom_kernel_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_kernel_cuda(const float* A, const float* B, float* C, int b, int i, int j, int l, int k) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int b_idx = idx / (i * j * k);
    int remaining = idx % (i * j * k);
    int i_idx = remaining / (j * k);
    int remaining2 = remaining % (j * k);
    int j_idx = remaining2 / k;
    int k_idx = remaining2 % k;

    float sum = 0.0f;
    for (int l_idx = 0; l_idx < l; ++l_idx) {
        sum += A[b_idx * i * j * l + i_idx * j * l + j_idx * l + l_idx] * B[l_idx * k + k_idx];
    }
    C[b_idx * i * j * k + i_idx * j * k + j_idx * k + k_idx] = sum;
}
\"\"\"

custom_kernel_cpp_source = \"\"\"
void custom_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor C, int b, int i, int j, int l, int k) {
    int total_elements = b * i * j * k;
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    custom_kernel_cuda<<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), b, i, j, l, k);
}
\"\"\"

custom_kernel = load_inline(
    name="custom_kernel",
    cpp_sources=custom_kernel_cpp_source,
    cuda_sources=custom_kernel_source,
    functions=["custom_kernel"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.custom_kernel = custom_kernel

    def forward(self, A, B):
        b, i, j, l = A.shape
        _, k = B.shape
        C = torch.zeros((b, i, j, k), device=A.device, dtype=A.dtype)
        self.custom_kernel(A, B, C, b, i, j, l, k)
        return C
```
"""

print("Calling Modal evaluator...")
t0 = time.time()
rewards = kernelbench_reward_fn(
    prompts=[problem["prompt"]],
    completions=[completion],
    ref_arch_src=[problem["ref_arch_src"]],
)
elapsed = time.time() - t0
print(f"Reward: {rewards}")
print(f"Eval time: {elapsed:.1f}s")
