"""
Test the static checker on a kernel completion.

Usage:
    python train/test_static_checker.py
"""

from kernelbench.kernel_static_checker import validate_kernel_static

kernel_code = """
import torch
import torch.nn as nn
import torch.utils.cpp_extension
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hardsigmoid_cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float value = 0.2f * x + 0.5f;
        value = fminf(fmaxf(value, 0.0f), 1.0f);
        output[idx] = value;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    hardsigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
\"\"\"

hardsigmoid_cpp_source = (
    "torch::Tensor hardsigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for HardSigmoid
hardsigmoid = load_inline(
    name="hardsigmoid",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_cuda_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
    extra_cflags=["-DFORCE fp32"],
    extra_ldflags=["-L/usr/local/cuda/lib64 -lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hardsigmoid = hardsigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid.hardsigmoid_cuda(x)
"""

STRICT_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",
    "lazy_eval",
    "cuda_impl",
    "torch_computation_ops",
]

valid, errors, warnings = validate_kernel_static(
    code=kernel_code,
    backend="cuda",
    precision="fp32",
    forbidden=STRICT_CHECKS,
    warnings=[],
)

print("valid:", valid)
print("errors:", errors)
print("warnings:", warnings)
