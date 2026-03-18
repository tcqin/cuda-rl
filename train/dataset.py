"""
KernelBench dataset for GRPO training.

Loads Level 1 problems from HuggingFace, formats them as prompts,
and returns a HuggingFace Dataset with columns:
  - prompt: list[dict]  (chat-formatted for the model)
  - ref_arch_src: str   (reference PyTorch code, passed to reward fn)
  - problem_name: str
"""

from __future__ import annotations

import sys
from pathlib import Path


from datasets import Dataset

# Allow importing kernelbench from the cloned repo
KERNELBENCH_SRC = Path(__file__).parent.parent / "KernelBench" / "src"
if str(KERNELBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(KERNELBENCH_SRC))

from kernelbench.prompt_constructor_toml import get_prompt_for_backend


SYSTEM_PROMPT = (
    "You are an expert CUDA kernel engineer optimizing PyTorch models for NVIDIA A100 "
    "using shared memory, kernel fusion, warp primitives, and vectorization. "
    "Think briefly (at most 3-5 short paragraphs) about the key operation and best CUDA strategy. "
    "Do not re-explain the problem or list alternatives. "
    "The code must be as fast as possible. "
    "Rules: "
    "Do not use torch.nn except for Parameter, containers, and init. "
    "Inputs and outputs must be on the CUDA device. "
    "The C++ declaration in cpp_sources must exactly match your CUDA function signature."
)

# Level 1 problems sorted by difficulty (easiest first), based on KernelBench
# leaderboard rank-1 speedups. Problems not in this list are excluded when
# difficulty_sort=True (no model has solved them on the leaderboard).
LEVEL1_DIFFICULTY_ORDER = [
    # Tier 1: Element-wise activations — trivial formulas, one thread per element
    "19_ReLU",
    "20_LeakyReLU",
    "32_HardTanh",
    "22_Tanh",
    "21_Sigmoid",
    "30_Softsign",
    "25_Swish",
    # Tier 2: Element-wise activations — slightly tricky formulas
    "31_ELU",
    "27_SELU_",
    "29_Softplus",
    "28_HardSigmoid",
    "26_GELU_",
    "88_MinGPTNewGelu",
    # Tier 3: Element-wise with simple indexing
    "5_Matrix_scalar_multiplication",
    "12_Matmul_with_diagonal_matrices_",
    # Tier 4: Matmul variants — naive O(N^3) impl is correct, just slow
    "4_Matrix_vector_multiplication_",
    "9_Tall_skinny_matrix_multiplication_",
    "17_Matmul_with_transposed_B",
    "3_Batched_matrix_multiplication",
    "1_Square_matrix_multiplication_",
    "2_Standard_matrix_multiplication_",
    "7_Matmul_with_small_K_dimension_",
    "13_Matmul_for_symmetric_matrices",
    "8_Matmul_with_irregular_shapes_",
    "16_Matmul_with_transposed_A",
    "10_3D_tensor_matrix_multiplication",
    "18_Matmul_with_transposed_both",
    "14_Matmul_for_upper_triangular_matrices",
    "15_Matmul_for_lower_triangular_matrices",
    "6_Matmul_with_large_K_dimension_",
    "11_4D_tensor_matrix_multiplication",
    # Tier 5: Simple reductions — need parallel reduction pattern
    "47_Sum_reduction_over_a_dimension",
    "48_Mean_reduction_over_a_dimension",
    "49_Max_reduction_over_a_dimension",
    "53_Min_reduction_over_a_dimension",
    "51_Argmax_over_a_dimension",
    "52_Argmin_over_a_dimension",
    # Tier 6: Reduction-based norms and losses — reduction + simple formula
    "94_MSELoss",
    "96_HuberLoss",
    "38_L1Norm_",
    "37_FrobeniusNorm_",
    "39_L2Norm_",
    # Tier 7: Numerically sensitive reductions
    "23_Softmax",
    "24_LogSoftmax",
    "98_KLDivLoss",
    "95_CrossEntropyLoss",
    "99_TripletMarginLoss",
    # Tier 8: Multi-pass normalizations — mean + variance + normalize
    "36_RMSNorm_",
    "40_LayerNorm",
    "34_InstanceNorm",
    "33_BatchNorm",
    # Tier 9: Spatial pooling — correct output size + padding formulas
    "41_Max_Pooling_1D",
    "45_Average_Pooling_2D",
    "42_Max_Pooling_2D",
    "43_Max_Pooling_3D",
    # Tier 10: Sequential scans — inherently hard to parallelize
    "89_cumsum",
    "90_cumprod",
    "91_cumsum_reverse",
    "93_masked_cumsum",
    # Tier 11: Convolutions — most complex
    "54_conv_standard_3D__square_input__square_kernel",
    "66_conv_standard_3D__asymmetric_input__asymmetric_kernel",
    "69_conv_transposed_2D__asymmetric_input__asymmetric_kernel",
]


def load_kernelbench_dataset(
    level: int = 1,
    dataset_src: str = "huggingface",
    test_fraction: float = 0.1,
    seed: int = 42,
    difficulty_sort: bool = False,
) -> tuple[Dataset, Dataset]:
    """
    Load KernelBench problems as a HuggingFace Dataset.

    Args:
        difficulty_sort: If True (level=1 only), filter to problems with known
            leaderboard solutions and sort easiest first by rank-1 speedup.

    Returns (train_dataset, test_dataset), each with columns:
        prompt, ref_arch_src, problem_name
    """
    from datasets import load_dataset

    if dataset_src == "huggingface":
        hf_dataset = load_dataset("ScalingIntelligence/KernelBench")
        problems = list(hf_dataset[f"level_{level}"])
    else:
        raise ValueError(f"Unsupported dataset_src: {dataset_src}")

    if difficulty_sort and level == 1:
        order = {name: i for i, name in enumerate(LEVEL1_DIFFICULTY_ORDER)}
        problems = [p for p in problems if p["name"] in order]
        problems = sorted(problems, key=lambda p: order[p["name"]])

    rows = []
    for problem in problems:
        ref_arch_src = problem["code"]
        problem_name = problem["name"]

        # Build the one-shot prompt using KernelBench's prompt constructor
        prompt_text = get_prompt_for_backend(
            ref_arch_src, backend="cuda", option="one_shot"
        )

        # Format as a chat (system + user turn)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        rows.append(
            {
                "prompt": prompt,
                "ref_arch_src": ref_arch_src,
                "problem_name": problem_name,
            }
        )

    full_dataset = Dataset.from_list(rows)

    # Random train/test split (problems distributed evenly across both sets)
    split = full_dataset.train_test_split(
        test_size=test_fraction, seed=seed, shuffle=True
    )
    train_data = split["train"]
    test_data = split["test"]

    # Sort train set easiest to hardest
    if difficulty_sort:
        order = {name: i for i, name in enumerate(LEVEL1_DIFFICULTY_ORDER)}
        train_sorted = sorted(
            range(len(train_data)), key=lambda j: order[train_data[j]["problem_name"]]
        )
        train_data = train_data.select(train_sorted)

    return train_data, test_data


if __name__ == "__main__":
    train_ds, test_ds = load_kernelbench_dataset(level=1, difficulty_sort=True)
    print(f"Train: {len(train_ds)} problems, Test: {len(test_ds)} problems")
    print("Problems in order:")
    for p in train_ds:
        print(f"  {p['problem_name']}")
