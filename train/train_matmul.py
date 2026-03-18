"""
Focused GRPO training on 1_Square_matrix_multiplication_ only,
starting from a LoRA checkpoint.

Temperature sweeps from 0.10 to 1.00 in steps of 0.05 (19 epochs total).
Learning rate is constant at 8e-6.

Launch:
    modal run train/train_matmul.py
"""

from __future__ import annotations

import os

import modal

# =============================================================================
# Modal Infrastructure
# =============================================================================

runs_volume = modal.Volume.from_name("kernelrl-runs", create_if_missing=True)

training_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "gcc", "g++", "clang")
    .pip_install(
        "torch==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.47.0",
        "trl>=0.12.0",
        "peft>=0.13.0",
        "accelerate>=1.2.0",
        "wandb",
        "weave",
        "modal",
        "python-dotenv",
        "rich",
        "kernelbench @ git+https://github.com/ScalingIntelligence/KernelBench.git@main",
    )
    .pip_install("wheel", "packaging", "ninja")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .add_local_dir(
        local_path=os.path.dirname(__file__),
        remote_path="/root/train",
    )
)

app = modal.App("kernelrl-matmul")

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
LEVEL = 1
MAX_NEW_TOKENS = 16384
NUM_GENERATIONS = 8
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LORA_RANK = 64
WANDB_PROJECT = "kernelrl"

TIER4_PROBLEMS = [
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
]

TIER5_PROBLEMS = [
    "47_Sum_reduction_over_a_dimension",
    "48_Mean_reduction_over_a_dimension",
    "49_Max_reduction_over_a_dimension",
    "53_Min_reduction_over_a_dimension",
    "51_Argmax_over_a_dimension",
    "52_Argmin_over_a_dimension",
]

TIER6_PROBLEMS = [
    "94_MSELoss",
    "96_HuberLoss",
    "38_L1Norm_",
    "37_FrobeniusNorm_",
    "39_L2Norm_",
]

TARGET_PROBLEMS = {"1_Square_matrix_multiplication_"}


# =============================================================================
# Training Function
# =============================================================================


@app.function(
    image=training_image,
    gpu="A100-80GB",
    timeout=3600 * 24,
    volumes={"/runs": runs_volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
)
def run_training(
    run_name: str = "qwen3-8b-matmul-v2-tv-8em6lrv",
    base_checkpoint: str = "qwen3-8b-kbl1-v3-0p45tv-6em6lrv/reset_3",
):
    """Temperature sweep on 1_Square_matrix_multiplication_ from a LoRA checkpoint."""
    import sys

    sys.path.insert(0, "/root/train")

    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    import reward as rw
    from reward import kernelbench_reward_fn
    from dataset import load_kernelbench_dataset

    # ------------------------------------------------------------------
    # 1. W&B
    # ------------------------------------------------------------------
    os.environ["WANDB_HTTP_TIMEOUT"] = "60"
    os.environ["WANDB_GRAPHQL_TIMEOUT"] = "60"
    LEARNING_RATE = 8e-6
    TEMPERATURES = [round(0.10 + 0.05 * i, 2) for i in range(19)]  # 0.10 to 1.00

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "model": MODEL_NAME,
            "base_checkpoint": base_checkpoint,
            "learning_rate": LEARNING_RATE,
            "num_epochs": len(TEMPERATURES),
            "temperatures": TEMPERATURES,
            "lora_rank": LORA_RANK,
            "focus": "1_Square_matrix_multiplication_",
            "num_generations": NUM_GENERATIONS,
        },
    )

    # ------------------------------------------------------------------
    # 2. Model + tokenizer
    # ------------------------------------------------------------------
    print(f"Loading base model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    checkpoint_path = f"/runs/{base_checkpoint}"
    print(f"Loading LoRA checkpoint from {checkpoint_path}...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=True)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 3. Dataset — filter to tier 4 + tier 5 train problems only
    # ------------------------------------------------------------------
    train_dataset, _ = load_kernelbench_dataset(level=LEVEL, difficulty_sort=True)
    target_indices = [
        i for i, p in enumerate(train_dataset) if p["problem_name"] in TARGET_PROBLEMS
    ]
    train_dataset = train_dataset.select(target_indices)

    print(f"\nFiltered to {len(train_dataset)} problems:")
    for p in train_dataset:
        print(f"  {p['problem_name']}")

    # ------------------------------------------------------------------
    # 4. Epoch loop
    # ------------------------------------------------------------------
    output_dir = f"/runs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    num_epochs = len(TEMPERATURES)

    for epoch, temperature in enumerate(TEMPERATURES):
        print(f"\n{'='*60}")
        print(
            f"Epoch {epoch + 1}/{num_epochs} | temp={temperature:.2f} | lr={LEARNING_RATE:.2e}"
        )
        print(f"{'='*60}")

        # Reset epoch compile tracking in reward module
        rw._epoch_compiled_count = 0
        rw._epoch_total_count = 0

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_NEW_TOKENS,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            report_to="wandb",
            save_strategy="no",
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            steps_per_generation=NUM_GENERATIONS,
            log_completions=False,
            chat_template_kwargs={"enable_thinking": True, "thinking_budget": 2048},
            max_grad_norm=0.5,
            lr_scheduler_type="constant",
            beta=0.0,
            epsilon_high=0.3,
            shuffle_dataset=False,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            reward_funcs=[kernelbench_reward_fn],
            processing_class=tokenizer,
        )

        trainer.train()

        # Log epoch summary
        epoch_compile_rate = (
            rw._epoch_compiled_count / rw._epoch_total_count
            if rw._epoch_total_count > 0
            else 0.0
        )
        print(
            f"\n[epoch {epoch + 1}] compile_rate={epoch_compile_rate:.3f} "
            f"({rw._epoch_compiled_count}/{rw._epoch_total_count}), "
            f"temp={temperature:.2f}"
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "epoch_compile_rate": epoch_compile_rate,
                "epoch_temperature": temperature,
                "epoch_lr": LEARNING_RATE,
            }
        )

        # Save checkpoint after each epoch
        trainer.save_model(f"{output_dir}/epoch_{epoch + 1}")
        runs_volume.commit()
        print(f"Saved checkpoint: {output_dir}/epoch_{epoch + 1}")

    # ------------------------------------------------------------------
    # 5. Final checkpoint
    # ------------------------------------------------------------------
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    runs_volume.commit()
    print(f"\nDone. Final checkpoint at {output_dir}/final")
    wandb.finish()


# =============================================================================
# Local entrypoint
# =============================================================================


@app.local_entrypoint()
def main(run_name: str = "qwen3-8b-matmul-v2-tv-8em6lrv"):
    run_training.remote(run_name=run_name)
