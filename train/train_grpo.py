"""
GRPO training for KernelBench using Qwen3-8B on Modal.

This script:
1. Defines a Modal training job (runs on A100-80GB)
2. Loads Qwen3-8B with LoRA via PEFT
3. Uses TRL GRPOTrainer with a KernelBench reward function
4. Logs to W&B
5. Saves checkpoints to a Modal Volume

Deploy the kernel evaluator first:
    cd kernelbench-tinker
    modal deploy src/kernelbench_tinker/modal/app.py

Create Modal secrets:
    modal secret create wandb-secret WANDB_API_KEY=...
    modal secret create hf-secret HF_TOKEN=...
    modal secret create modal-secret MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...

Then launch training:
    cd ~/projects/cuda-rl
    modal run train/train_grpo.py
"""

from __future__ import annotations

import os

import modal

# =============================================================================
# Modal Infrastructure
# =============================================================================

# Volume for persisting checkpoints across runs
runs_volume = modal.Volume.from_name("kernelrl-runs", create_if_missing=True)

# Training image: CUDA 12.4 + PyTorch + TRL + PEFT + W&B
training_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "gcc", "g++", "clang")
    # PyTorch with CUDA 12.4 support
    .pip_install(
        "torch==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    # ML + logging stack
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
        # KernelBench for prompt construction + static checker + code extraction
        "kernelbench @ git+https://github.com/ScalingIntelligence/KernelBench.git@main",
    )
    .pip_install("wheel", "packaging", "ninja")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    # Bake local train/ files into image so reward.py is importable
    .add_local_dir(
        local_path=os.path.dirname(__file__),
        remote_path="/root/train",
    )
)

app = modal.App("kernelrl-training")

# =============================================================================
# Training Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
LEVEL = 1
MAX_NEW_TOKENS = 16384
NUM_GENERATIONS = 8  # Rollouts per problem (GRPO group size)
BATCH_SIZE = 1  # Prompts per gradient step
GRAD_ACCUM_STEPS = 8  # Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LORA_RANK = 64
LEARNING_RATE = 6e-6
NUM_EPOCHS = 5
WANDB_PROJECT = "kernelrl"


# =============================================================================
# Training Function
# =============================================================================


@app.function(
    image=training_image,
    gpu="A100-80GB",
    timeout=3600 * 24,
    volumes={"/runs": runs_volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),  # WANDB_API_KEY
        modal.Secret.from_name("hf-secret"),  # HF_TOKEN
    ],
)
def run_training(run_name: str = "qwen3-8b-grpo-l1"):
    """Main training function. Runs inside the Modal A100-80GB container."""
    import sys

    # Make our local train/ module importable
    sys.path.insert(0, "/root/train")

    import torch
    import wandb
    import weave
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback, TrainerControl, TrainerState

    # Import from mounted train/ directory
    import reward as rw
    from reward import kernelbench_reward_fn
    from dataset import load_kernelbench_dataset

    # ------------------------------------------------------------------
    # 1. W&B
    # ------------------------------------------------------------------
    os.environ["WANDB_HTTP_TIMEOUT"] = "60"
    os.environ["WANDB_GRAPHQL_TIMEOUT"] = "60"
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "model": MODEL_NAME,
            "level": LEVEL,
            "lora_rank": LORA_RANK,
            "lr": LEARNING_RATE,
            "num_generations": NUM_GENERATIONS,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    )
    weave.init(WANDB_PROJECT)

    # ------------------------------------------------------------------
    # 2. Model + tokenizer (loaded once, reused across curriculum resets)
    # ------------------------------------------------------------------
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA once — reused across resets
    model = get_peft_model(model, lora_config)

    # ------------------------------------------------------------------
    # 3. Curriculum training loop
    # ------------------------------------------------------------------
    output_dir = f"/runs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    temperature = 0.45
    learning_rate = LEARNING_RATE
    reset_count = 0
    MIN_TEMPERATURE = 0.2
    MAX_TEMPERATURE = 0.9

    class CurriculumResetCallback(TrainerCallback):
        """Stops training when reward.py signals 3 consecutive stuck steps."""

        def on_step_end(
            self,
            args: GRPOConfig,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> TrainerControl:
            if rw._curriculum_reset_flag:
                control.should_training_stop = True
            return control

    while True:
        # Fresh dataset from easiest problem each (re)start
        print(
            f"Loading dataset (temperature={temperature:.2f}, reset #{reset_count})..."
        )
        train_dataset, _ = load_kernelbench_dataset(level=LEVEL, difficulty_sort=True)
        print(f"Train: {len(train_dataset)} problems")

        # Reset curriculum state
        rw._curriculum_reset_flag = False
        rw._consecutive_stuck_steps = 0
        rw._stuck_compiled_count = 0
        rw._stuck_total_count = 0

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_NEW_TOKENS,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            learning_rate=learning_rate,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            report_to="wandb",
            save_strategy="steps",
            save_steps=20,
            save_total_limit=3,
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
            # No peft_config — LoRA already applied to model
            processing_class=tokenizer,
            callbacks=[CurriculumResetCallback()],
        )

        print("Starting GRPO training...")
        trainer.train()

        if rw._curriculum_reset_flag:
            reset_count += 1
            compile_rate = (
                rw._stuck_compiled_count / rw._stuck_total_count
                if rw._stuck_total_count > 0
                else 0.0
            )
            delta = -0.05 + 0.10 * compile_rate
            temperature = max(
                MIN_TEMPERATURE, min(MAX_TEMPERATURE, temperature + delta)
            )
            learning_rate *= 0.95
            rw._stuck_compiled_count = 0
            rw._stuck_total_count = 0
            print(
                f"[curriculum] Reset #{reset_count} — compile_rate={compile_rate:.3f}, "
                f"delta={delta:+.3f}, temperature -> {temperature:.3f}, "
                f"lr -> {learning_rate:.2e}"
            )
            trainer.save_model(f"{output_dir}/reset_{reset_count}")
            runs_volume.commit()
        else:
            # Completed normally
            break

    # Save final checkpoint
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    runs_volume.commit()
    print(f"Done. Checkpoint at {output_dir}/final")
    wandb.finish()


# =============================================================================
# Local entrypoint
# =============================================================================


@app.local_entrypoint()
def main(run_name: str = "qwen3-8b-kbl1-v3-0p45tv-6em6lrv"):
    run_training.remote(run_name=run_name)
