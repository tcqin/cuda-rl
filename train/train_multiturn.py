"""
Multi-turn GRPO for KernelBench — full Level 1 curriculum.

Iterates through LEVEL1_DIFFICULTY_ORDER easiest → hardest (cycling).
Constant temperature=0.4, LR=3e-5.
If 3 consecutive steps have no correct kernel across any trajectory,
checkpoints the weights and restarts from the easiest problem.

Launch:
    modal run train/train_multiturn.py
"""

from __future__ import annotations

import json
import os
import sys
import time

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
        "modal",
        "kernelbench @ git+https://github.com/ScalingIntelligence/KernelBench.git@main",
    )
    .pip_install("wheel", "packaging", "ninja")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .add_local_dir(
        local_path=os.path.dirname(__file__),
        remote_path="/root/train",
    )
)

app = modal.App("kernelrl-multiturn")

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
NUM_GENERATIONS = 8
NUM_TURNS = 4
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.45
LEARNING_RATE = 3e-5
GRAD_ACCUM_STEPS = 1
EPSILON_HIGH = 0.3
MAX_GRAD_NORM = 0.5
LORA_RANK = 64
WANDB_PROJECT = "kernelrl"
SAVE_STEPS = 1
GAMMA = 0.4  # discount factor for multi-turn returns

# Reward threshold above which a kernel is considered "correct"
# (gap between 0.02=incorrect and 0.3=correct makes 0.1 a clean separator)
CORRECT_THRESHOLD = 0.1

# Consecutive steps with no correct kernel before checkpointing and restarting
STUCK_THRESHOLD = 3

# Eval params (match reward.py)
EVAL_GPU = "A100-80GB"
EVAL_GPU_ARCH = ["Ampere"]
NUM_CORRECT_TRIALS = 6
NUM_PERF_TRIALS = 40
TIMING_METHOD = "cuda_event"
PRECISION = "fp32"
BACKEND = "cuda"
CORRECTNESS_WEIGHT = 0.3
MAX_SPEED_REWARD = 10.0
STRICT_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",
    "lazy_eval",
    "cuda_impl",
    "torch_computation_ops",
]

CHAT_KWARGS = {"enable_thinking": True, "thinking_budget": 1024}

# System prompt for multi-turn: adds iterative-improvement framing and summary requirement.
MULTITURN_SYSTEM_PROMPT = (
    "You are an expert CUDA kernel engineer optimizing PyTorch models for NVIDIA A100 "
    "using shared memory, kernel fusion, warp primitives, and vectorization. "
    "Your internal reasoning should be brief (at most 3-5 short paragraphs). "
    "Your response must follow this exact format, with nothing before or after:\n"
    "1. A single ```python ... ``` code block containing your complete kernel.\n"
    "2. A <summary>...</summary> block with 3-5 sentences describing your implementation approach and key design choices.\n"
    "Rules: "
    "Do not use torch.nn except for Parameter, containers, and init. "
    "Inputs and outputs must be on the CUDA device. "
    "The C++ declaration in cpp_sources must exactly match your CUDA function signature. "
    "Your kernel will be evaluated for correctness and speedup, and you will receive "
    "feedback so you can keep iterating — your goal is to maximize speedup."
)


# =============================================================================
# Reward + Feedback helpers (module-level so they're importable inside Modal)
# =============================================================================


def _compute_reward(eval_result: dict, kernel_code: str = "") -> float:
    """Reward formula matching reward.py."""
    from kernelbench.kernel_static_checker import validate_kernel_static

    if not eval_result.get("format_ok", False):
        return 0.0
    if kernel_code:
        valid, _, _ = validate_kernel_static(
            code=kernel_code,
            backend=BACKEND,
            precision=PRECISION,
            forbidden=STRICT_CHECKS,
            warnings=[],
        )
        if not valid:
            return 0.0
    if not eval_result.get("compiled", False):
        return 0.01
    if not eval_result.get("correctness", False):
        return 0.02
    speedup = eval_result.get("speedup") or 0.0
    if speedup <= 1.0:
        return CORRECTNESS_WEIGHT + (1 - CORRECTNESS_WEIGHT) * speedup
    return min(speedup, MAX_SPEED_REWARD)


def _extract_summary(text: str) -> str:
    """Extract the <summary>...</summary> block; falls back to empty string."""
    import re

    m = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _is_truncated(text: str, kernel: str | None) -> bool:
    """Return True if the completion was cut off before the code block closed."""
    if kernel is not None:
        return False
    return "```python" in text or "```cpp" in text


def _feedback_message(eval_result: dict, kernel: str | None, text: str) -> str:
    """User message for the next turn, based on the previous turn's result."""
    if kernel is None:
        if _is_truncated(text, kernel):
            return (
                "Your response was cut off before the code block finished. "
                "Please write a shorter chain-of-thought (at most 3-5 short paragraphs) "
                "and provide a complete, self-contained kernel in a single "
                "```python ... ``` block."
            )
        return (
            "Your response did not contain a Python code block. "
            "Please provide your complete kernel inside a ```python ... ``` block."
        )
    if not eval_result.get("compiled", False):
        err = eval_result.get("error_message") or "Unknown error"
        if "returned None" in err or "lock file" in err:
            return (
                "The evaluation infrastructure encountered a transient error (not a code bug). "
                "Please resubmit your previous kernel unchanged."
            )
        hint = ""
        if "illegal memory access" in err or "cudaErrorIllegalAddress" in err:
            hint = (
                "This is typically caused by an out-of-bounds array access — "
                "check your thread/block index calculations and ensure all "
                "pointers are within bounds.\n"
            )
        return (
            "Your kernel failed to compile or run with the following error:\n"
            f"```\n{err}\n```\n"
            f"{hint}"
            "Please fix the error and provide a corrected, complete kernel."
        )
    if not eval_result.get("correctness", False):
        passed = eval_result.get("tests_passed", 0)
        total = eval_result.get("tests_total", 0)
        meta = eval_result.get("metadata", {})
        runtime_err = meta.get("runtime_error") or meta.get("correctness_issue")
        msg = (
            f"Your kernel compiled but failed correctness checks "
            f"({passed}/{total} tests passed).\n"
        )
        if runtime_err:
            msg += f"Runtime error during testing:\n```\n{runtime_err}\n```\n"
        msg += "Please debug and fix the correctness issue."
        return msg
    spd = eval_result.get("speedup") or 0.0
    return (
        f"Your kernel is correct with speedup {spd:.3f}x vs PyTorch. "
        "Please optimize it further to achieve higher speedup."
    )


# =============================================================================
# Training Function
# =============================================================================


@app.function(
    image=training_image,
    gpu="H200",
    timeout=3600 * 24,
    volumes={"/runs": runs_volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
)
def run_training(
    run_name: str = "qwen3-8b-kbl1-multiturn-v1-0p45t-3em5lr",
    resume_checkpoint: str = "",
):
    """Multi-turn GRPO training over all Level 1 problems, easiest → hardest."""
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, "/root/train")

    import datetime
    import zoneinfo as _zi

    import torch
    import torch.nn.functional as F
    import wandb
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kernelbench.utils import extract_last_code
    from dataset import load_kernelbench_dataset

    _tz = _zi.ZoneInfo("America/New_York")

    def _now_et() -> str:
        return datetime.datetime.now(_tz).strftime("%Y-%m-%d %H:%M:%S ET")

    # ── W&B ──────────────────────────────────────────────────────────────────
    os.environ["WANDB_HTTP_TIMEOUT"] = "60"
    os.environ["WANDB_GRAPHQL_TIMEOUT"] = "60"
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "model": MODEL_NAME,
            "lr": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "num_generations": NUM_GENERATIONS,
            "num_turns": NUM_TURNS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "epsilon_high": EPSILON_HIGH,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "gamma": GAMMA,
            "correct_threshold": CORRECT_THRESHOLD,
            "stuck_threshold": STUCK_THRESHOLD,
            "dataset": "KernelBench Level 1 (difficulty order)",
        },
    )

    # ── Model + tokenizer ────────────────────────────────────────────────────
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if resume_checkpoint:
        checkpoint_path = f"/runs/{run_name}/{resume_checkpoint}"
        print(f"Resuming LoRA weights from {checkpoint_path}")
        model = PeftModel.from_pretrained(
            base_model, checkpoint_path, is_trainable=True
        )
    else:
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
        model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds, _ = load_kernelbench_dataset(level=1, difficulty_sort=True)
    print(f"\nDataset: {len(train_ds)} problems (difficulty order)")
    for p in train_ds:
        print(f"  {p['problem_name']}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ── Modal evaluator ──────────────────────────────────────────────────────
    EvalCls = modal.Cls.from_name("kernel-rl-evaluator", "KernelEvaluator")
    evaluator = EvalCls.with_options(gpu=EVAL_GPU, timeout=240)()
    _zero = {
        "format_ok": True,
        "compiled": False,
        "correctness": False,
        "speedup": None,
    }

    def _strip_verbose(k: str) -> str:
        import re

        return re.sub(r"\bverbose\s*=\s*True", "verbose=False", k)

    def _eval_batch(kernels: list[str | None], ref_code: str) -> list[dict]:
        def _args(k):
            return (
                ref_code,
                _strip_verbose(k) if k else "",
                BACKEND,
                NUM_CORRECT_TRIALS,
                True,
                NUM_PERF_TRIALS,
                EVAL_GPU_ARCH,
                PRECISION,
                TIMING_METHOD,
                True,
                10.0,
            )

        eval_indices = [i for i, k in enumerate(kernels) if k is not None]
        results = [_zero.copy() for _ in kernels]
        if not eval_indices:
            return results

        def _run_starmap(indices):
            raw = list(
                evaluator.evaluate.starmap(
                    [_args(kernels[i]) for i in indices],
                    return_exceptions=True,
                    wrap_returned_exceptions=False,
                )
            )
            return {i: (r if isinstance(r, dict) else _zero.copy()) for i, r in zip(indices, raw)}

        first_pass = _run_starmap(eval_indices)
        for i, r in first_pass.items():
            results[i] = r

        # Retry any None-result (lock file / transient infra failure)
        retry_indices = [
            i for i, r in first_pass.items()
            if r.get("error_message", "").startswith("Evaluation returned None")
        ]
        if retry_indices:
            print(f"  [eval] retrying {len(retry_indices)} transient None result(s): {retry_indices}")
            retry_pass = _run_starmap(retry_indices)
            for i, r in retry_pass.items():
                results[i] = r

        return results

    # ── Generation helper ────────────────────────────────────────────────────

    def _generate(
        conversations: list[list[dict]], max_new_tokens: int, temperature: float
    ) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS
            )
            for conv in conversations
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion_texts, prompt_ids_list, comp_ids_list = [], [], []
        for i in range(len(conversations)):
            pad_len = (inputs["attention_mask"][i] == 0).sum().item()
            prompt_tok = inputs["input_ids"][i, pad_len:]
            comp_tok = output_ids[i, input_len:]

            eos_pos = (comp_tok == tokenizer.eos_token_id).nonzero()
            if len(eos_pos):
                comp_tok = comp_tok[: eos_pos[0, 0] + 1]

            completion_texts.append(
                tokenizer.decode(comp_tok, skip_special_tokens=False)
            )
            prompt_ids_list.append(prompt_tok.unsqueeze(0))
            comp_ids_list.append(comp_tok.unsqueeze(0))

        return completion_texts, prompt_ids_list, comp_ids_list

    # ── Log-prob helpers ─────────────────────────────────────────────────────

    def _sum_logp_nograd(
        prompt_ids: torch.Tensor, comp_ids: torch.Tensor
    ) -> tuple[float, int]:
        n_tok = comp_ids.shape[1]
        p_len = prompt_ids.shape[1]
        full_ids = torch.cat([prompt_ids, comp_ids], dim=-1)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(full_ids).logits[0, p_len - 1 : p_len - 1 + n_tok].clone()
        vocab_size = logits.shape[-1]
        valid = comp_ids[0] < vocab_size
        safe_ids = comp_ids[0].clamp(0, vocab_size - 1)
        lps = F.log_softmax(logits, dim=-1)
        per_tok = lps.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1) * valid
        return per_tok.sum().item(), n_tok

    def _grpo_loss_single(
        prompt_ids: torch.Tensor,
        comp_ids: torch.Tensor,
        old_sum_lp: float,
        n_tok: int,
        advantage: torch.Tensor,
    ) -> torch.Tensor:
        full_ids = torch.cat([prompt_ids, comp_ids], dim=-1)
        p_len = prompt_ids.shape[1]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(full_ids).logits[0, p_len - 1 : p_len - 1 + n_tok].clone()
        vocab_size = logits.shape[-1]
        valid = comp_ids[0] < vocab_size
        safe_ids = comp_ids[0].clamp(0, vocab_size - 1)
        lps = F.log_softmax(logits, dim=-1)
        per_tok = lps.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1) * valid
        new_sum_lp = per_tok.sum()
        ratio = torch.exp((new_sum_lp - old_sum_lp) / max(n_tok, 1))
        clipped = torch.clamp(ratio, 1.0 - EPSILON_HIGH, 1.0 + EPSILON_HIGH)
        return -torch.min(ratio * advantage, clipped * advantage)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assistant_memory(kernel: str | None, summary: str) -> str:
        kernel_block = (
            f"```python\n{kernel}\n```" if kernel else "(kernel not extracted)"
        )
        summary_block = (
            f"<summary>\n{summary}\n</summary>"
            if summary
            else "<summary>(No summary provided.)</summary>"
        )
        return f"{kernel_block}\n\n{summary_block}"

    def _grpo_advantages(returns: torch.Tensor) -> torch.Tensor | None:
        if returns.std() < 1e-6:
            return None
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def _print_turn(step, turn, traj, conv, completion, reward):
        bar = "=" * 72
        print(f"\n{bar}")
        print(f"STEP {step}  TURN {turn}  TRAJECTORY {traj}  reward={reward:.4f}")
        print(f"{bar}")
        print("── CONTEXT (conversation sent to model) ──")
        for msg in conv:
            role = msg["role"].upper()
            print(f"\n[{role}]\n{msg['content']}")
        print("\n── COMPLETION ──")
        print(completion)
        print(bar)

    # ── Training state ────────────────────────────────────────────────────────
    output_dir = f"/runs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    optimizer.zero_grad()

    consecutive_stuck = 0
    reset_count = 0
    if resume_checkpoint:
        state_path = f"/runs/{run_name}/{resume_checkpoint}/state.json"
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            global_step = state["global_step"]
            problem_idx = state["problem_idx"]
            print(
                f"Resumed from state.json: global_step={global_step}, problem_idx={problem_idx}"
            )
        else:
            # No state.json — infer from checkpoint name (e.g. "step_20" → 20)
            try:
                global_step = int(resume_checkpoint.rsplit("_", 1)[-1])
            except ValueError:
                raise ValueError(
                    f"Cannot infer step number from checkpoint name '{resume_checkpoint}'. "
                    "Expected format: step_N"
                )
            problem_idx = global_step % len(train_ds)
            print(
                f"No state.json found — inferred global_step={global_step}, problem_idx={problem_idx}"
            )
        print(f"  Starting on problem: {train_ds[problem_idx]['problem_name']}")
    else:
        problem_idx = 0
        global_step = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    while True:
        problem = train_ds[problem_idx]
        ref_code = problem["ref_arch_src"]
        _user_content = problem["prompt"][1]["content"].replace(
            "Just output the new model code, no other text, and NO testing code!",
            "Output your response in exactly this format:\n"
            "1. A single ```python ... ``` code block with your complete ModelNew implementation.\n"
            "2. A <summary>...</summary> block with 3-5 sentences describing your approach and key design choices.\n"
            "Nothing else.",
        )
        base_conv = [
            {"role": "system", "content": MULTITURN_SYSTEM_PROMPT},
            {"role": "user", "content": _user_content},
        ]
        problem_name = problem["problem_name"]
        t0 = time.time()

        print(f"\n{'='*80}")
        print(
            f"STEP {global_step + 1} START  ({problem_name})  [problem {problem_idx + 1}/{len(train_ds)}]"
        )
        print(f"temp={TEMPERATURE:.2f}  lr={LEARNING_RATE:.2e}  {_now_et()}")
        print(f"{'='*80}")

        # Per-turn accumulators — indexed [turn_idx][traj_idx]
        all_texts: list[list[str]] = []
        all_kernels: list[list[str | None]] = []
        all_summaries: list[list[str]] = []
        all_evals: list[list[dict]] = []
        all_rewards: list[list[float]] = []
        all_prompt_ids: list[list[torch.Tensor]] = []
        all_comp_ids: list[list[torch.Tensor]] = []
        all_old_logps: list[list[tuple]] = []

        # ── Multi-turn rollout ───────────────────────────────────────────────
        for turn_idx in range(NUM_TURNS):
            turn_num = turn_idx + 1

            if turn_idx == 0:
                convs = [base_conv] * NUM_GENERATIONS
            else:
                convs = []
                for i in range(NUM_GENERATIONS):
                    history = []
                    for prev in range(turn_idx):
                        history.append(
                            {
                                "role": "assistant",
                                "content": _assistant_memory(
                                    all_kernels[prev][i], all_summaries[prev][i]
                                ),
                            }
                        )
                        history.append(
                            {
                                "role": "user",
                                "content": _feedback_message(
                                    all_evals[prev][i],
                                    all_kernels[prev][i],
                                    all_texts[prev][i],
                                ),
                            }
                        )
                    convs.append(base_conv + history)

            print(f"\n{'='*80}")
            print(
                f"STEP {global_step + 1} TURN {turn_num}/{NUM_TURNS}  ({problem_name})"
            )
            print(f"temp={TEMPERATURE:.2f}  lr={LEARNING_RATE:.2e}  {_now_et()}")
            print(f"{'='*80}")

            texts, prompt_ids, comp_ids = _generate(convs, MAX_NEW_TOKENS, TEMPERATURE)
            kernels = [extract_last_code(t, ["python", "cpp"]) for t in texts]
            summaries = [_extract_summary(t) for t in texts]
            old_logps = [_sum_logp_nograd(p, c) for p, c in zip(prompt_ids, comp_ids)]
            evals = _eval_batch(kernels, ref_code)
            rewards = [_compute_reward(e, k or "") for e, k in zip(evals, kernels)]

            if turn_idx == NUM_TURNS - 1:
                for i in range(NUM_GENERATIONS):
                    _print_turn(
                        global_step + 1, turn_num, i, convs[i], texts[i], rewards[i]
                    )

            all_texts.append(texts)
            all_kernels.append(kernels)
            all_summaries.append(summaries)
            all_evals.append(evals)
            all_rewards.append(rewards)
            all_prompt_ids.append(prompt_ids)
            all_comp_ids.append(comp_ids)
            all_old_logps.append(old_logps)

        # ── Discounted returns ───────────────────────────────────────────────
        G: list[list[float]] = [[0.0] * NUM_GENERATIONS for _ in range(NUM_TURNS)]
        for i in range(NUM_GENERATIONS):
            G[NUM_TURNS - 1][i] = all_rewards[NUM_TURNS - 1][i]
            for t in range(NUM_TURNS - 2, -1, -1):
                G[t][i] = all_rewards[t][i] + GAMMA * G[t + 1][i]

        advantages: list[torch.Tensor | None] = []
        for t in range(NUM_TURNS):
            G_t = torch.tensor(G[t], dtype=torch.float32, device=model.device)
            advantages.append(_grpo_advantages(G_t))

        # ── Per-trajectory stats ─────────────────────────────────────────────
        max_rewards = [
            max(all_rewards[t][i] for t in range(NUM_TURNS))
            for i in range(NUM_GENERATIONS)
        ]
        best_turns = [
            max(range(NUM_TURNS), key=lambda t: all_rewards[t][i])
            for i in range(NUM_GENERATIONS)
        ]
        t1_rewards = [all_rewards[0][i] for i in range(NUM_GENERATIONS)]

        t1_correct = sum(1 for r in t1_rewards if r > CORRECT_THRESHOLD)
        any_correct = sum(1 for r in max_rewards if r > CORRECT_THRESHOLD)
        mt_recover = sum(
            1
            for i in range(NUM_GENERATIONS)
            if t1_rewards[i] <= CORRECT_THRESHOLD and max_rewards[i] > CORRECT_THRESHOLD
        )
        mean_t1 = sum(t1_rewards) / NUM_GENERATIONS
        mean_best = sum(max_rewards) / NUM_GENERATIONS
        max_best = max(max_rewards)
        best_lengths = [
            all_comp_ids[best_turns[i]][i].shape[1] for i in range(NUM_GENERATIONS)
        ]
        mean_best_len = sum(best_lengths) / NUM_GENERATIONS

        # ── GRPO backward pass ───────────────────────────────────────────────
        grad_norm_val: float | None = None
        if all(adv is None for adv in advantages):
            print(
                f"[step {global_step + 1}] Zero variance in all turns — skipping update"
            )
            optimizer.zero_grad()
        else:
            torch.cuda.empty_cache()
            peak_mem_gb = 0.0
            for i in range(NUM_GENERATIONS):
                torch.cuda.empty_cache()
                for t in range(NUM_TURNS):
                    if advantages[t] is None:
                        continue
                    sum_lp, n_tok = all_old_logps[t][i]
                    torch.cuda.reset_peak_memory_stats()
                    loss = _grpo_loss_single(
                        all_prompt_ids[t][i],
                        all_comp_ids[t][i],
                        sum_lp,
                        n_tok,
                        advantages[t][i],
                    )
                    (loss / NUM_GENERATIONS).backward()
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9
                    seq_len = (
                        all_prompt_ids[t][i].shape[1] + all_comp_ids[t][i].shape[1]
                    )
                    print(
                        f"  [mem] traj={i} turn={t+1} seq_len={seq_len} peak={mem_gb:.2f} GB"
                    )
                    peak_mem_gb = max(peak_mem_gb, mem_gb)
            print(
                f"  [mem] overall peak across all backward passes: {peak_mem_gb:.2f} GB"
            )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM
            )
            optimizer.step()
            optimizer.zero_grad()
            grad_norm_val = grad_norm.item()

        # ── Summary table ────────────────────────────────────────────────────
        bar = "=" * 80
        _summary_time = _now_et()
        print(f"\n{bar}")
        print(
            f"STEP {global_step + 1} SUMMARY  ({problem_name})  [problem {problem_idx + 1}/{len(train_ds)}]"
        )
        print(f"temp={TEMPERATURE:.2f}  lr={LEARNING_RATE:.2e}  {_summary_time}")
        print(f"{bar}")
        print(f"  T1 correct   : {t1_correct} / {NUM_GENERATIONS}")
        print(f"  Any correct  : {any_correct} / {NUM_GENERATIONS}")
        print(f"  MT recover   : {mt_recover}")
        print(f"  Mean T1 rwd  : {mean_t1:.4f}")
        print(f"  Mean best    : {mean_best:.4f}")
        print(f"  Max best     : {max_best:.4f}")
        print(
            f"  Grad norm    : {grad_norm_val:.4f}"
            if grad_norm_val is not None
            else "  Grad norm    : N/A (skipped)"
        )
        print(f"  Mean best L  : {mean_best_len:.0f} tokens")
        print(
            f"  Stuck streak : {0 if any_correct > 0 else consecutive_stuck + 1} / {STUCK_THRESHOLD}"
        )
        print(f"  Step time    : {(time.time() - t0) / 60:.1f} min")
        print(f"-" * 80)

        col = 9
        turn_labels = "  ".join(
            f"{'Turn ' + str(t + 1):>{col}}" for t in range(NUM_TURNS)
        )
        print(f"{'':15}{turn_labels}  {'BEST':>{col}}")
        for i in range(NUM_GENERATIONS):
            comp_lens = [all_comp_ids[t][i].shape[1] for t in range(NUM_TURNS)]
            best_t = best_turns[i]
            best_adv = (
                f"{advantages[best_t][i].item():>{col}.4f}"
                if advantages[best_t] is not None
                else f"{'N/A':>{col}}"
            )
            rewards_str = "  ".join(
                f"{all_rewards[t][i]:>{col}.4f}" for t in range(NUM_TURNS)
            )
            adjusted_str = "  ".join(f"{G[t][i]:>{col}.4f}" for t in range(NUM_TURNS))
            adv_str = "  ".join(
                (
                    f"{advantages[t][i].item():>{col}.4f}"
                    if advantages[t] is not None
                    else f"{'N/A':>{col}}"
                )
                for t in range(NUM_TURNS)
            )
            len_str = "  ".join(f"{comp_lens[t]:>{col}d}" for t in range(NUM_TURNS))
            print(
                f"Traj {i:2d}  raw:  {rewards_str}  {all_rewards[best_t][i]:>{col}.4f}"
            )
            print(f"         adj:  {adjusted_str}  {G[best_t][i]:>{col}.4f}")
            print(f"         adv:  {adv_str}  {best_adv}")
            print(f"         len:  {len_str}  {comp_lens[best_t]:>{col}d}")
        print(bar)

        # ── W&B logging ──────────────────────────────────────────────────────
        log_dict: dict = {
            "step": global_step + 1,
            "problem_idx": problem_idx,
            "problem_name_idx": problem_idx,  # numeric for plotting
            "t1_correct": t1_correct,
            "any_correct": any_correct,
            "mt_recover": mt_recover,
            "mean_t1_reward": mean_t1,
            "mean_best_reward": mean_best,
            "max_best_reward": max_best,
            "grad_norm": grad_norm_val if grad_norm_val is not None else 0.0,
            "mean_best_len": mean_best_len,
            "consecutive_stuck": consecutive_stuck,
            "step_time_min": (time.time() - t0) / 60,
            "reset_count": reset_count,
        }
        for t in range(NUM_TURNS):
            r_t = [all_rewards[t][i] for i in range(NUM_GENERATIONS)]
            log_dict[f"t{t + 1}_reward_mean"] = sum(r_t) / len(r_t)
            log_dict[f"t{t + 1}_compile_rate"] = (
                sum(1 for e in all_evals[t] if e.get("compiled")) / NUM_GENERATIONS
            )
            log_dict[f"G{t + 1}_mean"] = sum(G[t]) / NUM_GENERATIONS
        wandb.log(log_dict, step=global_step, commit=True)

        # ── Stuck detection ──────────────────────────────────────────────────
        if any_correct > 0:
            consecutive_stuck = 0
        else:
            consecutive_stuck += 1
            print(
                f"[step {global_step + 1}] No correct kernel — stuck streak {consecutive_stuck}/{STUCK_THRESHOLD}"
            )
            if consecutive_stuck >= STUCK_THRESHOLD:
                ckpt = f"{output_dir}/reset_{reset_count}"
                print(
                    f"[step {global_step + 1}] {STUCK_THRESHOLD} consecutive stuck steps — "
                    f"saving checkpoint to {ckpt} and restarting from easiest problem"
                )
                model.save_pretrained(ckpt)
                with open(f"{ckpt}/state.json", "w") as f:
                    json.dump(
                        {
                            "global_step": global_step,
                            "problem_idx": 0,
                            "reset_count": reset_count,
                        },
                        f,
                    )
                runs_volume.commit()
                reset_count += 1
                consecutive_stuck = 0
                global_step += 1
                problem_idx = 0  # restart from easiest
                continue

        global_step += 1
        problem_idx = (problem_idx + 1) % len(train_ds)

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if global_step % SAVE_STEPS == 0:
            ckpt = f"{output_dir}/step_{global_step}"
            model.save_pretrained(ckpt)
            with open(f"{ckpt}/state.json", "w") as f:
                json.dump({"global_step": global_step, "problem_idx": problem_idx}, f)
            runs_volume.commit()
            print(
                f"[step {global_step}] Saved checkpoint: {ckpt} (problem_idx={problem_idx})"
            )

    # (unreachable — training runs until manually stopped or Modal timeout)
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    runs_volume.commit()
    wandb.finish()


# =============================================================================
# Local entrypoint
# =============================================================================


@app.local_entrypoint()
def main(
    run_name: str = "qwen3-8b-kbl1-multiturn-v1-0p45t-3em5lr",
    resume_checkpoint: str = "",
):
    run_training.remote(run_name=run_name, resume_checkpoint=resume_checkpoint)
