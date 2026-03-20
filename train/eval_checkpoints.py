"""
eval_checkpoints.py — evaluate model checkpoints on KernelBench Level 1 test set.

For each model:
  - 8 trajectories × 4 turns per test problem
  - Per-problem score = mean of top-{TOP_K_SCORE} trajectory best rewards
  - Final score = mean of per-problem scores across the test set

Launch (single checkpoint):
    modal run train/eval_checkpoints.py --run-name <run_name> --label s28

Launch (base model):
    modal run train/eval_checkpoints.py --run-name <run_name> --label base

Checkpoints are loaded from the Modal volume at /runs/<run_name>/step_<N>/.
"""

from __future__ import annotations

import json
import os
import sys
import time

import modal

# =============================================================================
# Modal infrastructure
# =============================================================================

runs_volume = modal.Volume.from_name("kernelrl-runs", create_if_missing=True)

eval_image = (
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
        "modal",
        "kernelbench @ git+https://github.com/ScalingIntelligence/KernelBench.git@main",
    )
    .pip_install("wheel", "packaging", "ninja")
    .add_local_dir(
        local_path=os.path.dirname(__file__),
        remote_path="/root/train",
    )
)

app = modal.App("kernelrl-eval")

# =============================================================================
# Configuration
# =============================================================================

# Generation
MODEL_NAME      = "Qwen/Qwen3-8B"
NUM_GENERATIONS = 8      # parallel trajectories per problem
NUM_TURNS       = 4      # max turns per trajectory
MAX_NEW_TOKENS  = 8192
TEMPERATURE     = 0.45
CHAT_KWARGS     = {"enable_thinking": True, "thinking_budget": 1024}

# Scoring
TOP_K_SCORE = 2   # per-problem score = mean of top-K trajectory best rewards

# Kernel evaluation (must match reward.py)
EVAL_GPU           = "A100-80GB"
EVAL_GPU_ARCH      = ["Ampere"]
NUM_CORRECT_TRIALS = 6
NUM_PERF_TRIALS    = 40
TIMING_METHOD      = "cuda_event"
PRECISION          = "fp32"
BACKEND            = "cuda"
CORRECTNESS_WEIGHT = 0.3
MAX_SPEED_REWARD   = 10.0
STRICT_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",
    "lazy_eval",
    "cuda_impl",
    "torch_computation_ops",
]

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
# Reward + feedback helpers (defined at module level so Modal can serialize them)
# =============================================================================


def _compute_reward(eval_result: dict, kernel_code: str = "") -> float:
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


def _is_truncated(text: str, kernel: str | None) -> bool:
    if kernel is not None:
        return False
    return "```python" in text or "```cpp" in text


def _extract_summary(text: str) -> str:
    import re
    m = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _feedback_message(eval_result: dict, kernel: str | None, text: str) -> str:
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
                "check your thread/block index calculations.\n"
            )
        return (
            f"Your kernel failed to compile or run with the following error:\n"
            f"```\n{err}\n```\n{hint}"
            "Please fix the error and provide a corrected, complete kernel."
        )
    if not eval_result.get("correctness", False):
        passed = eval_result.get("tests_passed", 0)
        total  = eval_result.get("tests_total", 0)
        meta   = eval_result.get("metadata", {})
        runtime_err = meta.get("runtime_error") or meta.get("correctness_issue")
        msg = f"Your kernel compiled but failed correctness checks ({passed}/{total} tests passed).\n"
        if runtime_err:
            msg += f"Runtime error during testing:\n```\n{runtime_err}\n```\n"
        msg += "Please debug and fix the correctness issue."
        return msg
    spd = eval_result.get("speedup") or 0.0
    return (
        f"Your kernel is correct with speedup {spd:.3f}x vs PyTorch. "
        "Please optimize it further to achieve higher speedup."
    )


def _assistant_memory(kernel: str | None, summary: str) -> str:
    kernel_block = f"```python\n{kernel}\n```" if kernel else "(kernel not extracted)"
    summary_block = (
        f"<summary>\n{summary}\n</summary>"
        if summary
        else "<summary>(No summary provided.)</summary>"
    )
    return f"{kernel_block}\n\n{summary_block}"


# =============================================================================
# Evaluation function (one Modal container per model)
# =============================================================================


@app.function(
    image=eval_image,
    gpu="H200",
    timeout=3600 * 12,
    volumes={"/runs": runs_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def evaluate_model(run_name: str, label: str, checkpoint: str | None) -> dict:
    """Evaluate one model on the full Level 1 test set and return results."""
    import re
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kernelbench.utils import extract_last_code

    sys.path.insert(0, "/root/train")
    from dataset import load_kernelbench_dataset

    print(f"\n{'='*70}")
    print(f"EVAL  label={label}  checkpoint={checkpoint}")
    print(f"{'='*70}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if checkpoint is not None:
        ckpt_path = f"/runs/{run_name}/{checkpoint}"
        print(f"Loading LoRA from {ckpt_path}")
        model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=False)
    else:
        print("Using base model (no LoRA)")
        model = base_model

    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    _, test_ds = load_kernelbench_dataset(level=1, difficulty_sort=True)
    print(f"Test set: {len(test_ds)} problems\n")

    # ── Modal evaluator ───────────────────────────────────────────────────────
    EvalCls   = modal.Cls.from_name("kernel-rl-evaluator", "KernelEvaluator")
    evaluator = EvalCls.with_options(gpu=EVAL_GPU, timeout=240)()
    _zero     = {"format_ok": True, "compiled": False, "correctness": False, "speedup": None}

    def _strip_verbose(k: str) -> str:
        return re.sub(r"\bverbose\s*=\s*True", "verbose=False", k)

    def _eval_batch(kernels: list[str | None], ref_code: str) -> list[dict]:
        def _args(k):
            return (
                ref_code,
                _strip_verbose(k) if k else "",
                BACKEND, NUM_CORRECT_TRIALS, True, NUM_PERF_TRIALS,
                EVAL_GPU_ARCH, PRECISION, TIMING_METHOD, True, 10.0,
            )

        eval_indices = [i for i, k in enumerate(kernels) if k is not None]
        results = [_zero.copy() for _ in kernels]
        if not eval_indices:
            return results

        def _run(indices):
            raw = list(
                evaluator.evaluate.starmap(
                    [_args(kernels[i]) for i in indices],
                    return_exceptions=True,
                    wrap_returned_exceptions=False,
                )
            )
            return {i: (r if isinstance(r, dict) else _zero.copy()) for i, r in zip(indices, raw)}

        first = _run(eval_indices)
        for i, r in first.items():
            results[i] = r

        retry = [
            i for i, r in first.items()
            if (r.get("error_message") or "").startswith("Evaluation returned None")
        ]
        if retry:
            print(f"  [eval] retrying {len(retry)} transient None result(s)")
            for i, r in _run(retry).items():
                results[i] = r

        return results

    def _generate(convs: list[list[dict]]) -> tuple[list[str], list[int]]:
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS
            )
            for conv in convs
        ]
        inputs   = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        comp_texts, comp_lens = [], []
        for i in range(len(convs)):
            comp_tok = output_ids[i, input_len:]
            eos_pos  = (comp_tok == tokenizer.eos_token_id).nonzero()
            if len(eos_pos):
                comp_tok = comp_tok[: eos_pos[0, 0] + 1]
            comp_texts.append(tokenizer.decode(comp_tok, skip_special_tokens=False))
            comp_lens.append(comp_tok.shape[0])

        return comp_texts, comp_lens

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir  = f"/runs/{run_name}/eval_results"
    out_file = f"{out_dir}/{label}.json"
    os.makedirs(out_dir, exist_ok=True)

    # ── Problem loop ──────────────────────────────────────────────────────────
    problem_scores: list[float] = []
    all_results:    list[dict]  = []

    for prob_idx, problem in enumerate(test_ds):
        problem_name = problem["problem_name"]
        ref_code     = problem["ref_arch_src"]
        t0 = time.time()

        user_content = problem["prompt"][1]["content"].replace(
            "Just output the new model code, no other text, and NO testing code!",
            "Output your response in exactly this format:\n"
            "1. A single ```python ... ``` code block with your complete ModelNew implementation.\n"
            "2. A <summary>...</summary> block with 3-5 sentences describing your approach and key design choices.\n"
            "Nothing else.",
        )
        base_conv = [
            {"role": "system", "content": MULTITURN_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        print(f"[{label}] Problem {prob_idx+1}/{len(test_ds)}: {problem_name}")

        # Per-trajectory accumulators
        traj_texts:     list[list[str]]      = [[] for _ in range(NUM_GENERATIONS)]
        traj_kernels:   list[list[str|None]] = [[] for _ in range(NUM_GENERATIONS)]
        traj_summaries: list[list[str]]      = [[] for _ in range(NUM_GENERATIONS)]
        traj_evals:     list[list[dict]]     = [[] for _ in range(NUM_GENERATIONS)]
        traj_rewards:   list[list[float]]    = [[] for _ in range(NUM_GENERATIONS)]

        for turn_idx in range(NUM_TURNS):
            turn_num = turn_idx + 1

            if turn_idx == 0:
                convs = [list(base_conv) for _ in range(NUM_GENERATIONS)]
            else:
                convs = []
                for i in range(NUM_GENERATIONS):
                    history = []
                    for prev in range(turn_idx):
                        history.append({
                            "role":    "assistant",
                            "content": _assistant_memory(traj_kernels[i][prev], traj_summaries[i][prev]),
                        })
                        history.append({
                            "role":    "user",
                            "content": _feedback_message(traj_evals[i][prev], traj_kernels[i][prev], traj_texts[i][prev]),
                        })
                    convs.append(base_conv + history)

            texts, comp_lens = _generate(convs)
            kernels   = [extract_last_code(t, ["python", "cpp"]) for t in texts]
            summaries = [_extract_summary(t) for t in texts]
            evals     = _eval_batch(kernels, ref_code)
            rewards   = [_compute_reward(e, k or "") for e, k in zip(evals, kernels)]

            for i in range(NUM_GENERATIONS):
                traj_texts[i].append(texts[i])
                traj_kernels[i].append(kernels[i])
                traj_summaries[i].append(summaries[i])
                traj_evals[i].append(evals[i])
                traj_rewards[i].append(rewards[i])

            correct   = sum(1 for r in rewards if r > 0.1)
            mean_rwd  = sum(rewards) / NUM_GENERATIONS
            mean_len  = sum(comp_lens) / NUM_GENERATIONS
            print(
                f"  turn {turn_num}: correct={correct}/{NUM_GENERATIONS} "
                f"mean_reward={mean_rwd:.4f}  mean_len={mean_len:.0f} tok"
            )

        # Per-trajectory best = max reward across all turns
        traj_bests        = [max(traj_rewards[i]) for i in range(NUM_GENERATIONS)]
        traj_bests_sorted = sorted(traj_bests, reverse=True)
        top_k             = traj_bests_sorted[:TOP_K_SCORE]
        prob_score        = sum(top_k) / len(top_k)
        elapsed           = time.time() - t0

        # ── Problem summary ───────────────────────────────────────────────────
        bar = "=" * 72
        chk_str = checkpoint or "base"
        running_avg = (sum(problem_scores) + prob_score) / (prob_idx + 1)
        print(f"\n{bar}")
        print(f"  [{label} / {chk_str}]  Problem {prob_idx+1}/{len(test_ds)}: {problem_name}")
        print(f"  Elapsed: {elapsed/60:.1f} min   Running avg score: {running_avg:.4f}")
        print(bar)

        # All 32 rewards in a grid: rows = trajectories, cols = turns
        col = 9
        header = "            " + "  ".join(f"{'Turn '+str(t+1):>{col}}" for t in range(NUM_TURNS)) + f"  {'Best':>{col}}"
        print(header)
        print(f"  {'-'*68}")
        for i in range(NUM_GENERATIONS):
            row = "  ".join(f"{traj_rewards[i][t]:>{col}.4f}" for t in range(NUM_TURNS))
            best_marker = " *" if traj_bests[i] in top_k else "  "
            print(f"  Traj {i:<3d}   {row}  {traj_bests[i]:>{col}.4f}{best_marker}")

        print(f"  {'-'*68}")
        print(f"  Bests (sorted): {[f'{r:.4f}' for r in traj_bests_sorted]}")
        print(f"  Top-{TOP_K_SCORE} avg (performance score): {prob_score:.4f}")
        print(f"{bar}\n")

        problem_scores.append(prob_score)
        all_results.append({
            "problem_name":  problem_name,
            "problem_score": prob_score,
            "top_k":         top_k,
            "traj_bests":    traj_bests_sorted,
        })

        # Save progress after each problem so results aren't lost if interrupted
        partial = {
            "label":           label,
            "checkpoint":      checkpoint,
            "problems_done":   prob_idx + 1,
            "problems_total":  len(test_ds),
            "running_score":   sum(problem_scores) / len(problem_scores),
            "top_k":           TOP_K_SCORE,
            "problem_results": all_results,
        }
        with open(out_file, "w") as f:
            json.dump(partial, f, indent=2)
        runs_volume.commit()

    final_score = sum(problem_scores) / len(problem_scores)

    print(f"\n{'='*70}")
    print(f"FINAL SCORE  label={label}")
    print(f"  {final_score:.4f}  ({len(test_ds)} problems, top-{TOP_K_SCORE} avg)")
    print(f"{'='*70}\n")

    results = {
        "label":           label,
        "checkpoint":      checkpoint,
        "final_score":     final_score,
        "num_problems":    len(test_ds),
        "top_k":           TOP_K_SCORE,
        "problem_results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    runs_volume.commit()
    print(f"Results saved to {out_file}")

    return results


# =============================================================================
# Local entrypoint — launches all models in parallel
# =============================================================================


def _resolve(label: str) -> dict:
    """
    Resolve a label to a {label, checkpoint} dict.
      "base"  -> checkpoint=None
      "s28"   -> checkpoint="step_28"
    """
    import re
    if label == "base":
        return {"label": "base", "checkpoint": None}
    m = re.fullmatch(r"s(\d+)", label)
    if m:
        return {"label": label, "checkpoint": f"step_{m.group(1)}"}
    raise ValueError(
        f"Cannot resolve label '{label}'. Use 'base' or 's<N>' (e.g. 's28')."
    )


@app.local_entrypoint()
def main(run_name: str, label: str):
    """
    Evaluate a checkpoint on the Level 1 test set.

    Examples:
        modal run train/eval_checkpoints.py --run-name qwen3-8b-kbl1-mt-v2-0p6t-2em5lr --label s28
        modal run train/eval_checkpoints.py --run-name qwen3-8b-kbl1-mt-v2-0p6t-2em5lr --label base
    """
    model = _resolve(label)
    print(f"Launching evaluation: run_name={run_name}  label={model['label']}  checkpoint={model['checkpoint']}")
    result = evaluate_model.remote(run_name, model["label"], model["checkpoint"])

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY  (top-{TOP_K_SCORE} avg per problem)")
    print("=" * 60)
    chk = result["checkpoint"] or "none (base)"
    print(f"{result['label']:<12}  {chk:<20}  {result['final_score']:>8.4f}  {result['num_problems']:>12} problems")
    print("=" * 60)
