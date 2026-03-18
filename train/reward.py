"""
Reward function for KernelBench GRPO training.

Calls the deployed Modal eval app (kernel-rl-evaluator) in parallel,
then scores each result using the reward formula from Kevin-32B:
    reward = 0.3 * correct + speedup   (if correct)
    reward = 0                          (if incorrect or format/compile fail)

The reward function signature matches what TRL GRPOTrainer expects:
    fn(prompts, completions, **kwargs) -> list[float]

where kwargs contains any extra columns from the dataset
(we use "ref_arch_src" passed through via the dataset).
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path
from typing import Any

# Must be set before wandb is imported or initialized
os.environ.setdefault("WANDB_HTTP_TIMEOUT", "60")
os.environ.setdefault("WANDB_GRAPHQL_TIMEOUT", "60")

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

import modal

# Allow importing kernelbench from the cloned repo
KERNELBENCH_SRC = Path(__file__).parent.parent / "KernelBench" / "src"
if str(KERNELBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(KERNELBENCH_SRC))

from kernelbench.utils import extract_last_code
from kernelbench.kernel_static_checker import validate_kernel_static


# GPU to use for kernel evaluation (A100 for Ampere arch)
EVAL_GPU = "A100-80GB"
EVAL_GPU_ARCH = ["Ampere"]

# Eval parameters (match Kevin-32B setup)
NUM_CORRECT_TRIALS = 6
NUM_PERF_TRIALS = 40
TIMING_METHOD = "cuda_event"
PRECISION = "fp32"
BACKEND = "cuda"

# Reward weights (match Kevin-32B)
CORRECTNESS_WEIGHT = 0.3
MAX_SPEED_REWARD = 10.0  # cap to prevent outlier kernels dominating

_reward_call_count = 0

# Curriculum reset state — checked by CurriculumResetCallback in train_grpo.py
_consecutive_stuck_steps = 0
_curriculum_reset_flag = False
_stuck_compiled_count = 0  # compiled completions across current stuck streak
_stuck_total_count = 0  # total completions across current stuck streak

# Epoch compile tracking — used by train_matmul_reductions.py
_epoch_compiled_count = 0
_epoch_total_count = 0


# Strict checks: all default strict checks + cuda_impl + torch computation ops
# torch_computation_ops is normally a warning, but we promote it to strict
# to prevent reward hacking via torch.matmul / torch.nn.functional calls
STRICT_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",
    "lazy_eval",
    "cuda_impl",  # must actually use CUDA, not just torch ops
    "torch_computation_ops",  # promoted from warning: no torch.matmul etc.
]


def _is_truncated(text: str, kernel: str | None) -> bool:
    """Return True if the completion was cut off mid-code-block."""
    if kernel is not None:
        return False  # complete code block was extracted
    return "```python" in text or "```cpp" in text


def _preflight_check(kernel_code: str | None) -> bool:
    """
    Fast local check before sending to Modal. Returns False if the kernel
    is obviously invalid — saving an nvcc compilation round-trip.
    Checks:
      - kernel_code is not empty
      - parses as valid Python
      - defines a ModelNew class
    """
    if not kernel_code:
        return False
    try:
        tree = ast.parse(kernel_code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.ClassDef) and node.name == "ModelNew"
        for node in ast.walk(tree)
    )


def _compute_reward(
    eval_result: dict[str, Any], kernel_code: str = "", truncated: bool = False
) -> float:
    """
    Convert a KernelEvaluator result dict into a scalar reward.

    Reward ladder:
      0.0        : format invalid, cheating detected, or truncated (code cut off)
      0.01       : complete code block written, but doesn't compile
      0.02       : compiles but incorrect
      0.3 - 1.0  : correct, speedup in [0, 1] (linear interpolation)
      speedup    : correct, speedup > 1.0 (capped at 10.0)
    """
    if not eval_result.get("format_ok", False):
        return 0.0

    # Static check: zero reward for reward hacking patterns
    if kernel_code:
        valid, errors, _ = validate_kernel_static(
            code=kernel_code,
            backend=BACKEND,
            precision=PRECISION,
            forbidden=STRICT_CHECKS,
            warnings=[],
        )
        if not valid:
            return 0.0

    if truncated:
        return 0.0

    # Complete code block written — small reward for not truncating
    if not eval_result.get("compiled", False):
        return 0.01

    if not eval_result.get("correctness", False):
        return 0.02

    # Correct kernel — speedup-based reward
    speedup = eval_result.get("speedup") or 0.0

    if speedup <= 1.0:
        # Linear interpolation: CORRECTNESS_WEIGHT at speedup=0.0, 1.0 at speedup=1.0
        return CORRECTNESS_WEIGHT + (1 - CORRECTNESS_WEIGHT) * speedup
    else:
        return min(speedup, MAX_SPEED_REWARD)


def _make_eval_args(
    ref_arch_src: str,
    kernel_code: str | None,
    num_correct_trials: int = NUM_CORRECT_TRIALS,
    measure_performance: bool = False,
) -> dict[str, Any]:
    """Build the kwargs dict for KernelEvaluator.evaluate.remote()."""
    return dict(
        ref_code=ref_arch_src,
        kernel_code=kernel_code or "",
        backend=BACKEND,
        num_correct_trials=num_correct_trials,
        measure_performance=measure_performance,
        num_perf_trials=NUM_PERF_TRIALS,
        gpu_arch=EVAL_GPU_ARCH,
        precision=PRECISION,
        timing_method=TIMING_METHOD,
        check_for_excessive_speedup=True,
        excessive_speedup_threshold=10.0,
    )


def kernelbench_reward_fn(
    prompts: list[Any],
    completions: list[Any],
    ref_arch_src: list[str],
    **kwargs,
) -> list[float]:
    """
    TRL-compatible reward function. Called once per batch.

    Args:
        prompts: List of chat-formatted prompts (unused, but required by TRL).
        completions: List of model completions (chat or string format).
        ref_arch_src: List of reference PyTorch source strings (from dataset).
        **kwargs: Any other dataset columns passed through by TRL.

    Returns:
        List of scalar rewards, one per completion.
    """

    # TRL may pass completions as chat dicts or raw strings
    def _get_text(c: Any) -> str:
        if isinstance(c, list):
            # Chat format: last message is the assistant response
            return c[-1]["content"] if c else ""
        return str(c)

    completion_texts = [_get_text(c) for c in completions]

    # Extract CUDA kernel code blocks from completions
    kernels = [extract_last_code(text, ["python", "cpp"]) for text in completion_texts]

    EvalCls = modal.Cls.from_name("kernel-rl-evaluator", "KernelEvaluator")
    evaluator = EvalCls.with_options(gpu=EVAL_GPU, timeout=180)()

    _zero_result = {
        "format_ok": True,
        "compiled": False,
        "correctness": False,
        "speedup": None,
    }

    def _run_starmap(args_list: list) -> list[dict]:
        raw = list(
            evaluator.evaluate.starmap(
                args_list, return_exceptions=True, wrap_returned_exceptions=False
            )
        )
        results = [r if isinstance(r, dict) else _zero_result.copy() for r in raw]
        # Retry transient None results (lock file / infra failure)
        retry_indices = [
            i for i, r in enumerate(results)
            if (r.get("error_message") or "").startswith("Evaluation returned None")
        ]
        if retry_indices:
            print(f"[reward] retrying {len(retry_indices)} transient None result(s)")
            retry_raw = list(
                evaluator.evaluate.starmap(
                    [args_list[i] for i in retry_indices],
                    return_exceptions=True,
                    wrap_returned_exceptions=False,
                )
            )
            for i, r in zip(retry_indices, retry_raw):
                results[i] = r if isinstance(r, dict) else _zero_result.copy()
        return results

    def _to_starmap_tuple(a: dict) -> tuple:
        return (
            a["ref_code"],
            a["kernel_code"],
            a["backend"],
            a["num_correct_trials"],
            a["measure_performance"],
            a["num_perf_trials"],
            a["gpu_arch"],
            a["precision"],
            a["timing_method"],
            a["check_for_excessive_speedup"],
            a["excessive_speedup_threshold"],
        )

    # Pre-flight: skip Modal entirely for kernels that fail basic Python checks
    preflight_pass = [_preflight_check(k) for k in kernels]
    results = [_zero_result.copy() for _ in kernels]

    # Single pass: correctness + perf together (KernelBench skips perf for incorrect kernels)
    eval_indices = [i for i, ok in enumerate(preflight_pass) if ok]
    if eval_indices:
        eval_args = [
            _make_eval_args(ref_arch_src[i], kernels[i], measure_performance=True)
            for i in eval_indices
        ]
        eval_results = _run_starmap([_to_starmap_tuple(a) for a in eval_args])
        for i, r in zip(eval_indices, eval_results):
            results[i] = r

    truncated_flags = [
        _is_truncated(text, k) for text, k in zip(completion_texts, kernels)
    ]

    global _reward_call_count
    _reward_call_count += 1
    problem_names = kwargs.get("problem_name", [])
    problem_name = problem_names[0] if problem_names else "unknown"
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[step {_reward_call_count}] [problem_name {problem_name}] [time {ts}]")
    p0 = prompts[0]
    user_content = p0[1]["content"] if isinstance(p0, list) and len(p0) > 1 else str(p0)
    arch_marker = "You are given the following architecture:"
    arch_start = user_content.find(arch_marker)
    # print(user_content[arch_start:] if arch_start != -1 else user_content)
    for i, r in enumerate(results):
        print(
            f"[eval {i}] compiled={r.get('compiled')} "
            f"correctness={r.get('correctness')} "
            f"speedup={r.get('speedup')} "
            f"truncated={truncated_flags[i]} "
            f"error={(r.get('error_message') or '')[:120]}"
        )
    rewards = [
        _compute_reward(r, k or "", truncated=t)
        for r, k, t in zip(results, kernels, truncated_flags)
    ]

    # Epoch compile tracking
    global _epoch_compiled_count, _epoch_total_count
    _epoch_compiled_count += sum(1 for r in results if r.get("compiled", False))
    _epoch_total_count += len(results)

    # Curriculum reset tracking
    global _consecutive_stuck_steps, _curriculum_reset_flag
    global _stuck_compiled_count, _stuck_total_count
    if max(rewards) <= 0.02:
        _consecutive_stuck_steps += 1
        _stuck_compiled_count += sum(1 for r in results if r.get("compiled", False))
        _stuck_total_count += len(results)
        compile_rate = _stuck_compiled_count / _stuck_total_count
        print(
            f"[curriculum] stuck step {_consecutive_stuck_steps}/3 "
            f"(compile_rate={compile_rate:.3f} over stuck steps)"
        )
        if _consecutive_stuck_steps >= 3:
            _curriculum_reset_flag = True
            _consecutive_stuck_steps = 0
            print("[curriculum] 3 consecutive stuck steps — reset flag set")
    else:
        _consecutive_stuck_steps = 0
        _stuck_compiled_count = 0
        _stuck_total_count = 0

    # Log all completions to W&B as a persistent table (one row per completion)
    DESCRIPTION_CHAR_LIMIT = 2048
    if _wandb is not None and _wandb.run is not None:
        step = _wandb.run.step

        def _get_prompt_text(p: Any) -> str:
            if isinstance(p, list):
                user_msg = next((m for m in p if m.get("role") == "user"), None)
                return user_msg["content"] if user_msg else ""
            return str(p)

        def _extract_thinking(text: str) -> str:
            start = text.find("<think>")
            end = text.find("</think>")
            if start != -1 and end != -1:
                return text[start + len("<think>") : end].strip()[
                    :DESCRIPTION_CHAR_LIMIT
                ]
            return ""

        table = _wandb.Table(
            columns=[
                "step",
                "prompt",
                "compiled",
                "correctness",
                "speedup",
                "reward",
                "error",
                "reasoning",
                "kernel",
            ]
        )
        for r, k, reward, text, prompt in zip(
            results, kernels, rewards, completion_texts, prompts
        ):
            table.add_data(
                step,
                _get_prompt_text(prompt)[:DESCRIPTION_CHAR_LIMIT],
                r.get("compiled"),
                r.get("correctness"),
                r.get("speedup"),
                reward,
                (r.get("error_message") or ""),
                _extract_thinking(text),
                (k or ""),
            )
        _wandb.log({"completions_log": table}, commit=False)

    return rewards
