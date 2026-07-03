#!/usr/bin/env python3
"""Quality-first benchmark helpers for FABQ-RC experiments.

Perplexity is useful as a diagnostic, but the primary comparison for this
project should be task quality: answer accuracy and generation health at a
given size/runtime point.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


DEFAULT_QUALITY_TASKS = [
    {
        "id": "arithmetic_2_plus_2",
        "prompt": "Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nAnswer:",
        "choices": {"A": " A", "B": " B", "C": " C"},
        "answer": "B",
    },
    {
        "id": "capital_france",
        "prompt": "Question: What is the capital of France?\nA. Berlin\nB. Madrid\nC. Paris\nAnswer:",
        "choices": {"A": " A", "B": " B", "C": " C"},
        "answer": "C",
    },
    {
        "id": "opposite_hot",
        "prompt": "Question: Which word is the opposite of hot?\nA. Cold\nB. Tall\nC. Fast\nAnswer:",
        "choices": {"A": " A", "B": " B", "C": " C"},
        "answer": "A",
    },
]


def choose_lowest_loss(choice_scores: dict[str, dict[str, float]]) -> str:
    """Return the choice with the lowest average continuation loss."""
    if not choice_scores:
        raise ValueError("choice_scores must not be empty")
    return min(
        choice_scores,
        key=lambda key: choice_scores[key]["loss"] / max(choice_scores[key].get("tokens", 1), 1),
    )


def aggregate_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize task quality records with accuracy as the primary metric."""
    total = len(records)
    correct = sum(1 for row in records if row.get("prediction") == row.get("answer"))
    return {
        "primary_metric": "task_accuracy",
        "perplexity_role": "diagnostic_only",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


def quality_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    *,
    max_accuracy_drop: float = 0.05,
    min_tasks: int = 20,
) -> dict[str, Any]:
    """Evaluate whether a candidate meets the quality gate."""
    total = int(candidate.get("total", 0))
    if total < min_tasks:
        return {
            "pass": False,
            "accuracy_drop": None,
            "reason": f"only {total} quality tasks scored; minimum is {min_tasks}",
        }

    if baseline is None:
        return {
            "pass": True,
            "accuracy_drop": None,
            "reason": "no baseline supplied; recorded candidate quality only",
        }

    candidate_accuracy = float(candidate.get("accuracy", 0.0))
    baseline_accuracy = float(baseline.get("accuracy", 0.0))
    drop = round(max(0.0, baseline_accuracy - candidate_accuracy), 10)
    passed = drop <= max_accuracy_drop
    if passed:
        reason = "candidate quality is within the allowed baseline drop"
    else:
        reason = f"accuracy drop {drop:.4f} exceeds allowed drop {max_accuracy_drop:.4f}"
    return {"pass": passed, "accuracy_drop": drop, "reason": reason}


def format_quality_report(
    summary: dict[str, Any],
    gate: dict[str, Any],
    *,
    perplexity: dict[str, Any] | None = None,
) -> str:
    """Return a compact human-readable quality report."""
    lines = [
        "Primary metric: task_accuracy",
        f"Accuracy: {summary.get('accuracy', 0.0):.4f} ({summary.get('correct', 0)}/{summary.get('total', 0)})",
        f"Quality gate: {'PASS' if gate.get('pass') else 'FAIL'} - {gate.get('reason')}",
    ]
    if gate.get("accuracy_drop") is not None:
        lines.append(f"Accuracy drop vs baseline: {gate['accuracy_drop']:.4f}")
    if perplexity is not None and "perplexity" in perplexity:
        lines.append(f"PPL diagnostic: {float(perplexity['perplexity']):.4f}")
    return "\n".join(lines)


def load_quality_tasks(path: str | None, max_tasks: int = 0) -> list[dict[str, Any]]:
    if not path:
        tasks = list(DEFAULT_QUALITY_TASKS)
    else:
        text = Path(path).read_text(encoding="utf-8")
        if path.endswith(".jsonl"):
            tasks = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            loaded = json.loads(text)
            tasks = loaded["tasks"] if isinstance(loaded, dict) and "tasks" in loaded else loaded
    if max_tasks:
        tasks = tasks[:max_tasks]
    return tasks


def score_continuation_loss(model, tokenizer, prompt: str, continuation: str) -> dict[str, float]:
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False)["input_ids"]
    continuation_tokens = full_ids.shape[1] - prompt_ids.shape[1]
    if continuation_tokens <= 0:
        raise ValueError("continuation must add at least one token")

    device = next(model.parameters()).device
    full_ids = full_ids.to(device)
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.inference_mode():
        out = model(input_ids=full_ids, labels=labels)
    return {"loss": float(out.loss.detach().cpu()) * continuation_tokens, "tokens": continuation_tokens}


def score_multiple_choice_task(model, tokenizer, task: dict[str, Any]) -> dict[str, Any]:
    scores = {
        choice: score_continuation_loss(model, tokenizer, task["prompt"], continuation)
        for choice, continuation in task["choices"].items()
    }
    prediction = choose_lowest_loss(scores)
    return {
        "id": task["id"],
        "answer": task["answer"],
        "prediction": prediction,
        "correct": prediction == task["answer"],
        "choice_scores": scores,
    }


def load_baseline_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "quality" in data:
        return data["quality"]
    return data


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hf-home", default=str(Path.cwd() / ".hf_cache"))
    ap.add_argument("--tasks")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--baseline-json")
    ap.add_argument("--max-accuracy-drop", type=float, default=0.05)
    ap.add_argument("--min-tasks", type=int, default=3)
    ap.add_argument("--out", default="results/quality_benchmark.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tasks = load_quality_tasks(args.tasks, args.max_tasks)
    records = [score_multiple_choice_task(model, tokenizer, task) for task in tasks]
    quality = aggregate_quality(records)
    baseline = load_baseline_summary(args.baseline_json)
    gate = quality_gate(
        quality,
        baseline,
        max_accuracy_drop=args.max_accuracy_drop,
        min_tasks=args.min_tasks,
    )
    result = {
        "repo_id": args.repo_id,
        "benchmark_kind": "quality_task_accuracy",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
        },
        "quality": quality,
        "quality_gate": gate,
        "tasks": records,
        "elapsed_sec": time.perf_counter() - started,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(format_quality_report(quality, gate))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
