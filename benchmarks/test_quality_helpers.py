from benchmark_quality import (
    aggregate_quality,
    choose_lowest_loss,
    format_quality_report,
    quality_gate,
)


def test_choose_lowest_loss_returns_lowest_average_loss_choice():
    prediction = choose_lowest_loss(
        {
            "A": {"loss": 2.0, "tokens": 1},
            "B": {"loss": 3.0, "tokens": 3},
            "C": {"loss": 0.8, "tokens": 1},
        }
    )

    assert prediction == "C"


def test_aggregate_quality_uses_accuracy_as_primary_metric():
    summary = aggregate_quality(
        [
            {"id": "one", "answer": "A", "prediction": "A"},
            {"id": "two", "answer": "B", "prediction": "A"},
            {"id": "three", "answer": "C", "prediction": "C"},
        ]
    )

    assert summary["primary_metric"] == "task_accuracy"
    assert summary["correct"] == 2
    assert summary["total"] == 3
    assert abs(summary["accuracy"] - (2 / 3)) < 1e-12


def test_quality_gate_compares_accuracy_delta_to_baseline():
    candidate = {"accuracy": 0.72, "total": 25}
    baseline = {"accuracy": 0.80, "total": 25}

    gate = quality_gate(candidate, baseline, max_accuracy_drop=0.05, min_tasks=20)

    assert not gate["pass"]
    assert gate["accuracy_drop"] == 0.08
    assert "exceeds" in gate["reason"]


def test_quality_gate_passes_when_candidate_is_within_budget():
    candidate = {"accuracy": 0.77, "total": 25}
    baseline = {"accuracy": 0.80, "total": 25}

    gate = quality_gate(candidate, baseline, max_accuracy_drop=0.05, min_tasks=20)

    assert gate["pass"]
    assert gate["reason"] == "candidate quality is within the allowed baseline drop"


def test_format_quality_report_keeps_ppl_secondary():
    report = format_quality_report(
        {"accuracy": 0.75, "correct": 3, "total": 4},
        {"pass": True, "accuracy_drop": 0.02, "reason": "ok"},
        perplexity={"perplexity": 42.0},
    )

    assert "Primary metric: task_accuracy" in report
    assert "PPL diagnostic: 42.0000" in report
    assert "PPL diagnostic" in report.splitlines()[-1]
