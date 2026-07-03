#!/usr/bin/env python3
"""Generate Substack-ready PNG charts for the FABQ-RC project story."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent / "assets"


def save_storage_audit() -> None:
    blocks = [64, 128, 256, 512]
    bpw = [1.73, 1.55, 1.46, 1.42]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.plot(blocks, bpw, marker="o", linewidth=3, color="#1f6f8b")
    ax.axhline(1.21, color="#b33a3a", linestyle="--", linewidth=2, label="old 1.21 bpw claim")
    ax.axhline(1.18, color="#d47c1f", linestyle=":", linewidth=2, label="old 1.18 bpw claim")
    ax.set_xscale("log", base=2)
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("Binary block size", labelpad=10)
    ax.set_ylabel("Physical bits per weight", labelpad=10)
    ax.set_title("The storage audit changed the story", pad=16, weight="bold")
    ax.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False)
    for x, y in zip(blocks, bpw):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fabq_storage_audit.png", bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_tradeoff() -> None:
    methods = ["Q1 b64", "Q1 b128", "Q1 b256", "Q1 b512", "FABQ-RC-lite"]
    mse = [7.627237e-05, 7.701788e-05, 7.751190e-05, 7.792983e-05, 6.615134e-05]
    bpw = [1.2500, 1.1250, 1.0625, 1.0322, 1.4010]
    colors = ["#8d99ae"] * 4 + ["#1f6f8b"]
    sizes = [110] * 4 + [170]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.scatter(bpw, mse, s=sizes, color=colors)
    for x, y, label in zip(bpw, mse, methods):
        ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points")
    ax.set_xlabel("Bits per weight", labelpad=10)
    ax.set_ylabel("Weight reconstruction MSE", labelpad=10)
    ax.set_title("FABQ-RC-lite improved reconstruction, not quality", pad=16, weight="bold")
    ax.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fabq_reconstruction_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def save_quality_collapse() -> None:
    labels = ["Qwen3.5\ndense", "Qwen3.5\nFABQ-lite", "Qwen3\ndense", "Qwen3\nFABQ-lite"]
    ppl = [26.5952, 677505.3533, 35.2165, 3676448.8825]
    colors = ["#1f6f8b", "#b33a3a", "#1f6f8b", "#b33a3a"]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.bar(labels, ppl, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Perplexity, log scale", labelpad=10)
    ax.set_title("The model still ran, but quality collapsed", pad=16, weight="bold")
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.7)
    for idx, val in enumerate(ppl):
        label = f"{val:,.0f}" if val > 1000 else f"{val:.1f}"
        ax.annotate(label, (idx, val), xytext=(0, 5), textcoords="offset points", ha="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fabq_quality_collapse.png", bbox_inches="tight")
    plt.close(fig)


def save_variable_precision_cliff() -> None:
    estimated_bpw = [3.1151, 4.1432, 4.5255]
    ppl = [3269.7708, 67.4850, 42.5027]
    dense_ppl = 35.2165

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.plot(estimated_bpw, ppl, marker="o", linewidth=3, color="#1f6f8b", label="FABQ-VP/EBQ")
    ax.axhline(dense_ppl, color="#3f7d3f", linestyle="--", linewidth=2, label="dense baseline")
    ax.set_yscale("log")
    ax.set_xlabel("Estimated physical bits per weight", labelpad=10)
    ax.set_ylabel("Perplexity, log scale", labelpad=10)
    ax.set_title("Variable precision backed away from the cliff", pad=16, weight="bold")
    ax.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False)
    for x, y in zip(estimated_bpw, ppl):
        ax.annotate(f"{y:,.1f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fabq_variable_precision_cliff.png", bbox_inches="tight")
    plt.close(fig)


def save_evidence_map() -> None:
    points = [
        ("Dense baseline", 0.10, 0.88, "#3f7d3f"),
        ("FABQ-RC-lite", 0.86, 0.08, "#b33a3a"),
        ("Unified 3.1 bpw", 0.62, 0.20, "#d47c1f"),
        ("Unified 4.5 bpw", 0.42, 0.68, "#1f6f8b"),
        ("Old 1.18 bpw claim", 0.95, 0.05, "#6b5b95"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=180)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.axvline(0.5, color="#cfcfcf", linewidth=1.5)
    ax.axhline(0.5, color="#cfcfcf", linewidth=1.5)
    for label, x, y, color in points:
        ax.scatter([x], [y], s=170, color=color)
        ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points")
    ax.text(0.03, 0.95, "Less compressed,\nbetter evidence", transform=ax.transAxes, va="top", color="#555555")
    ax.text(0.63, 0.95, "Strong compression,\nstill unproven", transform=ax.transAxes, va="top", color="#555555")
    ax.text(0.03, 0.08, "Not the target", transform=ax.transAxes, color="#555555")
    ax.text(0.64, 0.08, "Compression trap", transform=ax.transAxes, color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Compression pressure", labelpad=10)
    ax.set_ylabel("Quality evidence", labelpad=10)
    ax.set_title("Where the evidence actually sits", pad=16, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fabq_evidence_map.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_storage_audit()
    save_reconstruction_tradeoff()
    save_quality_collapse()
    save_variable_precision_cliff()
    save_evidence_map()
    print(f"Wrote charts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
