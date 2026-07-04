# Qwen3-0.6B Global Allocator Quality and PPL Smoke

Date: 2026-07-03

Model: `results/qwen3_06b_bin_checkpoint`

Dataset for PPL diagnostic: `wikitext/wikitext-2-raw-v1/test`

Quality suite: default 3-task multiple-choice smoke from `benchmarks/benchmark_quality.py`

These runs use dense-dequantized validation. They do not prove native packed-kernel throughput.

## Results

| Variant | Estimated bpw | Task accuracy | Quality gate | PPL | Scored PPL tokens |
|---|---:|---:|---|---:|---:|
| Dense BF16 baseline | n/a | 0.6667 (2/3) | n/a | 50.7062 | 1016 |
| Global allocator target 3.0 | 3.1151 | 0.0000 (0/3) | FAIL | 4739.2258 | 1016 |
| Global allocator target 4.5 | 4.5255 | 0.6667 (2/3) | PASS | 61.6809 | 1016 |

## Readout

The target-3.0 point is not usable on this local smoke. It preserves a low estimated bpw, but task accuracy collapses and the PPL diagnostic is catastrophic.

The target-4.5 point is the viable local point after the allocator fix. It matches the dense 3-task smoke accuracy and passes the baseline-relative quality gate, but PPL is still worse than dense BF16: 61.6809 vs 50.7062 on the same 1016-token slice.

This is still a smoke, not a publishable quality claim. The quality suite has only 3 tasks, and PPL remains diagnostic rather than the primary acceptance metric for this project.

## Commands

```powershell
python benchmarks\benchmark_unified_fabq.py --repo-id results\qwen3_06b_bin_checkpoint --out results\qwen3_06b_global_allocator_bpw3_quality_ppl_smoke.json --target-bpw 3.0 --max-eval-tokens 1024 --imatrix-tokens 1024 --imatrix-batches 4 --quant-blocksize 128 --forward-repeats 2 --max-new-tokens 24 --run-quality --baseline-quality-json results\qwen3_06b_dense_quality_smoke.json --min-quality-tasks 3
```

```powershell
python benchmarks\benchmark_unified_fabq.py --repo-id results\qwen3_06b_bin_checkpoint --out results\qwen3_06b_global_allocator_bpw45_quality_ppl_smoke.json --target-bpw 4.5 --max-eval-tokens 1024 --imatrix-tokens 1024 --imatrix-batches 4 --quant-blocksize 128 --forward-repeats 2 --max-new-tokens 24 --run-quality --baseline-quality-json results\qwen3_06b_dense_quality_smoke.json --min-quality-tasks 3
```

## Artifacts

- `results/qwen3_06b_global_allocator_bpw3_quality_ppl_smoke.json`
- `results/qwen3_06b_global_allocator_bpw45_quality_ppl_smoke.json`
- Dense baseline references:
  - `results/qwen3_06b_dense_quality_smoke.json`
  - `results/qwen3_06b_baseline_1024_report.md`
