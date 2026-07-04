---
license: apache-2.0
base_model:
- Qwen/Qwen3-0.6B
- Qwen/Qwen3.5-0.8B
tags:
- quantization
- low-bit
- post-training-quantization
- research
- negative-results
- qwen
- fabq-rc
pipeline_tag: other
library_name: fabq-rc
---

# FABQ-RC Research Prototype

FABQ-RC, Fisher-Adaptive Binary Quantization with Residual Codebooks, is an
active research prototype for sub-byte post-training quantization. This card
describes the reproducible evidence currently checked into the repository. It
is not a model-release card for a validated GGUF or safetensors artifact.

## Status

The current evidence supports a conservative prototype-and-negative-result
claim:

- FABQ-RC-lite improves weight reconstruction versus fixed binary block
  quantizers on Qwen3.5-0.8B, but it fails language-modeling quality when
  evaluated as dense-dequantized runtime weights.
- The unified FABQ-VP/EBQ prototype recovers quality only when the storage
  budget rises into the 4-bit range.
- The repository does not currently validate a production 1-bit model, a
  quality win over BiLLM, or a 27B GGUF release.

## Measured Evidence

The most relevant reproducible results are in `paper/FABQ_RC_preprint.md`,
`results/qwen35_08b_weight_quant.md`, `results/fabq_runtime_validation_report.md`,
and the June 26, 2026 unified benchmark JSONs.

### Qwen3.5-0.8B Weight Reconstruction

This is a weight-level reconstruction benchmark, not a perplexity or task
benchmark.

| Method | MSE | SQNR dB | bpw |
|---|---:|---:|---:|
| Q1 block64 | 7.627237e-05 | 4.2685 | 1.2500 |
| Q1 block128 | 7.701788e-05 | 4.2263 | 1.1250 |
| FABQ-RC-lite | 6.615134e-05 | 4.8868 | 1.4010 |

FABQ-RC-lite improves MSE versus the fixed Q1 block baselines, but at a higher
storage budget and without validating language quality.

### Runtime Quality Smoke Results

All rows below are dense-dequantized validation runs. They test functional
loading, quantization, forward execution, and short generation, not native
compressed inference.

| Model | Variant | Dataset | Estimated bpw | PPL | Readout |
|---|---|---|---:|---:|---|
| Qwen/Qwen3-0.6B | Dense BF16 baseline | WikiText-2 slice | n/a | 35.2165 | Small-slice baseline |
| Qwen/Qwen3-0.6B | FABQ-RC-lite | WikiText-2 slice | 1.4004 | 3,676,448.8825 | Negative result |
| Qwen/Qwen3-0.6B | Unified VP/EBQ target 3.0 | WikiText-2 slice | 3.1151 | 3269.7708 | Quality failure |
| Qwen/Qwen3-0.6B | Unified VP/EBQ target 4.0 | WikiText-2 slice | 4.1432 | 67.4850 | Improved but still degraded |
| Qwen/Qwen3-0.6B | Unified VP/EBQ target 4.5 | WikiText-2 slice | 4.5255 | 42.5027 | Best current local smoke |
| Qwen3.5-2B local checkpoint | Unified VP/EBQ target 3.0 | Inline fallback corpus | 3.1089 | 188.5031 | Functional smoke only |

The Qwen3.5-2B run used an inline fallback corpus and is not directly
comparable to the WikiText-2 runs.

## What This Card Does Not Claim

- No validated 1-bit or near-1-bit production LLM artifact is claimed.
- No quality win over BiLLM, GPTQ, AWQ, or other prior work is claimed.
- No checked-in result validates any 27B Qwen GGUF artifact.
- No native compressed-kernel throughput or memory-speedup claim is made.
- The measured PPL values are small-slice smoke measurements, not leaderboard
  estimates.

## Method Summary

FABQ-RC proposes four components:

1. Fisher-weighted channel or row importance.
2. Mixed-precision allocation for important rows.
3. Per-layer or per-block quantization choices.
4. Residual correction using codebook-style reconstruction.

The current runtime results do not yet validate the full Fisher calibration
plus residual-codebook design. The strongest checked-in positive result is
weight-level reconstruction improvement. The strongest checked-in negative
result is that the simplified near-binary runtime path fails language-modeling
quality.

## Reproducibility Files

| File | Purpose |
|---|---|
| `paper/FABQ_RC_preprint.md` | Conservative technical report and claim boundary |
| `docs/validation/VALIDATION_MEMO.md` | Validation audit and unsupported-claim notes |
| `results/qwen35_08b_weight_quant.md` | Qwen3.5-0.8B weight reconstruction summary |
| `results/fabq_runtime_validation_report.md` | FABQ-RC-lite runtime negative result |
| `results/runtime_validation_report.md` | Dense runtime baseline |
| `results/qwen3_06b_unified_fabq_benchmark.json` | June 26 target 3.0 bpw unified result |
| `results/qwen3_06b_unified_fabq_bpw4_benchmark.json` | June 26 target 4.0 bpw unified result |
| `results/qwen3_06b_unified_fabq_bpw45_benchmark.json` | June 26 target 4.5 bpw unified result |

## License

Apache 2.0.

## Citation

```bibtex
@misc{fabqrc2026,
  author = {Zach Maronek},
  title = {FABQ-RC: Fisher-Adaptive Binary Quantization with Residual Codebooks},
  year = {2026},
  url = {https://github.com/toxzak/fabq-rc}
}
```
