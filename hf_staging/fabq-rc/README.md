---
language:
- en
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

# FABQ-RC: Reproducible Research Prototype

FABQ-RC, Fisher-Adaptive Binary Quantization with Residual Codebooks, is an
active research prototype for sub-byte LLM quantization. This Hugging Face card
is intentionally conservative: it summarizes the checked-in evidence and known
negative results rather than advertising a validated compressed model release.

## Current Claim Boundary

The repository currently supports this claim:

> FABQ-RC-lite improves weight reconstruction over fixed binary block
> quantizers, but the near-binary runtime path fails language-modeling quality.
> A variable-precision prototype improves quality at roughly 4.5 bpw, making
> calibrated variable precision a stronger near-term direction than a pure
> 1-bit release.

The repository does not currently support claims that FABQ-RC is a validated
1-bit production model, has a validated BiLLM comparison win, or provides
validated 27B GGUF quality.

## Evidence Summary

### Weight Reconstruction

Source: `results/qwen35_08b_weight_quant.md`

| Model | Method | MSE | SQNR dB | bpw | Interpretation |
|---|---|---:|---:|---:|---|
| Qwen/Qwen3.5-0.8B | Q1 block64 | 7.627237e-05 | 4.2685 | 1.2500 | Fixed binary baseline |
| Qwen/Qwen3.5-0.8B | Q1 block128 | 7.701788e-05 | 4.2263 | 1.1250 | Fixed binary baseline |
| Qwen/Qwen3.5-0.8B | FABQ-RC-lite | 6.615134e-05 | 4.8868 | 1.4010 | Better reconstruction, higher storage |

This is not a perplexity, downstream task, or generation-quality result.

### Runtime Quality Smokes

Sources: `paper/FABQ_RC_preprint.md`,
`results/fabq_runtime_validation_report.md`, and the June 26, 2026 unified
benchmark JSONs.

| Model | Variant | Dataset | Estimated bpw | PPL | Readout |
|---|---|---|---:|---:|---|
| Qwen/Qwen3-0.6B | Dense BF16 baseline | WikiText-2 slice | n/a | 35.2165 | Small-slice baseline |
| Qwen/Qwen3-0.6B | FABQ-RC-lite dequantized | WikiText-2 slice | 1.4004 | 3,676,448.8825 | Negative result |
| Qwen/Qwen3-0.6B | Unified VP/EBQ target 3.0 | WikiText-2 slice | 3.1151 | 3269.7708 | Quality failure |
| Qwen/Qwen3-0.6B | Unified VP/EBQ target 4.0 | WikiText-2 slice | 4.1432 | 67.4850 | Improved but still degraded |
| Qwen/Qwen3-0.6B | Unified VP/EBQ target 4.5 | WikiText-2 slice | 4.5255 | 42.5027 | Best current local smoke |
| Qwen3.5-2B local checkpoint | Unified VP/EBQ target 3.0 | Inline fallback corpus | 3.1089 | 188.5031 | Functional smoke only |

All runtime rows are dense-dequantized validation runs. They validate that the
harness can load, quantize, dequantize, run forward passes, and generate short
samples. They do not validate native compressed kernels, file-size savings in a
published model artifact, or leaderboard-quality perplexity.

## Method Notes

FABQ-RC proposes:

1. Fisher-weighted channel or row importance.
2. Mixed-precision allocation for high-importance rows.
3. Adaptive quantization choices rather than one global format.
4. Residual correction using codebook-style reconstruction.

The checked-in runtime runs use simplified proxies: row energy for FABQ-RC-lite
and forward-only imatrix calibration for the unified VP/EBQ prototype. The full
Fisher calibration plus residual-codebook path remains future work.

## Not A Model-Artifact Inference Card

This card intentionally does not include llama.cpp or `from_pretrained`
inference commands for a named FABQ-RC model file. The repository does not
currently contain a validated HF or GGUF release artifact whose quality is
supported by the checked-in benchmarks.

Before publishing a model artifact, add artifact-specific commands only after
recording:

- exact model files and checksums,
- physical bits-per-weight or file-size accounting,
- matched dense and quantized quality metrics,
- downstream quality/task checks,
- native runtime memory and throughput measurements, if claimed.

## Reproducibility Files

| File | Purpose |
|---|---|
| `paper/FABQ_RC_preprint.md` | Conservative technical report and claim boundary |
| `docs/validation/VALIDATION_MEMO.md` | Validation audit and unsupported-claim notes |
| `results/qwen35_08b_weight_quant.md` | Weight reconstruction benchmark |
| `results/fabq_runtime_validation_report.md` | Near-binary runtime negative result |
| `results/runtime_validation_report.md` | Dense runtime baseline |
| `results/qwen3_06b_unified_fabq_benchmark.json` | June 26 target 3.0 bpw unified result |
| `results/qwen3_06b_unified_fabq_bpw4_benchmark.json` | June 26 target 4.0 bpw unified result |
| `results/qwen3_06b_unified_fabq_bpw45_benchmark.json` | June 26 target 4.5 bpw unified result |

## Limitations

- The PPL numbers are 254-token small-slice smoke measurements.
- Qwen3.5-2B used an inline fallback corpus and is not comparable to the
  WikiText-2 Qwen3-0.6B rows.
- The full Fisher gradient calibration path is not validated in the runtime
  results.
- The full residual-codebook design is not validated in the runtime results.
- Throughput numbers are for dense-dequantized CPU execution, not native
  compressed inference.

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
