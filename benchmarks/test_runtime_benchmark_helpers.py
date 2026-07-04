import math
import sys
from types import SimpleNamespace

from benchmark_qwen35_runtime import aggregate_loss, auto_model_kind, load_eval_text, tokens_per_second


def test_tokens_per_second_counts_generated_tokens_only():
    assert tokens_per_second(prompt_tokens=5, total_tokens=13, elapsed_sec=2.0) == 4.0


def test_tokens_per_second_returns_zero_for_nonpositive_elapsed():
    assert tokens_per_second(prompt_tokens=5, total_tokens=13, elapsed_sec=0.0) == 0.0


def test_aggregate_loss_returns_perplexity_from_weighted_losses():
    result = aggregate_loss([(math.log(2.0), 10), (math.log(4.0), 10)])
    assert result["tokens"] == 20
    assert abs(result["loss"] - math.log(math.sqrt(8.0))) < 1e-12
    assert abs(result["perplexity"] - math.sqrt(8.0)) < 1e-12


def test_auto_model_kind_uses_multimodal_for_conditional_generation():
    assert auto_model_kind("qwen3_5", ["Qwen3_5ForConditionalGeneration"]) == "image_text_to_text"


def test_auto_model_kind_uses_causal_lm_for_text_generation():
    assert auto_model_kind("qwen3", ["Qwen3ForCausalLM"]) == "causal_lm"


def test_load_eval_text_uses_local_hf_cache_before_inline_fallback(tmp_path, monkeypatch):
    snapshot = (
        tmp_path
        / "hub"
        / "datasets--wikitext"
        / "snapshots"
        / "abc123"
        / "wikitext-2-raw-v1"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "test-00000-of-00001.jsonl").write_text(
        '{"text": ""}\n'
        '{"text": "  WikiText cached first article.  "}\n'
        '{"text": "WikiText cached second article."}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))),
    )

    text, dataset_id = load_eval_text(200, "wikitext", "wikitext-2-raw-v1", "test")

    assert dataset_id == "wikitext/wikitext-2-raw-v1/test"
    assert "WikiText cached first article." in text
    assert "WikiText cached second article." in text
    assert "Language models compress patterns" not in text
