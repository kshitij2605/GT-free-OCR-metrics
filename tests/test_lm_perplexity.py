"""Tests for LM perplexity scorer."""
from reference_free_ocr_metric.metrics.lm_perplexity.perplexity_scorer import (
    LMPerplexityMetric,
)


def test_metric_name():
    metric = LMPerplexityMetric()
    assert metric.name == "lm_perplexity"


def test_good_text_higher_score():
    metric = LMPerplexityMetric()
    good_text = "The quick brown fox jumps over the lazy dog. This is a perfectly normal English sentence that makes complete sense."
    garbled_text = "Teh qiuck brwon fxo jmups oevr teh lzay dgo. Tihs si a prfectly nromal Enlgish snetnece taht mkeas cmoplete snese."
    good_score = metric.compute(good_text)
    garbled_score = metric.compute(garbled_text)
    assert good_score > garbled_score


def test_compute_detailed_keys():
    metric = LMPerplexityMetric()
    result = metric.compute_detailed("Hello world, this is a test sentence.")
    assert "ngram_score" in result
    assert "transformer_score" in result
    assert "perplexity" in result
    assert "composite" in result


def test_composite_score_range():
    metric = LMPerplexityMetric()
    score = metric.compute("The weather is nice today and the sun is shining brightly.")
    assert 0.0 <= score <= 1.0


def test_empty_text():
    metric = LMPerplexityMetric()
    score = metric.compute("")
    assert isinstance(score, float)


def test_ngram_score_range():
    metric = LMPerplexityMetric()
    score = metric._ngram_score("Hello world this is a test of n-gram scoring")
    assert 0.0 <= score <= 1.0


def test_long_text_sliding_window():
    """Verify perplexity computation works for texts exceeding 1024 tokens."""
    metric = LMPerplexityMetric()
    # Generate text longer than 1024 tokens (~4 chars per token on average)
    long_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~9000 chars
    result = metric.compute_detailed(long_text)
    assert result["perplexity"] > 0
    assert result["perplexity"] < float("inf")
    assert 0.0 <= result["composite"] <= 1.0
