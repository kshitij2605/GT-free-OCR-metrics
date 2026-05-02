"""Tests for VLM comparison metric."""
from reference_free_ocr_metric.metrics.vlm_compare.vlm_similarity import (
    VLMSimilarityMetric,
)


def test_metric_name():
    metric = VLMSimilarityMetric(
        api_base="http://test:9000/v1",
        api_key="test",
        model_name="test-model",
    )
    assert metric.name == "vlm_similarity"


def test_parse_score_decimal():
    metric = VLMSimilarityMetric(
        api_base="http://test:9000/v1",
        api_key="test",
        model_name="test-model",
    )
    assert metric._parse_score("0.85") == 0.85
    assert metric._parse_score("Score: 0.72") == 0.72


def test_parse_score_fraction():
    metric = VLMSimilarityMetric(
        api_base="http://test:9000/v1",
        api_key="test",
        model_name="test-model",
    )
    assert metric._parse_score("7/10") == 0.7
    assert metric._parse_score("8/10") == 0.8


def test_parse_score_fallback():
    metric = VLMSimilarityMetric(
        api_base="http://test:9000/v1",
        api_key="test",
        model_name="test-model",
    )
    assert metric._parse_score("no number here") == 0.5


def test_parse_score_verbose_response():
    metric = VLMSimilarityMetric(
        api_base="http://test:9000/v1",
        api_key="test",
        model_name="test-model",
    )
    score = metric._parse_score("The similarity between the two images is approximately 0.78 based on text alignment.")
    assert abs(score - 0.78) < 0.01
