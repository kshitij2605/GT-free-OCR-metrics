"""Tests for OpenCLIP comparison metric."""
import numpy as np
from PIL import Image
from reference_free_ocr_metric.metrics.clip_compare.clip_similarity import (
    CLIPSimilarityMetric,
)


def test_metric_name():
    metric = CLIPSimilarityMetric()
    assert metric.name == "clip_similarity"


def test_identical_images_high_score():
    metric = CLIPSimilarityMetric()
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    score = metric.compute_from_images(img, img.copy())
    assert score["clip_cosine"] > 0.95


def test_score_returns_float():
    metric = CLIPSimilarityMetric()
    img1 = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    score = metric.compute_from_images(img1, img2)
    assert isinstance(score["clip_cosine"], float)
    assert -1.0 <= score["clip_cosine"] <= 1.0


def test_preprocess_produces_rgb():
    metric = CLIPSimilarityMetric()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    processed = metric._preprocess_for_comparison(img)
    assert processed.mode == "RGB"
