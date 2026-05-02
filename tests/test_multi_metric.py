"""Tests for multi-metric visual reconstruction."""
import numpy as np
from PIL import Image, ImageDraw
from reference_free_ocr_metric.metrics.multi_metric.visual_reconstruction import (
    VisualReconstructionMetric,
)


def test_metric_name():
    metric = VisualReconstructionMetric()
    assert metric.name == "visual_reconstruction"


def test_identical_images_high_score():
    metric = VisualReconstructionMetric()
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    score = metric.compute_from_images(img, img.copy())
    assert score["ssim"] > 0.99
    assert score["composite"] > 0.9


def test_different_images_lower_score():
    metric = VisualReconstructionMetric()
    img1 = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
    score = metric.compute_from_images(img1, img2)
    assert score["ssim"] < 0.1


def test_compute_from_images_returns_all_keys():
    metric = VisualReconstructionMetric()
    img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    score = metric.compute_from_images(img, img.copy())
    assert "ssim" in score
    assert "mse" in score
    assert "lpips" in score
    assert "composite" in score


def test_composite_score_range():
    metric = VisualReconstructionMetric()
    img1 = Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8))
    score = metric.compute_from_images(img1, img2)
    assert 0.0 <= score["composite"] <= 1.0


def test_compute_from_parsed():
    """compute_from_parsed should accept a ParsedDocument directly."""
    from reference_free_ocr_metric.reconstruction.html_parser import (
        ParsedDocument, TextElement,
    )

    metric = VisualReconstructionMetric()
    original = Image.new("RGB", (500, 500), "white")
    draw = ImageDraw.Draw(original)
    draw.text((100, 100), "Test text", fill="black")

    parsed = ParsedDocument(
        text_elements=[
            TextElement(text="Test text", bbox=(200, 200, 600, 260), tag="span"),
        ],
        image_regions=[],
        table_regions=[],
        plain_text="Test text",
    )
    result = metric.compute_from_parsed(
        parsed_doc=parsed,
        original_image=original,
        image_width=500,
        image_height=500,
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
