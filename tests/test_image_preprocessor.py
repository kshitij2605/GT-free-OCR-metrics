"""Tests for image preprocessor."""
import numpy as np
from PIL import Image
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor
from reference_free_ocr_metric.reconstruction.html_parser import Region


def test_mask_regions():
    preprocessor = ImagePreprocessor()
    img = Image.new("RGB", (1000, 1000), "black")
    regions = [Region(bbox=(100, 100, 400, 400), region_type="image")]
    masked = preprocessor.mask_regions(img, regions, fill_color=(255, 255, 255))
    assert masked.size == (1000, 1000)
    # Check that the masked region is white
    arr = np.array(masked)
    assert arr[200, 200, 0] == 255  # inside the masked region


def test_to_grayscale():
    preprocessor = ImagePreprocessor()
    img = Image.new("RGB", (100, 100), "red")
    gray = preprocessor.to_grayscale(img)
    assert gray.mode == "L"


def test_optimize_for_comparison():
    preprocessor = ImagePreprocessor()
    img = Image.new("RGB", (100, 100), "gray")
    optimized = preprocessor.optimize_for_comparison(img)
    assert optimized.mode == "L"
