"""Tests for PDF to image converter."""

import os
from unittest.mock import patch

from PIL import Image

from reference_free_ocr_metric.utils.pdf_converter import (
    convert_pdf_to_images,
    pdf_page_paths,
)


def _make_fake_images(count=2):
    return [Image.new("RGB", (100, 100), "white") for _ in range(count)]


@patch("reference_free_ocr_metric.utils.pdf_converter.convert_from_path")
def test_convert_returns_pil_images(mock_convert):
    mock_convert.return_value = _make_fake_images(3)
    result = convert_pdf_to_images("fake.pdf")
    assert len(result) == 3
    assert all(isinstance(img, Image.Image) for img in result)


@patch("reference_free_ocr_metric.utils.pdf_converter.convert_from_path")
def test_convert_passes_dpi(mock_convert):
    mock_convert.return_value = _make_fake_images(1)
    convert_pdf_to_images("fake.pdf", dpi=150)
    mock_convert.assert_called_once_with("fake.pdf", dpi=150, first_page=None, last_page=None)


@patch("reference_free_ocr_metric.utils.pdf_converter.convert_from_path")
def test_convert_passes_page_range(mock_convert):
    mock_convert.return_value = _make_fake_images(1)
    convert_pdf_to_images("fake.pdf", first_page=2, last_page=5)
    mock_convert.assert_called_once_with("fake.pdf", dpi=300, first_page=2, last_page=5)


@patch("reference_free_ocr_metric.utils.pdf_converter.convert_from_path")
def test_convert_empty_pdf(mock_convert):
    mock_convert.return_value = []
    result = convert_pdf_to_images("empty.pdf")
    assert result == []


# --- pdf_page_paths tests ---


@patch("reference_free_ocr_metric.utils.pdf_converter.convert_from_path")
def test_pdf_page_paths_yields_paths(mock_convert):
    """Context manager yields a list of existing PNG file paths."""
    mock_convert.return_value = _make_fake_images(3)
    with pdf_page_paths("fake.pdf") as paths:
        assert len(paths) == 3
        for p in paths:
            assert p.endswith(".png")
            assert os.path.isfile(p)


@patch("reference_free_ocr_metric.utils.pdf_converter.convert_from_path")
def test_pdf_page_paths_cleans_up(mock_convert):
    """Temp files are removed after context manager exits."""
    mock_convert.return_value = _make_fake_images(2)
    with pdf_page_paths("fake.pdf") as paths:
        saved_paths = list(paths)
    for p in saved_paths:
        assert not os.path.exists(p)


@patch("reference_free_ocr_metric.utils.pdf_converter.convert_from_path")
def test_pdf_page_paths_passes_args(mock_convert):
    """Arguments are forwarded to convert_pdf_to_images."""
    mock_convert.return_value = _make_fake_images(1)
    with pdf_page_paths("fake.pdf", dpi=150, first_page=2, last_page=3) as _paths:
        pass
    mock_convert.assert_called_once_with("fake.pdf", dpi=150, first_page=2, last_page=3)


@patch("reference_free_ocr_metric.utils.pdf_converter.convert_from_path")
def test_pdf_page_paths_empty_pdf(mock_convert):
    """Empty PDF yields an empty list."""
    mock_convert.return_value = []
    with pdf_page_paths("empty.pdf") as paths:
        assert paths == []
