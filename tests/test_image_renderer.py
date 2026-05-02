"""Tests for image renderer."""
import numpy as np
from PIL import Image
from reference_free_ocr_metric.reconstruction.image_renderer import ImageRenderer
from reference_free_ocr_metric.reconstruction.html_parser import TextElement


def test_render_text_creates_image():
    renderer = ImageRenderer()
    elements = [
        TextElement(text="Hello World", bbox=(100, 100, 500, 140), tag="p"),
    ]
    img = renderer.render_text_image(
        text_elements=elements, width=1000, height=1000, background="white"
    )
    assert isinstance(img, Image.Image)
    assert img.size == (1000, 1000)


def test_render_text_not_blank():
    renderer = ImageRenderer()
    elements = [
        TextElement(text="Hello World", bbox=(100, 100, 500, 140), tag="p"),
    ]
    img = renderer.render_text_image(
        text_elements=elements, width=1000, height=1000, background="white"
    )
    arr = np.array(img.convert("L"))
    assert arr.min() < 255


def test_render_empty_elements():
    renderer = ImageRenderer()
    img = renderer.render_text_image(
        text_elements=[], width=500, height=500, background="white"
    )
    arr = np.array(img.convert("L"))
    assert arr.min() == 255  # all white


def test_render_plain_text_fallback():
    renderer = ImageRenderer()
    img = renderer.render_plain_text(
        text="Hello World\nSecond line of text",
        width=1000,
        height=1000,
    )
    assert isinstance(img, Image.Image)
    assert img.size == (1000, 1000)
    arr = np.array(img.convert("L"))
    assert arr.min() < 255  # not blank


def test_render_plain_text_empty():
    renderer = ImageRenderer()
    img = renderer.render_plain_text(text="", width=500, height=500)
    arr = np.array(img.convert("L"))
    assert arr.min() == 255  # blank for empty text


# --- Task 8: render_analyzed_document tests ---


def test_render_analyzed_document():
    """render_analyzed_document uses style properties."""
    from reference_free_ocr_metric.reconstruction.document_analyzer import (
        AnalyzedDocument,
        AnalyzedElement,
        TextStyle,
    )

    renderer = ImageRenderer()
    style = TextStyle(
        font_size_pt=14.0,
        font_category="sans-serif",
        is_bold=False,
        alignment="left",
        line_height_px=20,
        lines=[],
    )
    elem = AnalyzedElement(
        text_element=TextElement(text="Test text", bbox=(50, 50, 400, 100), tag="p"),
        style=style,
    )
    doc = AnalyzedDocument(
        elements=[elem],
        image_regions=[],
        table_regions=[],
        page_dpi=300,
    )
    img = renderer.render_analyzed_document(doc, width=500, height=200)
    assert isinstance(img, Image.Image)
    assert img.size == (500, 200)
    arr = np.array(img.convert("L"))
    assert arr.min() < 255  # not blank
