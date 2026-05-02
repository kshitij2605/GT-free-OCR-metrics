"""Tests for multi-line and multi-column rendering in ImageRenderer."""
import numpy as np
from PIL import Image

from reference_free_ocr_metric.reconstruction.html_parser import TextElement
from reference_free_ocr_metric.reconstruction.image_renderer import ImageRenderer


def test_multiline_text_fits_within_bbox():
    """Long text that needs multiple lines should stay within bbox bounds."""
    renderer = ImageRenderer()
    elements = [
        TextElement(
            text="This is a very long text string that should wrap to multiple lines within the bounding box area",
            bbox=(50, 50, 450, 150),
            tag="p",
        ),
    ]
    img = renderer.render_text_image(elements, width=1000, height=1000)
    arr = np.array(img.convert("L"))
    bbox_region = arr[50:150, 50:450]
    assert bbox_region.min() < 255, "Text should be rendered in the bbox region"
    below_region = arr[160:200, 50:450]
    assert below_region.min() >= 250, "Text should not overflow below the bbox"


def test_small_font_for_multiline():
    renderer = ImageRenderer()
    elements = [
        TextElement(
            text="Word " * 50,
            bbox=(50, 50, 450, 90),
            tag="p",
        ),
    ]
    img = renderer.render_text_image(elements, width=1000, height=1000)
    assert isinstance(img, Image.Image)
    arr = np.array(img.convert("L"))
    bbox_region = arr[50:90, 50:450]
    assert bbox_region.min() < 255


def test_single_line_text_uses_bbox_height():
    renderer = ImageRenderer()
    elements = [
        TextElement(text="Hi", bbox=(50, 50, 200, 80), tag="p"),
    ]
    img = renderer.render_text_image(elements, width=1000, height=1000)
    arr = np.array(img.convert("L"))
    bbox_region = arr[50:80, 50:200]
    assert bbox_region.min() < 255


def test_empty_text_renders_nothing():
    renderer = ImageRenderer()
    elements = [
        TextElement(text="", bbox=(50, 50, 200, 80), tag="p"),
    ]
    img = renderer.render_text_image(elements, width=1000, height=1000)
    arr = np.array(img.convert("L"))
    assert arr.min() == 255


def test_multiple_elements_rendered():
    renderer = ImageRenderer()
    elements = [
        TextElement(text="First element", bbox=(50, 50, 300, 80), tag="p"),
        TextElement(text="Second element", bbox=(50, 100, 300, 130), tag="p"),
        TextElement(text="Third element with longer text that wraps", bbox=(50, 150, 300, 200), tag="p"),
    ]
    img = renderer.render_text_image(elements, width=1000, height=1000)
    arr = np.array(img.convert("L"))
    assert arr[50:80, 50:300].min() < 255, "First element"
    assert arr[100:130, 50:300].min() < 255, "Second element"
    assert arr[150:200, 50:300].min() < 255, "Third element"
