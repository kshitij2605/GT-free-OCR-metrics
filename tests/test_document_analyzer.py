"""Tests for document analyzer."""

import numpy as np
from PIL import Image, ImageDraw

from reference_free_ocr_metric.reconstruction.document_analyzer import (
    AnalyzedDocument,
    AnalyzedElement,
    TextStyle,
    detect_alignment,
    detect_alignment_from_crop,
    detect_bold,
    detect_first_line_indent,
    detect_font_category,
    detect_line_positions,
    estimate_font_size_pt,
    estimate_last_line_fill,
    measure_glyph_height,
    measure_mean_stroke_width,
)
from reference_free_ocr_metric.reconstruction.html_parser import (
    TextElement,
)


# --- Task 1: Dataclass tests ---


def test_text_style_defaults():
    style = TextStyle()
    assert style.font_size_pt == 12.0
    assert style.font_category == "sans-serif"
    assert style.is_bold is False
    assert style.alignment == "left"
    assert style.line_height_px == 0
    assert style.indent_px == 0
    assert style.lines == []


def test_analyzed_element_holds_style_and_text():
    te = TextElement(text="Hello", bbox=(0, 0, 100, 50), tag="p")
    style = TextStyle(font_size_pt=14.0, font_category="serif", is_bold=True)
    elem = AnalyzedElement(text_element=te, style=style)
    assert elem.style.font_size_pt == 14.0
    assert elem.text_element.text == "Hello"


def test_analyzed_document_holds_elements():
    doc = AnalyzedDocument(
        elements=[], image_regions=[], table_regions=[], page_dpi=300
    )
    assert doc.page_dpi == 300
    assert doc.elements == []


# --- Task 2: Font size estimation tests ---


def test_estimate_font_size_pt_300dpi():
    """At 300 DPI, a 50px glyph height (cap height) -> ~18.5pt after cap-height correction."""
    result = estimate_font_size_pt(bbox_height_px=50, dpi=300)
    assert abs(result - 18.5) < 0.2


def test_estimate_font_size_pt_150dpi():
    """At 150 DPI, a 50px glyph height -> ~36.9pt after cap-height correction."""
    result = estimate_font_size_pt(bbox_height_px=50, dpi=150)
    assert abs(result - 36.9) < 0.2


def test_measure_glyph_height_returns_positive():
    """measure_glyph_height returns a positive int for a non-blank region."""
    img = Image.new("L", (200, 60), 255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Test", fill=0)
    crop = np.array(img)
    h = measure_glyph_height(crop)
    assert h > 0
    assert h < 60


def test_measure_glyph_height_blank_returns_zero():
    """Blank (all white) region returns 0."""
    blank = np.full((60, 200), 255, dtype=np.uint8)
    assert measure_glyph_height(blank) == 0


# --- Task 3: Font category detection tests ---


def test_detect_font_category_returns_string():
    """detect_font_category returns 'serif' or 'sans-serif'."""
    crop = np.full((50, 200), 255, dtype=np.uint8)  # blank
    result = detect_font_category(crop)
    assert result in ("serif", "sans-serif")


def test_detect_font_category_blank_defaults_to_sans():
    """Blank image defaults to sans-serif."""
    blank = np.full((50, 200), 255, dtype=np.uint8)
    assert detect_font_category(blank) == "sans-serif"


# --- Task 4: Bold detection tests ---


def test_measure_mean_stroke_width_positive():
    """Non-blank crop returns positive stroke width."""
    img = Image.new("L", (200, 60), 255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Bold", fill=0)
    crop = np.array(img)
    w = measure_mean_stroke_width(crop)
    assert w > 0


def test_measure_mean_stroke_width_blank():
    """Blank crop returns 0."""
    blank = np.full((50, 200), 255, dtype=np.uint8)
    assert measure_mean_stroke_width(blank) == 0.0


def test_detect_bold_above_threshold():
    """Stroke/height ratio above 0.17 is bold."""
    # stroke_width=4, glyph_height=20 → ratio=0.2 → bold
    assert detect_bold(stroke_width=4.0, glyph_height=20) is True


def test_detect_bold_below_threshold():
    """Stroke/height ratio below 0.17 is not bold."""
    # stroke_width=2, glyph_height=20 → ratio=0.1 → not bold
    assert detect_bold(stroke_width=2.0, glyph_height=20) is False


# --- Task 5: Text alignment detection tests ---


def test_detect_alignment_left():
    """Elements with same left edge -> left-aligned."""
    bboxes = [(100, 0, 500, 30), (100, 40, 450, 70), (100, 80, 480, 110)]
    assert detect_alignment(bboxes, page_width=1000) == "left"


def test_detect_alignment_center():
    """Elements centered on same point -> center-aligned."""
    bboxes = [(200, 0, 800, 30), (250, 40, 750, 70), (300, 80, 700, 110)]
    assert detect_alignment(bboxes, page_width=1000) == "center"


def test_detect_alignment_right():
    """Elements with same right edge -> right-aligned."""
    bboxes = [(500, 0, 900, 30), (550, 40, 900, 70), (450, 80, 900, 110)]
    assert detect_alignment(bboxes, page_width=1000) == "right"


def test_detect_alignment_single_element():
    """Single element defaults to left."""
    bboxes = [(100, 0, 500, 30)]
    assert detect_alignment(bboxes, page_width=1000) == "left"


# --- Task 6: Line break detection tests ---


def test_detect_line_positions_multiline():
    """Synthetic 3-line image returns 3 line y-positions."""
    img = np.full((120, 200), 255, dtype=np.uint8)
    img[10:25, 20:180] = 0  # line 1
    img[45:60, 20:180] = 0  # line 2
    img[80:95, 20:180] = 0  # line 3
    positions = detect_line_positions(img)
    assert len(positions) == 3


def test_detect_line_positions_blank():
    """Blank image returns empty list."""
    blank = np.full((60, 200), 255, dtype=np.uint8)
    assert detect_line_positions(blank) == []


def test_detect_line_positions_single_line():
    """Single line of text returns 1 position."""
    img = np.full((40, 200), 255, dtype=np.uint8)
    img[10:30, 20:180] = 0
    positions = detect_line_positions(img)
    assert len(positions) == 1


# --- Task 7: DocumentAnalyzer orchestrator tests ---


def test_analyzer_returns_analyzed_document():
    """analyze() returns AnalyzedDocument with same number of elements."""
    from reference_free_ocr_metric.reconstruction.document_analyzer import (
        DocumentAnalyzer,
    )
    from reference_free_ocr_metric.reconstruction.html_parser import ParsedDocument

    img = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Hello", fill="black")

    parsed = ParsedDocument(
        text_elements=[TextElement(text="Hello", bbox=(10, 10, 100, 40), tag="p")],
        image_regions=[],
        table_regions=[],
        plain_text="Hello",
    )
    analyzer = DocumentAnalyzer()
    result = analyzer.analyze(img, parsed)
    assert isinstance(result, AnalyzedDocument)
    assert len(result.elements) == 1
    assert result.elements[0].style.font_size_pt > 0
    assert result.elements[0].style.font_category in ("serif", "sans-serif", "monospace", "handwriting")


# --- Task 8: New feature function tests ---


def test_detect_first_line_indent_no_indent():
    """Paragraph where all lines start at the same column returns 0."""
    img = np.full((80, 200), 255, dtype=np.uint8)
    img[5:20, 10:180] = 0
    img[25:40, 10:175] = 0
    img[45:60, 10:170] = 0
    assert detect_first_line_indent(img) == 0


def test_detect_first_line_indent_with_indent():
    """First line starts 20px further right than body lines."""
    img = np.full((80, 200), 255, dtype=np.uint8)
    img[5:20, 30:180] = 0   # first line indented to col 30
    img[25:40, 10:175] = 0  # body at col 10
    img[45:60, 10:170] = 0
    indent = detect_first_line_indent(img)
    assert indent >= 15


def test_estimate_last_line_fill_blank():
    """Blank image returns 0.0."""
    blank = np.full((60, 200), 255, dtype=np.uint8)
    assert estimate_last_line_fill(blank) == 0.0


def test_estimate_last_line_fill_full():
    """Single full-width line returns fill close to 1.0."""
    img = np.full((40, 200), 255, dtype=np.uint8)
    img[10:30, 5:195] = 0
    fill = estimate_last_line_fill(img)
    assert fill > 0.8


def test_detect_alignment_from_crop_left():
    """Uniform left-aligned lines detected as 'left'."""
    img = np.full((100, 300), 255, dtype=np.uint8)
    for y in [5, 30, 55, 75]:
        img[y : y + 15, 10:250] = 0
    result = detect_alignment_from_crop(img)
    assert result in ("left", "justified")


def test_detect_alignment_from_crop_blank():
    """Blank crop defaults to 'left'."""
    blank = np.full((60, 200), 255, dtype=np.uint8)
    assert detect_alignment_from_crop(blank) == "left"
