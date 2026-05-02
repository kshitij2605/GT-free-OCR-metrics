"""Tests for OmniDocBench comparison bridge."""
from reference_free_ocr_metric.comparison.omnidocbench_bridge import (
    compute_correlation,
    ComparisonResult,
    OmniDocBenchBridge,
)


def test_compute_correlation_perfect():
    ref_scores = [0.1, 0.5, 0.9]
    our_scores = [0.1, 0.5, 0.9]
    result = compute_correlation(ref_scores, our_scores)
    assert result.pearson > 0.99
    assert result.spearman > 0.99


def test_compute_correlation_inverse():
    ref_scores = [0.1, 0.5, 0.9]
    our_scores = [0.9, 0.5, 0.1]
    result = compute_correlation(ref_scores, our_scores)
    assert result.pearson < -0.99


def test_comparison_result_fields():
    result = ComparisonResult(pearson=0.85, spearman=0.82, n_samples=10)
    assert result.pearson == 0.85
    assert result.spearman == 0.82
    assert result.n_samples == 10


def test_edit_distance_identical():
    score = OmniDocBenchBridge.compute_edit_distance("hello world", "hello world")
    assert score == 0.0


def test_edit_distance_different():
    score = OmniDocBenchBridge.compute_edit_distance("hello", "world")
    assert 0.0 < score <= 1.0


def test_edit_distance_empty():
    score = OmniDocBenchBridge.compute_edit_distance("", "")
    assert score == 0.0


def test_extract_text_blocks():
    bridge = OmniDocBenchBridge("/tmp/omnidocbench")
    page = {
        "layout_dets": [
            {"category_type": "text_block", "text": "Hello"},
            {"category_type": "title", "text": "Title"},
            {"category_type": "figure", "text": ""},
        ]
    }
    blocks = bridge.extract_text_blocks(page)
    assert blocks == ["Hello", "Title"]


def test_get_page_text():
    bridge = OmniDocBenchBridge("/tmp/omnidocbench")
    page = {
        "layout_dets": [
            {"category_type": "text_block", "text": "Line 1"},
            {"category_type": "text_block", "text": "Line 2"},
        ]
    }
    text = bridge.get_page_text(page)
    assert "Line 1" in text
    assert "Line 2" in text


def test_too_few_samples():
    result = compute_correlation([0.5], [0.5])
    assert result.pearson == 0.0
    assert result.n_samples == 1


# --- textblock2unicode ---

def test_textblock2unicode_plain_text():
    result = OmniDocBenchBridge._textblock2unicode("Hello world")
    assert result == "Hello world"


def test_textblock2unicode_inline_formula():
    result = OmniDocBenchBridge._textblock2unicode(r"$\alpha$")
    # pylatexenc converts \alpha to α; clean_string then keeps it (Unicode word char)
    assert "α" in result


def test_textblock2unicode_removes_escape_chars():
    result = OmniDocBenchBridge._textblock2unicode(r"a\_b\&c")
    assert "\\" not in result
    assert "_" not in result
    assert "&" not in result


def test_clean_string_with_inline_latex():
    # $\alpha$ → textblock2unicode → "α" → clean_string keeps it (Unicode \w)
    result = OmniDocBenchBridge._clean_string(r"$\alpha$")
    assert len(result) > 0


# --- _merge_truncated_dets ---

def test_merge_truncated_no_relations():
    page = {
        "layout_dets": [
            {"anno_id": 1, "category_type": "text_block", "text": "A", "order": 1},
            {"anno_id": 2, "category_type": "text_block", "text": "B", "order": 2},
        ],
        "extra": {"relation": []},
    }
    bridge = OmniDocBenchBridge()
    result = bridge._merge_truncated_dets(page)
    assert len(result) == 2


def test_merge_truncated_two_blocks():
    page = {
        "layout_dets": [
            {"anno_id": 1, "category_type": "text_block", "text": "Hello ", "order": 1},
            {"anno_id": 2, "category_type": "text_block", "text": "world", "order": 2},
        ],
        "extra": {
            "relation": [{"relation_type": "truncated", "source_anno_id": 1, "target_anno_id": 2}]
        },
    }
    bridge = OmniDocBenchBridge()
    result = bridge._merge_truncated_dets(page)
    assert len(result) == 1
    assert result[0]["text"] == "Hello world"


def test_merge_truncated_preserves_category():
    page = {
        "layout_dets": [
            {"anno_id": 1, "category_type": "title", "text": "Part 1 ", "order": 1},
            {"anno_id": 2, "category_type": "title", "text": "Part 2", "order": 2},
        ],
        "extra": {
            "relation": [{"relation_type": "truncated", "source_anno_id": 1, "target_anno_id": 2}]
        },
    }
    bridge = OmniDocBenchBridge()
    result = bridge._merge_truncated_dets(page)
    assert result[0]["category_type"] == "title"


def test_merge_truncated_ignores_other_relations():
    page = {
        "layout_dets": [
            {"anno_id": 1, "category_type": "text_block", "text": "A", "order": 1},
            {"anno_id": 2, "category_type": "text_block", "text": "B", "order": 2},
        ],
        "extra": {
            "relation": [{"relation_type": "continuation", "source_anno_id": 1, "target_anno_id": 2}]
        },
    }
    bridge = OmniDocBenchBridge()
    result = bridge._merge_truncated_dets(page)
    assert len(result) == 2


# --- _poly_to_bbox ---

def test_poly_to_bbox_rectangle():
    poly = [10.0, 20.0, 100.0, 20.0, 100.0, 80.0, 10.0, 80.0]
    bbox = OmniDocBenchBridge._poly_to_bbox(poly)
    assert bbox == (10.0, 20.0, 100.0, 80.0)


# --- _iou ---

def test_iou_perfect_overlap():
    box = (0.0, 0.0, 10.0, 10.0)
    assert OmniDocBenchBridge._iou(box, box) == 1.0


def test_iou_no_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (20.0, 20.0, 30.0, 30.0)
    assert OmniDocBenchBridge._iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    # inter = 5x5=25, a_area=100, b_area=100, union=175
    iou = OmniDocBenchBridge._iou(a, b)
    assert abs(iou - 25 / 175) < 1e-6


# --- compute_page_edit_distance ---

def _make_page(dets):
    return {"layout_dets": dets, "extra": {"relation": []}}


def test_compute_page_edit_distance_perfect_match():
    bridge = OmniDocBenchBridge()
    page = _make_page([{
        "anno_id": 1, "category_type": "text_block", "text": "Hello world",
        "order": 1, "poly": [0, 0, 100, 0, 100, 50, 0, 50],
    }])
    ocr = [{"text": "Hello world", "bbox": [0, 0, 100, 50]}]
    assert bridge.compute_page_edit_distance(page, ocr) == 0.0


def test_compute_page_edit_distance_no_ocr_match():
    bridge = OmniDocBenchBridge()
    page = _make_page([{
        "anno_id": 1, "category_type": "text_block", "text": "Hello world",
        "order": 1, "poly": [0, 0, 100, 0, 100, 50, 0, 50],
    }])
    # Text is completely different — Hungarian cost > 0.7 → unmatched → GT vs empty
    ocr = [{"text": "xyzxyzxyz", "bbox": [0, 0, 100, 50]}]
    score = bridge.compute_page_edit_distance(page, ocr)
    assert score == 1.0  # GT vs empty → all chars missing


def test_compute_page_edit_distance_granularity_mismatch():
    """Hungarian matches a GT paragraph to a larger OCR block containing it."""
    bridge = OmniDocBenchBridge()
    page = _make_page([{
        "anno_id": 1, "category_type": "text_block", "text": "hello world",
        "order": 1, "poly": [0, 0, 100, 0, 100, 20, 0, 20],  # small GT bbox
    }])
    # OCR element bbox is much larger (IoU would be < 0.5), but text matches
    ocr = [{"text": "hello world", "bbox": [0, 0, 500, 200]}]
    assert bridge.compute_page_edit_distance(page, ocr) == 0.0


def test_compute_page_edit_distance_aggregation():
    """sum(edits) / sum(max_lens) across elements."""
    bridge = OmniDocBenchBridge()
    page = _make_page([
        {"anno_id": 1, "category_type": "text_block", "text": "ab",
         "order": 1, "poly": [0, 0, 10, 0, 10, 10, 0, 10]},
        {"anno_id": 2, "category_type": "text_block", "text": "abcd",
         "order": 2, "poly": [0, 20, 10, 20, 10, 30, 0, 30]},
    ])
    # OCR only covers first element
    ocr = [{"text": "ab", "bbox": [0, 0, 10, 10]}]
    # el1: edit("ab","ab")=0, max_len=2; el2: no match, edit("abcd","")=4, max_len=4
    # score = 4 / 6
    score = bridge.compute_page_edit_distance(page, ocr)
    assert abs(score - 4 / 6) < 1e-6


def test_compute_page_edit_distance_with_truncated():
    """Truncated blocks are merged before matching."""
    bridge = OmniDocBenchBridge()
    page = {
        "layout_dets": [
            {"anno_id": 1, "category_type": "text_block", "text": "Hello ",
             "order": 1, "poly": [0, 0, 100, 0, 100, 20, 0, 20]},
            {"anno_id": 2, "category_type": "text_block", "text": "world",
             "order": 2, "poly": [0, 25, 100, 25, 100, 45, 0, 45]},
        ],
        "extra": {
            "relation": [{"relation_type": "truncated", "source_anno_id": 1, "target_anno_id": 2}]
        },
    }
    # OCR covers the area of the first (merged) block
    ocr = [{"text": "Hello world", "bbox": [0, 0, 100, 20]}]
    score = bridge.compute_page_edit_distance(page, ocr)
    assert score == 0.0  # merged "Hello world" matches OCR "Hello world"


def test_compute_page_edit_distance_non_text_category_ignored():
    bridge = OmniDocBenchBridge()
    page = _make_page([
        {"anno_id": 1, "category_type": "figure_caption", "text": "Fig 1",
         "order": 1, "poly": [0, 0, 100, 0, 100, 50, 0, 50]},
    ])
    ocr = []
    # figure_caption is not in _TEXT_CATEGORIES → score 0 (no elements to evaluate)
    assert bridge.compute_page_edit_distance(page, ocr) == 0.0


# --- _normalized_formula tests ---

def test_normalized_formula_strips_delimiters():
    bridge = OmniDocBenchBridge()
    # spaces are in _FORMULA_FILTER so are removed; $$ stripped; lowercased
    result = bridge._normalized_formula("$$\\Alpha + \\Beta$$")
    assert "$$" not in result
    assert result == result.lower()


def test_normalized_formula_lowercases():
    bridge = OmniDocBenchBridge()
    result = bridge._normalized_formula("X^2 + Y^2")
    assert result == result.lower()


def test_normalized_formula_removes_macros():
    bridge = OmniDocBenchBridge()
    result = bridge._normalized_formula("\\mathbf{A} + \\mathrm{B}")
    assert "\\mathbf" not in result
    assert "\\mathrm" not in result


def test_normalized_formula_extracts_bracket_content():
    bridge = OmniDocBenchBridge()
    result = bridge._normalized_formula("\\[\\int_0^1 f(x) dx\\]")
    # \[ \] stripped; spaces removed by filter; key tokens preserved
    assert "\\int_0^1" in result
    assert "f(x)" in result


def test_normalized_formula_removes_tag_hspace():
    bridge = OmniDocBenchBridge()
    result = bridge._normalized_formula("\\tag{1}\\hspace{1cm}x=1")
    assert "\\tag" not in result
    assert "\\hspace" not in result


# --- _extract_table_text tests ---

def test_extract_table_text_simple():
    bridge = OmniDocBenchBridge()
    html = "<table><tr><td>milk</td><td>man</td></tr><tr><td>night</td><td>nice</td></tr></table>"
    result = bridge._extract_table_text(html)
    assert "milk" in result
    assert "man" in result
    assert "night" in result


def test_extract_table_text_empty():
    bridge = OmniDocBenchBridge()
    result = bridge._extract_table_text("<table></table>")
    assert result == ""


# --- compute_page_formula_edit_distance tests ---

def _make_page_with_formula(latex: str) -> dict:
    return _make_page([{
        "anno_id": 1, "category_type": "equation_isolated",
        "latex": latex, "text": "", "order": 1,
        "poly": [0, 0, 100, 0, 100, 50, 0, 50], "ignore": False,
    }])


def test_formula_edit_distance_perfect_match():
    bridge = OmniDocBenchBridge()
    page = _make_page_with_formula("$$\\int_0^1 f(x) dx$$")
    ocr = [{"text": "$$\\int_0^1 f(x) dx$$", "bbox": [0, 0, 100, 50]}]
    assert bridge.compute_page_formula_edit_distance(page, ocr) == 0.0


def test_formula_edit_distance_no_gt_formulas():
    bridge = OmniDocBenchBridge()
    page = _make_page([{
        "anno_id": 1, "category_type": "text_block", "text": "Hello",
        "order": 1, "poly": [0, 0, 100, 0, 100, 50, 0, 50], "ignore": False,
    }])
    ocr = [{"text": "$$x=1$$", "bbox": [0, 0, 100, 50]}]
    assert bridge.compute_page_formula_edit_distance(page, ocr) == 0.0


def test_formula_edit_distance_no_ocr():
    bridge = OmniDocBenchBridge()
    page = _make_page_with_formula("$$x + y = 1$$")
    # No OCR formulas → worst case (GT vs empty)
    score = bridge.compute_page_formula_edit_distance(page, [])
    assert 0.0 < score <= 1.0


# --- compute_page_table_edit_distance tests ---

def _make_page_with_table(html: str) -> dict:
    return _make_page([{
        "anno_id": 1, "category_type": "table",
        "html": html, "text": "", "order": 1,
        "poly": [0, 0, 200, 0, 200, 100, 0, 100], "ignore": False,
    }])


def test_table_edit_distance_perfect_match():
    bridge = OmniDocBenchBridge()
    html = "<table><tr><td>milk</td><td>man</td></tr></table>"
    page = _make_page_with_table(html)
    ocr = [{"html": html, "bbox": [0, 0, 200, 100]}]
    assert bridge.compute_page_table_edit_distance(page, ocr) == 0.0


def test_table_edit_distance_no_gt_tables():
    bridge = OmniDocBenchBridge()
    page = _make_page([{
        "anno_id": 1, "category_type": "text_block", "text": "Hello",
        "order": 1, "poly": [0, 0, 100, 0, 100, 50, 0, 50], "ignore": False,
    }])
    ocr = [{"html": "<table><tr><td>data</td></tr></table>", "bbox": [0, 0, 100, 50]}]
    assert bridge.compute_page_table_edit_distance(page, ocr) == 0.0


def test_table_edit_distance_no_ocr():
    bridge = OmniDocBenchBridge()
    html = "<table><tr><td>milk</td><td>man</td></tr></table>"
    page = _make_page_with_table(html)
    score = bridge.compute_page_table_edit_distance(page, [])
    assert 0.0 < score <= 1.0


# --- compute_page_teds tests ---

def test_compute_page_teds_no_gt_tables():
    bridge = OmniDocBenchBridge()
    page = _make_page([{
        "anno_id": 1, "category_type": "text_block", "text": "Hello",
        "order": 1, "poly": [0, 0, 100, 0, 100, 50, 0, 50], "ignore": False,
    }])
    # No GT tables → None (dimension not applicable)
    assert bridge.compute_page_teds(page, []) is None


def test_compute_page_teds_perfect_match():
    bridge = OmniDocBenchBridge()
    html = "<table><tr><td>milk</td><td>man</td></tr></table>"
    page = _make_page_with_table(html)
    ocr = [{"html": html, "bbox": [0, 0, 200, 100]}]
    score = bridge.compute_page_teds(page, ocr)
    assert score == 1.0


def test_compute_page_teds_no_ocr():
    bridge = OmniDocBenchBridge()
    html = "<table><tr><td>milk</td><td>man</td></tr></table>"
    page = _make_page_with_table(html)
    # No OCR tables → GT unmatched → TEDS = 0
    score = bridge.compute_page_teds(page, [])
    assert score == 0.0


# --- compute_page_cdm tests ---

def test_compute_page_cdm_no_gt_formulas():
    bridge = OmniDocBenchBridge()
    page = _make_page([{
        "anno_id": 1, "category_type": "text_block", "text": "Hello",
        "order": 1, "poly": [0, 0, 100, 0, 100, 50, 0, 50], "ignore": False,
    }])
    # No GT formulas → None (dimension not applicable)
    assert bridge.compute_page_cdm(page, []) is None


def test_compute_page_cdm_perfect_match():
    bridge = OmniDocBenchBridge()
    page = _make_page_with_formula("$$x + y = 1$$")
    ocr = [{"text": "$$x + y = 1$$", "bbox": [0, 0, 100, 50]}]
    score = bridge.compute_page_cdm(page, ocr)
    # Same formula → identical renders → all CCs match → CDM = 1.0
    assert score == 1.0


def test_compute_page_cdm_no_ocr():
    bridge = OmniDocBenchBridge()
    page = _make_page_with_formula("$$x + y = 1$$")
    # No OCR formulas → GT unmatched → CDM = 0
    score = bridge.compute_page_cdm(page, [])
    assert score == 0.0


