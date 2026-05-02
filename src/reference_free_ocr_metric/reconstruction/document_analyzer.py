"""Analyze original document images to extract precise text styling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import re

import cv2
import numpy as np

from reference_free_ocr_metric.reconstruction.html_parser import (
    Region,
    TableElement,
    TextElement,
)

if TYPE_CHECKING:
    from PIL import Image

    from reference_free_ocr_metric.reconstruction.html_parser import ParsedDocument


@dataclass
class TextStyle:
    """Visual style properties measured from the original image."""

    font_size_pt: float = 12.0
    font_category: str = "sans-serif"  # "serif" | "sans-serif" | "monospace" | "handwriting"
    is_bold: bool = False
    is_italic: bool = False
    font_weight: str = "regular"  # thin/light/regular/medium/bold/heavy/black
    alignment: str = "left"  # "left" | "center" | "right" | "justified"
    line_height_px: int = 0
    indent_px: int = 0
    lines: list[str] = field(default_factory=list)


@dataclass
class AnalyzedElement:
    """A text element paired with its measured style."""

    text_element: TextElement
    style: TextStyle


@dataclass
class AnalyzedDocument:
    """Fully analyzed document ready for precise rendering."""

    elements: list[AnalyzedElement]
    image_regions: list[Region]
    table_regions: list[Region]
    table_elements: list[TableElement] = field(default_factory=list)
    page_dpi: int = 300


# --- Task 1: DPI auto-detection ---


def estimate_dpi(image_width: int, image_height: int) -> int:
    """Estimate effective DPI from image dimensions by matching standard page sizes."""
    # Work in portrait orientation
    w = min(image_width, image_height)
    h = max(image_width, image_height)
    aspect = h / w

    # (width_in, height_in) for common page sizes in portrait
    candidates = [
        (8.5, 11.0),    # US Letter
        (8.27, 11.69),  # A4
        (6.0, 9.0),     # Compact book
        (7.0, 10.0),    # Trade book
        (5.5, 8.5),     # Digest / half-letter
        (6.69, 9.61),   # B5
    ]
    best_dpi = 150
    best_diff = float("inf")
    for pw, ph in candidates:
        diff = abs(aspect - ph / pw)
        if diff < best_diff:
            best_diff = diff
            best_dpi = int(round(w / pw))

    return max(72, min(600, best_dpi))


# --- Task 2: Font size estimation ---


# Cap height is ~65% of em for common Latin fonts (Liberation Serif/Sans).
# measure_glyph_height returns ink height ≈ cap height, so we divide by
# this fraction to recover the true em-square (point) size.
_CAP_HEIGHT_RATIO = 0.65


def estimate_font_size_pt(bbox_height_px: int, dpi: int = 300) -> float:
    """Convert ink glyph height (pixels) to point size.

    Applies a cap-height correction: glyph height from connected components
    approximates cap height (~65% of the em square for Liberation fonts).
    Dividing by _CAP_HEIGHT_RATIO recovers the correct point size.
    """
    return round((bbox_height_px / dpi) * 72 / _CAP_HEIGHT_RATIO, 1)


def measure_glyph_height(gray_crop: np.ndarray) -> int:
    """Measure glyph height in a grayscale crop via connected components.

    Uses the 85th-percentile CC height to target cap/ascender heights while
    excluding inflated values from merged multi-line components (which can
    span ascender+descender across adjacent lines and skew the maximum).
    Returns 0 for blank crops.
    """
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    if binary.sum() == 0:
        return 0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels <= 1:
        return 0
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    # Filter out tiny CCs (punctuation dots, serifs, noise) that deflate the
    # percentile estimate and cause regular text to be misclassified as bold.
    min_h = max(3, gray_crop.shape[0] * 0.15)
    significant = heights[heights >= min_h]
    if len(significant) >= 2:
        heights = significant
    return int(np.percentile(heights, 85))


# --- Task 3: Font category detection ---


def _thin(binary: np.ndarray) -> np.ndarray:
    """Morphological thinning fallback using OpenCV erosion-based approach."""
    # Use cv2.ximgproc.thinning if available, else use a simple approach
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(binary)
    # Simple skeletonization via iterative erosion
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = binary.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel


def detect_font_category(gray_crop: np.ndarray) -> str:
    """Classify text as 'serif' or 'sans-serif' using stroke width variance.

    Serif fonts (e.g. Mincho) have variable stroke widths due to serifs.
    Sans-serif fonts (e.g. Gothic) have uniform stroke widths.
    """
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    if binary.sum() == 0:
        return "sans-serif"

    # Skeletonize and measure stroke widths via distance transform.
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton = _thin(binary)
    stroke_widths = dist[skeleton > 0]

    if len(stroke_widths) < 10:
        return "sans-serif"

    # Coefficient of variation: std / mean.
    cv_val = float(np.std(stroke_widths) / max(np.mean(stroke_widths), 1e-6))
    # Serif fonts have high stroke width variance: CV > 0.20 catches Computer Modern
    # and Times-like fonts. Sans-serif fonts typically have CV < 0.15.
    return "serif" if cv_val > 0.20 else "sans-serif"


# --- Task 3b: Italic detection ---


def detect_italic(gray_crop: np.ndarray) -> tuple[bool, float]:
    """Detect italic text via column-centroid slant regression.

    Fits a line through column-wise ink centroids; italic strokes produce a
    consistent vertical drift yielding |slant| > 5°.
    Returns (is_italic, slant_degrees).
    """
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1].astype(np.float32)
    H = binary.shape[0]
    rows = np.arange(H, dtype=np.float32)
    col_masses = binary.sum(axis=0)
    min_ink = H * 0.05
    valid = np.where(col_masses > min_ink)[0]
    if len(valid) < 4:
        return False, 0.0
    centroids = (rows[:, None] * binary[:, valid]).sum(axis=0) / col_masses[valid]
    coeffs = np.polyfit(valid.astype(np.float32), centroids, 1)
    slant = float(np.degrees(np.arctan(coeffs[0])))
    return abs(slant) > 5.0, slant


# --- Task 3c: Font weight gradation ---


def classify_font_weight(stroke_width: float, glyph_height: int) -> str:
    """Map stroke/glyph-height ratio to a weight name.

    Thresholds derived from typographic ratio research:
    regular body text ~7-12%, bold ~17-22%.
    """
    if glyph_height <= 0 or stroke_width <= 0:
        return "regular"
    ratio = stroke_width / glyph_height
    if ratio < 0.04:
        return "thin"
    if ratio < 0.07:
        return "light"
    if ratio < 0.12:
        return "regular"
    if ratio < 0.17:
        return "medium"
    if ratio < 0.22:
        return "bold"
    if ratio < 0.30:
        return "heavy"
    return "black"


# --- Task 3d: Monospace and handwriting detection ---


def _detect_monospace(gray_crop: np.ndarray) -> bool:
    """Return True if connected-component widths are uniform (monospace font)."""
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    h_crop, w_crop = gray_crop.shape
    min_area = 0.001 * h_crop * w_crop
    max_area = 0.5 * h_crop * w_crop
    widths = [
        stats[i, cv2.CC_STAT_WIDTH]
        for i in range(1, n_labels)
        if min_area < stats[i, cv2.CC_STAT_AREA] < max_area
        and stats[i, cv2.CC_STAT_HEIGHT] > 3
    ]
    if len(widths) < 4:
        return False
    arr = np.array(widths, dtype=float)
    cv_val = arr.std() / arr.mean() if arr.mean() > 0 else 1.0
    return bool(cv_val < 0.15)


def _looks_like_code(text: str) -> bool:
    """Return True if text looks like a programming code block."""
    if '\n' not in text:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    # Strip LaTeX math before counting to avoid false positives on theorem/proof text.
    text_no_math = re.sub(r'[$][^$]+[$]', '', text)
    code_chars = sum(
        ln.count('{') + ln.count('}') + ln.count(';') + ln.count('(') + ln.count(')')
        for ln in text_no_math.splitlines()
        if ln.strip()
    )
    return code_chars / len(lines) >= 1.0


_THEOREM_RE = re.compile(
    r"^(Theorem|Lemma|Proposition|Corollary|Claim)"
)


def _detect_code_indented_lines(
    binary: np.ndarray,
    line_positions: list[tuple[int, int]],
    ocr_text: str,
    bbox_w: int,
) -> list[str]:
    """Detect per-line left indent from image pixels and prepend spaces to OCR lines.

    For a monospace code block the OCR strips leading whitespace, so we recover
    indentation by measuring the leftmost ink x-position in each image line band.
    Uses detected line bands when counts match; falls back to equal-height grid
    slicing when detect_line_positions over/under-segments the crop.
    """
    ocr_lines = ocr_text.splitlines()
    n_ocr = len(ocr_lines)
    if n_ocr == 0:
        return []

    h, _w = binary.shape
    n_img = len(line_positions)

    # Build per-line ink bands: prefer detected positions, fall back to grid.
    left_offsets: list[int] = []
    if n_img > 0 and abs(n_img - n_ocr) <= max(2, int(n_ocr * 0.4)):
        for top, bot in line_positions:
            band = binary[top:bot, :]
            ink_cols = np.where(band.any(axis=0))[0]
            left_offsets.append(int(ink_cols[0]) if len(ink_cols) > 0 else 0)
    else:
        # Equal-height grid: slice crop into n_ocr strips
        lh = max(1, h // n_ocr)
        for i in range(n_ocr):
            top = i * lh
            bot = min(h, (i + 1) * lh)
            band = binary[top:bot, :]
            ink_cols = np.where(band.any(axis=0))[0]
            left_offsets.append(int(ink_cols[0]) if len(ink_cols) > 0 else 0)

    if not left_offsets:
        return []
    min_offset = min(left_offsets)
    max_chars = max((len(ln) for ln in ocr_lines if ln.strip()), default=1)
    char_w = max(1.0, bbox_w / max_chars)

    result: list[str] = []
    for i, ocr_line in enumerate(ocr_lines):
        offset = left_offsets[i] if i < len(left_offsets) else min_offset
        n_spaces = max(0, round((offset - min_offset) / char_w))
        result.append(' ' * n_spaces + ocr_line.strip())
    return result

def _detect_handwriting(gray_crop: np.ndarray) -> bool:
    """Return True if stroke widths are highly variable (handwriting)."""
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    if binary.sum() == 0:
        return False
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    sw = dist[binary > 0]
    if len(sw) < 50 or sw.mean() < 1e-6:
        return False
    return bool(float(sw.std() / sw.mean()) > 0.5)


def classify_font_category(gray_crop: np.ndarray) -> str:
    """Classify font into 'monospace' | 'handwriting' | 'serif' | 'sans-serif'."""
    if gray_crop.size == 0:
        return "sans-serif"
    if _detect_monospace(gray_crop):
        return "monospace"
    if _detect_handwriting(gray_crop):
        return "handwriting"
    return detect_font_category(gray_crop)


# --- Task 4: Bold detection ---


def measure_mean_stroke_width(gray_crop: np.ndarray) -> float:
    """Measure mean stroke width of text in a grayscale crop."""
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    if binary.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton = _thin(binary)
    widths = dist[skeleton > 0]
    if len(widths) == 0:
        return 0.0
    return float(np.mean(widths))


def detect_bold(stroke_width: float, glyph_height: int) -> bool:
    """Bold if stroke-to-glyph-height ratio exceeds threshold.

    Bold strokes are ~18-22% of cap height; regular strokes are ~10-14%.
    """
    if glyph_height <= 0 or stroke_width <= 0:
        return False
    return (stroke_width / glyph_height) > 0.17


# --- Task 5: Text alignment detection ---


def detect_alignment(
    bboxes: list[tuple[int, int, int, int]], page_width: int
) -> str:
    """Detect text alignment from a group of bounding boxes.

    Analyzes variance of left edges, right edges, and centers to classify
    as left, right, center, or justified.
    """
    if len(bboxes) <= 1:
        return "left"

    left_edges = [b[0] for b in bboxes]
    right_edges = [b[2] for b in bboxes]
    centers = [(b[0] + b[2]) / 2 for b in bboxes]

    # Threshold: 1% of page width.
    threshold = page_width * 0.01
    threshold_sq = threshold**2

    left_var = float(np.var(left_edges))
    right_var = float(np.var(right_edges))
    center_var = float(np.var(centers))

    if left_var < threshold_sq and right_var < threshold_sq:
        return "justified"
    if right_var < threshold_sq:
        return "right"
    if center_var < threshold_sq:
        return "center"
    return "left"


# --- Task 5b: First-line indent, last-line fill, and crop-based alignment ---


def detect_first_line_indent(gray_crop: np.ndarray) -> int:
    """Detect first-line paragraph indent via dual-projection method.

    Returns indent in pixels (0 if < 5 px or single-line element).
    """
    if gray_crop.size == 0:
        return 0
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    h_proj = np.sum(binary > 0, axis=1)
    if h_proj.max() == 0:
        return 0

    line_bands = detect_line_positions(gray_crop)
    # Drop bands too thin to be real text lines (tops of tall chars, descenders).
    _min_band_h = max(4, gray_crop.shape[0] * 0.15)
    line_bands = [(y0, y1) for y0, y1 in line_bands if y1 - y0 >= _min_band_h]
    if len(line_bands) < 2:
        return 0

    # Column projection on first line band → leftmost ink column.
    y0, y1 = line_bands[0]
    first_band = binary[y0:y1, :]
    col_proj = np.sum(first_band > 0, axis=0)
    ink_cols = np.where(col_proj > 0)[0]
    if len(ink_cols) == 0:
        return 0
    first_line_left = int(ink_cols[0])

    # Body left = median of leftmost ink column across remaining line bands.
    body_lefts: list[int] = []
    for y0b, y1b in line_bands[1:]:
        band = binary[y0b:y1b, :]
        cp = np.sum(band > 0, axis=0)
        ic = np.where(cp > 0)[0]
        if len(ic) > 0:
            body_lefts.append(int(ic[0]))
    if not body_lefts:
        return 0

    body_left = int(np.median(body_lefts))
    indent = first_line_left - body_left
    # If the first line starts more than 25% into the crop it is a mid-line
    # continuation from the previous element, not a real paragraph indent.
    if indent >= 5 and first_line_left <= gray_crop.shape[1] * 0.25:
        return indent
    return 0


def estimate_last_line_fill(gray_crop: np.ndarray) -> float:
    """Estimate how full the last text line is (0.0–1.0).

    Gaussian blur → Otsu binarise → bottom-up scan for last line band →
    vertical projection to find rightmost ink column.
    Returns 0.0 for blank crops.
    """
    if gray_crop.size == 0:
        return 0.0
    blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    line_bands = detect_line_positions(gray_crop)
    if not line_bands:
        return 0.0

    y0, y1 = line_bands[-1]
    last_band = binary[y0:y1, :]
    col_proj = np.sum(last_band > 0, axis=0).astype(float)
    ink_cols = np.where(col_proj > col_proj.max() * 0.05)[0] if col_proj.max() > 0 else np.array([])
    if len(ink_cols) == 0:
        return 0.0

    crop_width = gray_crop.shape[1]
    return min(1.0, float(ink_cols[-1]) / max(1, crop_width - 1))


def detect_alignment_from_crop(gray_crop: np.ndarray) -> str:
    """Detect text alignment from a grayscale paragraph crop.

    Uses detect_line_positions to find line bands, then computes per-line
    left/right ink edges.  Uses a fraction-based approach rather than
    variance so that paragraph-start indents and short last lines (which
    occur frequently in multi-paragraph OCR elements) don't pollute the
    signal:

    - left_fraction: share of body lines (excluding first) that start
      within tolerance of the minimum (body) left edge.
    - right_fraction: share of body lines (excluding last) that end
      within tolerance of the maximum right edge.

    tolerance = max(10, 3% of crop width).

    justified  → left_fraction >= 0.6 AND right_fraction >= 0.6
    left       → left_fraction >= 0.6
    right      → right_fraction >= 0.8
    center     → center_var < threshold_sq
    else       → left
    """
    if gray_crop.size == 0:
        return "left"
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    line_bands = detect_line_positions(gray_crop)
    if len(line_bands) <= 1:
        return "left"

    crop_width = gray_crop.shape[1]

    left_edges: list[float] = []
    right_edges: list[float] = []

    for y0, y1 in line_bands:
        if y1 - y0 < 5:
            continue  # skip thin fragment bands
        band = binary[y0:y1, :]
        col_proj = np.sum(band > 0, axis=0)
        ink_cols = np.where(col_proj > 0)[0]
        if len(ink_cols) == 0:
            continue
        left_edges.append(float(ink_cols[0]))
        right_edges.append(float(ink_cols[-1]))

    if len(left_edges) <= 1:
        return "left"

    # Tolerance = 3% of crop width (min 10 px).
    tolerance = max(10.0, crop_width * 0.03)

    # Left consistency: fraction of body lines (excluding first, which may be
    # indented) that start within tolerance of the minimum body left edge.
    # Paragraph-start indents in multi-paragraph elements are excluded this way.
    body_left = left_edges[1:]
    if body_left:
        min_left = float(min(body_left))
        left_fraction = sum(1 for l in body_left if l <= min_left + tolerance) / len(body_left)  # noqa: E741
    else:
        left_fraction = 0.0

    # Right consistency: fraction of body lines (excluding last, which is
    # typically a short paragraph-ending line) that end within tolerance of
    # the maximum right edge.
    body_right = right_edges[:-1] if len(right_edges) > 1 else right_edges
    if body_right:
        max_right = float(max(body_right))
        right_fraction = sum(1 for r in body_right if r >= max_right - tolerance) / len(body_right)
    else:
        right_fraction = 0.0

    # Center variance (kept for center-alignment detection).
    centers = [(l + r) / 2.0 for l, r in zip(left_edges, right_edges)]  # noqa: E741
    threshold_sq = max(100.0, (crop_width * 0.03) ** 2)
    center_var = float(np.var(centers))

    if left_fraction >= 0.6 and right_fraction >= 0.6:
        return "justified"
    if left_fraction >= 0.6:
        return "left"
    if right_fraction >= 0.8:
        return "right"
    if center_var < threshold_sq:
        return "center"
    return "left"


# --- Task 6: Line break detection ---


def detect_line_positions(gray_crop: np.ndarray) -> list[tuple[int, int]]:
    """Detect text line positions via horizontal projection profile.

    Returns list of (y_start, y_end) for each detected line.
    """
    binary = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    # Horizontal projection: count black pixels per row.
    projection = np.sum(binary > 0, axis=1)
    if projection.max() == 0:
        return []

    # Find runs of rows with ink (above a noise threshold).
    # 15% of max filters out isolated character fragments / thin serifs while
    # keeping full text-line bands intact.
    threshold = projection.max() * 0.15
    in_line = projection > threshold
    lines: list[tuple[int, int]] = []
    start = None
    for i, val in enumerate(in_line):
        if val and start is None:
            start = i
        elif not val and start is not None:
            lines.append((start, i))
            start = None
    if start is not None:
        lines.append((start, len(in_line)))
    return lines


# --- Task 7: DocumentAnalyzer orchestrator ---


def _group_into_columns(
    elements: list[AnalyzedElement],
) -> list[list[AnalyzedElement]]:
    """Group elements into columns by x-range overlap.

    Two elements belong to the same column if their x-ranges overlap by
    more than 40% of the narrower element's width.
    """
    col_ranges: list[tuple[int, int]] = []
    col_members: list[list[AnalyzedElement]] = []

    for elem in elements:
        x1, _, x2, _ = elem.text_element.bbox
        placed = False
        for i, (cx1, cx2) in enumerate(col_ranges):
            overlap = max(0, min(x2, cx2) - max(x1, cx1))
            if overlap / max(1, min(x2 - x1, cx2 - cx1)) > 0.4:
                col_members[i].append(elem)
                col_ranges[i] = (min(cx1, x1), max(cx2, x2))
                placed = True
                break
        if not placed:
            col_ranges.append((x1, x2))
            col_members.append([elem])

    return col_members


class DocumentAnalyzer:
    """Analyzes original document image to extract precise text styling."""

    def analyze(
        self, image: Image.Image, parsed: ParsedDocument
    ) -> AnalyzedDocument:
        """Analyze original image + parsed OCR to produce styled elements."""
        effective_dpi = estimate_dpi(image.width, image.height)
        gray = np.array(image.convert("L"))

        # First pass: measure stroke widths for all elements (for bold detection).
        stroke_widths: list[float] = []
        for te in parsed.text_elements:
            x1, y1, x2, y2 = te.bbox
            crop = gray[y1:y2, x1:x2]
            if crop.size > 0:
                stroke_widths.append(measure_mean_stroke_width(crop))
            else:
                stroke_widths.append(0.0)

        # Second pass: build analyzed elements.
        elements: list[AnalyzedElement] = []

        for i, te in enumerate(parsed.text_elements):
            x1, y1, x2, y2 = te.bbox
            bbox_h = max(1, y2 - y1)
            crop = gray[y1:y2, x1:x2]

            # Font size from DPI, refined with glyph height.
            glyph_h = measure_glyph_height(crop) if crop.size > 0 else bbox_h
            effective_h = glyph_h if glyph_h > 0 else bbox_h
            font_size_pt = estimate_font_size_pt(effective_h, effective_dpi)

            # Font category (monospace / handwriting / serif / sans-serif).
            font_category = classify_font_category(crop) if crop.size > 0 else "sans-serif"
            if font_category != "monospace" and _looks_like_code(te.text):
                font_category = "monospace"
            elif font_category == "monospace" and not _looks_like_code(te.text) and chr(10) not in te.text:
                font_category = detect_font_category(crop) if crop.size > 0 else "sans-serif"
            # Font weight and bold flag.
            font_weight = classify_font_weight(stroke_widths[i], glyph_h)
            is_bold = font_weight in {"bold", "heavy", "black"}
            # If the ENTIRE element text is enclosed in **, the whole element is bold.
            # Partial ** markers are left in te.text for inline bold rendering.
            if "**" in te.text and re.match(r"^\*\*[^*]+\*\*$", te.text.strip()):
                is_bold = True
                te.text = te.text.replace("**", "")

            # Italic detection.
            is_italic = False if font_category == "monospace" else (detect_italic(crop)[0] if crop.size > 0 else False)
            # Theorem/lemma text is nearly always italic in typeset math books;
            # image-based slant detection fails when inline formulas dilute the signal.
            if not is_italic and font_category == "serif" and _THEOREM_RE.match(te.text.strip()):
                is_italic = True

            # Line positions.
            line_positions = detect_line_positions(crop) if crop.size > 0 else []

            # Line height from detected lines.
            if len(line_positions) >= 2:
                line_height_px = line_positions[1][0] - line_positions[0][0]
            elif line_positions:
                line_height_px = line_positions[0][1] - line_positions[0][0]
            else:
                line_height_px = bbox_h

            # Per-crop alignment, indent, and last-line fill.
            alignment = detect_alignment_from_crop(crop) if crop.size > 0 else "left"
            indent_px = detect_first_line_indent(crop) if crop.size > 0 else 0

            # For monospace code blocks: recover per-line indentation from image pixels.
            code_lines: list[str] = []
            if font_category == "monospace" and '\n' in te.text and crop.size > 0 and line_positions:
                binary_crop = cv2.threshold(
                    crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )[1]
                code_lines = _detect_code_indented_lines(
                    binary_crop, line_positions, te.text, max(1, x2 - x1)
                )

            style = TextStyle(
                font_size_pt=font_size_pt,
                font_category=font_category,
                is_bold=is_bold,
                is_italic=is_italic,
                font_weight=font_weight,
                alignment=alignment,
                line_height_px=line_height_px,
                indent_px=indent_px,
                lines=code_lines,
            )
            elements.append(AnalyzedElement(text_element=te, style=style))

        return AnalyzedDocument(
            elements=elements,
            image_regions=parsed.image_regions,
            table_regions=parsed.table_regions,
            table_elements=parsed.table_elements,
            page_dpi=effective_dpi,
        )

    def _detect_alignments(
        self, elements: list[AnalyzedElement], page_width: int
    ) -> None:
        """Detect alignment per column group."""
        for col in _group_into_columns(elements):
            if not col:
                continue
            bboxes = [e.text_element.bbox for e in col]
            alignment = detect_alignment(bboxes, page_width)
            for e in col:
                e.style.alignment = alignment
