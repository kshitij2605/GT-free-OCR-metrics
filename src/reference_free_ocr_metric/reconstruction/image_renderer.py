"""Render text elements onto a blank image at bounding box positions."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

# Matches $...$ and $$...$$ inline/display LaTeX.
_LATEX_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]*?\$", re.DOTALL)

# Detects CJK Unified Ideographs and common CJK blocks.
_CJK_RE = re.compile(r"[\u2E80-\u9FFF\uF900-\uFAFF\uFE30-\uFE4F\uFF00-\uFFEF]")

# Glyphs that Liberation/DejaVu Latin fonts often lack (dingbats, arrows,
# geometric shapes, box drawing, miscellaneous symbols, CJK). When any of
# these appear we route to the CJK-comprehensive font stack which has
# fuller Unicode coverage.
_EXT_GLYPH_RE = re.compile(r"[\u2000-\u2BFF\u3000-\uFFFF]")


_BOLD_SPLIT_RE = re.compile(
    r"(\*\*.*?\*\*|\*[^*\n]+?\*|\$\$.*?\$\$|\$[^$\n]*?\$)", re.DOTALL
)
# Matches only actual markdown bold/italic markers (not bare * in code/math).
_HAS_INLINE_MARK_RE = re.compile(r"\*\*.*?\*\*|\*[^*\n]+?\*", re.DOTALL)



def _split_by_formula(text: str) -> list[tuple[str, str]]:
    """Split text into ('text', ...) and ('formula', ...) segments."""
    segs: list[tuple[str, str]] = []
    last = 0
    for m in _LATEX_RE.finditer(text):
        if m.start() > last:
            segs.append(("text", text[last : m.start()]))
        segs.append(("formula", m.group(0).strip("$")))
        last = m.end()
    if last < len(text):
        segs.append(("text", text[last:]))
    return segs

def _split_inline_content(text: str) -> list[tuple[str, str]]:
    """Split text into ('text',...), ('formula',...), and ('bold',...) segments."""
    segments: list[tuple[str, str]] = []
    last = 0
    for m in _BOLD_SPLIT_RE.finditer(text):
        if m.start() > last:
            segments.append(("text", text[last : m.start()]))
        raw = m.group(0)
        if raw.startswith("**"):
            segments.append(("bold", raw[2:-2]))
        elif raw.startswith("*"):
            segments.append(("italic", raw[1:-1]))
        else:
            segments.append(("formula", raw.strip("$")))
        last = m.end()
    if last < len(text):
        segments.append(("text", text[last:]))
    return segments

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402


def _cjk_mathtext_rc():
    """Return an rc_context that swaps mathtext's `rm` slot to a CJK-capable
    font so `\\text{<chinese>}` inside formulas renders glyphs instead of tofu.

    Other slots stay on DejaVu so math symbols (Greek letters, integrals, etc.)
    keep working — only the upright-text slot needs CJK coverage.
    """
    import matplotlib
    return matplotlib.rc_context({
        "mathtext.fontset": "custom",
        "mathtext.rm": "WenQuanYi Zen Hei",
        "mathtext.it": "DejaVu Serif:italic",
        "mathtext.bf": "DejaVu Serif:bold",
        "mathtext.cal": "DejaVu Serif:italic",
        "mathtext.sf": "DejaVu Sans",
        "mathtext.tt": "DejaVu Sans Mono",
    })

from reference_free_ocr_metric.reconstruction.html_parser import TableElement, TextElement  # noqa: E402

if TYPE_CHECKING:
    from reference_free_ocr_metric.reconstruction.document_analyzer import (
        AnalyzedDocument,
    )

# CJK-capable fonts searched in priority order.
_FONT_PATHS = [
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
]

# CJK-capable fonts in priority order (used when CJK characters are detected).
# WenQuanYi leads because the dataset is mostly simplified Chinese (标 头 误
# 电 are missing from IPAGothic). For glyphs WenQuanYi lacks (✓ ✗), the
# per-character fallback in _draw_with_cjk_fallback below routes individual
# chars to IPAGothic / Noto.
_CJK_FONT_PATHS: list[str] = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
]

_MIN_FONT_SIZE = 8
_MAX_FONT_SIZE = 120

# Test string with CJK, Latin, and descender for line-height measurement.
_LINE_HEIGHT_SAMPLE = "あAy"

# Styled font search paths by (category, bold, italic).
# Liberation fonts lead: metric-compatible substitutes for Times NR (serif)
# and Arial (sans-serif), ensuring correct glyph widths for Latin documents.
_FONT_MAP: dict[tuple[str, bool, bool], list[str]] = {
    ("sans-serif", False, False): [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    ],
    ("sans-serif", True, False): [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    ],
    ("sans-serif", False, True): [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    ],
    ("sans-serif", True, True): [
        "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    ("serif", False, False): [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-mincho.ttf",
    ],
    ("serif", True, False): [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-mincho.ttf",
    ],
    ("serif", False, True): [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ],
    ("serif", True, True): [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ],
    ("monospace", False, False): [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    ],
    ("monospace", True, False): [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ],
    ("monospace", False, True): [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ],
    ("monospace", True, True): [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ],
    # Handwriting: no dedicated TTF on most servers; fall through to sans-serif.
    ("handwriting", False, False): [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    ("handwriting", True, False): [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    ("handwriting", False, True): [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    ],
    ("handwriting", True, True): [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    ],
}


def _find_font_path() -> str | None:
    """Return the first available font path from the search list."""
    for path in _FONT_PATHS:
        if os.path.exists(path):
            return path
    return None


# Resolved once at import time.
_RESOLVED_FONT_PATH = _find_font_path()


class ImageRenderer:
    """Renders OCR text elements onto a blank image."""

    def __init__(self) -> None:
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._styled_font_cache: dict[
            tuple[int, str, bool, bool, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont
        ] = {}

    def render_text_image(
        self,
        text_elements: list[TextElement],
        width: int,
        height: int,
        background: str = "white",
    ) -> Image.Image:
        """Create an image with text drawn at bounding box positions.

        For each element, estimates the appropriate font size so text
        wraps within the bbox boundaries without overflow.
        """
        img = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(img)

        for element in text_elements:
            x1, y1, x2, y2 = element.bbox
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            if element.tag == "formula":
                formula_img = self._render_formula_patch(element.text, bbox_w, bbox_h)
                if formula_img is not None:
                    img.paste(formula_img, (x1, y1))
                else:
                    draw.rectangle([x1, y1, x2, y2], fill=(220, 220, 220))
                continue
            # Use plain text (LaTeX stripped) only for font-size estimation.
            text_plain = _LATEX_RE.sub("", element.text).strip()
            estimate = text_plain or element.text.strip("$").strip()
            if not estimate:
                continue
            cjk = bool(_CJK_RE.search(element.text))
            font_size = self._estimate_font_size(estimate, bbox_w, bbox_h)
            font = self._get_styled_font(font_size, "sans-serif", False, False, cjk)
            self._draw_mixed_text(img, draw, element.text, x1, y1, bbox_w, bbox_h, font, font_size)

        return img

    def render_analyzed_document(
        self,
        doc: AnalyzedDocument,
        width: int,
        height: int,
        background: str = "white",
        render_formulas: bool = True,
        render_tables: bool = True,
        render_text: bool = True,
        original_image: Image.Image | None = None,
        canvas: Image.Image | None = None,
    ) -> Image.Image:
        """Render using precise style info from DocumentAnalyzer.

        When `canvas` is provided, render onto a copy of it (so the page
        background of the original document is preserved between elements);
        otherwise create a blank `background`-colored image.
        """
        if canvas is not None:
            img = canvas.copy().convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height))
        else:
            img = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(img)

        for elem in doc.elements:
            te = elem.text_element
            style = elem.style
            x1, y1, x2, y2 = te.bbox
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            if te.tag == "formula":
                if render_formulas:
                    formula_img = self._render_formula_patch(te.text, bbox_w, bbox_h)
                    if formula_img is not None:
                        img.paste(formula_img, (x1, y1))
                    else:
                        draw.rectangle([x1, y1, x2, y2], fill=(220, 220, 220))
                continue
            if not render_text:
                continue
            text_plain = _LATEX_RE.sub("", te.text).replace("**", "").strip()
            estimate = text_plain or te.text.strip("$").strip()
            if not estimate:
                continue

            cjk = bool(_CJK_RE.search(te.text))

            # Use measured font size (convert pt -> px at page DPI).
            font_size_px = max(
                _MIN_FONT_SIZE, int(style.font_size_pt * doc.page_dpi / 72)
            )
            font_size_px = min(font_size_px, _MAX_FONT_SIZE)
            initial_font_px = font_size_px
            font = self._get_styled_font(
                font_size_px, style.font_category, style.is_bold, style.is_italic, cjk
            )

            # Build shrink_estimate: replace each inline formula with X-char proxies
            # whose total pixel width matches the actual rendered formula image.
            # Use the same formula_h as _draw_mixed_text (cap_h * 0.75) so the
            # proxy widths match what the renderer will actually produce.
            if _LATEX_RE.search(te.text):
                try:
                    _cap_h = font.getbbox("H")[3]
                except Exception:
                    _cap_h = int(font_size_px * 0.7)
                _ref_lh = max(8, int(_cap_h * 0.75))
                _x_w = max(1, self._text_width("X", font, font_size_px))
                _fcache: dict[str, int] = {}
                def _fproxy(m: re.Match) -> str:  # noqa: E731
                    latex = m.group(0).strip("$")
                    if latex not in _fcache:
                        fi = self._render_inline_formula(latex, _ref_lh)
                        _fcache[latex] = max(1, (fi.width + _x_w - 1) // _x_w) if fi else max(2, len(latex) * 3 // 4)
                    return "X" * _fcache[latex]
                shrink_estimate = _LATEX_RE.sub(_fproxy, te.text)
            elif "**" in te.text:
                shrink_estimate = estimate.replace("**", "")
            else:
                shrink_estimate = estimate

            # Pre-render unique formulas once for accurate line counting.
            # _fw_raw[latex] = fi.width at _ref_lh; scaled each shrink iteration.
            _has_formulas = bool(_LATEX_RE.search(te.text)) and not style.lines
            _fw_raw: dict[str, int] = {}
            if _has_formulas:
                for _m in _LATEX_RE.finditer(te.text):
                    _lat = _m.group(0).strip("$")
                    if _lat not in _fw_raw:
                        _fi = self._render_inline_formula(_lat, _ref_lh)
                        _fw_raw[_lat] = _fi.width if _fi else max(4, int(len(_lat) * _x_w * 0.5))

            def _get_trial_lines(fs: int, fnt: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> list[str]:
                """Return list whose len() equals wrapped line count at font size fs."""
                if style.lines:
                    return style.lines
                if _has_formulas:
                    try:
                        _cc = fnt.getbbox("H")[3]
                    except Exception:
                        _cc = int(fs * 0.7)
                    _cfh = max(8, int(_cc * 0.75))
                    _scale = _cfh / max(1, _ref_lh)
                    _scaled = {k: int(v * _scale) + 2 for k, v in _fw_raw.items()}
                    return [""] * self._count_mixed_lines(te.text, bbox_w, fnt, fs, _cfh, _scaled)
                return self._wrap_text_chars(shrink_estimate, bbox_w, fnt, fs)

            # Shrink font until wrapped text fits vertically in the bbox.
            # Uses _count_mixed_lines for formula elements (mirrors _draw_mixed_text
            # token-level wrapping exactly), _wrap_text_chars otherwise.
            while font_size_px > _MIN_FONT_SIZE:
                font = self._get_styled_font(font_size_px, style.font_category, style.is_bold, style.is_italic, cjk)
                trial_lines = _get_trial_lines(font_size_px, font)
                if isinstance(font, ImageFont.FreeTypeFont):
                    try:
                        trial_lh = int(font.getbbox(_LINE_HEIGHT_SAMPLE)[3]) + int(font_size_px * 0.15)
                    except Exception:
                        ascent, descent = font.getmetrics()
                        trial_lh = ascent + abs(descent) + int(font_size_px * 0.15)
                else:
                    trial_lh = int(font_size_px * 1.15)
                if (len(trial_lines) - 1) * trial_lh + font_size_px <= bbox_h:
                    break
                font_size_px -= 1

            # If the shrink loop reduced the font, allow reverting to the initial
            # size only if the text fits exactly (no overflow).
            if initial_font_px > font_size_px:
                test_font = self._get_styled_font(initial_font_px, style.font_category, style.is_bold, style.is_italic, cjk)
                test_lines = _get_trial_lines(initial_font_px, test_font)
                if isinstance(test_font, ImageFont.FreeTypeFont):
                    try:
                        test_lh = int(test_font.getbbox(_LINE_HEIGHT_SAMPLE)[3]) + int(initial_font_px * 0.15)
                    except Exception:
                        t_asc, t_desc = test_font.getmetrics()
                        test_lh = t_asc + abs(t_desc) + int(initial_font_px * 0.15)
                else:
                    test_lh = int(initial_font_px * 1.15)
                if (len(test_lines) - 1) * test_lh + initial_font_px <= bbox_h:
                    font_size_px = initial_font_px
                    font = test_font

            font = self._get_styled_font(font_size_px, style.font_category, style.is_bold, style.is_italic, cjk)

            # For single-line elements, grow font to the max that still fits on one
            # line within bbox_w. OCR bboxes for single lines are tight vertical
            # crops of the ink, so bbox_h < actual line height — the shrink loop
            # undersizes relative to surrounding multi-line elements.
            # Uses shrink_estimate (formula-augmented) to avoid overshooting width.
            trial_lines = style.lines if style.lines else self._wrap_text_chars(shrink_estimate, bbox_w, font, font_size_px)
            if len(trial_lines) == 1:
                upper = _MAX_FONT_SIZE
                lo, hi = font_size_px + 1, upper
                while lo <= hi:
                    mid = (lo + hi) // 2
                    test_font = self._get_styled_font(mid, style.font_category, style.is_bold, style.is_italic, cjk)
                    test_lines = self._wrap_text_chars(shrink_estimate, bbox_w, test_font, mid)
                    if isinstance(test_font, ImageFont.FreeTypeFont):
                        t_asc, t_desc = test_font.getmetrics()
                        test_lh = t_asc + abs(t_desc)
                    else:
                        test_lh = int(mid * 1.15)
                    if len(test_lines) == 1 and test_lh <= bbox_h:
                        font_size_px = mid
                        font = test_font
                        lo = mid + 1
                    else:
                        hi = mid - 1

            # If element has inline formulas or markdown bold/italic, use mixed renderer.
            # Skip for code blocks (style.lines set or monospace) — * is an operator there.
            # But when LaTeX is present, the element is math content (not code) and the
            # mixed renderer must run so $...$ doesn't render as raw text.
            _has_latex = bool(_LATEX_RE.search(te.text))
            _is_code = (bool(style.lines) or style.font_category == "monospace") and not _has_latex
            if not _is_code and (_has_latex or _HAS_INLINE_MARK_RE.search(te.text)):
                self._draw_mixed_text(img, draw, te.text, x1, y1, bbox_w, bbox_h, font, font_size_px, style.alignment, style.indent_px, style.font_category, style.is_italic)
            else:
                lines = style.lines if style.lines else self._wrap_text_chars(text_plain, bbox_w, font, font_size_px)
                if isinstance(font, ImageFont.FreeTypeFont):
                    try:
                        natural_lh = int(font.getbbox(_LINE_HEIGHT_SAMPLE)[3]) + int(font_size_px * 0.15)
                    except Exception:
                        ascent, descent = font.getmetrics()
                        natural_lh = ascent + abs(descent)
                else:
                    natural_lh = int(font_size_px * 1.15)
                # Expand line spacing to fill bbox (proportional leading), capped at 1.5×
                if len(lines) > 1:
                    line_height = max(natural_lh, min(bbox_h // len(lines), int(natural_lh * 1.5)))
                else:
                    line_height = natural_lh
                # Pre-indented code lines: draw left-aligned with no extra indent offset.
                alignment = "left" if style.lines else style.alignment
                indent = 0 if style.lines else style.indent_px
                self._draw_aligned_lines(draw, lines, x1, y1, bbox_w, bbox_h, font, line_height, alignment, indent)

        if render_tables:
            for te in doc.table_elements:
                self._render_table(img, draw, te, original_image, doc.page_dpi)

        return img


    def _detect_table_grid(
        self,
        crop: Image.Image,
        n_rows: int,
        n_cols: int,
    ) -> tuple[list[int] | None, list[int] | None]:
        """Detect column-divider x-positions and row-divider y-positions
        from a cropped table image.

        Uses gradient + morphology to find continuous horizontal/vertical
        edges. Catches both thin grid rules (top/bottom of a 1-px black
        line both produce gradient rows) and colour-block boundaries (the
        pink→white transition between a styled header and a data row also
        produces a gradient row). Text strokes also produce gradients but
        only over short spans, so the morphological opening with a
        kernel sized to 30% of the table extent filters them out. Adds
        outer borders heuristically when the first/last detected line is
        well inside the crop. Returns (col_xs, row_ys) — sorted positions
        including outer borders, each with length n_cols+1 / n_rows+1 —
        or (None, None) when the detected counts don't match the HTML.
        """
        try:
            import numpy as np
            from scipy.ndimage import binary_opening
        except Exception:
            return None, None
        gray = np.array(crop.convert("L")).astype(np.int16)
        H, W = gray.shape
        if H < 20 or W < 20:
            return None, None
        # Trim to ink bounding box: the OCR bbox is often loose and
        # leaves blank page margin below/right of the table. The grid
        # detector gets confused by that margin (the bottom border
        # heuristic adds a phantom line where there's only whitespace).
        # Use a tight bbox computed from non-white content.
        nonwhite = gray < 240
        if nonwhite.any():
            ry = np.where(nonwhite.any(axis=1))[0]
            rx = np.where(nonwhite.any(axis=0))[0]
            ty1, ty2 = int(ry[0]), int(ry[-1]) + 1
            tx1, tx2 = int(rx[0]), int(rx[-1]) + 1
            tight = gray[ty1:ty2, tx1:tx2]
            if tight.shape[0] >= 20 and tight.shape[1] >= 20:
                gray = tight
                H, W = gray.shape
                tight_offset_y, tight_offset_x = ty1, tx1
            else:
                tight_offset_y, tight_offset_x = 0, 0
        else:
            tight_offset_y, tight_offset_x = 0, 0

        ydif = np.abs(np.diff(gray, axis=0))  # (H-1, W)
        xdif = np.abs(np.diff(gray, axis=1))  # (H, W-1)
        # Gradient threshold: 10 catches faint coloured-block transitions
        # (e.g. pink → white ≈ 30 grayscale units) without blowing up on
        # JPEG noise (typically <8).
        ymask = ydif > 10
        xmask = xdif > 10
        # Kernel = 30% of table extent: long enough that text stroke
        # gradients (~tens of pixels per word) don't survive opening, but
        # short enough that a real divider whose full-width run is broken
        # by occasional noise pixels still survives.
        hk = max(20, int(0.3 * W))
        vk = max(20, int(0.3 * H))
        h_edges = binary_opening(ymask, structure=np.ones((1, hk)))
        v_edges = binary_opening(xmask, structure=np.ones((vk, 1)))

        def cluster(positions: np.ndarray, min_gap: int = 8) -> list[int]:
            if len(positions) == 0:
                return []
            out: list[list[int]] = [[int(positions[0]), int(positions[0])]]
            for v in positions[1:]:
                vi = int(v)
                if vi - out[-1][1] <= min_gap:
                    out[-1][1] = vi
                else:
                    out.append([vi, vi])
            return [(s + e) // 2 for s, e in out]

        h_lines = cluster(np.where(h_edges.any(axis=1))[0])
        v_lines = cluster(np.where(v_edges.any(axis=0))[0])

        def _drop_close(lines: list[int], target: int) -> list[int]:
            # Detected lines whose neighbour gap is < 1/3 of the median
            # gap are almost always spurious (e.g. internal stripes inside
            # a coloured header band). Drop the smaller-margin one only
            # when we already have at least the target number of lines —
            # otherwise the extra line might be a real divider that the
            # rest of the heuristic still needs.
            while len(lines) >= max(3, target):
                gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
                med = sorted(gaps)[len(gaps) // 2]
                worst_idx = min(range(len(gaps)), key=lambda i: gaps[i])
                if gaps[worst_idx] < max(8, med // 3):
                    # The pair (worst_idx, worst_idx+1) is suspiciously
                    # close. Prefer to keep a line that sits at the very
                    # edge of the crop (a real outer border) and drop its
                    # spurious internal partner.
                    a, b = lines[worst_idx], lines[worst_idx + 1]
                    extent = lines[-1] + 1
                    if a <= 2:
                        lines = lines[: worst_idx + 1] + lines[worst_idx + 2 :]
                    elif b >= extent - 2:
                        lines = lines[: worst_idx] + lines[worst_idx + 1 :]
                    elif worst_idx == 0:
                        lines = lines[1:]
                    elif worst_idx == len(gaps) - 1:
                        lines = lines[:-1]
                    else:
                        if gaps[worst_idx - 1] < gaps[worst_idx + 1]:
                            lines = lines[: worst_idx] + lines[worst_idx + 1 :]
                        else:
                            lines = lines[: worst_idx + 1] + lines[worst_idx + 2 :]
                else:
                    break
            return lines

        h_lines = _drop_close(h_lines, n_rows + 1)
        v_lines = _drop_close(v_lines, n_cols + 1)

        # Heuristically add missing outer borders. Only add an outer
        # border when the gap at the start/end is no larger than the
        # internal cell pitch — that ensures a loose OCR bbox (which
        # often has 10-30 px of empty page below the actual table)
        # does not get a phantom bottom border that inflates line count.
        # Outer-border heuristic with ink-density gate. We append a
        # phantom outer border only when (a) the gap is no larger than
        # one cell pitch — eliminates unrelated content far from the
        # table — AND (b) the gap region contains appreciable ink
        # (real continuing content), not just bbox slack.
        gray_for_density = gray
        def _ink_density(y_lo: int, y_hi: int, x_lo: int, x_hi: int) -> float:
            y_lo = max(0, y_lo); y_hi = min(gray_for_density.shape[0], y_hi)
            x_lo = max(0, x_lo); x_hi = min(gray_for_density.shape[1], x_hi)
            if y_hi <= y_lo or x_hi <= x_lo:
                return 0.0
            sub = gray_for_density[y_lo:y_hi, x_lo:x_hi]
            return float((sub < 200).mean())
        def _add_outer_h(lines: list[int], target: int) -> list[int]:
            if not lines:
                return lines
            if len(lines) >= 2:
                gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
                pitch = sorted(gaps)[len(gaps) // 2]
            else:
                pitch = H
            slots = target - len(lines)
            # When we need more lines (slots > 0), drop the pitch upper
            # bound on the gap — a real missing outer border can sit
            # several cells away (e.g. only one v_line pair detected for
            # a 2-col table where the right border has no clear edge).
            head_pitch_ok = slots > 0 or lines[0] <= max(10, int(pitch * 1.2))
            # Don't add a head border that would create a stub row much
            # smaller than the median pitch — that's almost always a
            # spurious gradient inside an actual cell (e.g. bottom edge
            # of a coloured header band).
            head_not_stub = lines[0] >= max(10, int(pitch * 0.5))
            need_head = (
                lines[0] > 5
                and head_pitch_ok
                and head_not_stub
                and _ink_density(0, lines[0], 0, W) > 0.02
            )
            tail = H - 1 - lines[-1]
            tail_pitch_ok = slots > 0 or tail <= max(10, int(pitch * 1.2))
            tail_not_stub = tail >= max(10, int(pitch * 0.5))
            need_tail = (
                tail > 5
                and tail_pitch_ok
                and tail_not_stub
                and _ink_density(lines[-1] + 1, H, 0, W) > 0.02
            )
            adds = (1 if need_head else 0) + (1 if need_tail else 0)
            if adds > slots:
                # Smaller gap is more likely a real missing border at the
                # edge; larger gap is more likely bbox slack. Drop the side
                # with the larger gap.
                if lines[0] <= tail:
                    need_tail = False
                else:
                    need_head = False
            if need_head:
                lines = [0] + lines
            if need_tail:
                lines = lines + [H - 1]
            return lines
        def _add_outer_v(lines: list[int], target: int) -> list[int]:
            if not lines:
                return lines
            if len(lines) >= 2:
                gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
                pitch = sorted(gaps)[len(gaps) // 2]
            else:
                pitch = W
            slots = target - len(lines)
            head_pitch_ok = slots > 0 or lines[0] <= max(10, int(pitch * 1.2))
            head_not_stub = lines[0] >= max(10, int(pitch * 0.5))
            need_head = (
                lines[0] > 5
                and head_pitch_ok
                and head_not_stub
                and _ink_density(0, H, 0, lines[0]) > 0.02
            )
            tail = W - 1 - lines[-1]
            tail_pitch_ok = slots > 0 or tail <= max(10, int(pitch * 1.2))
            tail_not_stub = tail >= max(10, int(pitch * 0.5))
            need_tail = (
                tail > 5
                and tail_pitch_ok
                and tail_not_stub
                and _ink_density(0, H, lines[-1] + 1, W) > 0.02
            )
            adds = (1 if need_head else 0) + (1 if need_tail else 0)
            if adds > slots:
                # Smaller gap is more likely a real missing border at the
                # edge; larger gap is more likely bbox slack. Drop the side
                # with the larger gap.
                if lines[0] <= tail:
                    need_tail = False
                else:
                    need_head = False
            if need_head:
                lines = [0] + lines
            if need_tail:
                lines = lines + [W - 1]
            return lines
        h_lines = _add_outer_h(h_lines, n_rows + 1)
        v_lines = _add_outer_v(v_lines, n_cols + 1)

        if len(h_lines) != n_rows + 1 or len(v_lines) != n_cols + 1:
            return None, None
        h_lines = [v + tight_offset_y for v in h_lines]
        v_lines = [v + tight_offset_x for v in v_lines]
        return v_lines, h_lines

    def _detect_cell_alignment(
        self,
        original_image: Image.Image,
        cx1: int,
        cy1: int,
        cx2: int,
        cy2: int,
    ) -> str:
        """Detect text alignment by analyzing horizontal ink position
        in the cell crop from the original page. Returns 'left', 'center',
        'right', or empty string when ambiguous.
        """
        try:
            import numpy as np
        except Exception:
            return ''
        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw < 20 or ch < 8:
            return ''
        # Inset enough to skip the cell border + a thick block divider
        # (sudoku-style 2–3 px rules). 6% / minimum 4 px handles cell
        # widths up to ~70 px.
        ins_x = max(4, int(cw * 0.06))
        ins_y = max(4, int(ch * 0.06))
        crop = original_image.crop(
            (cx1 + ins_x, cy1 + ins_y, cx2 - ins_x, cy2 - ins_y)
        ).convert('L')
        arr = np.array(crop)
        if arr.size == 0:
            return ''
        ink = arr < 128
        if not ink.any():
            return ''
        cols = np.where(ink.any(axis=0))[0]
        if len(cols) == 0:
            return ''
        w = arr.shape[1]
        ink_first = float(cols[0])
        ink_last = float(cols[-1])
        content_w = ink_last - ink_first + 1
        free_space = (w - content_w) / 2.0
        if free_space <= 4:
            # Content fills the cell — alignment cannot be inferred from
            # padding asymmetry. Caller falls back to a content heuristic.
            return ''
        content_center = (ink_first + ink_last) / 2.0
        cell_center = (w - 1) / 2.0
        # offset > 0 means content sits right of center; < 0 means left.
        # Normalize by free_space so the threshold is meaningful: a digit
        # centered in a 60-px cell has free_space ~24, so a ±5-px wobble
        # is ratio ~0.2 — still center.
        ratio = (content_center - cell_center) / free_space
        if abs(ratio) <= 0.25:
            return 'center'
        if ratio < 0:
            return 'left'
        return 'right'

    def _render_table(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        te: "TableElement",
        original_image: Image.Image | None = None,
        page_dpi: int = 300,
        _skip_rotation: bool = False,
    ) -> None:
        """Render an HTML table into the image at the given bbox."""

        x1, y1, x2, y2 = te.bbox
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)

        soup = BeautifulSoup(te.html_content, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return

        # Collect all cells; handle colspan/rowspan for grid sizing only.
        n_rows = len(rows)
        n_cols = max(
            sum(int(td.get("colspan", 1)) for td in row.find_all(["td", "th"]))
            for row in rows
        )
        if n_cols == 0:
            return

        # Rotation detection: wide content (n_cols ≫ n_rows) jammed into a
        # portrait bbox means the source table was rotated 90° CCW on the page
        # (common for landscape tables on portrait pages — especially CJK).
        # OCR captured the logical landscape order; we render landscape on a
        # sub-canvas, rotate CCW, then paste into the portrait bbox.
        if not _skip_rotation:
            logical_aspect = n_cols / max(1, n_rows)
            bbox_aspect = bbox_w / bbox_h
            if logical_aspect > 1.5 and bbox_aspect < 1.0:
                sub_w, sub_h = bbox_h, bbox_w
                sub_img = Image.new("RGB", (sub_w, sub_h), "white")
                sub_draw = ImageDraw.Draw(sub_img)
                sub_original: Image.Image | None = None
                if original_image is not None:
                    sub_original = original_image.crop((x1, y1, x2, y2)).rotate(-90, expand=True)
                sub_te = TableElement(html_content=te.html_content, bbox=(0, 0, sub_w, sub_h))
                self._render_table(
                    sub_img, sub_draw, sub_te, sub_original, page_dpi, _skip_rotation=True
                )
                img.paste(sub_img.rotate(90, expand=True), (x1, y1))
                return

        # Detect actual grid positions from the source crop so that column
        # widths and row heights match the page (header rows are typically
        # short, body rows tall, columns vary by content). Falls back to
        # uniform spacing if line detection can't recover n_rows+1 /
        # n_cols+1 dividers.
        grid_col_xs: list[int] | None = None
        grid_row_ys: list[int] | None = None
        # Visible-cell layout: when every row has the same colspan pattern
        # (and no rowspan), detected vertical edges in the source image
        # may match the visible-cell count rather than the logical n_cols
        # (a uniform colspan=2 in column 0 means there's no internal
        # divider between logical cols 0 and 1, so detection finds
        # n_visible_cols+1 lines instead of n_cols+1). When that case
        # holds we switch to visible-cell-index layout: each row's cells
        # land in their position within the row, not in their logical
        # col_cursor index.
        row_colspan_tuples = [
            tuple(int(c.get('colspan', 1)) for c in row.find_all(['td', 'th']))
            for row in rows
        ]
        has_rowspan = any(
            int(c.get('rowspan', 1)) > 1
            for row in rows for c in row.find_all(['td', 'th'])
        )
        uniform_colspan = (
            len(set(row_colspan_tuples)) == 1
            and not has_rowspan
            and len(row_colspan_tuples[0]) != n_cols
        )
        n_visible_cols = len(row_colspan_tuples[0]) if row_colspan_tuples else n_cols
        visible_col_xs: list[int] | None = None
        if original_image is not None:
            try:
                grid_col_xs, grid_row_ys = self._detect_table_grid(
                    original_image.crop((x1, y1, x2, y2)), n_rows, n_cols
                )
            except Exception:
                grid_col_xs = grid_row_ys = None
            if grid_col_xs is None and uniform_colspan:
                try:
                    vcx, vry = self._detect_table_grid(
                        original_image.crop((x1, y1, x2, y2)),
                        n_rows, n_visible_cols,
                    )
                    if vcx is not None and vry is not None:
                        visible_col_xs = vcx
                        grid_row_ys = vry
                except Exception:
                    visible_col_xs = None

        def _col_x(ci: int) -> int:
            if grid_col_xs is not None:
                return x1 + grid_col_xs[ci]
            return int(x1 + ci * (bbox_w / n_cols))

        def _row_y(ri: int) -> int:
            if grid_row_ys is not None:
                return y1 + grid_row_ys[ri]
            return int(y1 + ri * (bbox_h / n_rows))

        # Mean cell dims used only for sizing pads / fall-back fits where a
        # representative magnitude is enough.
        avg_cell_w = bbox_w / n_cols
        avg_cell_h = bbox_h / n_rows
        # Use the broader detector so non-Latin symbols (✓, →, …) also pick
        # up the CJK-comprehensive font stack instead of LiberationSerif
        # which lacks them and renders tofu boxes.
        cjk = bool(_EXT_GLYPH_RE.search(te.html_content))


        # Pixel-level baseline font size: measure ink-height in the original
        # table region and convert to px the same way text elements do.
        baseline_px = _MAX_FONT_SIZE
        if original_image is not None:
            try:
                import numpy as np
                from reference_free_ocr_metric.reconstruction.document_analyzer import (
                    estimate_font_size_pt,
                    measure_glyph_height,
                )
                crop = original_image.crop((x1, y1, x2, y2)).convert("L")
                gray = np.array(crop)
                glyph_h = measure_glyph_height(gray)
                if glyph_h > 0:
                    pt = estimate_font_size_pt(glyph_h, page_dpi)
                    baseline_px = max(_MIN_FONT_SIZE, min(_MAX_FONT_SIZE, int(pt * page_dpi / 72)))
            except Exception:
                pass

        # Draw outer border at the detected table edges so a loose OCR
        # bbox (slightly larger than the actual table) doesn't paint a
        # second rectangle outside the leftmost/rightmost cell borders.
        if visible_col_xs is not None:
            ox1 = x1 + visible_col_xs[0]
            ox2 = x1 + visible_col_xs[-1]
        else:
            ox1, ox2 = _col_x(0), _col_x(n_cols)
        oy1, oy2 = _row_y(0), _row_y(n_rows)
        draw.rectangle([ox1, oy1, ox2 - 1, oy2 - 1], outline=(0, 0, 0), width=1)

        # Track grid occupancy from earlier rows' rowspans so that, e.g., a
        # rowspan=4 cell in column 0 keeps subsequent rows' first cell from
        # being placed back into column 0.
        occupied: set[tuple[int, int]] = set()

        def _cell_text(cell) -> str:
            """Extract cell text, preserving <br/> as newlines and collapsing
            other whitespace within each line."""
            for br in cell.find_all("br"):
                br.replace_with("\n")
            raw = cell.get_text(separator=" ")
            lines = [" ".join(ln.split()) for ln in raw.split("\n")]
            return "\n".join(ln for ln in lines if ln)

        def _fit_size(text: str, max_w: int, max_h: int, hi_cap: int) -> int:
            """Largest font size where text (mixed text+formula, multi-line) fits.

            For LaTeX-bearing text, mirrors _draw_mixed_text's token-fill
            algorithm via _count_mixed_lines so the chosen font matches what
            actually renders. Pre-renders each unique formula once at a
            reference cap_h and scales pixel-widths proportionally during
            the binary search to avoid re-rendering matplotlib at each step.
            """
            has_latex = bool(_LATEX_RE.search(text))
            segments_n = text.split("\n")

            # Pre-render formulas once at reference cap_h tied to hi_cap so
            # the scale factor stays in [0,1] during search.
            ref_fs = max(_MIN_FONT_SIZE, hi_cap)
            ref_font = self._get_styled_font(ref_fs, "serif", False, False, cjk)
            try:
                ref_cap = ref_font.getbbox("H")[3]
            except Exception:
                ref_cap = int(ref_fs * 0.7)
            ref_fh = max(8, int(ref_cap * 0.75))
            fw_raw: dict[str, int] = {}
            if has_latex:
                for m in _LATEX_RE.finditer(text):
                    lat = m.group(0).strip("$")
                    if lat not in fw_raw:
                        fi = self._render_inline_formula(lat, ref_fh)
                        fw_raw[lat] = (fi.width + 2) if fi else max(6, int(ref_fs * max(1, len(lat) // 4) * 0.6))

            lo, hi, best = _MIN_FONT_SIZE, max(_MIN_FONT_SIZE, min(hi_cap, max_h)), _MIN_FONT_SIZE
            while lo <= hi:
                mid = (lo + hi) // 2
                f = self._get_styled_font(mid, "serif", False, False, cjk)
                if has_latex:
                    try:
                        cap = f.getbbox("H")[3]
                    except Exception:
                        cap = int(mid * 0.7)
                    fh = max(8, int(cap * 0.75))
                    scale = fh / max(1, ref_fh)
                    fw_scaled = {k: max(2, int(v * scale)) for k, v in fw_raw.items()}
                    total_lines = sum(
                        self._count_mixed_lines(seg, max_w, f, mid, fh, fw_scaled)
                        for seg in segments_n
                    )
                    try:
                        text_lh = int(f.getbbox(_LINE_HEIGHT_SAMPLE)[3]) + int(mid * 0.15)
                    except Exception:
                        text_lh = int(mid * 1.15)
                    lh = max(text_lh, fh + 2)
                else:
                    total_lines = sum(
                        len(self._wrap_text_chars(seg, max_w, f, mid))
                        for seg in segments_n
                    )
                    try:
                        lh = int(f.getbbox(_LINE_HEIGHT_SAMPLE)[3]) + int(mid * 0.15)
                    except Exception:
                        lh = int(mid * 1.15)
                if total_lines * lh <= max_h:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            return best

        # Pass 1: lay out the grid and gather cell rects + text. Two passes are
        # needed because the per-cell font size has to be capped at the smallest
        # font that any cell would have picked individually — otherwise a short
        # header cell renders huge while a long data cell underneath uses a
        # smaller font, breaking visual consistency the source page maintains.
        cell_rects: list[tuple[int, int, int, int, str]] = []
        for ri, row in enumerate(rows):
            cells = row.find_all(["td", "th"])
            col_cursor = 0
            for vci, cell in enumerate(cells):
                while (ri, col_cursor) in occupied and col_cursor < n_cols:
                    col_cursor += 1
                if col_cursor >= n_cols:
                    break
                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                for dr in range(rowspan):
                    for dc in range(colspan):
                        occupied.add((ri + dr, col_cursor + dc))
                if visible_col_xs is not None:
                    cx1 = x1 + visible_col_xs[vci]
                    cx2 = x1 + visible_col_xs[vci + 1]
                else:
                    cx1 = _col_x(col_cursor)
                    cx2 = _col_x(col_cursor + colspan)
                cy1 = _row_y(ri)
                cy2 = _row_y(ri + rowspan)
                draw.rectangle([cx1, cy1, cx2 - 1, cy2 - 1], outline=(0, 0, 0), width=1)
                text = _cell_text(cell)
                cell_rects.append((cx1, cy1, cx2, cy2, text))
                col_cursor += colspan

        # Pass 2a: compute the smallest fits-this-cell size across all cells,
        # capped by the pixel-derived baseline. Use that as the uniform size.
        global_fs = baseline_px
        for cx1, cy1, cx2, cy2, text in cell_rects:
            if not text:
                continue
            pad = max(2, int((cx2 - cx1) * 0.05))
            cw = max(1, cx2 - cx1 - 2 * pad)
            ch = max(1, cy2 - cy1 - 4)
            global_fs = min(global_fs, _fit_size(text, cw, ch, baseline_px))
        global_fs = max(_MIN_FONT_SIZE, global_fs)
        cell_font = self._get_styled_font(global_fs, "serif", False, False, cjk)
        try:
            line_h = int(cell_font.getbbox(_LINE_HEIGHT_SAMPLE)[3]) + int(global_fs * 0.15)
        except Exception:
            line_h = int(global_fs * 1.15)

        # Re-parse cell HTML so we can read style="text-align: …" attrs;
        # cell_rects only carries text. Build a parallel list of style hints.
        style_hints: list[str] = []
        occ2: set[tuple[int, int]] = set()
        for ri2, row in enumerate(rows):
            cc = 0
            for cell in row.find_all(["td", "th"]):
                while (ri2, cc) in occ2 and cc < n_cols:
                    cc += 1
                if cc >= n_cols:
                    break
                cspan = int(cell.get("colspan", 1))
                rspan = int(cell.get("rowspan", 1))
                for dr in range(rspan):
                    for dc in range(cspan):
                        occ2.add((ri2 + dr, cc + dc))
                style = (cell.get("style") or "").lower()
                hint = ""
                if "text-align:" in style:
                    if "center" in style:
                        hint = "center"
                    elif "right" in style:
                        hint = "right"
                    elif "left" in style:
                        hint = "left"
                style_hints.append(hint)
                cc += cspan

        for idx, (cx1, cy1, cx2, cy2, text) in enumerate(cell_rects):
            if not text:
                continue
            pad = max(2, int((cx2 - cx1) * 0.05))
            cw = max(1, cx2 - cx1 - 2 * pad)

            # Resolve alignment: HTML hint > content-based heuristic.
            # The image-based detector is unreliable here because OCR table
            # bboxes are loose (often include page-margin artifacts that
            # throw off uniform row division), so a content heuristic is
            # the most consistent fallback: short single-line cells are
            # almost always centered (table headers, value cells), while
            # multi-line or long cells are left-aligned body text.
            alignment = style_hints[idx] if idx < len(style_hints) and style_hints[idx] else ""
            grid_ok = grid_col_xs is not None or visible_col_xs is not None
            if not alignment and grid_ok and original_image is not None:
                alignment = self._detect_cell_alignment(
                    original_image, cx1, cy1, cx2, cy2
                )
            if not alignment:
                # row index from cy1 — find which row this cell starts in
                cell_ri = next(
                    (ri_ for ri_ in range(n_rows) if cy1 < _row_y(ri_ + 1)),
                    n_rows - 1,
                )
                if cell_ri == 0:
                    alignment = "left" if ("\n" in text or len(text) > 30) else "center"
                else:
                    alignment = "left"

            if _LATEX_RE.search(text):
                self._draw_mixed_text(
                    img, draw, text, cx1 + pad, cy1 + 2, cw, max(1, cy2 - cy1 - 4),
                    cell_font, global_fs, alignment=alignment,
                )
                continue

            lines: list[str] = []
            for seg in text.split("\n"):
                lines.extend(self._wrap_text_chars(seg, cw, cell_font, global_fs))
            if not lines:
                continue
            total_h = len(lines) * line_h
            ty = cy1 + max(0, (cy2 - cy1 - total_h) // 2)
            for line in lines:
                if ty + line_h > cy2:
                    break
                if cjk:
                    tw = self._measure_cjk_aware(line, cell_font, global_fs)
                else:
                    try:
                        tw = int(cell_font.getbbox(line)[2] - cell_font.getbbox(line)[0])
                    except Exception:
                        tw = 0
                if alignment == "right":
                    tx = cx2 - tw - pad
                elif alignment == "center":
                    tx = cx1 + max(pad, (cx2 - cx1 - tw) // 2)
                else:
                    tx = cx1 + pad
                if cjk:
                    self._draw_with_cjk_fallback(draw, (tx, ty), line, cell_font, global_fs)
                else:
                    draw.text((tx, ty), line, fill=(0, 0, 0), font=cell_font)
                ty += line_h

    def _render_inline_formula(self, latex: str, line_h: int) -> Image.Image | None:
        """Render a LaTeX expression as an RGBA image scaled to line_h pixels tall.

        Returns an RGBA image with white background converted to transparent so it
        composites naturally onto surrounding text.
        """
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        from matplotlib.mathtext import math_to_image

        if not latex.strip():
            return None
        try:
            import numpy as np

            buf = io.BytesIO()
            # Render at 4× target size; downsampling gives thinner, sharper strokes.
            render_size = max(12, line_h * 4)
            prop = fm.FontProperties(size=render_size, weight="normal")
            # Substitute commands unsupported by matplotlib mathtext.
            _INLINE_SUBS = [(r"\bmod", r"\operatorname{mod}"), (r"\pmod", r"\operatorname{mod}")]
            def _apply(f):
                for a, b in _INLINE_SUBS:
                    f = f.replace(a, b)
                return f
            raw: Image.Image | None = None
            ctx = _cjk_mathtext_rc() if _CJK_RE.search(latex) else matplotlib.rc_context()
            try:
                with ctx:
                    math_to_image(f"${latex}$", buf, dpi=150, format="png", prop=prop)
            except Exception:
                buf = io.BytesIO()
                try:
                    with (_cjk_mathtext_rc() if _CJK_RE.search(latex) else matplotlib.rc_context()):
                        math_to_image(f"${_apply(latex)}$", buf, dpi=150, format="png", prop=prop)
                except Exception:
                    buf = None
            if buf is not None:
                buf.seek(0)
                raw = Image.open(buf).convert("RGBA")
            else:
                # mathtext can't render (e.g. \square, \blacksquare). Fall back to
                # pdflatex for inline symbols. Block envs are handled by the caller.
                if "begin{" in latex:
                    return None
                raw_pdf = self._render_formula_pdflatex(latex)
                if raw_pdf is None:
                    return None
                raw = raw_pdf.convert("RGBA")

            # Trim whitespace so we scale the actual glyph content, not blank padding.
            arr = np.array(raw)
            is_ink = (arr[:, :, 0] < 240) | (arr[:, :, 1] < 240) | (arr[:, :, 2] < 240)
            rows = np.where(is_ink.any(axis=1))[0]
            cols = np.where(is_ink.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                return None
            raw = raw.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))

            # Scale trimmed glyph height to line_h.
            if raw.height > 0 and raw.height != line_h:
                scale = line_h / raw.height
                raw = raw.resize(
                    (max(1, int(raw.width * scale)), line_h), Image.Resampling.LANCZOS
                )
            # Make near-white pixels transparent so formula blends with text.
            data = np.array(raw)
            white = (data[:, :, 0] >= 240) & (data[:, :, 1] >= 240) & (data[:, :, 2] >= 240)
            data[:, :, 3] = np.where(white, 0, 255)
            return Image.fromarray(data, "RGBA")
        except Exception:
            return None

    def _draw_mixed_text(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        max_width: int,
        max_height: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        font_size: int,
        alignment: str = "left",
        indent_px: int = 0,
        font_category: str = "serif",
        is_italic: bool = False,
    ) -> None:
        """Draw text that may contain inline $formula$ and **bold** expressions."""
        segments = _split_inline_content(text)
        if not any(seg_type in ("formula", "bold", "italic") for seg_type, _ in segments):
            clean = _LATEX_RE.sub("", text).replace("**", "").replace("*", "").strip()
            if clean:
                self._draw_wrapped_text(draw, clean, x, y, max_width, max_height, font, font_size)
            return
        cjk = bool(_CJK_RE.search(text))
        bold_font = self._get_styled_font(font_size, font_category, True, is_italic, cjk)
        italic_font = self._get_styled_font(font_size, font_category, False, True, cjk)

        try:
            line_height = int(font.getbbox(_LINE_HEIGHT_SAMPLE)[3]) + int(font_size * 0.15)
        except Exception:
            line_height = int(font_size * 1.15)

        # Formula height = cap-height * 1.25 (room for descenders in $g$, $p$, etc.)
        if isinstance(font, ImageFont.FreeTypeFont):
            try:
                cap_h = font.getbbox("H")[3]
            except Exception:
                cap_h = int(font_size * 0.7)
        else:
            cap_h = int(font_size * 0.7)
        formula_h: int = max(8, int(cap_h * 0.75))

        space_w = self._text_width(" ", font, font_size)

        # Build token stream: (type, payload, width)
        # type is "txt" | "sp" | "img"; payload is str or Image.Image
        tokens: list[tuple[str, str | Image.Image, int]] = []
        for seg_type, content in segments:
            if seg_type == "formula":
                fi = self._render_inline_formula(content, int(formula_h))
                if fi:
                    tokens.append(("img", fi, fi.width + 2))
                else:
                    # Fallback for block environments (e.g. bmatrix): try pdflatex first.
                    pdflatex_tried = 'begin{' in content
                    fi_big = self._render_formula_pdflatex(content) if pdflatex_tried else None
                    if fi_big is not None:
                        import numpy as np
                        # Use full height unless formula will wrap to a second line.
                        # Simulate line-wrap to get current x-position and line depth.
                        sim_x = 0
                        sim_line = 0
                        for _tt, _tp, _tw in tokens:
                            if sim_x + _tw <= max_width - 4:
                                sim_x += _tw
                            elif _tt != "sp":
                                sim_x = _tw
                                sim_line += 1
                            else:
                                sim_x = 0
                                sim_line += 1
                        avail_h_max = max(line_height, max_height - sim_line * line_height - 4)
                        formula_w_at_max = int(fi_big.width * avail_h_max / fi_big.height)
                        if sim_x + formula_w_at_max + 2 > max_width:
                            # Formula wraps to the next line; use its full available height.
                            avail_h = max(line_height, max_height - (sim_line + 1) * line_height)
                        else:
                            avail_h = avail_h_max
                        scale = min(1.0, avail_h / fi_big.height, (max_width - 4) / fi_big.width)
                        new_h = max(1, int(fi_big.height * scale))
                        new_w = max(1, int(fi_big.width * scale))
                        fi_big = fi_big.resize((new_w, new_h), Image.Resampling.LANCZOS).convert('RGBA')
                        arr = np.array(fi_big)
                        white = (arr[:, :, 0] >= 240) & (arr[:, :, 1] >= 240) & (arr[:, :, 2] >= 240)
                        arr[:, :, 3] = np.where(white, 0, 255)
                        fi_big = Image.fromarray(arr, 'RGBA')
                        tokens.append(("img", fi_big, fi_big.width + 2))
                    else:
                        # Plain-text fallback: strip LaTeX commands, show numbers.
                        import re as _re
                        fb = _re.sub(r'\\(begin|end)\{[^}]*\}', '', content)
                        fb = fb.replace('\\', ' ').replace('&', ' ')
                        fb = ' '.join(fb.split())
                        for _fi, _fw in enumerate(fb.split()):
                            if _fi > 0:
                                tokens.append(("sp", " ", space_w))
                            tokens.append(("txt", _fw, self._text_width(_fw, font, font_size)))
            elif seg_type == "bold":
                for sub_type, sub_content in _split_by_formula(content):
                    if sub_type == "formula":
                        fi = self._render_inline_formula(sub_content, int(formula_h))
                        if fi:
                            tokens.append(("img", fi, fi.width + 2))
                        else:
                            tokens.append(("btxt", sub_content, self._text_width(sub_content, bold_font, font_size)))
                    else:
                        words = sub_content.split(" ")
                        for i, word in enumerate(words):
                            if i > 0:
                                tokens.append(("sp", " ", space_w))
                            if word:
                                tokens.append(("btxt", word, self._text_width(word, bold_font, font_size)))
            elif seg_type == "italic":
                for sub_type, sub_content in _split_by_formula(content):
                    if sub_type == "formula":
                        fi = self._render_inline_formula(sub_content, int(formula_h))
                        if fi:
                            tokens.append(("img", fi, fi.width + 2))
                        else:
                            tokens.append(("itxt", sub_content, self._text_width(sub_content, italic_font, font_size)))
                    else:
                        words = sub_content.split(" ")
                        for i, word in enumerate(words):
                            if i > 0:
                                tokens.append(("sp", " ", space_w))
                            if word:
                                tokens.append(("itxt", word, self._text_width(word, italic_font, font_size)))
            else:
                words = content.split(" ")
                for i, word in enumerate(words):
                    if i > 0:
                        tokens.append(("sp", " ", space_w))
                    if word:
                        tokens.append(("txt", word, self._text_width(word, font, font_size)))

        # Group tokens into wrapped lines; store (type, payload, x_off, tok_w).
        lines: list[list[tuple[str, str | Image.Image, int, int]]] = []
        cur: list[tuple[str, str | Image.Image, int, int]] = []
        cur_w = 0

        for tok_type, payload, tok_w in tokens:
            if tok_type == "sp" and not cur:
                continue  # skip leading space on a new line
            if cur_w + tok_w <= max_width - 4:
                cur.append((tok_type, payload, cur_w, tok_w))
                cur_w += tok_w
            else:
                if cur:
                    lines.append(cur)
                if tok_type == "sp":
                    cur, cur_w = [], 0
                else:
                    cur, cur_w = [(tok_type, payload, 0, tok_w)], tok_w
        if cur:
            lines.append(cur)

        # Render each line.
        line_y = y + 1
        last_idx = len(lines) - 1
        for i, line in enumerate(lines):
            if line_y > y + max_height:
                break
            first_line_offset = indent_px if i == 0 else 0

            if alignment == "justified" and i < last_idx:
                # Expand inter-token gaps so the line fills max_width.
                content_w = sum(w for _, _, _, w in line if _ != "sp")
                n_gaps = sum(1 for t, _, _, _ in line if t == "sp")
                if n_gaps > 0:
                    available = max_width - 2 - first_line_offset
                    gap_w = max(float(space_w), (available - content_w) / n_gaps)
                else:
                    gap_w = float(space_w)
                # Recompute x offsets with expanded gap.
                cur_x = float(first_line_offset)
                for tok_type, payload, _, tok_w in line:
                    if tok_type == "sp":
                        cur_x += gap_w
                    else:
                        if isinstance(payload, Image.Image):
                            vert = max(0, (line_height - payload.height) // 2)
                            img.paste(payload, (x + 2 + int(cur_x), line_y + vert), payload)
                        elif isinstance(payload, str) and payload:
                            draw_font = bold_font if tok_type == "btxt" else italic_font if tok_type == "itxt" else font
                            draw.text((x + 2 + int(cur_x), line_y), payload, fill="black", font=draw_font)
                        cur_x += tok_w
            else:
                line_w = sum(w for _, _, _, w in line)
                if alignment == "center":
                    line_x_off = max(0, (max_width - line_w) // 2)
                elif alignment == "right":
                    line_x_off = max(0, max_width - line_w - 2)
                else:
                    line_x_off = first_line_offset
                for tok_type, payload, x_off, _ in line:
                    abs_x = x + 2 + line_x_off + x_off
                    if isinstance(payload, Image.Image):
                        vert = max(0, (line_height - payload.height) // 2)
                        img.paste(payload, (abs_x, line_y + vert), payload)
                    elif isinstance(payload, str) and payload:
                        draw_font = bold_font if tok_type == "btxt" else italic_font if tok_type == "itxt" else font
                        draw.text((abs_x, line_y), payload, fill="black", font=draw_font)
            line_y += line_height

    def _render_formula_patch(
        self, latex: str, target_w: int, target_h: int, dpi: int = 150
    ) -> Image.Image | None:
        """Render a LaTeX formula via matplotlib mathtext and return an RGB image
        sized to fit within (target_w, target_h). Returns None on failure."""
        import io

        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.mathtext import math_to_image

        # Detect leading equation number "(1) ", "(12) ", etc.
        import re as _re
        eqnum_match = _re.match(r"^\((\d+)\)\s*", latex.strip())
        eqnum_text = eqnum_match.group(0).rstrip() if eqnum_match else ""

        # Extract LaTeX from $...$ or $$...$$ delimiters if present.
        matches = _LATEX_RE.findall(latex)
        if matches:
            # Use the largest match (most likely the main formula).
            formula = max(matches, key=len).strip("$")
        else:
            formula = latex.strip().strip("$")
        if not formula:
            # Regex found only degenerate matches (e.g. bare "$$" with truncated content).
            # Fall back to raw latex with delimiters stripped.
            formula = latex.strip().strip("$")
        if not formula:
            return None

        # Strip HTML tag artifacts from truncated OCR output.  When the VLM's
        # LaTeX contains bare < > (e.g. "i<j \leq n}"), a browser or parser may
        # mangle the content into HTML tags like "<j div="" n=""></j>".  Remove
        # these before rendering.  We only strip tags that start with a word char
        # so standalone comparison operators like " x<y " are left intact.
        import re as _re
        formula = _re.sub(r'</?\w[^>]*>', '', formula)
        # Strip \tag{} and \label{}: unsupported by mathtext and cause
        # pdflatex to produce a full-page-width image (tag at right margin),
        # making the scaling factor width-limited and the formula too small.
        formula = _re.sub(r'\\tag\*?\{[^}]*\}', '', formula).strip()
        formula = _re.sub(r'\\label\{[^}]*\}', '', formula).strip()
        if not formula:
            return None

        # Substitute mathtext-unsupported commands with equivalents.
        _SUBS = [
            (r'\bmod', r'\mathrm{mod}'),
            (r'\pmod', r'\mathrm{mod}'),
        ]

        # STIX font set has all glyphs needed for LaTeX-style math.
        _has_cjk = bool(_CJK_RE.search(formula))
        def _try_render(f: str) -> Image.Image | None:
            try:
                buf = io.BytesIO()
                if _has_cjk or _CJK_RE.search(f):
                    ctx = _cjk_mathtext_rc()
                else:
                    ctx = matplotlib.rc_context({"mathtext.fontset": "stix"})
                with ctx:
                    math_to_image(f"${f}$", buf, dpi=dpi, format="png")
                buf.seek(0)
                return Image.open(buf).convert("RGBA")
            except Exception:
                return None

        def _apply_subs(f: str) -> str:
            for old, new in _SUBS:
                f = f.replace(old, new)
            return f

        import re as _re
        aligned_match = _re.search(r'\\begin\{align(?:ed)?\*?\}(.*?)\\end\{align(?:ed)?\*?\}', formula, _re.DOTALL)
        if aligned_match:
            # pdflatex handles \begin{aligned} correctly and preserves any
            # prefix content (e.g. "echelon form: [matrix]") that row-by-row
            # rendering would discard. Try it first.
            formula_img: Image.Image | None = self._render_formula_pdflatex(formula)
            if formula_img is None:
                # pdflatex failed: fall back to row-by-row matplotlib rendering.
                rows_raw = aligned_match.group(1).split(r'\\')
                row_imgs: list[Image.Image] = []
                for row in rows_raw:
                    row_f = _re.sub(r'&', ' ', row).strip().strip('\\').strip()
                    if not row_f:
                        continue
                    row_f = _apply_subs(row_f)
                    ri = _try_render(row_f)
                    if ri is not None:
                        row_imgs.append(ri)
                if row_imgs:
                    total_h = sum(r.height for r in row_imgs) + 4 * (len(row_imgs) - 1)
                    max_w = max(r.width for r in row_imgs)
                    stacked = Image.new("RGBA", (max_w, total_h), (255, 255, 255, 255))
                    cy = 0
                    for ri in row_imgs:
                        stacked.paste(ri, ((max_w - ri.width) // 2, cy))
                        cy += ri.height + 4
                    formula_img = stacked
        else:
            formula_img = _try_render(formula) or _try_render(_apply_subs(formula))
            if formula_img is None:
                # Truncated HTML may produce unclosed { braces. Close them and retry.
                depth = sum(1 if c == '{' else -1 if c == '}' else 0 for c in formula)
                if depth > 0:
                    closed = formula + '}' * depth
                    formula_img = _try_render(closed) or _try_render(_apply_subs(closed))

        if formula_img is None:
            formula_img = self._render_formula_pdflatex(formula)
            if formula_img is None:
                return None
            canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            if formula_img.width > 0 and formula_img.height > 0:
                scale = min(target_w / formula_img.width, target_h * 0.85 / formula_img.height)
                nw = max(1, int(formula_img.width * scale))
                nh = max(1, int(formula_img.height * scale))
                formula_img = formula_img.resize((nw, nh), Image.Resampling.LANCZOS)
            xo = 0
            yo = max(0, (target_h - formula_img.height) // 2)
            canvas.paste(formula_img.convert("RGB"), (xo, yo))
            return canvas

        # Scale to fill bbox (both up and down), preserving aspect ratio.
        if formula_img.width > 0 and formula_img.height > 0:
            scale = min(target_w / formula_img.width, target_h * 0.85 / formula_img.height)
            new_w = max(1, int(formula_img.width * scale))
            new_h = max(1, int(formula_img.height * scale))
            formula_img = formula_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        if eqnum_text:
            # Draw equation number at left; center formula in remaining space.
            font = self._get_font(max(_MIN_FONT_SIZE, int(target_h * 0.45)))
            try:
                num_w = int(font.getbbox(eqnum_text)[2] - font.getbbox(eqnum_text)[0]) + 6
            except Exception:
                num_w = int(len(eqnum_text) * target_h * 0.6)
            num_w = min(num_w, target_w // 3)
            draw.text((2, 1), eqnum_text, fill="black", font=font)
            remaining = target_w - num_w
            x_off = num_w + max(0, (remaining - formula_img.width) // 2)
        else:
            x_off = 0

        y_off = max(0, (target_h - formula_img.height) // 2)
        canvas.paste(formula_img.convert("RGB"), (x_off, y_off))
        return canvas

    def _render_formula_pdflatex(self, formula: str) -> Image.Image | None:
        """Render formula via pdflatex + pdftoppm. Handles array/matrix environments
        that matplotlib mathtext cannot parse."""
        import subprocess
        import tempfile

        _TEX = r"""\documentclass[12pt]{article}
\usepackage[margin=0.5cm,paperwidth=30cm,paperheight=20cm]{geometry}
\pagestyle{empty}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tex_path = f"{tmp}/formula.tex"
                with open(tex_path, "w") as f:
                    f.write(_TEX % formula)
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", f"-output-directory={tmp}", tex_path],
                    capture_output=True, timeout=15,
                )
                pdf_path = f"{tmp}/formula.pdf"
                import os
                if not os.path.exists(pdf_path):
                    return None
                subprocess.run(
                    ["pdftoppm", "-r", "200", "-png", pdf_path, f"{tmp}/out"],
                    capture_output=True, timeout=15,
                )
                from pathlib import Path
                pngs = sorted(Path(tmp).glob("out-*.png"))
                if not pngs:
                    return None
                img = Image.open(pngs[0]).convert("RGB")
                # Trim whitespace
                import numpy as np
                arr = np.array(img)
                mask = (arr < 240).any(axis=2)
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                if len(rows) == 0 or len(cols) == 0:
                    return None
                return img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))
        except Exception:
            return None

    def _estimate_font_size(self, text: str, bbox_w: int, bbox_h: int) -> int:
        """Estimate font size to fit text within bbox.

        Uses binary search for short text (<=30 chars) and a greedy
        top-down search for longer text. Font size is capped at
        _MAX_FONT_SIZE to avoid absurdly large text in tall boxes.
        """
        total_chars = len(text.replace("\n", ""))
        explicit_lines = text.split("\n")
        num_lines = len(explicit_lines)

        lo = _MIN_FONT_SIZE
        hi = min(_MAX_FONT_SIZE, bbox_h)

        if total_chars <= 30 and num_lines <= 2:
            # Binary search: find largest font where text fits.
            best = lo
            while lo <= hi:
                mid = (lo + hi) // 2
                font = self._get_font(mid)
                try:
                    longest = max(explicit_lines, key=len)
                    text_w = font.getbbox(longest)[2] - font.getbbox(longest)[0]
                    line_h = font.getbbox(_LINE_HEIGHT_SAMPLE)[3]
                    total_h = line_h * num_lines
                except Exception:
                    lo = mid + 1
                    continue

                if text_w <= bbox_w - 4 and total_h <= bbox_h:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            return max(_MIN_FONT_SIZE, best)

        # Greedy top-down: first size where wrapped text fits vertically.
        for size in range(hi, _MIN_FONT_SIZE - 1, -1):
            font = self._get_font(size)
            try:
                # CJK characters are roughly square.
                avg_char_w = max(1, size)
                chars_per_line = max(1, bbox_w // avg_char_w)
                lines_needed = max(
                    1, (total_chars + chars_per_line - 1) // chars_per_line
                )
                line_h = font.getbbox(_LINE_HEIGHT_SAMPLE)[3] + int(size * 0.15)
                if lines_needed * line_h <= bbox_h:
                    return size
            except Exception:
                continue

        return _MIN_FONT_SIZE

    def _draw_wrapped_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        max_width: int,
        max_height: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        font_size: int,
    ) -> None:
        """Draw character-wrapped text within a bounding box."""
        lines = self._wrap_text_chars(text, max_width, font, font_size)

        try:
            line_height = font.getbbox(_LINE_HEIGHT_SAMPLE)[3] + int(
                font_size * 0.15
            )
        except Exception:
            line_height = int(font_size * 1.15)

        line_y = y + 1
        for i, line in enumerate(lines):
            # Always render at least the first line.
            if i > 0 and line_y >= y + max_height:
                break
            draw.text((x + 2, line_y), line, fill="black", font=font)
            line_y += line_height

    def _draw_aligned_lines(
        self,
        draw: ImageDraw.ImageDraw,
        lines: list[str],
        x: int,
        y: int,
        max_width: int,
        max_height: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        line_height: int,
        alignment: str,
        indent_px: int = 0,
    ) -> None:
        """Draw text lines with alignment, justified spacing, and first-line indent."""
        last_idx = len(lines) - 1
        line_y = y + 1
        for i, line in enumerate(lines):
            if i > 0 and line_y >= y + max_height:
                break

            first_line_offset = indent_px if i == 0 else 0

            if alignment == "justified" and i < last_idx:
                # Expand word spacing to fill the full width (except last line).
                words = line.split()
                if len(words) > 1:
                    word_widths: list[int] = []
                    for w in words:
                        try:
                            word_widths.append(int(font.getbbox(w)[2] - font.getbbox(w)[0]))
                        except Exception:
                            word_widths.append(int(len(w) * line_height * 0.55))
                    total_word_w = sum(word_widths)
                    available = max_width - 4 - first_line_offset
                    gap_w = max(0.0, (available - total_word_w) / (len(words) - 1))
                    cur_x = float(x + 2 + first_line_offset)
                    for word, ww in zip(words, word_widths):
                        draw.text((int(cur_x), line_y), word, fill="black", font=font)
                        cur_x += ww + gap_w
                else:
                    draw.text((x + 2 + first_line_offset, line_y), line, fill="black", font=font)
            else:
                try:
                    text_w = font.getbbox(line)[2] - font.getbbox(line)[0]
                except Exception:
                    text_w = 0
                if alignment == "center":
                    line_x = x + (max_width - text_w) // 2
                elif alignment == "right":
                    line_x = x + max_width - text_w - 2
                else:
                    line_x = x + 2 + first_line_offset
                draw.text((line_x, line_y), line, fill="black", font=font)

            line_y += line_height

    def _count_mixed_lines(
        self,
        text: str,
        max_width: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        font_size: int,
        formula_h: int,
        fw_cache: dict[str, int] | None = None,
    ) -> int:
        """Count how many lines _draw_mixed_text would produce.

        Replicates the exact greedy token-fill algorithm used in rendering,
        so the shrink loop and renderer agree on line count.  formula_h is
        the same value computed in _draw_mixed_text (cap_h * 0.75).
        fw_cache optionally provides pre-computed formula pixel widths keyed
        by latex string, avoiding repeated renders during the shrink loop.
        """
        segments = _split_inline_content(text)
        space_w = self._text_width(" ", font, font_size)
        # Build flat list of (pixel_width, is_space)
        tok_w: list[int] = []
        tok_sp: list[bool] = []

        for seg_type, content in segments:
            if seg_type == "formula":
                if fw_cache is not None and content in fw_cache:
                    w = fw_cache[content]
                else:
                    fi = self._render_inline_formula(content, formula_h)
                    w = (fi.width + 2) if fi else max(6, int(font_size * max(1, len(content) // 4) * 0.6))
                    if fw_cache is not None:
                        fw_cache[content] = w
                tok_w.append(w)
                tok_sp.append(False)
            elif seg_type in ("bold", "italic"):
                for sub_t, sub_c in _split_by_formula(content):
                    if sub_t == "formula":
                        if fw_cache is not None and sub_c in fw_cache:
                            w = fw_cache[sub_c]
                        else:
                            fi = self._render_inline_formula(sub_c, formula_h)
                            w = (fi.width + 2) if fi else max(6, int(font_size * max(1, len(sub_c) // 4) * 0.6))
                            if fw_cache is not None:
                                fw_cache[sub_c] = w
                        tok_w.append(w)
                        tok_sp.append(False)
                    else:
                        for i, word in enumerate(sub_c.split(" ")):
                            if i > 0:
                                tok_w.append(space_w)
                                tok_sp.append(True)
                            if word:
                                tok_w.append(self._text_width(word, font, font_size))
                                tok_sp.append(False)
            else:
                for i, word in enumerate(content.split(" ")):
                    if i > 0:
                        tok_w.append(space_w)
                        tok_sp.append(True)
                    if word:
                        tok_w.append(self._text_width(word, font, font_size))
                        tok_sp.append(False)

        # Greedy line fill — mirrors _draw_mixed_text token loop exactly
        n_lines = 1
        cur_w = 0
        has_content = False
        for w, is_sp in zip(tok_w, tok_sp):
            if is_sp and not has_content:
                continue
            if cur_w + w <= max_width - 4:
                cur_w += w
                if not is_sp:
                    has_content = True
            else:
                if has_content:
                    n_lines += 1
                if is_sp:
                    cur_w, has_content = 0, False
                else:
                    cur_w, has_content = w, True
        return n_lines


    @staticmethod
    def _text_width(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, font_size: int) -> int:
        try:
            bb = font.getbbox(text)
            return int(bb[2] - bb[0])
        except Exception:
            return int(len(text) * font_size * 0.6)

    @staticmethod
    def _wrap_text_chars(
        text: str,
        max_width: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        font_size: int,
    ) -> list[str]:
        """Word-aware wrapping for Latin text; character-level fallback for CJK."""
        lines: list[str] = []

        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            # If it looks like CJK (no spaces / single chars), wrap by char.
            if len(words) <= 1 and len(paragraph) > 1:
                current = ""
                for char in paragraph:
                    test = current + char
                    if ImageRenderer._text_width(test, font, font_size) > max_width - 6 and current:
                        lines.append(current)
                        current = char
                    else:
                        current = test
                lines.append(current)
                continue

            # Word-level wrapping for Latin / mixed text.
            current_line = ""
            for word in words:
                candidate = (current_line + " " + word).lstrip() if current_line else word
                if ImageRenderer._text_width(candidate, font, font_size) <= max_width - 6:
                    current_line = candidate
                else:
                    if current_line:
                        lines.append(current_line)
                    # If single word is too wide, break it by character.
                    if ImageRenderer._text_width(word, font, font_size) > max_width - 6:
                        current = ""
                        for char in word:
                            test = current + char
                            if ImageRenderer._text_width(test, font, font_size) > max_width - 6 and current:
                                lines.append(current)
                                current = char
                            else:
                                current = test
                        current_line = current
                    else:
                        current_line = word
            if current_line:
                lines.append(current_line)

        return lines if lines else [""]

    def render_plain_text(
        self,
        text: str,
        width: int,
        height: int,
        background: str = "white",
        font_size: int = 16,
        margin: int = 20,
        canvas: Image.Image | None = None,
    ) -> Image.Image:
        """Render plain text in a simple top-to-bottom layout (fallback when no bboxes)."""
        if canvas is not None:
            img = canvas.copy().convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height))
        else:
            img = Image.new("RGB", (width, height), background)
        if not text.strip():
            return img

        draw = ImageDraw.Draw(img)
        font = self._get_font(font_size)
        y = margin
        max_w = width - 2 * margin

        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                y += font_size
                continue

            wrapped = self._wrap_text_chars(paragraph, max_w, font, font_size)
            try:
                line_height = font.getbbox(_LINE_HEIGHT_SAMPLE)[3] + int(
                    font_size * 0.15
                )
            except Exception:
                line_height = int(font_size * 1.15)

            for line in wrapped:
                if y > height - margin:
                    return img
                draw.text((margin, y), line, fill="black", font=font)
                y += line_height

        return img

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Return a cached font at the given size."""
        size = max(_MIN_FONT_SIZE, min(_MAX_FONT_SIZE, size))
        if size not in self._font_cache:
            try:
                if _RESOLVED_FONT_PATH:
                    self._font_cache[size] = ImageFont.truetype(
                        _RESOLVED_FONT_PATH, size
                    )
                else:
                    self._font_cache[size] = ImageFont.load_default()
            except (OSError, IOError):
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _get_styled_font(
        self, size: int, category: str, bold: bool, italic: bool = False, cjk: bool = False
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Return a cached font matching the requested style.

        When cjk=True, CJK-capable fonts are tried first so ideographs render
        correctly instead of appearing as empty boxes.
        """
        size = max(_MIN_FONT_SIZE, min(_MAX_FONT_SIZE, size))
        key = (size, category, bold, italic, cjk)
        if key not in self._styled_font_cache:
            styled_paths = (
                _FONT_MAP.get((category, bold, italic))
                or _FONT_MAP.get((category, bold, False))
                or _FONT_MAP[("sans-serif", False, False)]
            )
            # For CJK text, prepend CJK-capable fonts so they win the search.
            paths = (_CJK_FONT_PATHS + styled_paths) if cjk else styled_paths
            font = None
            for path in paths:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, size)
                        break
                    except (OSError, IOError):
                        continue
            if font is None:
                font = self._get_font(size)  # fallback to default
            self._styled_font_cache[key] = font
        return self._styled_font_cache[key]

    def _get_cjk_fallback_fonts(self, size: int) -> list[ImageFont.FreeTypeFont]:
        """Return all available CJK fonts at the given size, in priority order.
        Used by _draw_with_cjk_fallback for per-character glyph routing."""
        size = max(_MIN_FONT_SIZE, min(_MAX_FONT_SIZE, size))
        key = ("__cjk_list__", size)
        if key not in self._styled_font_cache:
            fonts: list[ImageFont.FreeTypeFont] = []
            for path in _CJK_FONT_PATHS:
                if os.path.exists(path):
                    try:
                        fonts.append(ImageFont.truetype(path, size))
                    except (OSError, IOError):
                        continue
            self._styled_font_cache[key] = fonts  # type: ignore[assignment]
        return self._styled_font_cache[key]  # type: ignore[return-value]

    def _font_has_glyph(self, font: ImageFont.FreeTypeFont, ch: str) -> bool:
        """Return True if `font` has a real glyph for `ch` (not a tofu .notdef).

        FreeType returns the .notdef glyph for missing chars, which PIL renders
        as a visible box. We detect that by comparing the rendered mask bytes
        for `ch` against the rendered mask for a known-missing PUA char in the
        same font; identical bytes mean both fell back to .notdef.
        """
        cache = getattr(self, "_glyph_cache", None)
        if cache is None:
            cache = {}
            self._glyph_cache = cache  # type: ignore[attr-defined]
        font_id = id(font)
        key = (font_id, ch)
        if key in cache:
            return cache[key]
        try:
            tofu_key = (font_id, "__tofu__")
            if tofu_key not in cache:
                cache[tofu_key] = bytes(font.getmask(""))  # type: ignore[assignment]
            tofu_bytes = cache[tofu_key]
            mask_bytes = bytes(font.getmask(ch))
            result = mask_bytes != tofu_bytes
        except Exception:
            result = True
        cache[key] = result
        return result

    def _pick_glyph_font(
        self, ch: str, primary: ImageFont.FreeTypeFont, fallbacks: list[ImageFont.FreeTypeFont]
    ) -> ImageFont.FreeTypeFont:
        """Return primary if it covers `ch`, else first fallback that does."""
        if self._font_has_glyph(primary, ch):
            return primary
        for fb in fallbacks:
            if fb is primary:
                continue
            if self._font_has_glyph(fb, ch):
                return fb
        return primary

    def _draw_with_cjk_fallback(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: str,
        primary_font: ImageFont.FreeTypeFont,
        size: int,
        fill: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Draw `text` char-by-char, switching to a covering CJK font for any
        char that primary_font lacks. ASCII chars stay on primary so the look
        is unchanged for non-CJK text."""
        if not isinstance(primary_font, ImageFont.FreeTypeFont):
            draw.text(xy, text, fill=fill, font=primary_font)
            return
        fallbacks = self._get_cjk_fallback_fonts(size)
        if not fallbacks:
            draw.text(xy, text, fill=fill, font=primary_font)
            return
        x, y = xy
        for ch in text:
            f = primary_font if ord(ch) < 0x80 else self._pick_glyph_font(ch, primary_font, fallbacks)
            draw.text((x, y), ch, fill=fill, font=f)
            try:
                x += int(f.getlength(ch))
            except Exception:
                try:
                    bb = f.getbbox(ch)
                    x += int(bb[2] - bb[0])
                except Exception:
                    x += size

    def _measure_cjk_aware(
        self, text: str, primary_font: ImageFont.FreeTypeFont, size: int
    ) -> int:
        """Total pixel width of `text` rendered with per-char CJK fallback."""
        if not isinstance(primary_font, ImageFont.FreeTypeFont):
            try:
                bb = primary_font.getbbox(text)
                return int(bb[2] - bb[0])
            except Exception:
                return len(text) * size
        fallbacks = self._get_cjk_fallback_fonts(size)
        if not fallbacks:
            try:
                return int(primary_font.getlength(text))
            except Exception:
                bb = primary_font.getbbox(text)
                return int(bb[2] - bb[0])
        total = 0
        for ch in text:
            f = primary_font if ord(ch) < 0x80 else self._pick_glyph_font(ch, primary_font, fallbacks)
            try:
                total += int(f.getlength(ch))
            except Exception:
                bb = f.getbbox(ch)
                total += int(bb[2] - bb[0])
        return total
