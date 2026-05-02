# Document Image Reconstruction Techniques

## Formula Rendering Options

### 1. Matplotlib mathtext (no TeX installation required)

Matplotlib ships a built-in TeX-subset parser called **mathtext** that renders math expressions to raster or vector images with no external dependency. The highest-level API is:

```python
import io
import matplotlib.mathtext as mathtext
from PIL import Image

buf = io.BytesIO()
mathtext.math_to_image(r'$\frac{-b \pm \sqrt{b^2-4ac}}{2a}$', buf, dpi=150, format='png')
buf.seek(0)
formula_img = Image.open(buf).convert("RGBA")
```

`math_to_image` clips the output tightly to the rendered formula bounding box, making it easy to paste into a reconstructed page at a known (x, y) position. For more control (e.g. adjusting the DPI to match the page resolution), use the lower-level approach with `matplotlib.figure.Figure`:

```python
import io, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def render_formula(latex_str: str, font_size: float, dpi: int = 150) -> Image.Image:
    fig = plt.figure(figsize=(0.01, 0.01))
    text = fig.text(0, 0, f"${latex_str}$", fontsize=font_size)
    fig.savefig(io.BytesIO(), dpi=dpi)          # force layout pass
    bbox = text.get_window_extent()
    w = bbox.width / dpi + 0.05
    h = bbox.height / dpi + 0.05
    fig.set_size_inches(w, h)
    text.set_position((0, -(bbox.ymin / dpi) / h))
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, transparent=True, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")
```

**Font size for formula matching**: the formula's `font_size` argument should be calibrated to the surrounding text. A rule of thumb is `font_size_pt = bbox_height_px * 72 / page_dpi`, matching the point size to the bounding box height observed in the document.

**Mathtext limitations**: supports a broad but not complete subset of LaTeX (no `\begin{align}`, limited macro expansion). Complex display formulas from academic PDFs may fail silently — falling back to a placeholder rectangle is safer than crashing.

### 2. External LaTeX pipeline (full fidelity)

For complete LaTeX support: compile a minimal `.tex` document with `pdflatex` or `xelatex`, then convert the output PDF crop to a PNG with Ghostscript or `pdf2image`. This handles any LaTeX formula but requires a TeX installation and is slow (~1 s per formula). Useful for offline batch preprocessing, not suitable for interactive rendering.

```bash
# Minimal wrapper
pdflatex -halt-on-error formula.tex && \
gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r150 -sOutputFile=formula.png formula.pdf
```

### 3. SymPy `preview()`

`sympy.preview(expr, viewer='file', filename='formula.png', dvipng=True)` generates images via `dvipng`; requires a LaTeX + dvipng install but produces high-quality output matching TeX fonts exactly. Best when the formula is already parsed as a SymPy expression rather than raw LaTeX strings.

### 4. Pillow-only fallback

For unrenderable formulas, a useful fallback is to draw a light-gray filled rectangle at the formula's bounding box coordinates. This preserves spatial presence in the reconstruction without leaving a blank hole:

```python
draw.rectangle([x1, y1, x2, y2], fill=(220, 220, 220), outline=(180, 180, 180))
```

This is strictly better than skipping the element (the current behavior in `image_renderer.py`), since it restores approximate pixel occupancy and reduces the SSIM/LPIPS penalty caused by large empty white regions where formulas belong.

---

## Document Reconstruction from OCR Output

There is no dedicated research literature specifically on "reconstructing a synthetic page image from OCR bounding box output for metric evaluation purposes." The task spans two adjacent areas:

**Layout-aware document understanding** (e.g., LayoutDIT, DoPTA) focuses on using bounding-box + text pairs as input to downstream NLP tasks (translation, QA), not on synthesizing images. These works validate that spatial layout is informationally important but do not address the visual rendering pipeline.

**Document image synthesis for training data** (e.g., SynthDoc, DocSynth) generates realistic document images from scratch using template engines or HTML/CSS rendering. The closest practical tool to our use case is **WeasyPrint** or **Playwright/Puppeteer**, which accept an HTML+CSS document as input and produce a pixel-accurate rendering. The reconstruction pipeline can therefore be expressed as:

1. Convert OCR output (text + bboxes + element types) → HTML with `position: absolute` CSS styling for each element.
2. For formulas, embed pre-rendered formula images (from matplotlib mathtext) as `<img>` tags at the correct position.
3. Render the HTML with WeasyPrint (Python-native, no browser) or Playwright (higher CSS fidelity) to produce a PNG.

WeasyPrint is particularly attractive because it is already available in Python environments, produces compact output, and handles multi-font layouts well. However, it is ~75x slower than warm Playwright per render, which matters only if rendering many images per second.

---

## Layout Fidelity Techniques

### Font size estimation from bbox height

The fundamental relationship is:

```
font_size_pt = bbox_height_px * 72 / page_dpi
```

For a page scanned at 150 DPI, a text line with a 25 px bounding box height corresponds to roughly 12 pt. The current `image_renderer.py` already implements this conversion in `render_analyzed_document` via `style.font_size_pt * doc.page_dpi / 72`. The main source of error is (a) DPI estimation (many scanned documents do not embed reliable DPI metadata) and (b) the "em square" vs. "cap height" distinction: font height in pixels typically matches the ascender+descender span, which is ~115–130% of the cap height. A correction factor of 0.8–0.85 applied to the raw bbox-to-pt conversion often improves fidelity.

### Iterative font size fitting with PIL

PIL's `draw.textbbox()` returns the exact pixel bounding box of a rendered string. The standard pattern for fitting text into an observed OCR bounding box is:

```python
def fit_font_size(draw, text, target_h, font_path, lo=8, hi=72):
    for size in range(hi, lo - 1, -1):
        font = ImageFont.truetype(font_path, size)
        bb = draw.textbbox((0, 0), text, font=font)
        if (bb[3] - bb[1]) <= target_h:
            return size, font
    return lo, ImageFont.truetype(font_path, lo)
```

Binary search is faster for large ranges; the current renderer already uses this strategy.

### Text wrapping and alignment

`draw.textlength()` gives sub-pixel accurate line widths, enabling correct word-wrap decisions. Using `anchor="lt"` (left-top) in `draw.text()` avoids the default baseline offset that otherwise shifts text a few pixels below the intended top of the bounding box. The current code applies a +1 px top margin and +2 px left margin which is reasonable; verify against actual rendering output.

### Multi-column and reading-order handling

OCR bounding boxes from Qwen3-VL are typically in reading order, so rendering them in list order naturally preserves left-to-right, top-to-bottom layout. The main risk is overlapping bounding boxes (a common artefact when the OCR model slightly over-estimates regions). Sorting elements by (top, left) and skipping elements whose boxes intersect significantly with already-rendered areas prevents pileup.

---

## OCR Evaluation via Image Comparison

### SSIM (Structural Similarity Index)

SSIM measures luminance, contrast, and structural coherence between patches. It is well-suited for evaluating layout preservation (column positions, text block presence/absence) but is insensitive to color changes and does not capture semantic content. For document reconstruction quality, SSIM scores in the 0.5–0.7 range are typical when text fonts do not exactly match; scores above 0.8 indicate close layout reproduction.

**Limitation relevant to our project**: missing formula regions create large white patches in the reconstruction. These patches have perfect local SSIM with a white background but mislead the global score by inflating it when the original page had dense formula content.

### LPIPS (Learned Perceptual Image Patch Similarity)

LPIPS uses features extracted from pre-trained CNNs (AlexNet, VGG, or SqueezeNet) to compute perceptual distance. Lower LPIPS = more similar. It captures texture and high-level appearance differences that SSIM misses, and correlates better with human judgments of reconstruction quality. The trade-off is higher compute cost and sensitivity to rendering style differences (e.g., font family mismatch).

### CLIP cosine similarity

CLIP encodes both images into a shared semantic space. CLIP cosine similarity between original and reconstructed pages captures whether the reconstructed page "looks like the same kind of document" at a semantic level, even when fonts differ. It is less sensitive to exact pixel alignment and more sensitive to overall content and structure. The `torchmetrics` library provides `CLIPScore` with the formula `max(100 * cos(E_I1, E_I2), 0)`.

**For our project**: CLIP similarity is the most forgiving metric for formula gaps — if the reconstruction preserves the surrounding text layout, CLIP may still assign a high score. This is a feature (robustness) and a limitation (it may not penalize poor formula rendering enough).

### CDM (Character Detection Matching)

CDM, proposed in arXiv:2409.03643, evaluates formula recognition by comparing rendered images rather than LaTeX source strings. It was developed to fix the unfairness of BLEU and edit distance metrics that penalize semantically identical formulas expressed in different LaTeX syntax.

**How it works** (four stages):

1. **Color-coded rendering**: Each LaTeX token is rendered in a unique RGB color (5,832 distinct colors via a 15-unit grid). Pixel coordinate clusters identify precise per-token bounding boxes.
2. **Bipartite matching**: The Hungarian algorithm pairs predicted tokens to ground-truth tokens, minimizing a weighted cost: token identity cost + L1 position cost + sequence order cost.
3. **RANSAC outlier filtering**: Geometrically inconsistent matches (those not fitting an affine transformation of the ground truth layout) are removed.
4. **F1 calculation**: `CDM = 2*TP / (2*TP + FP + FN)`, where TP is correctly matched tokens.

CDM achieves 96% consistency with human evaluation vs. lower rates for BLEU. The companion metric `ExpRate@CDM` counts formulas with CDM=1.0 (perfect match).

**Relevance to our project**: CDM is designed for formula-level evaluation (predicted LaTeX vs. ground-truth LaTeX), not for whole-page reconstruction quality. However, its core idea — render both prediction and reference to images, then compare visual token positions — is directly applicable as a formula rendering quality check. The CDM codebase is available at `github.com/opendatalab/UniMERNet/tree/main/cdm`.

OmniDocBench (arXiv:2412.07626) uses CDM alongside normalized edit distance and BLEU specifically for formula evaluation within its broader document parsing benchmark.

### ViTScore and semantic metrics

ViTScore computes cosine similarity between Vision Transformer patch embeddings of two images, analogous to BERTScore in NLP. It has superior semantic sensitivity compared to SSIM while remaining reference-based. For document reconstruction, it could serve as a replacement for CLIP cosine similarity with better spatial resolution.

---

## Recommended Approach for Our Project

The highest-priority fix is **replacing `continue` for formula elements with actual rendering**. The current code in `image_renderer.py` (lines 101–102 in `render_text_image` and lines 129–130 in `render_analyzed_document`) skips formula elements entirely. This creates blank white regions where formulas exist in the original, systematically degrading all comparison metrics.

### Step 1: Add formula rendering via matplotlib mathtext

Add a method to `ImageRenderer` and call it in both render paths:

```python
def _render_formula_patch(
    self, latex: str, target_w: int, target_h: int, dpi: int = 150
) -> Image.Image | None:
    """Render a LaTeX formula to an RGBA image sized to (target_w, target_h).
    Returns None if rendering fails."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Strip outer $...$ wrappers if present; math_to_image adds its own.
    formula = latex.strip().strip("$")
    if not formula:
        return None
    try:
        buf = io.BytesIO()
        import matplotlib.mathtext as mathtext
        mathtext.math_to_image(f"${formula}$", buf, dpi=dpi, format="png")
        buf.seek(0)
        formula_img = Image.open(buf).convert("RGBA")
        # Scale to fit bbox while preserving aspect ratio.
        formula_img.thumbnail((target_w, target_h), Image.LANCZOS)
        # Paste onto white background of exact bbox size.
        canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
        canvas.paste(formula_img, (0, 0))
        return canvas.convert("RGB")
    except Exception:
        return None  # caller draws gray rectangle fallback
```

In the render loop, replace the `continue` with:

```python
if te.tag == "formula":
    formula_img = self._render_formula_patch(te.text, bbox_w, bbox_h)
    if formula_img is not None:
        img.paste(formula_img, (x1, y1))
    else:
        draw.rectangle([x1, y1, x2, y2], fill=(220, 220, 220))
    continue
```

### Step 2: DPI calibration for font size

The formula `font_size_pt = bbox_height_px * 72 / page_dpi` should use `0.8` as a correction factor (cap height ≈ 80% of em square): `font_size_pt = bbox_height_px * 72 / page_dpi * 0.8`. Evaluate visually on the four working images.

### Step 3: Maintain existing PIL text rendering

The current PIL-based text rendering pipeline in `image_renderer.py` is architecturally sound — binary search font fitting, word-level wrapping with CJK fallback, style-aware font selection. No structural changes needed; improvements to formula rendering and DPI calibration should be sufficient to meaningfully improve SSIM and LPIPS scores.

### Step 4: Metric selection

Keep the current SSIM + LPIPS + CLIP combination. SSIM is sensitive to layout changes (good for detecting structural failures). LPIPS captures perceptual texture differences (penalizes wrong fonts and missing formulas). CLIP cosine similarity gives a semantic-level signal robust to rendering artifacts. Consider adding CDM as a formula-specific metric if per-formula quality becomes a research question, using the `opendatalab/UniMERNet` CDM implementation.

---

## Italic Detection

Four training-free methods for detecting italic text from grayscale bbox crops (2025 research synthesis).

### Method 1 (Primary): Column-centroid Regression

For each column of the binarized crop, compute the vertical center of mass (weighted mean row). Upright text → flat centroid sequence; italic text → monotonically drifting centroids. Fit a line; slope encodes slant angle.

```python
def detect_italic(gray_crop: np.ndarray) -> tuple[bool, float]:
    binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1].astype(np.float32)
    H = binary.shape[0]
    rows = np.arange(H, dtype=np.float32)
    col_masses = binary.sum(axis=0)
    valid = np.where(col_masses > H * 0.05)[0]
    if len(valid) < 4:
        return False, 0.0
    centroids = (rows[:, None] * binary[:, valid]).sum(axis=0) / col_masses[valid]
    slope = np.polyfit(valid.astype(np.float32), centroids, 1)[0]
    slant = float(np.degrees(np.arctan(slope)))
    return abs(slant) > 5.0, slant
```

- **Threshold**: |slant| > 5° (italic fonts lean 10–15°; upright stays within ±4°)
- **Speed**: O(W×H) vectorized, fastest method
- **Limitations**: Does not distinguish page-level skew from character italic slant; needs at least 4 ink-bearing columns

### Method 2: Horizontal Projection Profile Variance Maximization

Apply virtual shear transforms at a range of angles; the angle maximizing row-sum variance is the text's slant direction. Slower than Method 1 but more accurate for noisy crops.

- **Threshold**: same 5° threshold
- **Limitation**: requires `scipy.ndimage.rotate`; downsample to ~150px wide first for speed

### Method 3: Radon Transform Sinogram Variance Peak

Projects image at multiple angles; column variance of sinogram peaks at the dominant stroke orientation. Highest accuracy but most expensive.

### Method 4: `cv2.minAreaRect` on Foreground Pixels

Fastest but least precise. Fits a rotated bounding box to all foreground pixel coordinates. OpenCV angle sign convention changed in 4.5.1 — always normalize explicitly.

**Combined decision logic**: Use column centroid (fast), fall back to projection profile if angle is 3–7° (ambiguous zone).

**Published references**: Fan & Huang (IEEE 2006), ScienceDirect slant estimation for OCR (Pal & Datta 2001), endolith Radon gist.

---

## Font Weight Gradation Detection

Three training-free methods mapping text crops to CSS weight tiers (100=Thin → 900=Black).

### Method 1 (Primary): Stroke/Glyph-Height Ratio

The typographic stem width as fraction of letter height is the most theoretically grounded measure. Use existing `mean_stroke_width` (from distance transform + skeleton) and `glyph_height` (from connected components).

```python
def classify_font_weight(stroke_width: float, glyph_height: int) -> str:
    if glyph_height <= 0 or stroke_width <= 0:
        return "regular"
    ratio = stroke_width / glyph_height
    if ratio < 0.04: return "thin"
    if ratio < 0.07: return "light"
    if ratio < 0.12: return "regular"
    if ratio < 0.17: return "medium"
    if ratio < 0.22: return "bold"
    if ratio < 0.30: return "heavy"
    return "black"
```

| Ratio | CSS weight |
|---|---|
| < 0.04 | 100 Thin |
| 0.04 – 0.07 | 300 Light |
| 0.07 – 0.12 | 400 Regular |
| 0.12 – 0.17 | 500 Medium |
| 0.17 – 0.22 | 700 Bold |
| 0.22 – 0.30 | 800 Heavy |
| > 0.30 | 900 Black |

Grounded in: regular body text ~10–15% (Funakawa 2000), bold ~17–22%, hairline <5%.

### Method 2: Ink Ratio (Pixel Density)

```python
def compute_ink_ratio(gray_crop: np.ndarray) -> float:
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return int(np.count_nonzero(binary)) / binary.size
```

| Ink ratio | Weight |
|---|---|
| < 0.05 | Thin (100) |
| 0.05 – 0.18 | Light→Regular |
| 0.18 – 0.35 | Medium→Bold |
| > 0.35 | Heavy→Black |

Fast fallback. Highly sensitive to bbox tightness.

### Method 3: Stroke Width Distribution Histogram

Sample distance transform at all foreground pixels (not just skeleton). The median of this distribution in absolute pixels at a known DPI provides finer gradation than the mean-only approach.

**Key references**: Epshtein et al. CVPR 2010 (SWT), Sai et al. NCVPRIPG 2013 (MOBDoB bold detection), Funakawa 2000 via Vision Research 2021.

---

## Font Family Detection

Four-tier detection: monospace → handwriting → serif → sans-serif. Priority order matters because some features (CV of component widths for monospace) conflict with handwriting detection.

### 1. Monospace Detection (CC Width Variance)

```python
def _detect_monospace(gray_crop: np.ndarray) -> bool:
    binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    h, w = gray_crop.shape
    widths = [stats[i, cv2.CC_STAT_WIDTH] for i in range(1, n_labels)
              if 0.001 * h * w < stats[i, cv2.CC_STAT_AREA] < 0.5 * h * w
              and stats[i, cv2.CC_STAT_HEIGHT] > 3]
    if len(widths) < 4:
        return False
    arr = np.array(widths, dtype=float)
    cv = arr.std() / arr.mean() if arr.mean() > 0 else 1.0
    return cv < 0.15
```

- **CV < 0.15** → monospace; **CV 0.15–0.22** → borderline; **CV > 0.22** → proportional
- **Limitation**: touching characters merge blobs and inflate variance; works best on single-word crops

### 2. Handwriting Detection (SWT Coefficient of Variation)

```python
def _detect_handwriting(gray_crop: np.ndarray) -> bool:
    binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    sw = dist[binary > 0]
    if len(sw) < 50 or sw.mean() < 1e-6:
        return False
    return float(sw.std() / sw.mean()) > 0.5
```

- **SWT CV > 0.5** → handwriting; **< 0.3** → printed; **0.3–0.5** → ambiguous
- Epshtein et al. (CVPR 2010): valid printed text has adjacent pixel SWT ratios between 0.33–3.0; handwriting violates this widely

### 3. Serif vs Sans-serif (Stroke Width CV + Horizontal Projection Spread)

Current implementation uses SWT CV > 0.35 for serif detection. Enhancement: add horizontal projection spread ratio (ink spread at top/bottom thirds vs. midline indicates serif feet):

```python
def _serif_spread_ratio(gray_crop: np.ndarray) -> float:
    binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    h = binary.shape[0]
    row_counts = np.sum(binary > 0, axis=1).astype(float)
    top_peak = np.percentile(row_counts[:h//3], 90) if h >= 9 else 0
    bot_peak = np.percentile(row_counts[2*h//3:], 90) if h >= 9 else 0
    mid_mean = np.mean(row_counts[h//3: 2*h//3]) if h >= 9 else 1
    return (top_peak + bot_peak) / (2 * mid_mean + 1e-6)
```

- **Spread ratio > 1.35** → serif feet detected (catches slab serifs that SWT CV misses)
- Combined rule: `has_serif = spread_ratio > 1.35 or swt_cv > 0.35`

### 4. Condensed vs. Expanded (Aspect Ratio)

Median width/height ratio of character blobs:
- **< 0.45** → condensed; **> 0.75** → expanded; otherwise normal
- Useful for selecting narrow or wide font variants

### Full Priority Chain

```
classify_font_category(crop):
    if _detect_monospace(crop): return "monospace"
    if _detect_handwriting(crop): return "handwriting"
    return detect_font_category(crop)  # serif vs sans-serif via SWT CV
```

**Key references**: Epshtein et al. CVPR 2010, Zagoris et al. PRImA 2013 (handwriting vs. printed), ApOFIS font clustering (2000).

---

## Specific Font Family Identification

### Overview

Fine-grained font family identification (Times New Roman vs. Georgia vs. Garamond) from image crops alone is genuinely hard — a 2025 COLM paper found the best VLMs achieve only ~31% accuracy. The practical recommendation is a **two-tier strategy**: classify into broad typographic categories using image features, then select the best open-source representative font for each category.

### 1. Feature-Based Classification (Training-Free)

#### Stroke Contrast Ratio

The p95/p10 ratio of distance-transform values at ink pixels:

```python
def stroke_contrast_ratio(crop_gray: np.ndarray) -> float:
    _, bw = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    ink_widths = dist[bw > 0]
    if len(ink_widths) == 0:
        return 1.0
    p95 = np.percentile(ink_widths, 95)
    p10 = np.percentile(ink_widths, 10)
    return float(p95 / max(p10, 0.5))
```

Thresholds:
- **< 1.4** → Geometric/Neo-Grotesque sans (Helvetica, Arial, Futura, Calibri)
- **1.4–2.0** → Humanist sans (Verdana, Gill Sans) or Old-style serif (Garamond, Caslon)
- **2.0–3.5** → Transitional serif (Times New Roman, Georgia, Baskerville)
- **> 3.5** → Didone/Modern serif (Bodoni, Didot)

Accuracy: ~70–80% on clean images. Breaks down on crops < 30px tall.

#### X-Height Ratio

The x-height/cap-height ratio distinguishes families within the same category:
- Times New Roman: ~0.45–0.50
- Georgia: ~0.51–0.57 (notably larger x-height)
- Garamond: ~0.38–0.44
- Helvetica/Arial: ~0.52–0.56
- Calibri/Verdana: ~0.57–0.62

Measure via horizontal projection profile: cap-line is the top ink row, x-height is the row where lowercase-only ink density transitions to high.

#### Combined Category Classification

```python
class FontCategory(Enum):
    TRANSITIONAL_SERIF  = "transitional_serif"   # Times NR, Georgia, Baskerville
    OLDSTYLE_SERIF      = "oldstyle_serif"        # Garamond, Caslon, Palatino
    DIDONE_SERIF        = "didone_serif"          # Bodoni, Didot
    SLAB_SERIF          = "slab_serif"            # Rockwell, Courier slab
    HUMANIST_SANS       = "humanist_sans"         # Calibri, Gill Sans, Verdana
    GROTESQUE_SANS      = "grotesque_sans"        # Helvetica, Arial, Univers
    GEOMETRIC_SANS      = "geometric_sans"        # Futura, Avenir
    MONOSPACE           = "monospace"             # Courier, Consolas

def classify_extended_category(has_serif, stroke_contrast, xheight_ratio, is_monospace):
    if is_monospace: return FontCategory.MONOSPACE
    if not has_serif:
        if stroke_contrast < 1.25: return FontCategory.GEOMETRIC_SANS
        if xheight_ratio > 0.57:   return FontCategory.HUMANIST_SANS
        return FontCategory.GROTESQUE_SANS
    if stroke_contrast > 3.5: return FontCategory.DIDONE_SERIF
    if stroke_contrast > 2.0: return FontCategory.TRANSITIONAL_SERIF
    if xheight_ratio < 0.46:  return FontCategory.OLDSTYLE_SERIF
    if stroke_contrast < 1.4: return FontCategory.SLAB_SERIF
    return FontCategory.TRANSITIONAL_SERIF
```

### 2. OCR / PDF Metadata Approach

**Best method when source is PDF (PyMuPDF):**

```python
import pymupdf

def extract_pdf_font_info(pdf_path: str) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    spans = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0: continue
            for line in block["lines"]:
                for span in line["spans"]:
                    spans.append({
                        "text": span["text"],
                        "font": span["font"],     # e.g. "TimesNewRomanPS-BoldMT"
                        "size": span["size"],
                        "is_bold": bool(span["flags"] & 16),
                        "is_italic": bool(span["flags"] & 2),
                        "is_serif": bool(span["flags"] & 4),
                        "is_monospace": bool(span["flags"] & 8),
                    })
    return spans
```

The `flags` field has dedicated serif (bit 2), monospace (bit 3), and bold (bit 4) bits. Font names are exact PostScript names (e.g., `"AAAAAB+CMR10"` for LaTeX Computer Modern — strip the subset prefix).

**Tesseract legacy engine (OEM 0):** Can return approximate font names via `--oem 0` with old tessdata. Reliability ~50–60%; only for coarse family discrimination.

### 3. Font Substitution Tables

#### Metric-Compatible Free Substitutes (identical advance widths, no layout reflow)

| Commercial Font | Free Substitute | Source |
|---|---|---|
| Times New Roman | **Liberation Serif** / **Tinos** | Red Hat / Google Fonts |
| Arial | **Liberation Sans** / **Arimo** | Red Hat / Google Fonts |
| Courier New | **Liberation Mono** / **Cousine** | Red Hat / Google Fonts |
| Calibri | **Carlito** | Google Fonts |
| Cambria | **Caladea** | Google Fonts |
| Georgia | **Gelasio** | Google Fonts (exact metric match) |
| Helvetica | **TeX Gyre Heros** / **Nimbus Sans** | CTAN / ghostscript |
| Palatino | **TeX Gyre Pagella** | CTAN |

#### Representative Fallbacks by Document Category

| Document Category | Body Font | Code Font |
|---|---|---|
| Academic paper | **TeX Gyre Termes** (Times-alike) | Inconsolata |
| LaTeX-generated | **Latin Modern Roman** | Latin Modern Mono |
| Book / novel | **EB Garamond** or **Libre Baskerville** | — |
| Corporate / report | **Carlito** (Calibri-alike) | Cousine |
| Newspaper | **Tinos** | — |

#### PostScript Name Lookup Table

```python
PDF_FONT_SUBSTITUTION = {
    "timesnewromanps-regularmt": "LiberationSerif-Regular.ttf",
    "timesnewromanps-boldmt":    "LiberationSerif-Bold.ttf",
    "timesnewromanps-italicmt":  "LiberationSerif-Italic.ttf",
    "arialmt":                   "LiberationSans-Regular.ttf",
    "arial-boldmt":              "LiberationSans-Bold.ttf",
    "helvetica":                 "texgyreheros-regular.otf",
    "calibri":                   "Carlito-Regular.ttf",
    "georgia":                   "Gelasio-Regular.ttf",
    "cambria":                   "Caladea-Regular.ttf",
    "cmr10":                     "lmroman10-regular.otf",     # LaTeX
    "cmmi10":                    "lmmathitalic10-regular.otf",
    "cmbx10":                    "lmroman10-bold.otf",
    "couriernewpsmt":            "LiberationMono-Regular.ttf",
    "palatino-roman":            "texgyrepagella-regular.otf",
    "garamond":                  "EBGaramond-Regular.ttf",
}

def lookup_font_substitute(pdf_fontname: str) -> str | None:
    key = pdf_fontname.lower().strip()
    if "+" in key:
        key = key.split("+", 1)[1]  # strip subset prefix
    return PDF_FONT_SUBSTITUTION.get(key)
```

**Key references**: Font-Agent CVPR 2025, Texture or Semantics COLM 2025, DeepFont (Adobe 2015), ArchLinux Metric-Compatible Fonts wiki, PyMuPDF text extraction docs.

---

## Font Size Estimation and Text Fitting

### 1. The Em Square vs. Rendered Glyph Problem

"Font size" (in points) refers to the *em square* — an abstract design container. Glyphs are drawn inside the em square at ratios set by the type designer. The OCR bbox tightly wraps rendered ink, not the em square. The result: the naive formula `bbox_height_px * 72 / dpi` over-estimates font size because it treats ink height as em height.

**Empirically measured ratios** (Latin fonts):

| Metric | Typical ratio vs. point size |
|---|---|
| Ascender + descender span (full cell) | 91–164%, avg ~120% |
| Cap height ("H" glyph) | 62–78%, avg ~70% |
| X-height ("x" glyph) | 46–55%, avg ~50% |

For reconstruction: the OCR bbox height ≈ rendered ascender+descender span ≈ 1.2× the point size for most common fonts. The correction factor is:

```python
BBOX_TO_EM_FACTOR = 1.0 / 1.2   # bbox spans ~1.2× the em
font_size_pt = (bbox_height_px / dpi) * 72 * BBOX_TO_EM_FACTOR
```

**Reading the exact ratio from the rendering font (fonttools):**

```python
from fontTools.ttLib import TTFont

def get_span_ratio(font_path: str) -> float:
    tt = TTFont(font_path)
    upm = tt["head"].unitsPerEm
    os2 = tt["OS/2"]
    span = os2.sTypoAscender + abs(os2.sTypoDescender)
    return span / upm

# DejaVu Sans:    ≈ 1.07
# Liberation Serif: ≈ 1.05
# Catamaran:      ≈ 1.64 (extreme outlier)
```

Compute once at startup per rendering font, then use as the correction divisor.

### 2. Multi-Line Bbox: Estimating Line Count

When a text block has N lines, its bbox height = N × line_height. Line height ("leading") ≈ 1.2× font size at `line-height: normal`. Never use a multi-line bbox height as a single-glyph height.

**Image-based line counting** (most reliable — use existing `detect_line_positions`):

```python
line_positions = detect_line_positions(crop)   # [(y_start, y_end), ...]
n_lines = len(line_positions)
if n_lines >= 2:
    line_height_px = line_positions[1][0] - line_positions[0][0]  # baseline-to-baseline
    glyph_height_px = line_positions[0][1] - line_positions[0][0]  # ink height only
    font_size_pt = (glyph_height_px / dpi) * 72 * BBOX_TO_EM_FACTOR
```

**Fallback (no image crop):**

```python
def font_size_from_multiline_bbox(bbox_h_px, n_lines, dpi,
                                   span_ratio=1.2, line_height_ratio=1.2):
    per_line_px = bbox_h_px / n_lines          # includes leading
    glyph_px    = per_line_px / line_height_ratio  # strip leading
    return glyph_px * 72 / (dpi * span_ratio)  # strip em overhead
```

### 3. PIL Font Metrics: Accurate Size Fitting

The four key PIL measurement primitives:

| Method | Returns | Use for |
|---|---|---|
| `font.getbbox(text)` | `(left, top, right, bottom)` | Tight ink bbox of a string |
| `font.getmetrics()` | `(ascent, descent)` | Full cell height = ascent + \|descent\| |
| `font.getlength(text)` | float | Horizontal advance width |
| `draw.multiline_textbbox(xy, text, font)` | `(l,t,r,b)` | Wrapped text bbox |

**Key: always use `getmetrics()` for cell height, not `getbbox("Ag")`.**

```python
ascent, descent = font.getmetrics()
cell_height = ascent + abs(descent)   # = rendered line height slot
```

**Binary search for font size matching a target cell height:**

```python
def find_size_for_cell_height(font_path: str, target_px: int) -> int:
    lo, hi, best = 1, 500, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        a, d = f.getmetrics()
        if a + abs(d) <= target_px:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best
```

### 4. Multi-Line Fitting: Binary Search with Text Wrapping

```python
def fit_font_to_box(text, font_path, target_w, target_h, spacing=4, lo=6, hi=144):
    """Largest font size where word-wrapped text fits in (target_w, target_h)."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        lines = _wrap_at_width(text, f, target_w)
        l, t, r, b = draw.multiline_textbbox((0,0), "\n".join(lines), font=f, spacing=spacing)
        if (r - l) <= target_w and (b - t) <= target_h:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best

def _wrap_at_width(text, font, max_w):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".lstrip() if current else word
        if font.getlength(candidate) <= max_w:
            current = candidate
        else:
            if current: lines.append(current)
            current = word
    if current: lines.append(current)
    return lines or [""]
```

This converges in ≤9 iterations (binary search over 1–500pt). The current greedy top-down loop in `_estimate_font_size` is O(n) and can be replaced with this pattern.

### 5. Single-Line Width Fitting

```python
def fit_font_to_width(text, font_path, target_w, lo=1, hi=500):
    """Largest font size where text fits on one line within target_w pixels."""
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        if f.getlength(text) <= target_w:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best
```

**Fast estimate without full binary search** (probe at one size, then scale):

```python
def estimate_size_for_width(text, font_path, target_w, probe_size=50):
    f = ImageFont.truetype(font_path, probe_size)
    w = f.getlength(text)
    return max(1, int(probe_size * target_w / w)) if w > 0 else probe_size
```

### 6. Integration Recommendation

The minimal fix for the current pipeline (`document_analyzer.py` line 87):

```python
# Current (over-estimates by ~20%):
font_size_pt = (bbox_height_px / dpi) * 72

# Fixed (account for em-square vs. cell-height mismatch):
RENDERING_FONT_SPAN_RATIO = 1.07   # for DejaVu Sans; compute via get_span_ratio()
font_size_pt = (glyph_height_px / dpi) * 72 / RENDERING_FONT_SPAN_RATIO
```

For multi-line bboxes where `measure_glyph_height` returns 0, add `/ line_height_ratio` (default 1.2).

**Also use `fit_font_to_box` instead of the greedy size loop in `ImageRenderer._estimate_font_size`** — it is faster, more accurate (re-wraps at each trial size), and eliminates the O(n) worst case.

**Key references**: tonsky.me "Font Size is Useless", FreeType glyph conventions, iamvdo.me CSS font metrics, PIL ImageFont docs, fonttools OS/2 table docs.

---

## Text Alignment Detection from Document Images

*Research via subagent, 2026-04-17*

### Overview

Two complementary approaches: (A) pixel-level analysis on a single paragraph crop, (B) geometric analysis across multiple bboxes in the same column.

### Algorithm A: Single-Crop Pixel-Level Edge Variance

1. Binarize with Otsu invert (ink = 255).
2. For each row, find leftmost and rightmost non-zero pixel (skip rows with fewer than ~5% of crop width in ink — inter-line gaps).
3. Compute `left_std = np.std(left_edges)`, `right_std = np.std(right_edges)`, `center_std = np.std(midpoints)`.
4. Classify with threshold T ≈ 5–10px at 150 DPI (scale with DPI):
   - `left_std < T and right_std < T` → **Justified** (needs disambiguation — see below)
   - `left_std < T and right_std >= T` → **Left-aligned**
   - `right_std < T and left_std >= T` → **Right-aligned**
   - both high, `center_std < T` → **Centered**

**Morphological cleanup before edge analysis:**
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```
This merges fragmented ink within a line, stabilizing edge position.

### Algorithm B: Multi-BBox Column-Level (when OCR gives line-level boxes)

Group bboxes by y-position into lines, then compute `np.std([line.x1])` and `np.std([line.x2])`. Apply same threshold logic. This is more noise-resistant than pixel analysis.

### Distinguishing Justified from Left-Aligned (the Hard Problem)

Both have consistent left edges. Three signals:

**Signal A — Last-line width ratio (most reliable):**
```python
last_line_ratio = last_line_width / mean_other_line_width
# last_line_ratio < 0.85 → very likely justified
```
Justified paragraphs conventionally leave the last line unjustified (shorter). If the last line fills ≥ 95% of width, the paragraph is probably left-aligned (or it happens to be a full last line).

**Signal B — Right-edge variance excluding the last line:**
```python
right_std_excl_last = np.std(right_edges[:-1])
# right_std_excl_last < T with last_line shorter → justified
```

**Signal C — Inter-word gap variance within lines:** Justified text has variable inter-word spacing from line to line (spaces stretched to fill). Measure average gap width per line using zero-runs between ink clusters; high variance across lines → justified.

**Recommended scoring heuristic:**
```python
score = 0
if right_std_excl_last < T:       score += 2
if last_line_ratio < 0.85:        score += 1
if inter_word_gap_variance > V:   score += 1
return "justified" if score >= 3 else "left-aligned"
```

### Practical Notes

- **DPI scaling**: T = 5px at 150 DPI → T = 10px at 300 DPI.
- **5th/95th percentile** of left/right edges per row is more robust than exact first/last pixel on noisy scans.
- **Current implementation** (`detect_alignment` in `document_analyzer.py`) already uses bbox-level edge variance — this research validates and extends it with the justified/left disambiguation via last-line width.

### Key References
- US8565474B2 (HP/Nuance patent): multi-feature paragraph alignment classification after OCR
- Kleber et al. (2014), Pattern Analysis and Applications: ε-threshold alignment grouping
- MDPI Applied Sciences (2025): paragraph structural features for alignment detection

---

## First-Line Paragraph Indent Detection

*Research via subagent, 2026-04-17*

### Overview

Detects whether the first line of a paragraph starts with leading whitespace (indent), and measures its width in pixels.

### Algorithm: Dual-Profile Method

**Step 1 — Binarize:**
```python
_, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

**Step 2 — Isolate first text line via horizontal projection:**
```python
row_profile = binary.sum(axis=1)
in_text = row_profile > row_profile.max() * 0.05
# Find first gap after the first text band
saw_text = False
for i, has_ink in enumerate(in_text):
    if has_ink: saw_text = True
    elif saw_text: first_gap = i; break
first_line = binary[:first_gap, :]
```

**Step 3 — Column projection on first-line strip:**
```python
col_profile = first_line.sum(axis=0)
dark_cols = np.where(col_profile > col_profile.max() * 0.02)[0]
first_line_left = int(dark_cols[0]) if len(dark_cols) else 0
```

**Step 4 — Body left edge (median across non-first-line rows):**
```python
body_lefts = [binary[r].nonzero()[0][0] for r in range(first_gap, H)
              if binary[r].sum() > 0]
body_left = int(np.median(body_lefts))
```

**Step 5 — Indent decision:**
```python
indent_px = first_line_left - body_left
is_indented = indent_px >= 5  # threshold ~5px minimum
```

### Robustness Notes

- **Median** over several rows (not just first row) for the first-line left edge, guards against noise.
- **Descenders from above** may appear in top rows of crop — skip first 10-15% of rows if crop is tight.
- **Hanging indent** (body indented, first line flush) → indent_px is negative; detect as a different style.
- **Zero indent ≠ failure** — fully-justified paragraphs correctly return indent_px ≈ 0.
- **Drop cap guard**: if `indent_px > 0.25 × crop_width`, likely a drop cap not a standard indent.

### Typographic Unit Conversion

```python
indent_pt = (indent_px / dpi) * 72
indent_em = indent_pt / font_size_pt
# Standard: 1–2 em indent; threshold for detection: > 0.5 em
```

### Application to Reconstruction

In reconstruction, a detected indent should be rendered as `text_x = x1 + indent_px` for the first line only. This avoids artificially extending the paragraph to fill a bbox that was intended for an indented paragraph.

### Key References
- Projection-Based Text Line Segmentation with Variable Threshold (AMCS 2017)
- US Patent 8565474B2: paragraph indent detection in OCR
- Butterick's Practical Typography: first-line indents (1–2 em is standard)

---

## Last-Line Text Extent Detection for Font-Size Calibration

*Research via subagent, 2026-04-17*

### Overview

By measuring how far the last line of a paragraph extends (its fill ratio), we can cross-validate the estimated font size: `chars_per_line = OCR_last_line_chars / fill_ratio`, then `avg_char_width_px = bbox_width / chars_per_line`, and `font_size_pt ≈ avg_char_width_px / 0.5 × 72/dpi` (using the typographic constant that avg char width ≈ 0.5 × em for Latin proportional fonts).

### Algorithm: Horizontal + Vertical Projection

**Step 1 — Binarize (Gaussian blur first to suppress noise):**
```python
blur = cv2.GaussianBlur(gray_crop, (3, 3), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

**Step 2 — Find last text line rows via horizontal projection (scan bottom-up):**
```python
h_proj = binary.sum(axis=1).astype(float)
h_thresh = 0.05 * h_proj.max()
ink_rows = np.where(h_proj > h_thresh)[0]
last_row = int(ink_rows[-1])
# Walk backward to find start of last line band
for r in range(last_row, -1, -1):
    if h_proj[r] <= h_thresh: first_row = r + 1; break
```

**Step 3 — Vertical projection on last-line rows:**
```python
last_line = binary[first_row:last_row + 1, :]
v_proj = last_line.sum(axis=0).astype(float)
ink_cols = np.where(v_proj > 0.05 * v_proj.max())[0]
rightmost = int(ink_cols[-1])
fill_ratio = rightmost / (binary.shape[1] - 1)
```

### Font Size Calibration Using Fill Ratio

```python
# From OCR text, count characters in last line
last_line_text = ocr_text.rstrip().split('\n')[-1]
chars_last_line = len(last_line_text)

# Estimate characters per full line
chars_per_full_line = chars_last_line / fill_ratio  # validity: expect 30-90 CPL

# Estimate average character width in pixels
avg_char_width_px = bbox_width_px / chars_per_full_line

# Font size estimate from character width (avg char width ≈ 0.5 × em for Latin)
font_size_px_from_width = avg_char_width_px / 0.5
font_size_pt_from_width = font_size_px_from_width / dpi * 72

# Cross-validate against line-height-based estimate
num_lines = len(detect_line_positions(crop))
font_size_pt_from_height = (bbox_height_px / num_lines) / dpi * 72
```

If both estimates agree (within ~20%), use the average. Large disagreement suggests an outlier (very short/long last line, OCR errors, multi-column layout).

### RLSA for Noisy Crops (Optional Enhancement)

Apply Run-Length Smearing Algorithm (RLSA) horizontally before vertical projection to bridge inter-character gaps:
```python
# Smear horizontal zero-runs shorter than C pixels
for i, row in enumerate(last_line):
    smeared = rlsa_1d(row, C=int(avg_char_width_px * 0.3))
    last_line[i] = smeared
```
This is optional but helps with math/formula text that has large inter-symbol spacing.

### Practical Validity Checks

- Fill ratio < 0.05 → last line is almost empty (orphan widow); treat as unreliable.
- Fill ratio > 0.98 → last line is nearly full (possibly justified); may not help discriminate.
- Estimated CPL outside [30, 90] → flag as unreliable measurement.
- Noise threshold should be proportional: `max(3, bbox_width * 0.01)` pixels per row minimum.

### Key References
- Wahl et al. (1982): Run-Length Smearing Algorithm (RLSA) — foundational text block segmentation
- Projection Profile Method (GeeksforGeeks): practical OpenCV implementation
- Pearsonified.com: CPL calibration (ideal 45–75 chars/line, optimum 66)
- Text Line Segmentation survey (arXiv:0704.1267): horizontal projection as baseline

---

## Background Normalisation for Original–Reconstruction Comparison

*Research via subagents, 2026-04-19*

The reconstructed page always has a white background. The original scan may have coloured paper, gradients, shadows, or noise. This mismatch inflates SSIM/LPIPS error and deflates CLIP cosine similarity even when the text is perfectly reproduced. Three complementary strategies exist.

---

### Strategy A — Remove Background from the Original (Make It White-on-Black-Ink)

The goal is to produce a clean black-ink-on-white image from the original scan so that it visually matches the reconstruction's white background.

#### Method A1: Gaussian Divide Normalisation (Recommended — 3 Lines of OpenCV)

Divide each pixel by a low-frequency estimate of the local illumination, then rescale to [0, 255]. This removes gradients, uneven lighting, and paper colour in one step.

```python
import cv2
import numpy as np

def gaussian_divide_normalize(img_bgr: np.ndarray, sigma: int = 51) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    normalized = cv2.divide(gray, bg, scale=255.0)
    return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
```

`sigma=51` targets illumination frequencies below ~1/(51px). For high-resolution scans (300+ DPI), increase to 101–151. Followed by Otsu or a fixed threshold (e.g. 200) to binarize.

**Limitations**: assumes the background is slowly varying. Fails on sharp-edged coloured panels or dark-background designs.

#### Method A2: Morphological Black-Hat

Subtracts the morphological *close* of the image from the original, isolating dark features (text) on any background. Effective for uniform but non-white backgrounds.

```python
def blackhat_normalize(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)   # black text on white
```

Kernel size ≈ 2× the tallest expected character. Computationally fast; less effective for gradient backgrounds than Gaussian divide.

#### Method A3: Sauvola / NICK Local Adaptive Thresholding

Per-block adaptive threshold using local mean and variance. Robust to spatially varying illumination at the cost of occasionally splitting large-font characters.

```python
from skimage.filters import threshold_sauvola

def sauvola_binarize(gray: np.ndarray, window: int = 25, k: float = 0.2) -> np.ndarray:
    thresh = threshold_sauvola(gray, window_size=window, k=k)
    binary = (gray > thresh).astype(np.uint8) * 255
    return binary   # white text on black → invert if needed
```

`k=0.2` is standard; increase toward 0.5 for very faint text.

#### Method A4: HSV Colour-Aware Thresholding (for Coloured Paper)

When paper is a known colour (cream, blue, etc.), isolate it in HSV and replace with white.

```python
def remove_colored_background(img_bgr: np.ndarray,
                               bg_lower=(0, 0, 180), bg_upper=(180, 60, 255)) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(bg_lower), np.array(bg_upper))
    result = img_bgr.copy()
    result[mask > 0] = [255, 255, 255]
    return result
```

`bg_lower/bg_upper` should be tuned per document type. Combine with a small morphological dilation of the mask to cover edge pixels.

#### Deep-Learning Background Removal (Batch / Offline)

| Model | Description | Link |
|---|---|---|
| **FourBi** (ICDAR 2024) | Frequency-domain binarization; best for gradients and watermarks | `github.com/fax004/FourBi` |
| **DocRes** (CVPR 2024) | Unified restoration (deblur, denoise, binarize, dewarp) | `github.com/zzzhang-jx/DocRes` |
| **SauvolaNet** | CNN that mimics Sauvola but learns optimal parameters | arXiv:2105.12899 |

These require inference (GPU recommended) but produce near-perfect binarization on degraded documents.

**Recommended pipeline for this project**: Apply Gaussian Divide + Otsu on the original image before computing any similarity metric. This is the lowest-effort change with the highest impact on SSIM/LPIPS stability.

---

### Strategy B — Transfer Background Style from Original to Reconstruction

Instead of cleaning the original, add the original's background texture to the (white) reconstruction, making both images visually similar in background appearance.

#### Method B1: Inpainting + Alpha Compositing (Recommended)

1. **Inpaint text regions** from the original to get a clean background estimate.
2. **Composite** the reconstructed text (as an RGBA layer with the text bbox mask as alpha) over the inpainted background.

```python
import cv2
import numpy as np
from PIL import Image

def build_text_mask(pixel_doc, width: int, height: int) -> np.ndarray:
    """Binary mask: 255 where text bboxes are, 0 elsewhere."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for te in pixel_doc.text_elements:
        x1, y1, x2, y2 = te.bbox
        mask[y1:y2, x1:x2] = 255
    return mask

def transfer_background(original_bgr: np.ndarray,
                         reconstructed_pil: Image.Image,
                         text_mask: np.ndarray,
                         inpaint_radius: int = 3) -> Image.Image:
    # Step 1: inpaint text regions in original to get background estimate
    bg_estimate = cv2.inpaint(original_bgr, text_mask, inpaint_radius, cv2.INPAINT_NS)

    # Step 2: convert reconstruction to RGBA, alpha = inverted background mask
    recon_rgba = reconstructed_pil.convert("RGBA")
    # Pixels that are non-white in reconstruction = text ink
    recon_arr = np.array(recon_rgba)
    ink_alpha = ((recon_arr[:, :, :3] < 240).any(axis=2)).astype(np.uint8) * 255
    recon_arr[:, :, 3] = ink_alpha

    # Step 3: paste reconstruction ink over inpainted background
    bg_pil = Image.fromarray(cv2.cvtColor(bg_estimate, cv2.COLOR_BGR2RGB))
    bg_pil.paste(Image.fromarray(recon_arr), mask=Image.fromarray(ink_alpha))
    return bg_pil
```

**Inpainting options**:
- `cv2.INPAINT_NS` (Navier-Stokes, ~60ms): good for thin text on uniform/gradient backgrounds
- `cv2.INPAINT_TELEA` (~40ms): slightly faster, slightly less accurate
- **LaMa** (Resolution-robust large mask inpainting): ~200ms on GPU, handles large text blocks and complex backgrounds. `pip install simple-lama-inpainting`

**Limitations**: Inpainting fails when text regions are very large or dense (tables, formula blocks), because there is too little background context remaining. Combine with `Strategy A` (Gaussian divide) as a fallback.

#### Method B2: Histogram Matching

Match the histogram of the reconstruction to the original. Simple but unreliable for documents: the histogram is strongly bimodal (ink vs. background) and matching it distorts both ink colour and background colour in unpredictable ways. **Not recommended**.

#### Method B3: Neural Style Transfer

Transfer the low-frequency texture style of the original to the reconstruction using Gram-matrix optimization (Gatys 2015) or AdaIN (Huang & Belongie 2017). Produces stylistically faithful results but: (a) ~5s per image even on GPU, (b) creates artifacts in high-frequency text strokes. **Overkill for this use case**.

---

### Strategy C — Background-Invariant Similarity Metrics

Rather than aligning the images, modify the similarity metric to ignore background appearance entirely.

#### Method C1: Alpha-CLIP (CVPR 2024) — Drop-in CLIP with Region Mask

Alpha-CLIP extends CLIP with an alpha channel input that focuses the encoder on a specified spatial region. Pass the union of all text bboxes as a binary alpha mask; the model ignores everything outside.

```python
# pip install git+https://github.com/SunzeY/AlphaCLIP
import alpha_clip
import torch
from PIL import Image
import numpy as np

model, preprocess = alpha_clip.load("ViT-L/14", alpha_vision_ckpt_pth="clip_l14_grit1m_fultune_6xe.pth")

def alpha_clip_similarity(img1: Image.Image, img2: Image.Image,
                           mask: np.ndarray) -> float:
    """mask: uint8 array [0,255], same size as images, white = focus region."""
    alpha = Image.fromarray(mask).convert("L")
    t1 = preprocess(img1).unsqueeze(0)
    t2 = preprocess(img2).unsqueeze(0)
    a = preprocess(alpha).unsqueeze(0)
    with torch.no_grad():
        e1 = model.visual(t1, a)
        e2 = model.visual(t2, a)
    return float(torch.nn.functional.cosine_similarity(e1, e2).item())
```

**Weights**: download from the Alpha-CLIP HuggingFace repo (`SunzeY/AlphaCLIP`). Requires a one-time 1.7 GB download. Achieves near-CLIP accuracy on masked regions while being completely insensitive to background.

**Key reference**: Sun et al., "Alpha-CLIP: A CLIP Model Focusing on Wherever You Want", CVPR 2024, arXiv:2312.03818.

#### Method C2: DINOv2 PCA Foreground Extraction (Training-Free)

The first principal component of DINOv2 patch features naturally separates foreground (text) from background. No training, no prompts — just extract features and threshold.

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Load DINOv2 (cached after first download)
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dino.eval()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def dino_foreground_mask(img: Image.Image, threshold: float = 0.0) -> np.ndarray:
    """Returns (224//14, 224//14) binary mask: True = foreground."""
    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        feats = dino.get_intermediate_layers(x, n=1)[0][0]   # (N_patches, D)
    pca_first = torch.pca_lowrank(feats, q=1)[0].squeeze(-1).numpy()
    mask = pca_first > threshold   # first PC positive = foreground
    if mask.mean() > 0.5:          # auto-flip if background is majority positive
        mask = ~mask
    return mask.reshape(224 // 14, 224 // 14)
```

Use this mask to extract patch-level features from both images, then compute cosine similarity only over foreground patches.

**Key reference**: Amir et al., "Deep ViT Features as Dense Visual Descriptors", ECCV 2022; Caron et al., "DINO: Self-supervised ViTs", ICCV 2021.

#### Method C3: DreamSim (NeurIPS 2023) — Inherently Foreground-Biased

DreamSim is a perceptual metric fine-tuned on human similarity judgments for natural images. Empirically, it is more sensitive to foreground content and less sensitive to background differences than SSIM or LPIPS, without any explicit masking.

```python
# pip install dreamsim
from dreamsim import dreamsim
import torch
from PIL import Image

model, preprocess = dreamsim(pretrained=True)

def dreamsim_distance(img1: Image.Image, img2: Image.Image) -> float:
    t1 = preprocess(img1).unsqueeze(0)
    t2 = preprocess(img2).unsqueeze(0)
    with torch.no_grad():
        dist = model(t1, t2)   # lower = more similar
    return float(dist.item())
```

DreamSim uses an ensemble of CLIP-ViT, DINO, and OpenCLIP features. It requires ~400MB of weights (auto-downloaded). For document images it provides a useful complement to SSIM/LPIPS.

**Key reference**: Fu et al., "DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data", NeurIPS 2023, arXiv:2306.09344.

#### Method C4: CLIP Surgery (Training-Free Text-Guided Saliency)

CLIP Surgery adds attention modifications to standard CLIP that produce per-pixel saliency maps conditioned on a text prompt. Use the prompt `"text, formulas, tables"` to generate a foreground mask, then compute CLIP similarity only over masked regions.

```python
# pip install git+https://github.com/xmed-lab/CLIP_Surgery
import clip_surgery as cs

model, preprocess = cs.load("ViT-B/16")

def clip_surgery_mask(img: Image.Image, prompt: str = "text formulas tables") -> np.ndarray:
    img_t = preprocess(img).unsqueeze(0)
    text_t = cs.tokenize([prompt])
    with torch.no_grad():
        saliency = cs.get_similarity_map(model, img_t, text_t)
    return (saliency.numpy() > 0.5).astype(np.uint8) * 255
```

**Key reference**: Li et al., "CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks", arXiv:2304.05653.

#### Method C5: CLIPSeg — Text-Prompted Zero-Shot Segmentation

CLIPSeg generates binary segmentation masks from text prompts, leveraging a lightweight decoder on top of frozen CLIP. More accurate than CLIP Surgery for spatial localization.

```python
# pip install transformers
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from PIL import Image

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def clipseg_mask(img: Image.Image, prompt: str = "text characters formulas") -> Image.Image:
    inputs = processor(text=[prompt], images=[img], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = torch.sigmoid(logits[0]).numpy()
    mask = (prob > 0.5).astype("uint8") * 255
    return Image.fromarray(mask).resize(img.size)
```

Weights: ~200MB, auto-downloaded from HuggingFace. Generates accurate text-region masks even for complex multi-column layouts.

**Key reference**: Lüddecke & Ecker, "Image Segmentation Using Text and Image Prompts", CVPR 2022.

#### Method C6: ST-LPIPS — Shift-Tolerant LPIPS

Standard LPIPS penalises sub-pixel misalignment heavily. ST-LPIPS uses spatially shifted feature comparisons to be tolerant of minor positional offsets between original and reconstruction (e.g. OCR bbox rounding errors).

```python
# pip install stlpips-pytorch
import stlpips
import torch
from PIL import Image
import torchvision.transforms.functional as TF

metric = stlpips.LPIPS(net="alex", variant="shift_tolerant")

def stlpips_distance(img1: Image.Image, img2: Image.Image) -> float:
    t1 = TF.to_tensor(img1).unsqueeze(0) * 2 - 1   # scale to [-1, 1]
    t2 = TF.to_tensor(img2).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(metric(t1, t2).item())
```

**Key reference**: Ghildyal & Liu, "Shift-tolerant Perceptual Similarity Metric", ECCV 2022.

---

### Implementation Recommendations for This Project

**Highest-impact, lowest-effort** (implement first):
1. **Gaussian Divide normalisation** on the original before computing SSIM/LPIPS. Three lines of OpenCV — eliminates background colour/gradient contribution entirely.
2. **Alpha-CLIP or text-bbox masking** for the CLIP metric. Pass the union of all text bboxes from `pixel_doc.text_elements` as the alpha mask. Eliminates white-background vs. coloured-background mismatch.

**Medium effort, high value**:
3. **Inpainting + alpha compositing** to generate a style-matched reconstruction (original background + reconstructed text). Best for visual inspection and human evaluation. Use `cv2.INPAINT_NS` as baseline; upgrade to LaMa for better quality on dense text regions.
4. **DreamSim** as a drop-in additional metric: `pip install dreamsim`, inherently foreground-biased, good complement to the current SSIM+LPIPS+CLIP triple.

**Research-grade / longer-term**:
5. DINOv2 PCA foreground masks + masked CLIP similarity.
6. CLIPSeg for high-quality document region segmentation.
7. ST-LPIPS to reduce positional sensitivity in the current LPIPS computation.

**Not recommended**: Histogram matching (bimodal problem), neural style transfer (slow and introduces artifacts).


---

