# DPI and Resolution Reference

## Summary

There is no single fixed DPI for the project. DPI depends on the source image and is then capped by the OCR model's pixel budget. In practice, **almost all images are sent to the OCR model at ~146–191 effective DPI**, regardless of their native resolution.

---

## DPI by Stage

### 1. PDF → PNG Conversion (`pdf_converter.py`)

```python
dpi: int = 300  # hardcoded default
```

When a PDF is the input, `pdf2image.convert_from_path` renders each page at **300 DPI**. This produces a PNG that is then processed like any other image. The 300 DPI default can be overridden per call but is never changed in the current pipeline.

### 2. Source Images (Pre-existing PNGs)

The 8 sample pages are provided as PNG files with **no embedded DPI metadata**. Their native resolution is estimated from pixel dimensions by matching against standard page sizes (see auto-detection below).

### 3. Image Sent to OCR Model — Pixel Budget Cap

The vLLM server applies `smart_resize` to fit within `max_pixels = 2048 × 32 × 32 = 2,097,152`. Images larger than this are **downscaled**, reducing effective DPI:

| Page | Native pixels | Native DPI (est.) | Sent to OCR | Effective DPI |
|------|--------------|-------------------|-------------|---------------|
| PPT_linalg | 2000×1500 | 176 | 1664×1248 | **146** |
| Puzzles & Problems | 1850×1225 | 204 | 1760×1152 | **191** |
| Complex Analysis | 2200×1700 | 200 | 1632×1248 | **146** |
| Color textbook (CN) | 1884×1334 | 161 | 1696×1216 | **146** |
| Eastmoney (CN) | 2339×1654 | 200 | 1696×1216 | **147** |
| Exam paper | 2200×1700 | 200 | 1632×1248 | **146** |
| The Economist | 5996×4559 | **536** | 1632×1248 | **146** |
| Newspaper | 3939×2754 | **393** | 1728×1184 | **168** |

The Economist (536 DPI native) and newspaper (393 DPI native) are the most heavily downscaled — losing ~3.6× and ~2.3× resolution respectively before reaching the OCR model.

**Practical implication**: the OCR model effectively sees all pages at ~146–191 DPI, regardless of source quality. Increasing `max_pixels` to `8192 × 32 × 32` (~8.4M) would allow processing at native resolution for all but the Economist page, at the cost of higher VRAM and latency.

### 4. DPI Auto-Detection for Font Size (`document_analyzer.py`)

Font sizes in the reconstructed image are estimated from bbox heights in pixels. To convert pixels → points, the pipeline needs to know DPI. Since images have no metadata, DPI is estimated from pixel dimensions:

```python
def estimate_dpi(image_width, image_height):
    # Fits aspect ratio against common page sizes (portrait)
    candidates = [
        (8.5, 11.0),    # US Letter
        (8.27, 11.69),  # A4
        (6.0, 9.0),     # Compact book
        (7.0, 10.0),    # Trade book
        (5.5, 8.5),     # Digest
        (6.69, 9.61),   # B5
    ]
    # Returns: min(image_width, image_height) / best_matching_page_width_inches
    # Clamped to [72, 600]
```

This runs on the **original image dimensions** (before pixel budget cap), so the estimated DPI matches the source image, not the downscaled version. The result is used in `run_experiment.py` via `DocumentAnalyzer(dpi=300)` as the constructor argument — but `DocumentAnalyzer.analyze()` overrides this with the auto-detected value internally.

### 5. Formula Rendering (`image_renderer.py`)

LaTeX formulas are rendered by matplotlib's `math_to_image` at a fixed **150 DPI**:

```python
math_to_image(f"${latex}$", buf, dpi=150, format="png", prop=prop)
```

This is independent of document DPI. At 150 DPI, formula images are rendered at a resolution similar to what the OCR model sees (~146 DPI effective), so formula patches should integrate visually with the reconstructed page.

---

## Key Takeaway

| Context | DPI |
|---------|-----|
| PDF conversion | 300 (fixed default) |
| Source PNGs (native, estimated) | 161–536 |
| Sent to OCR model (after pixel budget) | **146–191** (most at ~146) |
| Font size estimation | Auto-detected from native image dimensions |
| Formula rendering (matplotlib) | 150 (fixed) |

The effective OCR resolution is ~150 DPI for all pages. High-resolution source images (Economist at 536 DPI, newspaper at 393 DPI) lose the most information at the pixel budget boundary. This is a likely contributor to the poor OCR quality on the newspaper page.
