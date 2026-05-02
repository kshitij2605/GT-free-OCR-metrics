# System Design: Reference-Free OCR Metric

## Overview

This system evaluates OCR quality without requiring ground-truth text. The core principle: good OCR output, when rendered back as an image, should closely resemble the original document page.

## Pipeline

```
Original Image → Qwen3-VL OCR → HTML (text + bboxes)
                                        |
               +------------------------+------------------------+
               |                        |                        |
       Parse text bboxes        Parse image/table bboxes   Extract plain text
               |                        |                        |
               v                        v                        v
    Render text-only image     Mask original (white-fill)  LM Perplexity Scoring
               |                        |
               +---------- Compare -----+
                    (SSIM+MSE+LPIPS / CLIP cosine)
```

## OCR Model

**Qwen3.5-122B-A10B** via vLLM (OpenAI-compatible API endpoint).

- Hosted at: `<your-vllm-host>`
- Output format: HTML with `data-bbox="x1 y1 x2 y2"` attributes per element
- Element classes: `text`, `formula`, `table`, `image`
- Pixel budget: 2048×32×32 = 2,097,152 pixels (configurable via `max_pixels`)
- **Image encoding strategy**: images ≤ 4× pixel budget are sent as original bytes (PNG/JPEG) — the vLLM server handles resizing at high quality. Images > 4× budget (e.g. magazine scans) are resized client-side to avoid upload timeouts.

## Metrics

### Method 1: Visual Reconstruction (multi-metric composite)

Render OCR bounding boxes back onto a blank canvas and compare with the masked original.

| Component | Weight | Captures |
|-----------|--------|---------|
| SSIM | 40% | Structural / luminance similarity |
| MSE | 30% | Pixel-level difference |
| LPIPS | 30% | Perceptual (deep feature) distance |

Both SSIM and MSE operate on **binarized** images (removes font rendering differences). LPIPS uses continuous grayscale.

### Method 2: CLIP Cosine Similarity

Encode both original and reconstructed images with OpenCLIP (ViT-B/32, LAION-2B). Compute cosine similarity between CLS-token embeddings.

**Known limitation**: Global CLS token tracks layout semantics but is blind to character-level errors, so it can rank stylistically similar but character-wrong reconstructions too highly. Multi-metric (pixel-level) is expected to give better rank alignment.

### Method 3: LM Perplexity

Compute GPT-2 perplexity on the extracted OCR text. Lower perplexity = more natural text = better OCR. Combined with character n-gram uniformity score.

(Prior correlation numbers for all three methods have been removed — dataset under correction.)

## Planned Metric Improvements (Shift-Invariant)

Based on research in `research/position_invariant_similarity.md`, the visual metrics are hurt by positional shifts and font differences. Planned additions:

| Metric | Why | Priority |
|--------|-----|----------|
| **ST-LPIPS** | Drop-in LPIPS replacement tolerant to 1–30px shifts | Immediate |
| **MS-SSIM** | Multi-scale SSIM; drop-in for SSIM | Immediate |
| **Patch-CLIP best-match** | Per-patch cosine with flexible matching, not global CLS | Near-term |
| **DINOv2 dense patch matching** | Fine-grained 14px patch features, ink-weighted | Near-term |
| **TokenCLIP OT** | Optimal transport between CLIP patches and text tokens | Research |

## Comparison Bridge: OmniDocBench

Validates reference-free metrics by correlating against ground-truth metrics on 8 sample pages from the OmniDocBench dataset. Edit distance is the **primary** correlation baseline; CDM and TEDS are computed as diagnostic metrics.

```python
# run_comparison.py computes per page:
text_accuracy    = 1.0 - text_edit_distance      # higher = better
formula_accuracy = 1.0 - formula_edit_distance
table_accuracy   = 1.0 - table_edit_distance
end_to_end_accuracy = 1.0 - end_to_end_edit_distance  # average over present dims
teds = bridge.compute_page_teds(...)   # 0–1, table structure
cdm  = bridge.compute_page_cdm(...)   # 0–1, formula character matching
```

### Why Edit Distance Is the Primary Baseline

1. **Works for all pages** — every OmniDocBench page has text GT; formula/table GT is sparse.
2. **No sparse-data bias** — with 8 pages, most have no formula or table GT, making CDM/TEDS correlation unreliable.
3. **CDM is expensive** — pdflatex invocation per formula pair; infeasible at scale.
4. CDM and TEDS are better used on content-type-specific subsets once more annotated pages are available.

### Edit Distance Calculation

Edit distance is computed at **element level** using Hungarian text-similarity matching, mirroring OmniDocBench's methodology.

**Step 1 — Load GT elements** (`omnidocbench_bridge.py`):
- Read `layout_dets` from the OmniDocBench JSON for the page
- Merge elements linked by `relation_type == "truncated"` (sort by `order`, concatenate text)
- Keep only the 5 text categories scored by OmniDocBench: `text_block`, `title`, `code_txt`, `code_txt_caption`, `reference`
- Sort remaining elements by `order` field

**Step 2 — Load OCR elements** (`run_comparison.py`):
- Prefer `ocr_elements.json` / `ocr_formula_elements.json` / `ocr_table_elements.json` artifacts
- Fall back to parsing `ocr_html.html` with `QwenVLHTMLParser` when JSON artifacts are missing

**Step 3 — Text normalization** (both GT and OCR, per element):
```python
clean_string(textblock2unicode(text))
# textblock2unicode: converts $...$ and \(...\) inline LaTeX to Unicode via pylatexenc,
#                    strips escape chars (\, _, &, %, ^)
# clean_string:      strips tabs/newlines, keeps only \w and CJK (\u4e00-\u9fff)
```

**Step 4 — Hungarian matching**:
- Build cost matrix: `cost[i][j] = Levenshtein(clean(gt_i), clean(ocr_j)) / max(len(gt_i), len(ocr_j))`
- Solve with `scipy.optimize.linear_sum_assignment` for globally optimal assignment
- Pairs with cost > 0.7 treated as unmatched (GT element scored against empty string)

**Step 5 — Page-level aggregation**:
```python
edit_distance = sum(Levenshtein(gt_i, ocr_matched_i) for all GT elements)
              / sum(max(len(gt_i), len(ocr_matched_i)) for all GT elements)
```
Unmatched GT elements contribute their full length to the denominator and their full character count to the numerator.

**End-to-end score** (per page) is `avg(text_accuracy, CDM, TEDS)` over the dimensions that have GT content on the page. CDM and TEDS are `None` when no formula/table GT exists — excluded, not zeroed. OmniDocBench uses a fixed `/3` denominator but only at the dataset level (macro-average over pages that have each content type). Using fixed `/3` per page would suppress text-only pages to `text_acc / 3`, destroying the correlation signal.

**Why Hungarian over IoU spatial matching**: Our Qwen OCR groups paragraphs into column-level elements (one OCR block may cover many GT paragraphs). IoU matching fails in this case — a narrow GT paragraph has near-zero IoU with a tall OCR column block. Hungarian text-similarity matching handles granularity mismatches by finding the globally optimal text-based pairing regardless of bbox sizes.

**Polarity**: `ocr_accuracy = 1 - edit_distance` so that all correlation values are positive for well-performing metrics.

### CDM (Character Detection Matching)

Implemented in `comparison/cdm_scorer.py` — a faithful port of the OmniDocBench CDM algorithm.

**Algorithm:**
1. Tokenize GT and OCR LaTeX strings (split on whitespace and LaTeX commands)
2. Assign a unique RGB color to each token
3. Wrap each token in `\textcolor[RGB]{r,g,b}{token}` and compile with pdflatex + pdftoppm at 400 DPI
4. Extract per-token bounding boxes by finding pixels matching each unique color
5. Match GT↔OCR token sets using Hungarian assignment on centroid distance, then RANSAC for geometric consistency
6. F1 score over inlier matches = CDM score

**Key implementation details:**
- Math delimiters (`$...$`, `$$...$$`, `\[...\]`) are stripped before rendering (displaymath environment is used instead)
- DPI must be 400+ to avoid anti-aliasing color bleeding that destroys exact pixel matching
- skimage 0.26+ `ransac()` does not accept `random_state` parameter — omit it

**Public API**: `compute_cdm_pair(gt_latex: str, ocr_latex: str) -> float | None`

### TEDS (Tree Edit Distance Score)

Computed by `compute_page_teds()` in `omnidocbench_bridge.py` using the APTED algorithm on HTML parse trees.

```python
teds = 1 - apted_distance(gt_html_tree, ocr_html_tree) / max_nodes
```

Fractional rename costs apply to `<td>` content differences (text similarity rather than binary match), matching OmniDocBench's implementation.

## Experiment Tracking: MLflow

- **Backend**: Local SQLite database (`mlruns.db`)
- **Artifact Store**: Local filesystem (`mlartifacts/`)
- **UI**: `mlflow ui` for local visualization

## Directory Structure

```
src/reference_free_ocr_metric/
├── ocr/
│   └── qwen_client.py              # Qwen3-VL OpenAI-compatible client
├── reconstruction/
│   ├── html_parser.py              # Parse QwenVL HTML → ParsedDocument
│   ├── image_renderer.py           # Render text/formula/table elements
│   ├── image_preprocessor.py       # Mask image/table regions in original
│   └── document_analyzer.py        # Font selection, alignment detection
├── metrics/
│   ├── multi_metric/
│   │   └── visual_reconstruction.py  # SSIM+MSE+LPIPS composite
│   ├── clip_compare/
│   │   └── clip_similarity.py        # OpenCLIP cosine similarity
│   ├── vlm_compare/
│   │   └── vlm_similarity.py         # VLM-as-judge similarity
│   └── lm_perplexity/
│       └── perplexity_scorer.py      # GPT-2 perplexity + n-gram
├── comparison/
│   ├── omnidocbench_bridge.py      # Correlation with reference metrics
│   └── cdm_scorer.py               # CDM: per-token RGB colorization + Hungarian/RANSAC
└── experiment/
    └── tracker.py                  # MLflow tracking

scripts/
├── run_experiment.py               # Run OCR + all metrics on sample pages
└── run_comparison.py               # Correlate with OmniDocBench accuracy

results/
└── sample_pages/
    ├── results.json                # Per-page metric scores
    └── <page_name>/
        ├── ocr_html.html                # Raw Qwen HTML output (text + bboxes)
        ├── ocr_text.txt                 # Plain concatenated text (human-readable)
        ├── ocr_elements.json            # Text elements [{text, bbox}] for text edit distance
        ├── ocr_formula_elements.json    # Formula elements [{text, bbox}] for CDM / formula edit distance
        ├── ocr_table_elements.json      # Table elements [{html, bbox}] for TEDS / table edit distance
        ├── reconstructed.png
        ├── masked_original.png
        └── bbox_visualization.png

data/
└── annotations/
    └── sample_pages_annotations.json  # OmniDocBench ground truth for 8 pages
```
