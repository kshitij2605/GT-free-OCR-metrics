# Dataset Plan

## Primary Dataset: OmniDocBench

### Overview
OmniDocBench is a comprehensive document understanding benchmark containing 1355+ pages of diverse document types with detailed annotations including ground-truth text for edit-distance evaluation.

### Location
```
<omnidocbench-data-dir>/
```

### Document Types
- **Text blocks**: Paragraphs, columns, captions
- **Tables**: Simple and complex tables with various structures
- **Formulas**: Inline and display math expressions
- **Figures**: Charts, diagrams, photographs
- **Headings**: Section headers at various levels

---

## Ground Truth JSON Structure

Each page in the dataset is represented as a JSON object:

```json
{
    "page_info": {
        "image_path": "path/to/image.jpg",
        "page_attribute": {
            "text_language": "text_english",
            "text_background": "white",
            "page_difficulty": "medium"
        }
    },
    "layout_dets": [
        {
            "category_type": "title",
            "poly": [x1, y1, x2, y2, x3, y3, x4, y4],
            "ignore": false,
            "order": 1,
            "anno_id": 2,
            "text": "- Human Factors",
            "attribute": {
                "text_language": "text_english",
                "text_background": "white",
                "text_rotate": "normal"
            }
        }
    ],
    "extra": {
        "relation": [
            {
                "relation_type": "truncated",
                "source_anno_id": 2,
                "target_anno_id": 4
            }
        ]
    }
}
```

Key fields:
- **`category_type`** — one of 17 types (see below)
- **`order`** — integer reading order (sequential from 1)
- **`text`** — raw ground-truth text (normalization applied at evaluation time)
- **`poly`** — 8 coordinates defining a quadrilateral bounding box
- **`ignore`** — boolean flag for elements to skip
- **`relation`** — cross-element links (e.g. truncated blocks that must be merged)

---

## Category Types

All 17 `category_type` values in `layout_dets`:

| # | Category | Description |
|---|----------|-------------|
| 1 | `text_block` | Main body text |
| 2 | `title` | Document/section titles |
| 3 | `code_txt` | Inline source code |
| 4 | `code_txt_caption` | Code snippet captions |
| 5 | `reference` | Bibliographic references |
| 6 | `equation_caption` | Captions for equations |
| 7 | `figure_caption` | Figure captions |
| 8 | `figure_footnote` | Figure footnotes |
| 9 | `table_caption` | Table captions |
| 10 | `table_footnote` | Table footnotes |
| 11 | `code_algorithm` | Algorithm pseudocode |
| 12 | `code_algorithm_caption` | Algorithm captions |
| 13 | `header` | Page headers |
| 14 | `footer` | Page footers |
| 15 | `page_footnote` | Page-level footnotes |
| 16 | `page_number` | Page numbers |
| 17 | `equation_isolated` | Standalone display equations |

**Categories used for final text edit-distance scoring (5):**

```python
{"text_block", "title", "code_txt", "code_txt_caption", "reference"}
```

The other 12 categories (figure_caption, figure_footnote, table_caption, table_footnote, code_algorithm, code_algorithm_caption, header, footer, page_footnote, page_number, equation_caption, equation_isolated) are loaded during matching but filtered out before final scoring (source: `end2end_dataset.py` line 333).

---

## Text Normalization Pipeline

Normalization is category-dependent and applied before edit distance computation.

### Text categories (`text_block`, `title`, etc.)

```python
# clean_string(textblock2unicode(text))
# Step 1: textblock2unicode() — converts inline LaTeX to Unicode, removes escape chars (\, _, &, %, ^)
# Step 2: clean_string() — removes all whitespace/tabs, keeps only alphanumeric + Chinese (\u4e00-\u9fff)
cleaned = re.sub(r'[^\w\u4e00-\u9fff]', '', text.replace('\t','').replace('\n',''))
```

Result: no spaces, no punctuation — only alphanumeric and CJK characters.

### Formula category (`equation_isolated`)

```python
# normalized_formula(text)
# Strips $, $$, \[, \] delimiters
# Removes 44 formatting macros: \mathbf, \mathrm, \text, \quad, \qquad, etc.
# Removes \tag{}, \hspace{}, \begin{}, \end{}, \arraycolsep
# Lowercases entire formula
# Preserves spaces (unlike text normalization)
```

### Table category

```python
# normalized_html_table() via BeautifulSoup:
# - Converts <th> → <td>
# - Removes style/height/width/align/class attributes
# - Collapses multiple spaces to single
# - Normalizes Unicode (NFKC)
# - Removes <sup>, <sub>, <span>, <div>, <p> tags
```

---

## Edit Distance Formula

Source: `<omnidocbench-data-dir>/metrics/cal_metric.py` lines 139–184

```python
edit_dist = Levenshtein.distance(pred, gt)
normalized = edit_dist / max(len(pred), len(gt))
```

**Three aggregation levels:**

| Level | Formula |
|-------|---------|
| **Sample-level** | `edit_dist / max(len(pred), len(gt))` per element |
| **Page-level** | `sum(edits_for_page) / sum(max_lens_for_page)` |
| **Corpus-level** | `sum(all_edits) / sum(all_max_lens)` |

Our `run_comparison.py` uses sample-level, averaged across matched pairs per page.

**Polarity**: lower edit distance = better OCR. In `run_comparison.py` this is converted to `ocr_accuracy = 1 - edit_distance` so that all correlations are positive for well-performing reference-free metrics.

---

## Truncated Text Merging

Some text blocks span multiple layout elements (e.g. a paragraph split across columns). The `relation` field marks these with `relation_type == 'truncated'`.

Merging logic (source: `end2end_dataset.py` lines 58–91):
1. Collect all elements linked via `truncated` relations
2. Sort by `order` field
3. Concatenate text in order
4. Merged block inherits `order`, `category_type`, and `anno_id` from the first block

Our bridge (`omnidocbench_bridge.py`) does **not** currently handle truncated merging — a known gap.

---

## Reading Order Evaluation

OmniDocBench includes a separate reading order metric (not used in our correlation analysis):
- Each matched element pair has `gt_position` (index in GT list) and `pred_position` (index in pred list)
- GT indices are reordered according to the prediction sequence
- Edit distance is computed on the two index sequences, normalized by max-length
- Result is a 0–1 score where 0 = perfect reading order

Source: `end2end_dataset.py` lines 125–143

---

## Our Bridge vs. Full OmniDocBench

| Aspect | OmniDocBench full pipeline | Our `omnidocbench_bridge.py` |
|--------|---------------------------|------------------------------|
| Text categories | 15 loaded → 5 scored | 5 loaded and scored |
| Normalization | `clean_string(textblock2unicode(...))` | Same (matches exactly) |
| Truncated merging | Yes, via `relation` field | Not implemented |
| Reading order eval | Yes, separate metric | Not implemented |
| Edit distance normalization | max-length | max-length (matches) |
| Aggregation | Sample / page / corpus | Sample-level average per page |

---

## OmniDocBench Codebase File Index

| Purpose | File |
|---------|------|
| Edit distance computation | `metrics/cal_metric.py` lines 139–184 |
| Dataset loading (end-to-end) | `dataset/end2end_dataset.py` |
| Recognition evaluation | `task/recognition_eval.py` |
| Qwen inference | `tools/model_infer/Qwen3-VL-235B_img2md.py` |
| Markdown extraction pipeline | `utils/extract.py` function `md_tex_filter` lines 111–392 |
| Text normalization | `utils/data_preprocess.py` |
| Ground truth demo JSON | `demo_data/omnidocbench_demo/OmniDocBench_demo.json` |

All paths relative to `<omnidocbench-data-dir>/`.

---

## Sample Pages Subset (Active Development)

### Purpose
8 representative pages stored locally for rapid development and testing.

### Location
```
results/sample_pages/<page_name>/
```

Each page directory contains:
- `ocr_html.html` — raw Qwen3-VL OCR output (HTML with bboxes)
- `ocr_text.txt` — plain extracted text (used for edit-distance and perplexity)
- `reconstructed.png` — rendered reconstruction image
- `masked_original.png` — original with text regions masked
- `bbox_visualization.png` — bbox overlay on original

Experiment results JSON: `results/sample_pages/results.json`

### The 8 Sample Pages

| Page | Type | Characteristics |
|------|------|-----------------|
| `book_en_6.Complex.Analysis...page_051` | Academic math | Dense LaTeX formulas |
| `book_en_40.Puzzles.and.Problems...page_064` | Academic text | Mixed text + figures |
| `color_textbook_教材全解...page_067` | Chinese textbook | CJK text, mixed layout |
| `eastmoney_...pdf_1` | Financial report | Tables, Chinese text |
| `exam_paper_en-file-putnam...page_004` | Math exam | Heavy formula density |
| `magazine_TheEconomist...page_029` | Magazine | Multi-column, justified text |
| `newspaper_354720d9...` | Newspaper | Dense two-column layout |
| `PPT_lay_linalg5_01_05_page_004` | Presentation slide | Sparse layout, large text |

### Annotations
Ground-truth annotations for the 8 sample pages are stored at:
```
data/annotations/sample_pages_annotations.json
```
This is a subset of the full OmniDocBench JSON, pre-extracted for the 8 pages above.

---

## Usage Plan

1. **Development**: `results/sample_pages/` for rapid iteration (8 pages, all 8 matched to OmniDocBench)
2. **Correlation Analysis**: `scripts/run_comparison.py` — computes Pearson/Spearman between reference-free scores and OCR accuracy (`1 - edit_distance`) from OmniDocBench ground truth
3. **Validation**: Run full pipeline on complete OmniDocBench dataset (1355+ pages) before publication

---

## Current Correlation Results

Prior correlation results have been removed — the evaluation dataset is being corrected and metrics will be re-run before being reported here. Methodology (Hungarian text-similarity matching, `1 - edit_distance` for polarity) is unchanged.
