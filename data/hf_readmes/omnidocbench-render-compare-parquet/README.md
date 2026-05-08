---
license: cc-by-nc-4.0
language:
  - en
  - zh
task_categories:
  - other
tags:
  - ocr
  - document-understanding
  - render-and-compare
  - reference-free-metric
  - document-quality
  - visual-similarity
multilinguality:
  - multilingual
size_categories:
  - 1K<n<10K
configs:
  - config_name: ocr_all
    data_files: "ocr_all-*.parquet"
  - config_name: ocr_text
    data_files: "ocr_text-*.parquet"
  - config_name: ocr_formula
    data_files: "ocr_formula-*.parquet"
  - config_name: ocr_table
    data_files: "ocr_table-*.parquet"
  - config_name: ocr_all_no_mask
    data_files: "ocr_all_no_mask-*.parquet"
---

# OmniDocBench Render-and-Compare — Parquet Edition

Parquet-shard repackaging of
[`gt-free-ocr-metrics/omnidocbench-render-compare`](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare).

The original dataset stores per-page artifacts as ~23,600 individual files
(6 files × 5 OCR variants × 200–1,355 pages). Bulk download of that layout
trips HuggingFace's 5,000 resolver-cache requests / 5-minute rate limit and
needs an authenticated token to complete.

This repackaging stores the same artifacts as **64 zstd-compressed parquet
shards (~9.4 GB total)**, which downloads in 64 HTTP requests instead of
23,600 — well inside the unauthenticated quota. Round-trip extraction back
to the original per-page directory layout is byte-identical.

## Overview

The pipeline processes each page of [OmniDocBench](https://arxiv.org/abs/2412.07626)
through a Qwen3.5-122B-A10B OCR model, renders the structured output back
to a PNG via HTML (`reconstructed`), and compares it against the original
page scan (`masked_original`) using reference-free visual metrics.

Five OCR extraction variants are provided, each targeting a different subset
of document element types:

| Config | Recognised elements | Pages | Shards |
|---|---|---|---|
| `ocr_all` | text + formula + table (with region masking) | 1,355 | 19 |
| `ocr_all_no_mask` | text + formula + table (no masking) | 1,355 | 19 |
| `ocr_text` | text only | 1,349 | 18 |
| `ocr_formula` | formula only | 200 | 3 |
| `ocr_table` | table only | 351 | 5 |

## Schema (one row per page)

| Column | Parquet type | Description |
|---|---|---|
| `page_id` | string | OmniDocBench page identifier |
| `variant` | string | `ocr_all` / `ocr_text` / `ocr_formula` / `ocr_table` / `ocr_all_no_mask` |
| `masked_original` | binary | PNG bytes — original page with non-target regions masked |
| `reconstructed` | binary | PNG bytes — rendered reconstruction from OCR HTML |
| `ocr_html` | string | HTML rendered from OCR elements |
| `ocr_elements` | string | JSON list of detected text elements with bboxes |
| `ocr_formula_elements` | string | JSON list of detected formulas with bboxes + LaTeX |
| `ocr_table_elements` | string | JSON list of detected tables with bboxes + HTML |

## Usage

### Streaming via `datasets` library

```python
from datasets import load_dataset
ds = load_dataset("gt-free-ocr-metrics/omnidocbench-render-compare-parquet",
                  "ocr_all", split="train")
print(ds[0]["page_id"])

# Image bytes are straight PNG content
from PIL import Image
import io
img = Image.open(io.BytesIO(ds[0]["masked_original"]))
img.show()
```

### Materialising back to per-page directory layout

To run the official methods which read on-disk PNG/JSON files, materialise
the parquet shards into the original per-page layout:

```bash
git clone https://anonymous.4open.science/r/GT-free-OCR-metrics-E0A9
cd GT-free-OCR-metrics
bash download_data.sh    # default: parquet (this dataset) → extracts on-disk
```

`scripts/extract_parquet_to_disk.py` writes
`data/omnidocbench/ocr_<variant>/<page_id>/{masked_original.png, reconstructed.png, ocr_html.html, ocr_elements.json, ocr_formula_elements.json, ocr_table_elements.json}`.

## Dataset Structure

Files in this repository:

```
ocr_all-shard0000.parquet ... ocr_all-shard0018.parquet               (19 shards)
ocr_all_no_mask-shard0000.parquet ... ocr_all_no_mask-shard0018.parquet  (19 shards)
ocr_text-shard0000.parquet ... ocr_text-shard0017.parquet              (18 shards)
ocr_formula-shard0000.parquet ... ocr_formula-shard0002.parquet        (3 shards)
ocr_table-shard0000.parquet ... ocr_table-shard0004.parquet            (5 shards)
```

Each shard contains 75 rows except the last shard of each variant which may
contain fewer. `page_id` matches `OmniDocBench.json` identifiers (e.g.
`book_en_5.Advanced.Modern.Algebra_page_572`).

## Intended Use

- **Reference-free OCR evaluation research**: `masked_original` /
  `reconstructed` PNG pairs provide ground material for developing visual
  similarity metrics that do not require OCR ground truth.
- **Document visual quality research**: diverse real-world layouts (books,
  exams, slides, financial reports, scientific papers) in EN and ZH.
- **Training document similarity models**: pair the rendered images with
  page-level edit-distance ground truth (computed offline from OmniDocBench
  text annotations) to train perceptual similarity heads.

## Out-of-Scope Use

- The reconstructed images and OCR HTML outputs **must not** be used to
  extract or reproduce copyrighted text for redistribution.
- This is **not** a ground-truth OCR corpus; `reconstructed` images are
  model outputs, not verified transcripts.
- Not intended for clinical, legal, or safety-critical applications.

## Limitations and Biases

- **Single OCR model**: all reconstructions are produced by Qwen3.5-122B-A10B.
  Hallucinations, formula mis-renders, and other model-specific errors are
  systematically present across the entire dataset.
- **OCR misclassification ceiling**: when the OCR model misclassifies a region
  (e.g. tags a formula as text), both the masked original and the
  reconstruction erase the region consistently, so visual similarity stays
  artificially high while reference-based metrics correctly mark the page as
  wrong. This Case-2 misclassification creates an information-theoretic
  ceiling on reference-free correlation that cannot be lifted by better
  visual metrics alone.
- **Bounding-box detection noise**: Qwen3.5-122B-A10B detection is imperfect,
  particularly on information-dense pages such as newspapers and dense
  multi-column financial layouts; this lowers correlations with ground-truth
  metrics for those categories.
- **Element sparsity**: `ocr_formula` covers only the 200 pages where the OCR
  model detected at least one formula; `ocr_table` covers only 351 such
  pages. The other variants cover all 1,355 pages.
- **Render fidelity**: HTML→PNG rendering uses a headless browser at a fixed
  viewport; fonts, rendering artefacts, and DPI may differ from the original
  scans.
- **Language distribution**: OmniDocBench contains English and Chinese pages
  only. English documents are more numerous than Chinese.
- **No human verification**: all OCR outputs and rendered images are
  machine-generated; no manual quality checks were performed.

## Sensitive and Personal Information

OmniDocBench draws from public academic and professional documents.
No personally identifiable information (PII) was deliberately collected.
Financial report pages may incidentally reference company or executive names.
No medical, legal, or biometric data is present.

## Source Dataset

This dataset is derived from
[OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)
(OpenDataLab / Shanghai Jiao Tong University, CC-BY-NC-4.0).

## Citation

```bibtex
@article{omnidocbench2024,
  title   = {OmniDocBench: Benchmarking Document Parsing with Diverse Layouts on Real-World Data},
  author  = {Hu, Linke and others},
  journal = {arXiv},
  year    = {2024},
  eprint  = {2412.07626}
}
```

## License

Released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/),
inherited from the OmniDocBench source license.
