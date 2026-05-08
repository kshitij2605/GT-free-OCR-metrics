---
license: cc-by-nc-4.0
language:
  - en
  - zh
task_categories:
  - other
tags:
  - ocr
  - logprobs
  - confidence-estimation
  - render-and-compare
  - reference-free-metric
multilinguality:
  - multilingual
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files: "logprobs-*.parquet"
---

# OmniDocBench Qwen OCR Logprobs — Parquet Edition

Parquet-shard repackaging of
[`gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs`](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs).

The original dataset stores per-token and per-bbox logprob artifacts as
~4,065 individual files (3 files × 1,355 pages). Bulk download trips the
HuggingFace 5,000 resolver-cache requests / 5-min limit, the same problem
the render-compare dataset has.

This repackaging stores the same artifacts as **7 zstd-compressed parquet
shards (~115 MB total)**, which downloads in 7 HTTP requests instead of
4,065 — well inside the unauthenticated quota. Round-trip extraction back
to the original per-page directory layout is byte-identical.

## Overview

For each of the 1,355 OmniDocBench pages, this dataset contains the full
token-level log-probability stream emitted by Qwen3.5-122B-A10B during OCR
inference, plus a per-bbox aggregation of those logprobs. These are
intended as a **reference-free confidence signal** that can be combined
with the visual-similarity metrics in
[`gt-free-ocr-metrics/omnidocbench-render-compare-parquet`](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare-parquet).

## Schema (one row per page)

| Column | Parquet type | Description |
|---|---|---|
| `page_id` | string | OmniDocBench page identifier |
| `ocr_html` | string | HTML rendered from the logprobs run (matches the per-page run that produced the token stream) |
| `ocr_logprobs` | string | JSON: full token stream with top-N logprobs per token (≈100–500 KB per page typical) |
| `per_bbox_logprobs` | string | JSON: per-bbox aggregated entropy/logprob features computed from the token stream |

`page_id` matches the identifiers used in OmniDocBench.json and in the
render-and-compare dataset.

## Usage

### Streaming via `datasets` library

```python
from datasets import load_dataset
import json
ds = load_dataset("gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs-parquet", split="train")
row = ds[0]
print("page:", row["page_id"])

logprobs = json.loads(row["ocr_logprobs"])
print("first 3 tokens:")
for t in logprobs[:3]:
    print(f"  {t['token']!r}  lp={t['logprob']:.3f}  top={t['top_logprobs'][:3]}")
```

### Materialising back to per-page directory layout

To use these alongside the official methods which read per-bbox JSONs:

```bash
git clone https://anonymous.4open.science/r/GT-free-OCR-metrics-E0A9
cd GT-free-OCR-metrics
bash download_data.sh    # default downloads + extracts both datasets
```

`scripts/extract_logprobs_parquet.py` writes:
- `data/ocr_logprobs/<page_id>/{ocr_html.html, ocr_logprobs.json}`
- `data/ocr_logprobs_per_bbox/<page_id>/per_bbox_logprobs.json`

## Dataset Structure

```
logprobs-shard0000.parquet  (200 rows)
logprobs-shard0001.parquet  (200 rows)
logprobs-shard0002.parquet  (200 rows)
logprobs-shard0003.parquet  (200 rows)
logprobs-shard0004.parquet  (200 rows)
logprobs-shard0005.parquet  (200 rows)
logprobs-shard0006.parquet  (155 rows)
```

Total: 1,355 pages across 7 shards.

## Intended Use

- **Reference-free OCR confidence estimation**: aggregated bbox-level logprobs
  give an OCR-internal signal of certainty that is independent of visual
  similarity. The two signals are complementary.
- **Multi-signal metric design**: combine logprob entropy with visual
  similarity to build hybrid metrics that beat either signal alone.
- **OCR uncertainty studies**: per-token top-N logprobs let you study where
  the OCR model is hesitant and how that correlates with downstream errors.

## Out-of-Scope Use

- These logprobs are model-specific (Qwen3.5-122B-A10B) and **not** a
  ground-truth confidence signal — they reflect a single model's distribution
  and cannot be transferred directly to another OCR system.
- Not a substitute for human review of OCR output quality.

## Limitations and Biases

- **Single OCR model**: all logprobs are from Qwen3.5-122B-A10B. The
  miscalibration patterns, hallucination tendencies, and formula vs. text
  confidence biases of this specific model are baked into the data.
- **Top-N truncation**: per-token `top_logprobs` is capped at the model's
  serving-time `top_logprobs` setting (typically 5); the full vocabulary
  distribution is not available.
- **Per-bbox aggregation choices**: `per_bbox_logprobs` aggregations
  (entropy, min/mean logprob, etc.) are pre-computed; if you need a different
  aggregation, recompute from `ocr_logprobs`.
- **Element sparsity inheritance**: per-bbox features are absent for pages
  with no detected formula or table content where the corresponding render
  variant excluded them.

## Sensitive and Personal Information

The logprobs reflect tokens produced by a model reading public OmniDocBench
documents. No personally identifiable information was deliberately collected;
content sensitivity inherits from OmniDocBench.

## Source Dataset

Derived from
[OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)
(OpenDataLab / SJTU, CC-BY-NC-4.0). OCR inference was performed by
Qwen3.5-122B-A10B served via vLLM.

## License

CC-BY-NC-4.0, inherited from OmniDocBench.
