# Reproducing Results

This guide describes how to reproduce the results reported in the paper,
from data download through to the final leaderboard.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.9 | Tested on 3.11 |
| CUDA GPU | ≥ 16 GB VRAM recommended for DocSim/DINOv2 methods; CPU fallback available |
| Disk space | ~50–60 GB for data + model weights |
| `uv` or `pip` | We use `uv` in scripts; substitute `pip` if preferred |

---

## Step 1 — Install the package

```bash
git clone https://anonymous.4open.science/r/GT-free-OCR-metrics-E0A9
cd gt-free-ocr-metrics

# With uv (recommended):
pip install uv     # install uv package manager
uv sync            # creates .venv and installs all dependencies

# Or with plain pip:
pip install -e .
```

---

## Step 2 — Download pre-computed data

All OCR outputs are pre-computed and available on HuggingFace.
You do **not** need access to a Qwen inference endpoint.

```bash
bash download_data.sh
```

This downloads (default — what you need to reproduce the 146 method evaluations):
- `data/omnidocbench/OmniDocBench.json` — ground-truth annotations (~65 MB, 1 355 pages)
- `data/omnidocbench/ocr_{all,text,formula,table,all_no_mask}/<page_id>/` — per-variant OCR artifacts (~40 GB total):
  `masked_original.png`, `reconstructed.png`, `ocr_html.html`,
  `ocr_elements.json`, `ocr_formula_elements.json`, `ocr_table_elements.json`
- `data/ocr_logprobs/` — per-token log-probabilities (~2 GB)

The original page scans (`data/omnidocbench/images/`, ~1.2 GB) are **not required** for
the 146 method evaluations — the methods read pre-computed `masked_original.png` /
`reconstructed.png` from the render-compare dataset above. The originals are only
needed for the optional "Re-running OCR" flow at the bottom of this guide; opt in with:

```bash
WITH_IMAGES=1 bash download_data.sh
```

**Two data formats are available:**

- **`parquet` (default, recommended)** — zstd-compressed parquet shards. The
  render-compare artifacts ship as 64 shards (~9.4 GB) at
  [`gt-free-ocr-metrics/omnidocbench-render-compare-parquet`](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare-parquet),
  and the OCR logprobs as 7 shards (~115 MB) at
  [`gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs-parquet`](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs-parquet).
  `download_data.sh` fetches the shards and runs the extract scripts to
  materialise them into the same per-page directory layout the methods expect.
  71 HTTP requests instead of ~27,000 — well inside HuggingFace's unauthenticated
  rate limits, no `HF_TOKEN` needed for these two datasets.

- **`raw` (legacy)** — original per-page-directory layouts at
  [`gt-free-ocr-metrics/omnidocbench-render-compare`](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare)
  and
  [`gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs`](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs).
  Triggers `429 Too Many Requests` for unauthenticated users (5,000
  resolver-cache requests / 5-min window vs. ~23,600 + ~4,000 files).
  Supply an HF token to avoid the rate limit. The download is resumable.

> **Note on OmniDocBench (the upstream dataset):** the `opendatalab/OmniDocBench`
> source dataset contains 1,659 individual image files. Even unauthenticated
> users can hit a 1,000-API-requests/5-min limit while listing them.
> `download_data.sh` does not yet repackage this upstream dataset, so this step
> may need a single retry after a 5-min wait. Authenticating with
> `huggingface-cli login` (or `export HF_TOKEN=...`) avoids it entirely.

```bash
# Default (parquet):
bash download_data.sh

# Legacy raw layout:
DATA_FORMAT=raw bash download_data.sh

# With HF auth (helps for raw layout):
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DATA_FORMAT=raw bash download_data.sh
```

> **Quick exploration (no full download required):** a 60-page stratified sample (~370 MB)
> is available at
> [`gt-free-ocr-metrics/omnidocbench-render-compare-sample`](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare-sample).
> Open `explore_datasets.ipynb` in this repository for an interactive walkthrough
> that downloads only the sample data.
>
> **Sampling methodology:** 5 pages drawn uniformly at random from each of 12 document
> categories (slides, book, academic paper, financial, newspaper, exam, textbook/notes,
> notes, magazine, DocStructBench, colorful, other), inferred from `page_id` prefix
> (random seed 42). Up to 5 extra pages added to cover the sparse `ocr_formula` and
> `ocr_table` variants. Full per-category counts are in the sample dataset README.

---

## Step 3 — Download DocSim LoRA weights

DocSim is a LoRA-adapted CLIP + DINOv2 similarity head trained on 20 280 triplets.
Methods in the `P1_084` family and several others depend on it.

```bash
bash download_models.sh
```

Weights are placed in `models/docsim_lora/`.

---

## Step 4 — Run a single method

```bash
# Run the rank-1 method (per-element CLIP at P5 percentile, with table-bbox + text-elem-CLIP fixes):
# (Note: P2_209-P2_213 are §-equivalent at spearman_mean=0.4938 — same per-page scores; see Expected results below.)
bash scripts/run_method.sh P2_210_elem_p05_table_text_elemclip

# Run the baseline:
bash scripts/run_method.sh baseline

# Run on a specific GPU (default: 0):
bash scripts/run_method.sh P1_137_content_elem_p10_all 1
```

Output is written per variant. For each of the 5 variants (`ocr_all`, `ocr_text`, `ocr_formula`, `ocr_table`, `ocr_all_no_mask`), the runner produces three files:

```
results/method_runs/<variant>/<method_id>/results.json       — per-page metric scores (~1,349 entries)
results/method_runs/<variant>/<method_id>/comparison.json    — aggregated comparison data (matched pairs, by-category breakdown)
results/method_runs/<variant>/<method_id>/correlations.json  — Pearson and Spearman correlation summary
```

`correlations.json` contains Pearson and Spearman correlation with edit distance
for the `text`, `formula`, and `table` reference dimensions.

---

## Step 5 — Run all 146 methods

```bash
bash scripts/run_all_correlations.sh
```

This runs all method YAMLs sequentially. On a single RTX 6000 Ada (48 GB),
expect ~25–30 minutes per method (each method runs the full 1,355-page pipeline
across 5 OCR variants), ~50–70 hours total for the full 146-method sweep.
Use the per-GPU split below for parallelism, or run only the top-30 methods
listed in `results/leaderboard.json` for a faster sanity check.

For parallel execution across multiple GPUs, split the method list manually
and specify GPU IDs as the second argument to `run_method.sh`.

---

## Step 6 — Update and view the leaderboard

`update_leaderboard.py` is invoked **once per method** with the method's YAML
(this is also printed at the end of every `run_method.sh` run as `Next: …`):

```bash
# After each run_method.sh, append that method's row to the leaderboard:
python scripts/update_leaderboard.py --technique-yaml methods/<method_id>.yaml

# Then view:
python scripts/show_leaderboard.py
```

`results/leaderboard.json` is updated with the aggregated `spearman_mean`
across all variants. The pre-computed version committed to this repo matches
the results in the paper.

---

## Expected results

The top-5 methods by `spearman_mean` (averaged across all variants):

| Rank | Method | Spearman mean |
|---|---|---|
| 1 | `P2_210_elem_p05_table_text_elemclip` | 0.4938 |
| 2 | `P2_213_elem_p05_table_fixed`         | 0.4938 |
| 3 | `P2_212_dino_elem_p05_formula_fixed`  | 0.4938 |
| 4 | `P2_211_dino_elem_p05_formula`        | 0.4938 |
| 5 | `P2_209_elem_p05_table_elemclip`      | 0.4938 |

(Methods ranked 1-5 are §-equivalent — they share byte-identical per-page scores
to ≥4 decimal places across all 5 variants and differ only in bug-fix or
DINOv2-vs-CLIP implementation alternatives of the same composite stack.
See paper Table 3 for the full top-15.)

Exact values depend on the variant weighting used in `update_leaderboard.py`.
The `results/leaderboard.json` committed to this repo is the authoritative reference.

---

## Step 7 — Character-level sensitivity experiment (paper Appendix C)

A controlled perturbation study confirming that the reference-free metric
responds to character-level content, not only to gross page layout. Bounding
boxes and the rendering pipeline are held fixed; only the displayed
characters change across the five conditions.

```bash
# Run the perturbation sweep (~10 minutes on a single RTX 6000 Ada).
# Samples 100 OmniDocBench pages, corrupts OCR text at 0/5/10/20/50%
# character fractions, and computes per-element CLIP-cosine P10
# against the cached masked_original.
CUDA_VISIBLE_DEVICES=0 uv run python scripts/charsens_perturbation.py
# Writes results/charsens/results.json
```

The rendered figure (Appendix E in the paper) is already included in the
released PDF; rebuilding it requires the paper-side build script which is
kept locally (not in this code repository).

Expected output (seed 42, 100 pages):

| Corruption fraction | Mean per-element CLIP P10 | 95% bootstrap CI |
|---|---|---|
| 0% | 0.557 | [0.527, 0.584] |
| 5% | 0.543 | [0.515, 0.568] |
| 10% | 0.538 | [0.509, 0.563] |
| 20% | 0.517 | [0.487, 0.544] |
| 50% | 0.491 | [0.463, 0.516] |

The strict monotonic drop is evidence that the reference-free metric measures
character-level fidelity rather than only document-level structure, since
layout and rendering are identical across the five points on the curve.

---

## Re-running OCR (optional)

Pre-computed artifacts are sufficient for all 146 methods.
If you want to re-run OCR inference with your own model endpoint:

1. Copy `.env.example` to `.env` and fill in `OCR_ENDPOINT_URL`.
2. Run: `python scripts/run_omnidocbench_ocr.py`

This requires Qwen3.5-122B-A10B (or a compatible vision-language model) via an OpenAI-compatible API.

---

## Training DocSim from scratch (optional)

```bash
# 1. Build triplets from the render-compare dataset
python scripts/build_docsim_triplets.py

# 2. Train LoRA head (~2 hours on 1x A100)
python scripts/train_docsim.py
```

The trained weights should be placed in `models/docsim_lora/`.

---

## Troubleshooting

**`ModuleNotFoundError: reference_free_ocr_metric`**
→ Run `pip install -e .` from the repo root.

**CUDA out of memory**
→ Most methods run fine on 8 GB. DocSim/DINOv2 methods need ~12 GB.
  Use `--gpu_id N` to select a device with more VRAM.

**`FileNotFoundError: data/omnidocbench/OmniDocBench.json`**
→ Run `bash download_data.sh` first.

**`FileNotFoundError: models/docsim_lora/`**
→ Run `bash download_models.sh` first (required only for DocSim methods).
