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
git clone https://github.com/PLACEHOLDER/gt-free-ocr-metrics
cd gt-free-ocr-metrics

# With uv (recommended):
pip install uv
uv pip install -e .

# Or with pip:
pip install -e .
```

---

## Step 2 — Download pre-computed data

All OCR outputs are pre-computed and available on HuggingFace.
You do **not** need access to a Qwen inference endpoint.

```bash
bash download_data.sh
```

This downloads:
- `data/omnidocbench/OmniDocBench.json` — ground-truth annotations (1 355 pages)
- `data/omnidocbench/images/` — original page scans (~30 GB)
- `data/omnidocbench/ocr/` — pre-computed OCR artifacts for all 5 variants (~18 GB):
  `masked_original.png`, `reconstructed.png`, `ocr_html.html`,
  `ocr_elements.json`, `ocr_formula_elements.json`, `ocr_table_elements.json`
- `data/omnidocbench/ocr_logprobs/` — per-token log-probabilities (~2 GB)

**Note on HuggingFace auth:** if the datasets are gated, set `HF_TOKEN` first:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
bash download_data.sh
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
# Run the best-performing method (element-patch SSIM + text coverage β=0.10):
bash scripts/run_method.sh P2_213_elem_p05_table_fixed

# Run the baseline:
bash scripts/run_method.sh baseline

# Run on a specific GPU (default: 0):
bash scripts/run_method.sh P1_137_elem_p10_all 1
```

Output is written to:
```
results/method_runs/ocr_all/<method_id>/results.json
results/method_runs/ocr_all/<method_id>/correlations.json
results/method_runs/ocr_text/<method_id>/correlations.json
results/method_runs/ocr_formula/<method_id>/correlations.json
results/method_runs/ocr_table/<method_id>/correlations.json
```

`correlations.json` contains Pearson and Spearman correlation with edit distance
for the `text`, `formula`, and `table` reference dimensions.

---

## Step 5 — Run all 147 methods

```bash
bash scripts/run_all_correlations.sh
```

This runs all method YAMLs sequentially. On a single RTX 6000 Ada (48 GB),
expect ~2–5 minutes per method, ~4–8 hours total.

For parallel execution across multiple GPUs, split the method list manually
and specify GPU IDs as the second argument to `run_method.sh`.

---

## Step 6 — Update and view the leaderboard

```bash
python scripts/update_leaderboard.py
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
| 1 | `P2_213_elem_p05_table_fixed` | 0.4938 |
| 2 | `P2_210_elem_p05_table_text_elemclip` | 0.4938 |
| 3 | `P1_137_content_elem_p10_all` | 0.4932 |
| 4 | `P1_150_elem_p15` | 0.4930 |
| 5 | `P1_136_ssim_all_beta20` | 0.4928 |

Exact values depend on the variant weighting used in `update_leaderboard.py`.
The `results/leaderboard.json` committed to this repo is the authoritative reference.

---

## Re-running OCR (optional)

Pre-computed artifacts are sufficient for all 147 methods.
If you want to re-run OCR inference with your own model endpoint:

1. Copy `.env.example` to `.env` and fill in `OCR_ENDPOINT_URL`.
2. Run: `python scripts/run_omnidocbench_ocr.py`

This requires a Qwen2.5-72B-Instruct or compatible model via an OpenAI-compatible API.

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
