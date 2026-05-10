# GT-Free OCR Metrics

**Evaluate OCR quality without ground-truth text** using a render-and-compare visual
similarity framework. The pipeline renders OCR output back to an image and measures
how closely it matches the original page scan — no transcriptions needed.

Validated on [OmniDocBench](https://arxiv.org/abs/2412.07626) (1 355 pages, EN/ZH,
9 document categories) against reference-based edit distance.

**Best composite result:** Spearman ρ = 0.494 (`P2_210`, a five-mechanism composite stacking DocSim + Shannon-entropy + IQ + SSIM + per-element CLIP; mean across all 5 OCR-output variants).

---

## How It Works

```
Original image ──► Qwen OCR ──► HTML with bounding boxes
                                          │
             ┌────────────────────────────┤
             │ Mask non-target regions    │ Parse element bboxes
             ▼                            ▼
   masked_original.png       Render to reconstructed.png
             │                            │
             └──────── Compare ───────────┘
                   (SSIM · patches, LPIPS, CLIP,
                    DINOv2, logprobs, coverage, …)
```

Five OCR extraction variants:

| Variant | Elements extracted | Pages |
|---|---|---|
| `all` | text + formula + table (masked) | 1 355 |
| `all_no_mask` | text + formula + table (unmasked) | 1 355 |
| `text` | text only | 1 349 |
| `formula` | formula only | 200 |
| `table` | table only | 351 |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://anonymous.4open.science/r/GT-free-OCR-metrics-E0A9
cd gt-free-ocr-metrics
pip install -e .

# 2. Download pre-computed data (~50 GB)
bash download_data.sh

# 3. Download DocSim LoRA weights
bash download_models.sh

# 4. Run a method against all variants
bash scripts/run_method.sh P1_137_content_elem_p10_all

# 5. View the leaderboard
python scripts/show_leaderboard.py
```

See **[REPRODUCING.md](REPRODUCING.md)** for a full step-by-step reproduction guide.

---

## Methods

146 method implementations are included in `methods/` + `scripts/methods/`.
See [methods/README.md](methods/README.md) for a categorized index with headline scores.

Key result families (mean Spearman across 5 OCR-output variants, computed via paper Eq. (1)):

| Category | Best method | Spearman mean |
|---|---|---|
| Per-element CLIP P5 + table-bbox/text-elem-CLIP fixes (rank 1, paper Table 3) | `P2_210_elem_p05_table_text_elemclip` | **0.494** |
| Per-element CLIP P10 (Phase-1 anchor) | `P1_137_content_elem_p10_all` | 0.493 |
| Per-table-cell DocSim supplement (β=0.30) | `P1_107c_beta_table_cell_30` | 0.482 |
| DINOv2/CLIP encoder fusion (clip\_cosine slot) | `P1_084c_dinov2_clip_avg` | 0.368 |
| Baseline (per-variant best of CLIP cosine and SSIM+MSE+LPIPS multi-composite) | `baseline` | 0.339 |
| ST-LPIPS shift-tolerant (refuted as LPIPS replacement) | `P1_082_st_lpips` | 0.319 |

---

## Robustness check: character-level sensitivity

A controlled perturbation experiment (paper Appendix C) verifies that the
reference-free metric responds to **character-level** content, not just to
gross page layout. Holding bounding boxes and the rendering pipeline fixed,
we corrupt OCR text at increasing character edit distances and measure the
per-element CLIP cosine (P10 aggregation, the dominant signal in the top
methods). The score drops monotonically with the corruption fraction:

| Corruption | Mean P10 | 95% bootstrap CI |
|---|---|---|
| 0% | 0.557 | [0.527, 0.584] |
| 5% | 0.543 | [0.515, 0.568] |
| 10% | 0.538 | [0.509, 0.563] |
| 20% | 0.517 | [0.487, 0.544] |
| 50% | 0.491 | [0.463, 0.516] |

(`n = 100` OmniDocBench pages, seed 42, ~10 min on one RTX 6000 Ada).
Reproduction instructions are in [REPRODUCING.md § Step 7](REPRODUCING.md).

---

## Repository Structure

```
src/reference_free_ocr_metric/   # Python package (metrics, parsers, renderers)
  ocr/                           # Qwen OCR client
  reconstruction/                # HTML parser, image renderer, preprocessor
  metrics/                       # 10+ metric families
  comparison/                    # Edit distance, correlation bridge
  experiment/                    # MLflow experiment tracker

scripts/
  run_method.sh                  # Entry point: run one method, all variants
  run_comparison.py              # Correlate method scores with edit distance
  update_leaderboard.py          # Aggregate results/leaderboard.json
  show_leaderboard.py            # Print ranked leaderboard
  train_docsim.py                # Train the DocSim LoRA head
  build_docsim_triplets.py       # Build training triplets from render-compare pairs
  ...

methods/                         # 146 method spec YAML files
scripts/methods/                 # 146 method Python implementations

results/
  leaderboard.json               # Final ranked results
  method_runs/                   # Per-variant, per-method output artifacts

data/hf_readmes/                 # HuggingFace dataset cards (Croissant 1.1)
documentation/                   # Architecture, dataset, and metric documentation
research/                        # Literature review and technical notes
```

---

## Datasets

| Dataset | Description | HF link |
|---|---|---|
| OmniDocBench | 1 355 real-world document pages, GT annotations | [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) |
| Render-and-Compare (parquet, **recommended**) | 64 zstd-parquet shards (~9.4 GB) of all 5 variants - one HTTP request per shard, no rate limits | [gt-free-ocr-metrics/omnidocbench-render-compare-parquet](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare-parquet) |
| Render-and-Compare (raw per-page) | Original per-page directory layout | [gt-free-ocr-metrics/omnidocbench-render-compare](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare) |
| Render-and-Compare (sample) | 60-page stratified sample (~370 MB) for quick exploration | [gt-free-ocr-metrics/omnidocbench-render-compare-sample](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare-sample) |
| OCR Log-probabilities (parquet, **recommended**) | 7 zstd-parquet shards (~115 MB), token-level Qwen OCR confidence scores | [gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs-parquet](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs-parquet) |
| OCR Log-probabilities (raw per-page) | Original per-page directory layout | [gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs) |

**Collection page:** <https://huggingface.co/collections/gt-free-ocr-metrics/gt-free-ocr-metrics-datasets-and-models>

A **60-page stratified sample** of the full render-and-compare dataset (~370 MB) is available for reviewers who want to explore the data without downloading the full 10 GB dataset. An interactive exploration notebook (`explore_datasets.ipynb`) is included in this repository.

**How the sample was created:** pages were selected by stratified random sampling. Each of the 1 355 pages in `ocr_all` was assigned to one of 12 document categories (slides, book, academic paper, financial, newspaper, exam, textbook/notes, notes, magazine, DocStructBench, colorful, other) inferred from its `page_id` prefix. Five pages were drawn uniformly at random from each category (random seed 42), giving a 60-page base sample representative of the full distribution. Up to 5 additional pages were added to ensure the sparse `ocr_formula` and `ocr_table` variants each appear at least 5 times in the sample. Full per-category and per-variant counts are documented in the [sample dataset README](https://huggingface.co/datasets/gt-free-ocr-metrics/omnidocbench-render-compare-sample).

---

## License

Code: **Apache 2.0** — see [LICENSE](LICENSE).
Datasets: **CC-BY-NC-4.0** — see [NOTICE](NOTICE) and individual dataset cards.
