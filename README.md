# GT-Free OCR Metrics

**Evaluate OCR quality without ground-truth text** using a render-and-compare visual
similarity framework. The pipeline renders OCR output back to an image and measures
how closely it matches the original page scan — no transcriptions needed.

Validated on [OmniDocBench](https://arxiv.org/abs/2412.07626) (1 355 pages, EN/ZH,
10 document categories) against reference-based edit distance.

**Best single-metric result:** Spearman ρ ≈ 0.49 (element-patch SSIM + text coverage,
`P1_137_elem_p10_all`).

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
git clone https://github.com/PLACEHOLDER/gt-free-ocr-metrics
cd gt-free-ocr-metrics
pip install -e .

# 2. Download pre-computed data (~50 GB)
bash download_data.sh

# 3. Download DocSim LoRA weights
bash download_models.sh

# 4. Run a method against all variants
bash scripts/run_method.sh P1_137_elem_p10_all

# 5. View the leaderboard
python scripts/show_leaderboard.py
```

See **[REPRODUCING.md](REPRODUCING.md)** for a full step-by-step reproduction guide.

---

## Methods

109 method implementations are included in `methods/` + `scripts/methods/`.
See [methods/README.md](methods/README.md) for a categorized index with headline scores.

Key result families:

| Category | Best method | Spearman (`all` variant) |
|---|---|---|
| Element-patch SSIM | `P1_137_elem_p10_all` | **0.493** |
| DocSim LoRA (CLIP+DINOv2) | `P1_084` series | ~0.46 |
| ST-LPIPS shift-tolerant | `P1_082_st_lpips` | ~0.44 |
| OCR log-probabilities | `P1_107c` series | ~0.45 |
| Baseline (SSIM+MSE+LPIPS) | `baseline` | ~0.40 |

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

methods/                         # 109 method spec YAML files
scripts/methods/                 # 109 method Python implementations

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
| Render-and-Compare | Pre-computed OCR artifacts for all 5 variants | [puku128/omnidocbench-render-compare](https://huggingface.co/datasets/puku128/omnidocbench-render-compare) |
| OCR Log-probabilities | Token-level Qwen OCR confidence scores | [puku128/omnidocbench-qwen-ocr-logprobs](https://huggingface.co/datasets/puku128/omnidocbench-qwen-ocr-logprobs) |

---

## Citation

```bibtex
@misc{gtfreeocr2025,
  title   = {GT-Free OCR Metrics: Reference-Free Evaluation via Render-and-Compare},
  author  = {[Authors]},
  year    = {2025},
  note    = {Preprint},
}
```

---

## License

Code: **Apache 2.0** — see [LICENSE](LICENSE).
Datasets: **CC-BY-NC-4.0** — see [NOTICE](NOTICE) and individual dataset cards.
