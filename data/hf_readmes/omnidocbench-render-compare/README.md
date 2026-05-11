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
  - 10K<n<100K
configs:
  - config_name: ocr_all
    data_files:
      - split: train
        path: configs/ocr_all/metadata.jsonl
  - config_name: ocr_all_no_mask
    data_files:
      - split: train
        path: configs/ocr_all_no_mask/metadata.jsonl
  - config_name: ocr_text
    data_files:
      - split: train
        path: configs/ocr_text/metadata.jsonl
  - config_name: ocr_formula
    data_files:
      - split: train
        path: configs/ocr_formula/metadata.jsonl
  - config_name: ocr_table
    data_files:
      - split: train
        path: configs/ocr_table/metadata.jsonl
  - config_name: docsim_triplets
    data_files:
      - split: train
        path: configs/docsim_triplets/metadata.jsonl
---

# OmniDocBench Render-and-Compare

This dataset contains the **rendered HTML reconstructions** and **comparison images** produced
by the OmniDocBench Render-and-Compare pipeline — a reference-free visual similarity evaluation
framework for OCR systems.
It is released as part of the **OmniDocBench Render-and-Compare** research project.

## Overview

The pipeline processes each page of [OmniDocBench](https://arxiv.org/abs/2412.07626) through
a Qwen3.5-122B-A10B OCR model, renders the structured output back to a PNG via HTML
(reconstructed.png), and compares it against the original page scan (masked_original.png)
using reference-free visual metrics.

Five OCR extraction variants are provided, each targeting a different subset of document
element types:

| Variant | Recognised elements | Pages |
|---------|-------------------|-------|
| `ocr_all` | text + formula + table (with region masking) | 1 355 |
| `ocr_all_no_mask` | text + formula + table (no masking) | 1 355 |
| `ocr_text` | text only | 1 349 |
| `ocr_formula` | formula only | 200 |
| `ocr_table` | table only | 351 |

## Dataset Structure

```
<variant>/
  <page_id>/
    masked_original.png     # original scan with non-target regions grayed out
    reconstructed.png       # HTML render of the OCR output (screenshot at 1× DPI)
    ocr_html.html           # raw HTML produced by the OCR model
    ocr_elements.json       # extracted text bboxes + content  (text variants only)
    ocr_formula_elements.json  # extracted formula bboxes + LaTeX (formula variants)
    ocr_table_elements.json    # extracted table bboxes + HTML  (table variants)

docsim_triplets/
  manifest.jsonl            # 20 280 training triplets for the DocSim LoRA head

OmniDocBench.json           # original OmniDocBench ground-truth annotation file (GT text/formula/table)
```

`<page_id>` is the identifier used in `OmniDocBench.json` (e.g.
`book_en_5.Advanced.Modern.Algebra_page_572`).

Not all files are present in every variant:
- `ocr_all_no_mask` pages contain only `masked_original.png`, `ocr_html.html`, and
  `reconstructed.png` (no element JSON files, no bbox visualization).
- Formula/table element JSON files are absent when no such elements were detected on a page.

## DocSim Triplets

`docsim_triplets/manifest.jsonl` is a JSONL file with 20 280 training triplets used to
fine-tune a LoRA-adapted CLIP+DINOv2 similarity head (DocSim).
Each line is a JSON object:

```json
{
  "anchor_path":   "ocr_all_no_mask/<page_id>/masked_original.png",
  "positive_path": "ocr_all_no_mask/<page_id>/reconstructed.png",
  "negative_path": "ocr_all/<other_page_id>/reconstructed.png",
  "anchor_ed":   0.0,
  "positive_ed": 0.12,
  "negative_ed": 0.64,
  ...
}
```

Paths are relative to the root of this repository.
`anchor_ed` / `positive_ed` / `negative_ed` are Hungarian-matched edit distances
(lower = higher text fidelity) used as supervision during training.

## Intended Use

- **Research on reference-free OCR evaluation**: masked_original / reconstructed pairs provide
  ground material for developing visual similarity metrics that do not require OCR ground truth.
- **Document visual quality research**: diverse real-world layouts (books, exams, slides,
  financial reports, scientific papers) in EN and ZH.
- **Training document similarity models**: the DocSim triplet manifest can be used to train or
  evaluate learned perceptual similarity heads for document images.

## Out-of-Scope Use

- The reconstructed images and OCR outputs should **not** be used to extract or reproduce
  copyrighted text for redistribution.
- The dataset is **not** a ground-truth OCR corpus; the `reconstructed.png` images are model
  outputs, not verified transcripts.

## Limitations and Biases

- **Single OCR system**: all reconstructions come from Qwen3.5-122B-A10B.
  Errors and biases specific to this model (e.g. hallucinations, formula mis-renders) are
  present in the reconstruction side of every comparison pair.
- **Language distribution**: OmniDocBench contains English and Chinese pages; the ratio of
  each language is determined by the original benchmark.
- **Element sparsity**: formula and table variants cover only pages with relevant ground-truth
  content; pages with no formulas or tables are absent from those variants.
- **Render fidelity**: HTML-to-PNG rendering uses a headless browser at a fixed viewport;
  fonts, rendering artefacts, and DPI settings may differ from the original scans.
- **No human verification**: all OCR outputs and rendered images are machine-generated;
  no manual quality checks were performed.

## Sensitive and Personal Information

OmniDocBench draws from public academic and professional documents.
No personally identifiable information (PII) was deliberately collected.
Financial report pages may incidentally reference company names or executives.
No medical, legal, or biometric data is present.

## Source Dataset

This dataset is derived from
[OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)
(OpenDataLab / SJTU, CC-BY-NC-4.0).
The `OmniDocBench.json` annotation file is reproduced here for user convenience;
original page scans are available in the source dataset.

## Citation

If you use this dataset, please cite:

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

Released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The original OmniDocBench page images (`images/`) retain their source licensing terms;
users must comply with those when using the raw scans.
