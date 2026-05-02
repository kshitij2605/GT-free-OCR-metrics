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
  - logprobs
  - reference-free-metric
  - qwen
  - document-quality
multilinguality:
  - multilingual
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: metadata.jsonl
dataset_info:
  features:
    - name: page_id
      dtype: string
    - name: n_total_tokens
      dtype: int64
    - name: n_bboxes
      dtype: int64
    - name: n_text_bboxes
      dtype: int64
    - name: n_formula_bboxes
      dtype: int64
    - name: n_table_bboxes
      dtype: int64
    - name: logprob_mean
      dtype: float64
    - name: logprob_min
      dtype: float64
    - name: logprob_max
      dtype: float64
    - name: shannon_entropy_mean
      dtype: float64
    - name: shannon_entropy_max
      dtype: float64
    - name: ocr_logprobs_file
      dtype: string
    - name: per_bbox_logprobs_file
      dtype: string
  splits:
    - name: train
      num_examples: 1355
---

# OmniDocBench Qwen OCR Log-Probabilities

This dataset provides token-level and bounding-box-level OCR log-probabilities produced by
running Qwen3.5-122B-A10B (via vLLM) on the original page scans of the
[OmniDocBench](https://arxiv.org/abs/2412.07626) benchmark.
It is a **reference-free** auxiliary signal — no ground-truth text is used — and is released
as part of the **OmniDocBench Render-and-Compare** research project.

## Dataset Structure

```
ocr_logprobs/
  <page_id>/
    ocr_logprobs.json   # full per-token logprobs + top-5 alternatives
    ocr_html.html       # raw HTML output from the OCR model

ocr_logprobs_per_bbox/
  <page_id>/
    per_bbox_logprobs.json  # per-bounding-box aggregated statistics
```

`<page_id>` matches the page identifiers in the OmniDocBench annotation file (`OmniDocBench.json`).
There are **1 355 pages** in each sub-collection, covering the full OmniDocBench benchmark.

### `ocr_logprobs.json` schema

```json
{
  "tokens": [
    {
      "token": "string",
      "logprob": -0.0026,
      "top_logprobs": [
        {"token": "string", "logprob": -0.0026},
        ...
      ]
    },
    ...
  ]
}
```

Each element of `tokens` corresponds to one output token of the Qwen model.
`top_logprobs` lists the five highest-probability alternatives at that position.

### `per_bbox_logprobs.json` schema

```json
{
  "n_total_tokens": 551,
  "n_bboxes": 6,
  "bboxes": [
    {
      "bbox_id": 0,
      "div_class": "text",
      "data_bbox": [x0, y0, x1, y1],
      "n_tokens": 12,
      "stats": {
        "logprob_mean": -0.0098,
        "logprob_min": -0.0674,
        "logprob_max": -1.2e-05,
        "shannon_entropy_mean": 0.047,
        "shannon_entropy_max": 0.288
      }
    },
    ...
  ]
}
```

`div_class` is one of `"text"`, `"formula"`, or `"table"`.
`data_bbox` is in pixels relative to the reconstructed HTML render of the page.

## How the Data Was Generated

1. **Source images**: Original OmniDocBench page scans (`images/<page_id>.png`), **not** any
   reconstructed rendering.
2. **OCR model**: Qwen3.5-122B-A10B served via vLLM with `logprobs=True` and `top_logprobs=5`.
3. **Prompt**: The standard OmniDocBench HTML-output OCR prompt (system role + page image).
4. **Per-bbox mapping**: Tokens are attributed to bounding boxes by matching token offsets
   against the HTML `<div data-bbox="...">` structure produced by the OCR model.
5. **Reference-free**: Only the page image and model outputs are used — no ground-truth OCR text
   or annotations.

## Intended Use

- **Research on reference-free document quality metrics**: the logprob distribution is a
  model-internal signal of OCR difficulty and can serve as a feature for downstream metrics.
- **Study of OCR uncertainty on diverse document types**: books, exam papers, slides, financial
  reports, and academic papers in English and Chinese.

## Out-of-Scope Use

- This dataset should **not** be used to extract or reconstruct the text content of copyrighted
  documents; the source images belong to OmniDocBench's original licensing terms.
- It is **not** a ground-truth OCR dataset; logprobs reflect model uncertainty, not text accuracy.

## Limitations and Biases

- **Model-specific**: logprob magnitudes are calibrated to Qwen3.5-122B-A10B and may not
  transfer to other OCR models.
- **Language imbalance**: OmniDocBench skews toward English documents; Chinese-document
  calibration may differ.
- **Formula/table sparsity**: Many pages contain no formulas or tables; per-bbox stats for
  those element types are absent for such pages.
- **Tokenizer artefacts**: Qwen's tokenizer may split tokens at unexpected boundaries for
  mathematical symbols or CJK characters, affecting per-token statistics.

## Sensitive and Personal Information

OmniDocBench contains scanned document pages from public or semi-public sources
(academic books, exam papers, financial reports, slides).
No personally identifiable information (PII) such as names, addresses, or ID numbers was
intentionally included in the source benchmark.
Users should exercise caution when processing financial-report pages, which may name
individuals or organisations.

## Source Dataset

This dataset derives from
[OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)
(OpenDataLab / SJTU, released under CC-BY-NC-4.0).
Only the original page images were used as input; no annotation files were modified.

## Citation

If you use this dataset, please cite the OmniDocBench paper and our work:

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

This dataset is released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The source OmniDocBench images retain their original licensing terms; users must comply with
those terms when using the images directly.
