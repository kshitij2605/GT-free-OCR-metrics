# Correlation Metrics: Pearson vs Spearman

## What We Are Correlating

Reference-free metric scores (CLIP cosine, multi-metric composite, LM perplexity) are correlated against **OCR accuracy** (`1 - edit_distance`) from OmniDocBench ground truth. A correlation close to +1.0 means the reference-free metric reliably tracks OCR quality without needing ground truth.

## Pearson Correlation

Measures **linear** relationship. Both variables must move together at a constant rate across the full range.

```
OCR accuracy:  0.5  →  0.6  →  0.7  →  0.8  →  0.9
CLIP score:    0.4  →  0.5  →  0.6  →  0.7  →  0.8
                     +0.1 each step  ← linear
```

When Pearson is high, the relationship is proportional: a 0.1 increase in the reference-free score predicts a consistent 0.1 increase in OCR accuracy. This lets you:
- Draw a regression line and map any reference-free score to a predicted OCR accuracy
- Make absolute quality statements: "a CLIP score of 0.7 corresponds to ~80th percentile OCR quality"
- Threshold on scores: "reject pages with predicted accuracy below 0.7"

**Weakness with small samples**: a single outlier page can pull Pearson significantly, because it uses the raw values (not ranks).

## Spearman Correlation

Measures **monotonic** relationship — whether the ordering is correct, regardless of the size of the gaps.

```
OCR accuracy:  0.5  →  0.6  →  0.7  →  0.8  →  0.9
CLIP score:    0.4  →  0.41 →  0.60 →  0.61 →  0.80
                     uneven jumps ← monotonic but not linear
```

The ranking is correct (higher CLIP = higher accuracy) but the gaps between scores are not meaningful. A jump from 0.4 to 0.6 in CLIP might mean a tiny accuracy improvement or a large one.

Spearman converts both variables to ranks first, so one outlier page only shifts one rank — much more robust with small samples (like our 8 pages).

## Which Is More Important for This Project

**Spearman is the primary metric.** The core question is ranking: "does this page have better OCR than that one?" That is a monotonic ordering question. We do not currently need to make calibrated absolute predictions.

Use Pearson as a secondary check: if Pearson and Spearman diverge significantly for a metric, it signals that a small number of pages are driving the apparent correlation rather than the metric working consistently.

## When Pearson Would Become Primary

Pearson matters if we eventually want **calibrated scores** — mapping a reference-free score directly to a predicted OCR accuracy value. This would be needed for:
- Setting a hard rejection threshold (e.g., "discard any page with predicted accuracy < 0.70")
- Reporting absolute quality estimates to users
- Comparing scores across different document collections with different baseline difficulty

Until then, Spearman is the right anchor for metric selection and ranking.

## Current Results

Prior correlation results have been removed — the evaluation dataset is being corrected and metrics will be re-run before being reported here.

Reference baseline: **text edit distance** (Hungarian element-level matching, `1 - edit_distance` for polarity).

## CDM and TEDS as Diagnostic Metrics

CDM (Character Detection Matching) and TEDS (Tree Edit Distance Score) are now computed by `run_comparison.py` alongside edit distance, but are **not used as primary correlation targets**. Reasons:

1. **Sparse signal**: most of the 8 sample pages lack formula or table GT content. Pages without formula GT return CDM = 1.0 (default), making correlation meaningless.
2. **CDM cost**: pdflatex invoked per formula pair — prohibitively slow for large datasets.
3. **Small N**: with only ~2–3 pages having formula/table content, Pearson/Spearman computed on this subset are statistically unreliable.

CDM and TEDS results are saved in `results/comparison_results.json` under `matched_pairs[*].cdm` and `matched_pairs[*].teds` for manual inspection. Once more annotated pages are available, they can be used to evaluate formula and table OCR quality separately.

## Polarity Convention

All correlations in this project use `ocr_accuracy = 1 - edit_distance` as the reference, so:
- **Positive correlation** = metric correctly tracks OCR quality
- **Negative correlation** = metric is inversely related or uncorrelated
- **Near zero** = metric provides no signal

Never correlate raw edit distance (lower = better) against higher-is-better metrics — the resulting negative coefficient for a good metric is misleading.
