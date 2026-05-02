# Known Problems with Current Metrics

> Tracking document for issues, confounds, and failure modes identified in each metric.
> Each section covers one metric as used in `scripts/compute_variant_metrics.py` + `scripts/run_comparison.py`.

---

## 1. SSIM / MSE (`multi_composite`)

### 1a. Binarization does not remove font differences

**Problem**: The binarization step (`adaptive_binarize`) was intended to remove font rendering differences between the original document and the reconstruction, so SSIM would measure text placement rather than font style. This reasoning is wrong.

Binarization converts the image to pure black-on-white by thresholding, which removes soft antialiasing gradients at character edges. But it does **not** remove differences in glyph shapes. Different fonts (e.g. Times New Roman in the original vs the reconstruction's render font) have different stroke widths, proportions, serifs, and character outlines. After binarization, these shape differences remain as different black pixel patterns.

**Consequence**: SSIM on binarized images still conflates two distinct failure modes:
- The OCR transcribed the wrong text
- The OCR used a different font than the original document

The metric cannot distinguish between them. A perfectly correct transcription in a different font will score lower than it should.

**Status**: No fix applied. Potential directions: font-agnostic feature comparison (e.g. skeleton/stroke-based), or accepting font sensitivity as a known limitation since reconstruction always uses a fixed font.

---

### 1b. Full-page comparison dilutes signal for formula/table variants

**Problem**: For `ocr_formula` and `ocr_table` variants, pages are masked so only formula or table regions are visible — the rest is white. Formulas occupy on average ~5 crops per page at small sizes; tables occupy ~1.5 regions. The white background dominates the image area, so SSIM/MSE on the full page is dominated by white-on-white agreement rather than element fidelity.

**Consequence**: `multi_composite` Pearson correlation drops significantly for crop-focused variants compared to full-text pages. In the OmniDocBench experiment, `multi_composite` became **negatively correlated** when tested on formula/table crops, indicating the white background is actively inverting the signal.

**Status**: Documented. The crop-based approach was investigated and rejected (see `research/element_render_metrics.md`). Shift-invariant metrics are the recommended next step (see `research/position_invariant_similarity.md`).

---

## 2. LPIPS (`multi_composite`)

### 2a. Full-page white background dilution (same as 1b)

LPIPS operates on continuous grayscale (no binarization) at 512×512. For formula/table variants, the same white-background dilution applies — the learned perceptual distance is dominated by the large white regions rather than the small formula/table areas.

### 2b. Sensitivity to font rendering without binarization

Unlike SSIM, LPIPS is applied to continuous grayscale without binarization. It is therefore directly sensitive to both glyph shape differences and antialiasing differences between fonts. This compounds the font-sensitivity problem noted in 1a.

---

## 3. CLIP Cosine (`clip_cosine`)

### 3a. Preprocessing may suppress useful signal

Before passing to CLIP, images are converted to grayscale, sharpened (`ImageFilter.SHARPEN`), and autocontrasted. CLIP was pretrained on natural colour images. Forcing grayscale + aggressive autocontrast moves the distribution away from pretraining data and may degrade embedding quality for document images.

**Status**: No ablation done. Unknown how much this affects correlation.

### 3b. Full-page white background dilution (same as 1b)

CLIP operates on the full page before its own 224×224 resize. For formula/table variants, the embedding is dominated by the large white background. The ViT-B/32 patch size is 32px — at 224×224 the patches are coarse and formula/table regions may collapse into only a few patches.

### 3c. Cross-modal sensitivity gap

CLIP cosine correlation with formula/table GT metrics is consistently lower than `multi_composite` for full-page comparisons (e.g. `clip_cosine` Pearson +0.29 vs `multi_composite` Pearson +0.32 for formula). CLIP may not be well suited to detecting subtle formula transcription errors since it was not trained on document/formula similarity tasks.

---

## 4. LM Composite (`lm_composite`)

### 4a. Constant output — always NaN correlation

**Problem**: `lm_composite` scores are constant across all pages in every variant, causing `scipy.stats.pearsonr` and `spearmanr` to return `nan` with a `ConstantInputWarning`. Correlation is undefined when one input array has zero variance.

**Root cause**: The `lm_perplexity` scores stored in `results.json` are copied from the base `ocr/results.json` (not recomputed per variant), so all pages in all variants carry the same LM scores regardless of which elements were masked. If the base results also have low variance (e.g. LM scores cluster tightly), the correlation is undefined.

**Consequence**: `lm_composite` is effectively non-functional as a correlation signal. All reported Pearson/Spearman values for `lm_composite` are `nan` and should be ignored.

**Status**: Not fixed. Requires investigation into whether LM scores are actually being computed per variant, or whether the issue is low natural variance in the LM signal across OmniDocBench pages.

---

## 5. General / Cross-Cutting Issues

### 5a. Position mismatch for crop-based comparisons

When cropping `reconstructed.png` at GT bboxes, the OCR renderer places content at OCR-detected positions, not GT positions. This causes crops at GT coordinates to often sample white regions, inverting the correlation. See `research/element_render_metrics.md` for full analysis.

### 5b. OCR span granularity mismatch (text)

OmniDocBench GT text annotations are line-level. OCR systems vary in output granularity (line, paragraph, word). This makes element-level comparison unreliable for text. See `research/element_render_metrics.md`.

### 5c. Correlation polarity: edit distance requires inversion

Raw edit distance (lower = better) must be converted to `1 - edit_distance` before computing Pearson/Spearman so that positive correlation = good metric. This is applied correctly in `run_comparison.py` but is a non-obvious convention that must be preserved in any new correlation scripts.
