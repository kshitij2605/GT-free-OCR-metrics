# Reference-Free OCR Metrics: Inherent Limitations of OCR-Bbox Masking

## Context

Our reference-free pipeline evaluates OCR quality on per-component variants
(`ocr_all`, `ocr_text`, `ocr_formula`, `ocr_table`) by:

1. **Masked image** — original page with everything except the variant of
   interest masked out, using **OCR-derived bboxes** (production signal).
2. **Reconstructed image** — re-rendering of OCR output, restricted to the
   variant of interest.
3. **Score** — visual similarity between masked and reconstructed (CLIP
   cosine, SSIM+MSE+LPIPS composite, etc.).

The reference-free constraint forbids using ground-truth bboxes for masking —
that would smuggle GT into the metric and defeat the purpose.

## The Question

For `ocr_formula`: if a page has multiple formulas and OCR misses some,
won't the reference-free metric score artificially high while the
reference-based formula edit distance / CDM score is bad — dragging the
correlation down?

## Three Failure Modes for Any Variant

Take a single formula on a page and trace it through the pipeline:

### Case 1 — OCR completely fails to detect the formula

- No bbox is emitted in any category.
- Mask policy keeps the region **visible** (it isn't in `image_regs`,
  `table_regs`, `text_regs`, or formula list).
- Reconstruction renders nothing in that region.
- Visual diff: **high** (original ink vs blank reconstruction).
- Reference-based: **bad** (formula missing from OCR list).
- → Both metrics agree. Correlation preserved. ✓

### Case 2 — OCR detects but misclassifies the formula (e.g. as text)

- Bbox lands in `text_regs`.
- For `ocr_formula`: mask policy is "image + table + text → mask, formulas →
  keep". So the formula's bbox gets **masked away** (treated as text).
- Reconstruction (formula-only) skips the element (it's tagged `text`,
  not `formula`).
- Both images: **blank** in that spot. → Visual similarity stays **high**.
- Reference-based: **bad** (Hungarian match fails to pair this GT formula).
- → Metrics **disagree**. Page contributes a (high RF score, low RB score)
  pair that pulls down Pearson/Spearman. ✗

### Case 3 — OCR detects correctly but renders the LaTeX badly

- Bbox in `formula_regs`, content garbled.
- Masked image: formula visible.
- Reconstructed image: garbled formula at the right position.
- Visual diff: visible (different glyphs).
- Reference-based: bad (CDM compares LaTeX).
- → Both metrics agree. Correlation preserved. ✓

## The Real Failure Mode

Case 2 is the one that genuinely degrades correlation. It is **not** "OCR
missed the formula" in general — case 1 is fine. It is specifically
**misclassification**, where OCR sees a region but tags it as the wrong
component class.

Multiple misclassified formulas on the same page compound the effect: the
masked image and reconstruction can both lose 2 of 5 formulas, reading as
identical → very high similarity, while reference-based metrics see 2/5 GT
formulas unmatched and score the page poorly.

## Why This Is Inherent, Not a Bug

The reference-free metric only knows about content the OCR system has
**labeled**. If a region of the page never enters the OCR's typed output
under the right class, the metric has no way to notice it's missing — both
sides of the comparison erase it consistently.

GT-bbox masking would fix case 2, but it stops being reference-free.

This means there is an **information-theoretic ceiling** on reference-free
correlation with reference-based scores: pages where OCR systematically
misclassifies will always vote against correlation, regardless of how
sophisticated the visual similarity metric becomes.

## Implications for the Leaderboard

- Visual-similarity ceilings on `ocr_formula` (and similar variants) reflect
  partly metric weakness, partly the irreducible blind spot above.
- Methods that improve glyph-level matching (DINOv3, ST-LPIPS, MS-SSIM) can
  only address case 3 — they cannot recover case-2 information that the
  pipeline has already erased.
- To clear the ceiling we need signal **outside** the OCR labels.

## Concrete Attacks on the Ceiling (Ordered by EV)

### 1. Text-region cross-validation (highest EV, cheap)

For the `ocr_formula` score specifically, add a side signal:

- For every OCR **text** region, render the claimed text at its bbox and
  visually compare to the original crop.
- Aggregate per-page disagreement. High disagreement = OCR's "text"
  doesn't look like text → likely a misclassified formula.
- Final score: `formula_visual_sim × (1 − text_region_disagreement)`.

This directly attacks case 2: the misclassified formula sits in a text
region whose rendered text won't visually match the math glyphs in the
original crop.

**Cost:** no new dependencies, ~2× rendering cost. One new method on the
leaderboard.

### 2. Cross-variant disagreement penalty (almost-free)

We already compute all 4 variants. A page with **high `ocr_all`
dissimilarity** but **high `ocr_formula` similarity** signals "OCR rendered
something wrong, but my formula-only check thinks it's fine" — likely case
2. Subtract a penalty proportional to `(1 − ocr_all_sim)` from
`ocr_formula_sim`.

**Cost:** zero — pure post-hoc combination of existing leaderboard scores.
Worth trying as a one-liner before investing in #1.

### 3. Independent math-region detector (highest ceiling, more work)

Run a lightweight detector (math-symbol density heuristic, or a small
pretrained model like Pix2Text's MFD) over the original page. For each
Qwen text region, compute an "is this actually math?" probability. If high
but Qwen tagged it text → case-2 confirmed.

**Cost:** extra model integration, ~few hours to wire in. Most direct fix
for case 2 but worth doing only if #1 doesn't clear the gap.

### Recommended order

Run **#2 first** (re-aggregation of existing JSON outputs, ~30 min as a
notebook). If the cross-variant disagreement penalty improves Spearman on
`ocr_formula`, that confirms case 2 is the binding constraint and motivates
investing in #1. If #2 doesn't help, the ceiling is something else (metric
weakness on case 3) and effort should focus there instead.

## Other Mitigation Ideas (Brainstorm)

- **Residual-ink detector** — after masking, detect any non-blank ink in
  the masked image that the OCR claims should be blank. Catches case 1
  (full miss) primarily. Cheap.
- **LM perplexity** — for inline formulas misclassified as text, perplexity
  may spike at the boundary where the LaTeX would be and surface the issue
  without bbox supervision.
- **Whole-image visual comparison (no mask)** — accepts layout noise but
  is immune to mask-policy errors. Equivalent to the `ocr_all_no_mask`
  variant. Use as a complementary signal in cross-variant ensembles.

## Bottom Line

The reference-free metric will never perfectly correlate with the
reference-based one. The gap is not (only) metric quality — part of it
**is** the reference-free blind spot. We accept the constraint because the
goal is no-GT evaluation; the architectural fix is supplementing the
single-source (OCR-derived) signal with auxiliary signals that can see what
the OCR system missed.
