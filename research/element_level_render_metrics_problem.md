# Element-Level Render Comparison — Design Notes & Limitations

> Proposed approach for per-element visual quality assessment of OCR output.
> Deferred: not implemented due to fundamental measurement issues documented below.

---

## Motivation

Full-page visual metrics (SSIM, LPIPS, CLIP cosine on full `reconstructed.png` vs `masked_original.png`) work well for text-heavy pages but are a poor signal for formula- and table-heavy pages because:

- Formula/table pages are mostly white after masking non-target regions
- A single misrendered element is diluted across a large white background
- The metric rewards correct layout (white space) rather than element fidelity

The natural fix is to evaluate element-by-element rather than full-page.

---

## Proposed Approach

### Pipeline

1. **OCR side** — for each OCR-detected element on the page:
   - Formula: render the OCR latex string (`ocr_formula_elements.json[i]["text"]`) on a fresh white background sized to the OCR-detected bbox dimensions, using pdflatex (clean render, no CDM colorization)
   - Table: render the OCR HTML (`ocr_table_elements.json[i]["html"]`) on a white background at OCR bbox dimensions using weasyprint

2. **GT side** — for each GT element annotation in `OmniDocBench.json`:
   - Crop the original document image at the GT `poly` bbox (`equation_isolated` for formula, `table` for table)

3. **Matching** — pair OCR renders to GT crops by Hungarian assignment on centroid distances (one cost matrix per page, same algorithmic principle as CDM's token-level matching)

4. **Comparison** — SSIM / LPIPS / CLIP cosine on matched pairs, averaged per page

5. **Correlation** — correlate per-page scores against GT-based baselines (CDM, TEDS, edit distance) to assess metric validity

### Infrastructure available
- pdflatex rendering: reusable from `comparison/cdm_scorer.py` (`_TEX_TEMPLATE`, `_render_to_png`), dropping the token colorization step
- weasyprint: installed (`v68.1`), Cairo/Pango system libs confirmed present
- Element files: `ocr/<page>/ocr_formula_elements.json` and `ocr_table_elements.json` both carry `{"text": ..., "bbox": [x1,y1,x2,y2]}` and `{"html": ..., "bbox": [...]}` respectively
- Original images: `data/omnidocbench/images/<page>.png`, 1355 pages

---

## Why This Was Not Implemented — Core Limitations

### 1. Conflation of layout detection and element fidelity (formula & table)

The match step uses OCR-detected bboxes (from `ocr_formula_elements.json`) to locate where the OCR "found" each element, and GT bboxes (from `OmniDocBench.json`) to crop what the element looks like in the original document.

This means the metric is simultaneously measuring **two distinct things**:

- **Layout accuracy**: did the OCR detect the element at approximately the right position? A positional mismatch causes a bad centroid match, which degrades the score even if the OCR text itself is perfect.
- **Element fidelity**: does the rendered OCR output visually match the original element? This is the signal we actually want.

These two signals cannot be disentangled with this design. A system that detects formulas in wrong positions but transcribes them perfectly will score poorly. A system that detects positions correctly but transcribes poorly will also score poorly — but for a different reason. The metric cannot distinguish between the two failure modes.

In production (without GT annotations), layout detection quality further contaminates the score because you would use OCR bboxes on both sides, making the crop of the original image dependent on layout detection accuracy.

### 2. OCR span granularity mismatch (text)

For text elements the same approach would be:
- Crop original at GT text span bbox
- Render OCR text span on white background
- Compare

This breaks because different OCR systems output text at different granularities:
- Some output line-level spans
- Some output paragraph-level spans
- Some output word-level spans

OmniDocBench GT annotations are at **line level**. An OCR system that outputs paragraph-level spans will never produce a 1-to-1 match with line-level GT spans. Hungarian matching on centroids partially compensates but a paragraph OCR span centroid will match at most one line GT span, leaving all other GT lines unmatched and scored as zero — penalising paragraph-level OCR systems regardless of their actual text quality.

This granularity sensitivity makes element-level comparison unsuitable as a universal reference-free text metric and would introduce systematic bias favouring OCR systems whose granularity happens to match the benchmark annotations.

### 3. Rendering fidelity as a confound (formula)

Rendering OCR latex on a white background introduces its own fidelity ceiling: pdflatex rendering quality, font availability, and unsupported packages all affect how the rendered image looks, independent of whether the OCR correctly transcribed the formula. A correctly transcribed formula that uses a LaTeX package not available in the render environment will produce a degraded or failed render, scoring poorly despite correct transcription.

---

## Relationship to CDM

CDM (Character Detection Matching) addresses element fidelity at the token level and is agnostic to layout position — it directly compares rendered GT latex vs rendered OCR latex without any bbox matching. CDM is therefore the cleaner signal for formula fidelity. The element-render approach described here would add visual SSIM/CLIP on top, but CDM already captures the core quality signal for formulas and requires GT latex strings (reference-based), which is acceptable for validation experiments.

The element-render approach adds nothing for validation that CDM does not already provide, while adding substantial complexity and the confounds described above.

---

## Conclusion

Full-page visual metrics remain the recommended approach for this pipeline. The position-invariance limitation (GT bbox ≠ OCR render position when cropping `reconstructed.png`) is better addressed by shift-invariant image similarity metrics (ST-LPIPS, MS-SSIM, DINOv2 patch matching) rather than element-level decomposition. See `research/position_invariant_similarity.md`.
