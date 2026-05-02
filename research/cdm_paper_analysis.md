# CDM Paper Analysis: Character Detection Matching for OCR Evaluation

## Source
https://arxiv.org/html/2409.03643v2

**Full title:** "Image Over Text: Transforming Formula Recognition Evaluation with Character Detection Matching"
**Authors:** Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Botian Shi, Bo Zhang, Conghui He (Shanghai AI Lab / Shanghai Jiao Tong University)
**Code:** https://github.com/opendatalab/UniMERNet/tree/main/cdm

---

## Overview

CDM addresses a fundamental flaw in LaTeX formula recognition evaluation: the same mathematical expression can be written in many syntactically different but visually identical ways. Metrics like BLEU and Edit Distance compare raw text strings and therefore penalize semantically correct predictions that merely differ in LaTeX style. CDM bypasses this by operating entirely at the rendered-image level — both the predicted LaTeX and the ground-truth LaTeX are rendered into images, and character-level matching is performed on the visual output. The key claim is: if two rendered images look the same, the prediction is correct, regardless of the underlying LaTeX syntax.

The paper reports 96% consistency between CDM scores and human judgments, versus substantially lower consistency for BLEU and Edit Distance. On the UniMER-Test benchmark (23,757 formula samples), CDM correctly ranks models by visual quality where BLEU gives contradictory rankings due to training-distribution bias.

---

## Core Algorithm

CDM operates in four sequential stages:

**Stage 1 — Token-level colorization and rendering.**
The LaTeX source is first normalized — composite structures like `\frac ab` are decomposed into `\frac {a} {b}`, and each atomic token (numbers, Greek letters, operators, commands like `\sin`, `\alpha`) is assigned a unique RGB color from a fixed palette. The palette uses 15-unit intervals from (0,0,15) to (255,255,255), yielding 5,832 distinct colors. The coloring is applied via `\mathcolor[RGB]{r,g,b}` commands wrapping each token. The colored LaTeX is then rendered into an image.

**Stage 2 — Bounding box extraction.**
After rendering, pixel analysis finds all pixels matching each assigned color and computes bounding boxes `[x1, y1, x2, y2]` for every colored token. This produces two sets of elements: ground truth elements `y` and predicted elements `ŷ`, each with an associated token identity and spatial bounding box.

**Stage 3 — Bipartite matching via Hungarian algorithm.**
Predicted elements are matched to ground-truth elements by minimizing a combined cost across three terms:

```
L_match = W_t × L_t  +  W_p × L_p  +  W_o × L_o
```

- **L_t (token cost):** 0 for identical tokens; 0.05 for visually identical but syntactically different tokens (e.g., `(` vs `\left(`; `\leq` vs `\le`); 1 for different tokens.
- **L_p (positional proximity):** L1 norm between bounding box coordinates, normalized by a dimension constant `D_b`. Encodes spatial closeness.
- **L_o (order similarity):** Normalized L1 distance between token sequence positions. Penalizes matches where tokens appear in very different positions in the expression.

The Hungarian algorithm finds the global minimum-cost assignment. Note: the actual numerical weights W_t, W_p, W_o are not disclosed in the paper.

**Stage 4 — RANSAC outlier elimination and metric calculation.**
After the initial bipartite matching, RANSAC validates the matches by estimating an affine transformation `b̂_s(i) = A(b_i)` between ground-truth and predicted bounding boxes. Matched pairs that are inconsistent with this global transformation are discarded as outliers. The rotation component of the affine transform is fixed at 0° (formulas are horizontal), so only translation and scaling are estimated. After filtering, F1-score and ExpRate@CDM are computed:

- **F1-score:** Standard precision/recall F1 over matched vs unmatched elements.
- **ExpRate@CDM:** `sum(I(CDM_i = 1)) / N` — the fraction of formulas where every element is perfectly matched (CDM = 1.0). This is the strictest measure.

If LaTeX fails to render entirely, CDM = 0 for that sample.

---

## Formula/Equation Handling

Formula handling is the entire focus of the paper. Key design decisions:

- **Rendering-first philosophy:** Instead of comparing LaTeX strings, the system compares what the LaTeX *looks like*. This correctly handles the vast diversity of equivalent LaTeX representations.
- **Decomposition of nested structures:** Fractions, subscripts, and superscripts are decomposed during normalization so each atomic character token gets its own color and bounding box. The positional cost then naturally encodes the spatial relationships (e.g., a subscript token appears below and to the right of its base character).
- **Near-identical rendering tolerance:** L_t = 0.05 (rather than 0 or 1) for visually identical but syntactically different tokens handles cases like `\leq` vs `\le`, partially rewarding syntactic diversity without full penalty.
- **Rendering failure penalty:** Models that fail to produce valid LaTeX receive CDM = 0. Rendering success rates vary by model (Pix2tex: 96.6%, UniMERNet: 99.7%), and the paper treats this as a useful quality signal in itself.
- **Acknowledged edge case:** When different LaTeX commands render identically (e.g., `\mathcal{E}` vs `\varepsilon`) or when the same command renders differently in different contexts, CDM can err. These account for approximately 4% of failure cases.

---

## Document Element Handling

CDM is explicitly scoped to mathematical formulas only. The paper does not address:
- Plain text evaluation
- Table structure recognition
- Figure/image comparison
- Mixed-layout document evaluation

The document-level work (Tiny-Doc-Math dataset: 12 PDFs, 196 pages, 437 formulas) extracts only the formula regions for evaluation, ignoring all other document content. The extraction is performed via regular expressions on LaTeX source. Non-formula text quality is not evaluated.

For context: the paper notes that pixel-level image metrics (MSE, SSIM) are inappropriate for document formulas because "even slight character misalignments can result in significant penalties" — a sub-pixel shift in rendering makes SSIM collapse even when the formula is semantically identical. CDM is position-tolerant because it matches individual tokens spatially rather than comparing raw pixel grids.

---

## Metrics and Computation

The paper benchmarks CDM against four baseline metrics:

| Metric | Type | Key weakness |
|--------|------|-------------|
| BLEU | N-gram text overlap | Training-distribution bias; rewards style-matching over correctness |
| Edit Distance | Character-level string edit | Penalizes valid LaTeX variants; large distances for equivalent formulas |
| ExpRate | Exact string match | Coarsest baseline; binary pass/fail on full string |
| MSE/SSIM | Pixel-level image | Extreme sensitivity to rendering alignment; not semantically meaningful |

CDM results on UniMER-Test (four model tiers: SPE, CPE, SCE, HWE categories):
- Pix2tex: CDM 0.636
- Texify: CDM 0.755
- Mathpix: CDM 0.951
- UniMERNet: CDM 0.968

The critical finding is that BLEU ranks Mathpix above UniMERNet on Screenshot Expressions (BLEU: 0.8182 vs 0.8018) while CDM correctly reverses this (CDM: 0.9461 vs 0.9041 in favor of UniMERNet). The BLEU inversion occurs because Mathpix's training data distribution matches the test set's LaTeX style conventions — BLEU rewards style consistency rather than correctness.

Stability test: 50 formulas were each rewritten 5 times by GPT-4 into equivalent LaTeX. CDM assigned 1.0 to all variants; BLEU scores dispersed widely. This is the strongest argument in the paper for CDM's robustness.

Dataset-efficiency experiment: Models trained on only 18% of UniMER-1M (selected using CDM as a hard-case filter) matched full-dataset performance, suggesting CDM-guided curriculum selection is valuable for training.

---

## Key Insights

1. **Text metrics are fundamentally wrong for formula evaluation.** The LaTeX symbol space is too large and too ambiguous for string-matching metrics to be reliable. BLEU, edit distance, and ExpRate all encode syntactic style preferences rather than semantic correctness.

2. **Rendering first, then match.** By moving evaluation to the image domain, CDM sidesteps all LaTeX representation issues. This is a clean architectural insight that generalizes: any structured markup language with multiple valid surface forms (HTML, Markdown, LaTeX) should ideally be evaluated on rendered output rather than raw text.

3. **Character-level spatial matching beats pixel grids.** Direct pixel comparison (MSE/SSIM) is too sensitive to sub-pixel shifts and font rendering artifacts. Extracting individual character bounding boxes and matching them spatially is more robust and more semantically interpretable.

4. **Position + identity + order together.** None of the three cost components (L_t, L_p, L_o) alone is sufficient. Token identity tells you what the character is; positional proximity tells you where it sits in the layout; order similarity tells you whether the reading sequence is preserved. All three are needed for reliable matching.

5. **RANSAC for global consistency.** After local bipartite matching, RANSAC adds a global geometric consistency check. Locally plausible matches that contradict the global affine transform (translation + scaling of the whole expression) are rejected. This is a robust way to eliminate false positive matches in dense expressions.

6. **Rendering failures are informative.** A model that produces invalid LaTeX should be penalized beyond just mis-matching — CDM = 0 for rendering failures makes the metric reflect actual usability.

---

## Relevance to Our Project

CDM is specifically about formula evaluation and does not directly address the reconstruction-based evaluation approach this project uses. However, several insights apply:

**On the limits of pixel-level comparison (SSIM/MSE):** CDM's rejection of SSIM for formula evaluation is directly relevant. Our reconstruction pipeline computes SSIM + MSE between the reconstructed text image and the original. CDM shows these metrics are fragile when content is not pixel-perfectly aligned. If our OCR shifts a word by a few pixels or uses a slightly different font size, SSIM will report poor quality even when the text content is correct. This reinforces the case for our CLIP cosine similarity metric, which is position-tolerant.

**On CLIP cosine similarity:** CDM's approach is philosophically related to CLIP — both avoid raw pixel comparison and instead extract semantic representations. CDM uses character-level bounding boxes and spatial matching; CLIP uses global image embeddings. CDM has higher precision for character-level correctness but requires the structured LaTeX source to be available. CLIP is more general and requires no source markup, making it better suited for reference-free evaluation (our use case). CDM's finding that rendering-based semantic matching outperforms pixel metrics is consistent with CLIP's strong performance in our experiments.

**On edit distance:** CDM's critique of edit distance for formula evaluation partially applies to OCR text evaluation too. Edit distance (CER/WER) penalizes valid paraphrasing, OCR artifacts in ground-truth data, and format differences between OCR engines. Perplexity and CLIP are arguably more semantically appropriate for our reference-free setting.

**On font rendering and layout:** CDM's color-coding technique (assigning per-token colors, rendering, then extracting bounding boxes) is an elegant approach for getting character-level positions from a rendering engine. If we wanted to do fine-grained spatial accuracy evaluation of our reconstructed documents, a similar approach — render HTML with color-tagged spans, then diff bounding box positions — could be adapted from CDM's methodology.

**On the rendering-first principle:** CDM's central insight — evaluate what things *look like*, not what they *are encoded as* — directly validates our reconstruction-based approach. We reconstruct an image from OCR output and compare it to the original. CDM independently arrives at the same philosophy for a different sub-task.
