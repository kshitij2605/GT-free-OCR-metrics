# OCR Reference Metrics: Why Each Dimension Gets Its Own Metric

## The Core Problem: One Metric Doesn't Fit All Content Types

OCR documents contain different content types — plain text, math formulas, tables — and each type has a different notion of "correct". Using a single metric (edit distance) for all three flattens these differences and produces misleading correlation signals.

---

## Plain Text: Edit Distance Is Appropriate

For prose or structured text, the "correct" unit is the character. An OCR error that replaces `"the"` with `"tbe"` is exactly one character wrong regardless of layout. Element-level Levenshtein edit distance, with Hungarian matching to handle granularity mismatches between GT and OCR elements, captures this faithfully.

`ocr_accuracy = 1 - edit_distance` is the right reference signal for text.

---

## Math Formulas: Edit Distance Is Wrong

### The structural vs character problem

Consider two formula representations of the same fraction:

```
Ground truth:  \frac{a}{b}
OCR output:    a/b
```

**Character-level edit distance** computes this as roughly 50% similar — they share `a`, `b`, and `/`. It partially rewards the OCR output.

**What they actually render as:**

```
\frac{a}{b}  →      a        (fraction: a above a horizontal bar, b below)
                    ─
                    b

a/b          →   a/b         (inline slash: all on the same baseline)
```

These are completely different visual structures. A downstream system that re-renders the document, a student reading a PDF, or a CAS (computer algebra system) parsing the expression will see fundamentally different things. `a/b` is not a reasonable OCR approximation of `\frac{a}{b}` — it is a structural failure.

### Why edit distance misses this

Edit distance measures character identity. It does not ask: *do these two formulas render to the same visual layout?* So OCR that consistently writes inline expressions instead of fractions, roots, or summations can score acceptably on edit distance while being completely wrong structurally.

### What CDM measures instead

CDM (Character Detection Matching) renders **both** formulas to images and works in pixel space:

1. Tokenize: both formulas are split into individual LaTeX tokens (`\frac`, `a`, `b`, etc.)
2. Colorize: each token gets a unique RGB color wrapping (`\textcolor[RGB]{15,0,0}{\frac}`)
3. Render: pdflatex + pdftoppm at 400 DPI — the formula is rendered as it would actually appear
4. Locate: each token's bounding box is extracted by finding its exact color in the rendered image
5. Match: Hungarian assignment finds the globally optimal GT↔OCR token pairing by position + identity
6. Filter: RANSAC removes geometrically inconsistent matches
7. Score: F1 over matched inliers = CDM ∈ [0, 1]

For `\frac{a}{b}` vs `a/b`:
- In the GT render, `a` is centered above the bar (~y=10), `b` is below the bar (~y=40)
- In the OCR render, `a` is at baseline (~y=25), `/` is at baseline, `b` is at baseline
- Hungarian matches `a`→`a` but the position cost is high (y=10 vs y=25), and `b`→`b` similarly
- RANSAC finds the affine transform inconsistent → outliers flagged
- CDM ≈ 0 — correctly scores this as a structural failure

**CDM scores what matters for math: does the rendered formula look right?**

---

## Tables: TEDS Is Appropriate

For tables, the meaningful unit is not the character but the **cell structure**. An OCR that extracts all the right numbers but loses the row/column relationships has failed, even if its character edit distance is good.

TEDS (Tree Edit Distance Score) compares the HTML parse trees of GT and OCR tables:

```
teds = 1 - apted_distance(gt_html_tree, ocr_html_tree) / max_nodes
```

With fractional rename costs for `<td>` content, it penalizes both structural errors (missing rows/columns) and content errors (wrong cell text), in proportion to their severity.

---

## Implications for Correlation Baseline

Each content dimension should use the metric that best captures correctness for that type:

| Dimension | Correlation baseline | Rationale |
|-----------|---------------------|-----------|
| Text | `1 - text_edit_distance` | Character identity is the right unit for prose |
| Formula | CDM | Structural rendering correctness matters more than character similarity |
| Table | TEDS | Cell structure correctness matters more than character similarity |
| End-to-end | Average of (text accuracy, CDM, TEDS) over present dims | Each dim uses its own appropriate metric |

Using text edit distance as a proxy for formula quality is wrong for the same reason that counting pixels is a poor proxy for sentence meaning — the abstraction level is wrong for the content type.

---

## End-to-End Formula: Dataset Level vs Page Level

OmniDocBench computes its overall end-to-end score at the **dataset level**:

```
Overall (%) = ((1 − text_ED) × 100 + TEDS × 100 + CDM × 100) / 3
```

where each term is a macro-average over all pages in the dataset that have GT content of that type. The fixed `/3` denominator is intentional at this level: a well-rounded OCR benchmark should cover all three content types, and the score rewards systems that handle all of them.

**Why we don't use fixed `/3` at the page level:**

Our end-to-end is computed per page (each page is one data point in the Pearson/Spearman correlation). Most pages in our 8-page sample have no formula or table GT content. Using fixed `/3` with `0` for absent dims would give a text-only page `text_acc / 3` instead of `text_acc` — artificially suppressing scores and making all end-to-end values cluster near 1/3 of their true signal, destroying the correlation.

**Our per-page formula:**

```
end_to_end = avg(text_accuracy, CDM, TEDS)   over present dims only
```

- `text_accuracy = 1 - text_edit_distance` (always included — every page has text GT)
- CDM included only if the page has formula GT and at least one GT formula compiled
- TEDS included only if the page has table GT

This matches OmniDocBench's intent (each dimension computed only over pages that have that content) but applied per page rather than at dataset level. The scaling is identical (all three are 0–1); the only difference is the denominator reflects actual content present on the page.

---

## Practical Limitations

### Sparse data (8 pages)

Most of the 8 OmniDocBench sample pages have no formula GT content. CDM correlation computed on 2–3 pages is not statistically meaningful. This is a data limitation, not a reason to use a worse metric. As more annotated pages become available, CDM and TEDS become the correct primary signals for their respective dimensions.

### CDM computational cost

CDM invokes pdflatex once per formula pair. For large datasets this is expensive. At scale, approximate alternatives (image-level SSIM on rendered formula crops, or token-level string matching after KaTeX normalization) may be needed. For validation on small sample sets, full CDM is fine.

### CDM failure modes

CDM returns `None` (skip) if pdflatex fails to compile a formula. This happens for:
- Formulas with non-standard packages (`\upgreek`, custom macros)
- Malformed LaTeX that neither GT nor OCR can compile
- Complex environments (`\begin{cases}` with nested content) that trigger rendering bugs

When CDM returns None, the formula dimension is excluded from the end-to-end average (same treatment as absent GT content).

---

## Summary

> Edit distance is right for text. CDM is right for formulas. TEDS is right for tables.
> Using edit distance uniformly across all content types makes formula and table evaluation look better than it is, biasing the correlation signal in favor of OCR systems that get characters right but structure wrong.
