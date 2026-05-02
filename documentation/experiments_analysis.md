# Experiments Analysis

## Template
### [Study Name]
- **Date**: YYYY-MM-DD
- **Hypothesis**: What we're testing
- **Variables**: What was changed
- **Results**: Quantitative results table
- **Conclusion**: What we learned

---

### DeepSeek-OCR v1 Prompt Evaluation
- **Date**: 2026-03-02
- **Hypothesis**: Grounding prompts (`<|grounding|>`) produce per-text bounding boxes suitable for visual reconstruction scoring, while non-grounding prompts produce richer text for LM perplexity evaluation.
- **Variables**: 4 prompt types tested on a 2-page Japanese PDF (2022統合報告書-11-12.pdf, 2481x3508px @ 300 DPI)
  - `grounding_markdown`: `<image>\n<|grounding|>Convert the document to markdown.`
  - `free_ocr`: `<image>\nFree OCR.`
  - `ocr`: `<image>\nOCR this document.`
  - `extract_text`: `<image>\nExtract the text in the image.`
- **Setup**: DeepSeek-OCR (`deepseek-ai/DeepSeek-OCR`), PyTorch 2.5.1+cu124, RTX 6000 Ada, eager attention, eval_mode=True
- **Results**:

| Prompt | Grounding Rate | Avg Visual | Avg LM | Avg PPL | Avg Output Len |
|--------|---------------|------------|--------|---------|----------------|
| grounding_markdown | 2/2 (100%) | 0.7192 | nan | nan | 1047 |
| free_ocr | 0/2 (0%) | 0.0000 | 0.5169 | 8.74 | 6574 |
| ocr | 0/2 (0%) | 0.0000 | 0.5915 | 11.02 | 11256 |
| extract_text | 0/2 (0%) | 0.0000 | 0.5915 | 11.02 | 11256 |

- **Conclusion**:
  1. **Grounding prompt produces only image-level boxes**, not per-text bounding boxes. Page 1 returned 1 element ("image"), Page 2 returned 12 elements. Visual reconstruction score (0.7192) reflects coarse layout, not character-level accuracy.
  2. **`free_ocr` gives best text extraction** with lowest perplexity (8.74) and good output volume (6574 chars avg). Best candidate for LM-based quality evaluation.
  3. **`ocr` and `extract_text` produce identical output** — both return longer, more verbose text (11256 chars) but with higher perplexity (11.02).
  4. **LM perplexity is nan for grounding prompt** because grounding output contains mostly tags, not natural text.
  5. **PyTorch 2.10+cu128 is incompatible** with DeepSeek-OCR's SAM vision encoder (CUBLAS bf16 einsum bug). Must use PyTorch 2.5.x+cu124.
