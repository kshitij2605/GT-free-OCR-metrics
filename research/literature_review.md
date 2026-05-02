# Literature Review: Reference-Free OCR Evaluation

## Key Papers

### 1. F1_OCR-VE (arXiv:2602.13376)
**Title**: Reference-Free OCR Evaluation Framework
**Summary**: Combines OCR recall and visual entailment precision into a single F1-style metric for reference-free OCR evaluation. The framework uses a VLM to assess whether extracted text is visually entailed by the source image, while OCR recall measures coverage. Achieves Pearson r=0.94 with ground-truth-based metrics on the FlowVQA dataset.
**Relevance**: Directly addresses our core problem. The visual entailment approach is complementary to our visual reconstruction method.

### 2. HTR Eval Without Ground Truth (LREC 2022)
**Title**: Evaluating Handwritten Text Recognition without Ground Truth
**Summary**: Proposes ground-truth-free evaluation methods for handwritten text recognition using character n-gram frequency analysis and BERT perplexity scoring. Achieves Spearman r=0.90-0.99 correlation with Character Error Rate (CER) on historical document datasets.
**Relevance**: Validates the LM perplexity approach (our Method 2). The high correlation with CER suggests perplexity is a strong proxy for OCR quality.

### 3. ConfBERT (arXiv:2409.04117)
**Title**: ConfBERT: BERT-Based OCR Error Detection with Confidence Scores
**Summary**: Integrates OCR engine confidence scores into a BERT-based model for detecting OCR errors at the token level. Uses confidence as an additional feature alongside contextual embeddings.
**Relevance**: Demonstrates the value of combining OCR confidence with language model understanding. Could inform future extensions of our metric.

### 4. CDM (CVPR 2025)
**Title**: Character Detection Matching for Formula Evaluation
**Summary**: Proposes Character Detection Matching for evaluating OCR of mathematical formulas by performing character-level matching in image space rather than string space. Avoids ambiguities in LaTeX normalization.
**Relevance**: The image-space comparison approach aligns with our visual reconstruction method. Particularly relevant for formula-heavy documents.

### 5. HIP21 OCR Evaluation Survey
**Title**: Survey of Ground-Truth-Free OCR Evaluation Methods
**Summary**: Comprehensive survey of methods for evaluating OCR without ground truth, covering: confidence score modeling, lexicality checks (dictionary lookup), cross-OCR engine consistency, and language model-based approaches.
**Relevance**: Provides a taxonomy of existing approaches and identifies gaps. Our visual reconstruction method addresses the identified gap in visual-grounding approaches.

### 6. OCR-Quality Dataset (arXiv:2510.21774)
**Title**: OCR Quality Assessment Dataset
**Summary**: Introduces a dataset of 1000 document pages annotated with 4-level human quality scores (excellent, good, fair, poor). Provides a benchmark for evaluating OCR quality metrics.
**Relevance**: Potential validation dataset for our metric. Human quality scores provide ground truth for meta-evaluation.

### 7. Confidence Estimation Study (arXiv:2404.18722)
**Title**: OCR Confidence Estimation for Quality Assessment
**Summary**: Studies various methods for estimating OCR confidence, finding that averaging softmax probabilities per token provides strong correlation with recognition rate. Simple averaging outperforms more complex aggregation methods.
**Relevance**: Informs how we might incorporate OCR confidence as a baseline or supplementary signal in our metric.

### 8. Masked LM Scoring (Amazon Science)
**Title**: Pseudo-Log-Likelihood Scoring with Masked Language Models
**Summary**: Proposes using masked language models (e.g., BERT) for scoring text via pseudo-log-likelihood, where each token is masked and scored independently. Outperforms autoregressive models (GPT-2) on acceptability judgments and several other tasks.
**Relevance**: Suggests masked LM scoring may be superior to GPT-2 perplexity for our Method 2. Worth investigating as an alternative to autoregressive perplexity.

### 9. CLIPScore (EMNLP 2021)
**Title**: CLIPScore: A Reference-Free Evaluation Metric for Image Captioning
**Summary**: Uses CLIP embeddings to compute image-text cosine similarity as a reference-free metric for image captioning. Achieves strong correlation with human judgments without requiring reference captions.
**Relevance**: Directly applicable to our OpenCLIP-based sub-approach in Method 1. We adapt CLIPScore for OCR by comparing original images with reconstructed text renders.

### 10. FASTER (WACV 2025)
**Title**: FASTER: OCR Perceptual Loss Using Pre-trained Text Detector Features
**Summary**: Introduces a perceptual loss function for OCR using features from pre-trained text detection networks. Captures text-specific visual features better than generic perceptual losses (e.g., VGG-based LPIPS).
**Relevance**: Suggests text-specific perceptual features may improve our visual comparison. The pre-trained text detector features could supplement or replace generic LPIPS in our multi-metric approach.

---

## Foundational Techniques

### 11. SSIM — Structural Similarity Index Measure (IEEE TIP 2004)
**Title**: Image Quality Assessment: From Error Visibility to Structural Similarity
**Authors**: Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli
**Venue**: IEEE Transactions on Image Processing, Vol. 13, No. 4, 2004 (50,000+ citations)
**Summary**: Departs from MSE/PSNR by modeling quality degradation as loss of structural information. Decomposes comparison into luminance, contrast, and structure components computed over sliding Gaussian windows. Achieves ~63% agreement with human judgments on 2AFC tasks. Bounded in [-1, 1] where 1 = identical.
**Limitations**: Poor color handling (operates on grayscale), low discrimination at high quality, no texture awareness, struggles with noise/blur, uniform spatial pooling ignores regional importance.
**Relevance**: Core component of our multi-metric visual reconstruction comparison. Captures structural layout fidelity between original document images and OCR-reconstructed renders. Its limitations motivate combining with LPIPS and CLIP.

### 12. LPIPS — Learned Perceptual Image Patch Similarity (CVPR 2018)
**Title**: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
**Authors**: Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
**Venue**: CVPR 2018
**Summary**: Demonstrates that deep network features from pretrained CNNs (AlexNet, VGG, SqueezeNet) serve as effective perceptual similarity metrics — an emergent property of deep visual representations. Extracts intermediate features, normalizes per-channel, computes squared differences, applies learned channel weights, and sums across layers. Trained on BAPPS dataset of human perceptual judgments. Achieves ~69% agreement with humans on 2AFC tasks (vs. SSIM's ~63%).
**Limitations**: Sensitive to spatial shifts, vulnerable to adversarial attacks, no global context, computationally expensive (CNN forward pass), backbone trained on ImageNet — not text-aware.
**Relevance**: Deep perceptual complement to SSIM in our pipeline. Captures higher-level visual fidelity. However, ImageNet-trained features may underweight text-specific distortions — the FASTER (WACV 2025) and OCR-VQGAN (WACV 2023) papers found OCR-trained perceptual features substantially outperform generic LPIPS for text fidelity.

---

## Additional Papers (2026-03-20)

### 13. GLYPH-SR (arXiv 2025)
**Title**: GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?
**Authors**: Mingyu Sung, Seungjae Ham, Kangwoo Kim, Yeokyoung Yoon, Jae-Mo Kang, Il-Min Kim, Sangseok Yun
**Venue**: arXiv:2510.26339, October 2025
**Summary**: VLM-guided diffusion framework for super-resolving images while preserving text legibility. Uses dual-branch ControlNet (text-SR fusion) with a text-image balancing scheduler. Achieves large OCR F1 gains (e.g., OpenOCR F1 67.54 vs. DiffBIR's 38.73 on SVT 4x) without sacrificing perceptual quality. Demonstrates that standard metrics (PSNR, SSIM, LPIPS) are inadequate for text fidelity since text regions occupy <1% of image area.
**Relevance**: Validates our multi-metric approach. The finding that visual metrics systematically underweight text regions supports combining SSIM/LPIPS with text-specific signals (LM perplexity). Their use of CLIP-IQA for no-reference assessment parallels our CLIP similarity component.

### 14. OCR-VQGAN (WACV 2023)
**Title**: OCR-VQGAN: Taming Text-within-Image Generation
**Authors**: Juan A. Rodriguez, David Vazquez, Issam Laradji, Marco Pedersoli, Pau Rodriguez
**Venue**: WACV 2023, pp. 3689-3698
**Summary**: Extends VQGAN with an OCR perceptual loss using CRAFT text detector features alongside standard VGG-based LPIPS. Demonstrates that text-specific feature extractors outperform generic ImageNet features for measuring text fidelity. Introduces Paper2Fig100k dataset (100K+ figure images from arXiv papers). Also proposes "OCR SIM" metric comparing OCR detection features between original and reconstructed images.
**Relevance**: Directly validates a concern with our pipeline — standard LPIPS may miss text-specific distortions. The OCR SIM metric (comparing OCR features between images) is conceptually related to our reconstruction approach. The OCR perceptual loss concept could inform future extensions of our metric.

### 15. PECL: Embedding Similarity for License Plate SR (Neurocomputing 2025)
**Title**: Embedding Similarity Guided License Plate Super Resolution
**Authors**: Abderrezzaq Sendjasni, Mohamed-Chaker Larabi
**Venue**: Neurocomputing, Vol. 651, Article 130657, 2025
**Summary**: Introduces PECL (Pixel and Embedding Consistency Loss) combining pixel-level and embedding-level contrastive loss for license plate super-resolution at 8x. Achieves SSIM 0.8127 / LPIPS 0.106 on CCPD dataset (vs. SwinIR's 0.7477 / 0.147) with only 1.9M parameters. Downstream OCR exact match accuracy improves from 58.1% (SwinIR) to 62.8%.
**Relevance**: Validates our metric choices — their Table 3 shows SSIM/LPIPS improvements correlate with OCR accuracy improvements. The embedding similarity approach parallels our CLIP similarity component, providing evidence that learned embeddings capture OCR-relevant quality beyond pixel-level metrics.

---

## Render-and-Compare Prior Art (2026-03-20)

The following papers use a similar "render OCR/parsing output back to image, compare with original" paradigm. They were discovered during a novelty verification search and are critical for positioning our work.

### 16. MonkeyOCR v1.5 (arXiv, November 2025)
**Title**: MonkeyOCR v1.5 Technical Report
**Venue**: arXiv:2511.10390, November 2025
**Summary**: For table recognition, MonkeyOCR renders the model's OCR output (HTML/LaTeX tables) back into an image, then compares this rendered image to the original using a VLM-based reward model. This visual consistency signal is used as a reward for reinforcement learning (GRPO), enabling training on unlabeled data without ground truth.
**Overlap with our work**: Same core idea — reconstruct from OCR output, compare visually to original, use comparison as quality signal. **Key differences**: (1) Tables only, not full documents; (2) Used as RL training reward, not a standalone evaluation metric; (3) VLM-based comparison only, no multi-metric fusion.

### 17. dots.ocr / ISVGEN Score (RedNote, 2025-2026)
**Title**: dots.ocr Document Parser with ISVGEN Evaluation
**Link**: https://github.com/rednote-hilab/dots.ocr
**Summary**: Parses documents into structured code (SVG, Markdown), renders the parsed output back into an image, and computes the ISVGEN score (from UniSVG) between the rendered output and the original image using CLIP similarity.
**Overlap with our work**: Same paradigm — parse document, render back, compare visually. **Key differences**: (1) SVG output format, not text+bboxes; (2) CLIP-only comparison, no SSIM/LPIPS/LM perplexity; (3) Benchmark evaluation tool, not a standalone deployable metric.

### 18. Visual-ERM (InternLM, March 2026)
**Title**: Visual-ERM: Reward Modeling for Visual Equivalence
**Venue**: arXiv:2603.13224, March 2026
**Summary**: Reward model that evaluates vision-to-code tasks (chart-to-code, table-to-markdown, SVG-to-code) by comparing rendered output images to ground-truth images. Provides fine-grained visual discrepancy annotations. Achieves +2.7 points on OmniDocBench for table-to-markdown.
**Overlap with our work**: Render-and-compare for document parsing output including OCR-like tasks. **Key differences**: (1) Requires a trained reward model, not metric-based; (2) Applied to tables/charts, not full document text; (3) Published contemporaneously (March 2026).

### 19. CycleReward (ICCV 2025)
**Title**: CycleReward: Cycle Consistency as Reward for Vision-Language Models
**Venue**: arXiv:2506.02095 / ICCV 2025
**Summary**: Uses cycle consistency as a supervisory signal: image → caption → reconstructed image, then compares reconstructed image to original using CLIP similarity. Applied to image captioning, not OCR.
**Overlap with our work**: Identical conceptual framework (forward pass, render back, compare) but in a different domain. **Key differences**: (1) Image captioning, not document OCR; (2) Uses image generation model for reconstruction, not deterministic text rendering; (3) CLIP-only comparison.

---

## Image Super-Resolution & Text SR Evaluation (2026-03-20)

### 20. TextSR (arXiv, May 2025)
**Title**: TextSR: Diffusion Super-Resolution with Multilingual OCR Guidance
**Authors**: Keren Ye, Ignacio Garcia Dorado, Michalis Raptis, et al.
**Venue**: arXiv:2505.23119, May 2025
**Summary**: Multimodal diffusion model for multilingual scene text SR using OCR-extracted text as cross-attention guidance. Key finding: generic diffusion SR models (StableSR, SUPIR) *reduce* OCR accuracy from 40.6% baseline to 34-35%, while TextSR improves it. Evaluated with PSNR, SSIM, LPIPS, and downstream OCR accuracy.
**Relevance**: Demonstrates that visual quality and OCR quality can diverge — SR can make images look better but hurt OCR. Directly validates the need for text-specific evaluation metrics like ours.

### 21. TeReDiff — Text-Aware Image Restoration (arXiv, June 2025)
**Title**: Text-Aware Image Restoration with Diffusion Models
**Authors**: Jaewon Min, Jin Hyeon Kim, et al. (KAIST)
**Venue**: arXiv:2506.09993, June 2025
**Summary**: Introduces the TAIR task and TeReDiff model addressing "text-image hallucination" in diffusion restoration. Evaluated with comprehensive metric suite: PSNR, SSIM, LPIPS, DISTS, FID, NIQE, MANIQA, MUSIQ, CLIPIQA, plus text spotting F1. Text restoration does not compromise overall image quality.
**Relevance**: Uses nearly our exact metric set. Finding that text fidelity and perceptual quality can be independently evaluated parallels our multi-metric approach.

### 22. Rethinking Image Evaluation in SR (arXiv, March 2025)
**Title**: Rethinking Image Evaluation in Super-Resolution
**Authors**: Shaolin Su et al.
**Venue**: arXiv:2503.13074, March 2025
**Summary**: Shows GT images in SR datasets often have poor quality, causing biased evaluations. PSNR, SSIM, and LPIPS can give contradictory results when reference quality varies. Proposes Relative Quality Index (RQI) for comparing two images of arbitrary quality.
**Relevance**: Cautionary finding for our methodology — our "reference" (original document) may itself be degraded. RQI's approach of not assuming perfect reference quality could improve our metric design.

### 23. TextDiff (Pattern Recognition, 2024)
**Title**: TextDiff: Mask-Guided Residual Diffusion Models for Scene Text Image Super-Resolution
**Authors**: Baolin Liu et al.
**Venue**: Pattern Recognition, 2024 (arXiv:2308.06743)
**Summary**: First diffusion-based scene text SR framework using text masks to guide residual diffusion. Achieves best MANIQA (no-reference) and LPIPS scores. Also evaluated with downstream OCR accuracy.
**Relevance**: Uses both reference-based (LPIPS, SSIM) and no-reference (MANIQA) metrics together. MANIQA correlating with text quality suggests it could be an additional metric for our pipeline.

### 24. DocIQ (arXiv, September 2025)
**Title**: DocIQ: A Benchmark Dataset and Feature Fusion Network for Document Image Quality Assessment
**Authors**: Zhichao Ma et al.
**Venue**: arXiv:2509.17012, September 2025
**Summary**: Introduces DIQA-5000 (5,000 document images with human MOS scores across 3 dimensions). DocIQ model uses layout-aware downsampling and multi-level feature fusion. Correlation with OCR accuracy: CACC SRCC=0.9086, WACC SRCC=0.8989. Outperforms general-purpose IQA.
**Relevance**: High correlation between visual quality predictions and OCR accuracy validates our premise. Layout-aware quality assessment parallels our bbox-based analysis. DIQA-5000 could serve as a validation benchmark.

### 25. DIQA Survey (ACM Computing Surveys, 2023)
**Title**: Document Image Quality Assessment: A Survey
**Authors**: Alireza Alaei, Vinh Bui, David Doermann, Umapada Pal
**Venue**: ACM Computing Surveys, 2023
**Summary**: Comprehensive survey of no-reference, reduced-reference, and full-reference DIQA methods. Key finding: "OCR accuracy does not always correlate with visual quality" — OCR captures text content but not graphical elements. Notes no deep-learning FR-DIQA method existed at time of writing.
**Relevance**: Essential background. The observation that OCR accuracy and visual quality are linked but distinct directly motivates our multi-metric approach (visual + perceptual + linguistic).

---

## Image Reconstruction & Document Evaluation (2026-03-20)

### 26. Image Regeneration (AAAI 2025)
**Title**: Image Regeneration: Evaluating Text-to-Image Model via Generating Identical Image with MLLMs
**Authors**: Chutian Meng, Fan Ma, et al.
**Venue**: AAAI 2025 (arXiv:2411.09449)
**Summary**: Uses MLLM (GPT-4V) to understand an image, generates a structured description tree, then uses T2I model to regenerate it. Evaluates T2I quality by comparing regenerated vs. original. Pipeline: image → understanding → structured description → reconstruction → comparison.
**Relevance**: Closest conceptual parallel to our approach. Their pipeline mirrors ours: image → VLM OCR → text+bbox → rendered reconstruction → comparison. Both use reconstruction quality as a proxy for model understanding quality.

### 27. SCORE — Semantic Evaluation for Document Parsing (arXiv, September 2025)
**Title**: SCORE: A Semantic Evaluation Framework for Generative Document Parsing
**Authors**: Renyu Li et al.
**Venue**: arXiv:2509.19345, September 2025
**Summary**: Addresses evaluation gap for generative document parsers that produce semantically correct but structurally divergent outputs. Traditional metrics (CER, WER, TEDS) penalize valid outputs by 12-25% on ambiguous structures. Integrates adjusted edit distance, token-level hallucination/omission diagnostics, and hierarchy-aware checks.
**Relevance**: Validates our visual comparison approach — exact text matching penalizes valid interpretations. Our reconstruction-based comparison can assess overall fidelity without requiring exact string matches.

### 28. STRICT — Stress Test of Rendering Images Containing Text (EMNLP 2025)
**Title**: STRICT: Stress Test of Rendering Images Containing Text
**Authors**: Tianyu Zhang, Xinyu Wang, et al.
**Venue**: EMNLP 2025 (arXiv:2505.18985)
**Summary**: Benchmarks text-to-image models on text rendering quality. Generates text images, OCRs them, compares extracted text to target using NED, CER, WER. Tests 10+ models (GPT-4o, Gemini, FLUX, etc.).
**Relevance**: Inverse of our pipeline — they do text → render → OCR → compare text, we do image → OCR → render → compare image. Both validate the round-trip paradigm for quality evaluation.

### 29. LED — Layout Error Detection Benchmark (arXiv, July 2025)
**Title**: LED: Diagnosing Structural Layout Errors for Document Layout Analysis
**Authors**: Inbum Heo et al.
**Venue**: arXiv:2507.23295 / BigComp 2026
**Summary**: Defines 8 layout error types (Missing, Hallucination, Size Error, Split, Merge, Overlap, Duplicate, Misclassification). Conventional overlap metrics (IoU, mAP) fail to capture logical inconsistencies.
**Relevance**: Our reconstruction approach implicitly detects these same layout errors — when OCR misses a block, merges two blocks, or hallucinates content, the reconstruction diverges predictably. LED's taxonomy could help interpret what our visual metrics measure.

### 30. CG-DIQA — No-Reference Document IQA (ICDAR 2018)
**Title**: CG-DIQA: No-reference Document Image Quality Assessment Based on Character Gradient
**Authors**: Hongyu Li, Fan Zhu, Junhua Qin
**Venue**: ICDAR 2018 (arXiv:1807.04047)
**Summary**: No-reference document quality metric using character gradient features from MSER-detected patches. Uses OCR accuracy as ground-truth quality label. Achieves SROCC=0.9429.
**Relevance**: Bidirectional validation — CG-DIQA shows visual quality predicts OCR accuracy; our project shows OCR reconstruction quality predicts visual fidelity. Together they establish the bidirectional relationship between image quality and OCR quality.

---

## Novelty Assessment (Revised after 30-paper review)

### The Reconstruction-as-Evaluation Paradigm Is Not Novel

The core idea of "reconstruct from model output, compare to original" has multiple concurrent instances:

| Work | Domain | Reconstruction | Comparison | Purpose |
|------|--------|---------------|------------|---------|
| **MonkeyOCR v1.5** (#16) | Table OCR | Render HTML/LaTeX tables | VLM reward | RL training signal |
| **dots.ocr** (#17) | Document parsing | Render SVG | CLIP similarity | Benchmark metric |
| **Visual-ERM** (#18) | Tables/charts | Render code output | Fine-grained VLM | RL reward model |
| **CycleReward** (#19) | Image captioning | T2I from caption | CLIP similarity | RL reward |
| **Image Regeneration** (#26) | T2I evaluation | T2I from MLLM description | Image similarity | Model evaluation |
| **STRICT** (#28) | Text rendering | Text → image → OCR | Text comparison | Inverse round-trip |
| **Our approach** | Full-document OCR | Style-aware text+bbox render | SSIM+LPIPS+CLIP+LM | Standalone metric |

Image Regeneration (#26, AAAI 2025) is the closest conceptual parallel — it evaluates model quality through the cycle: image → understand → reconstruct → compare. The difference is they use neural reconstruction (diffusion model) to evaluate T2I quality, while we use deterministic styled rendering to evaluate OCR quality.

### What Remains Novel

1. **Full-document, all-element scope**: Prior works target specific sub-tasks (tables, SVG, charts, captions). We evaluate OCR across entire document pages including text, headings, columns, and mixed layouts of arbitrary type (textbooks, financial reports, newspapers, exams, presentations).

2. **Deterministic, style-aware reconstruction**: Our DocumentAnalyzer extracts font size, serif/sans-serif, bold, alignment, and line positions from the original image via computer vision, then renders with matching styles. Prior works use either neural reconstruction (Image Regeneration, CycleReward) or unstyled rendering (dots.ocr). Deterministic rendering means differences between original and reconstruction are directly attributable to OCR errors — not reconstruction model artifacts.

3. **Multi-modal metric fusion**: We combine 4 complementary signals:
   - Structural (SSIM) — layout fidelity
   - Perceptual (LPIPS) — deep visual similarity
   - Semantic (CLIP) — high-level content match
   - Linguistic (LM perplexity) — text coherence

   Prior works use single signals. TeReDiff (#21) uses a similar metric suite but for SR evaluation, not OCR evaluation. The DIQA Survey (#25) notes OCR accuracy and visual quality are "linked but distinct" — our fusion explicitly bridges both.

4. **Standalone, engine-agnostic evaluation metric**: Prior works use render-and-compare as RL training rewards (MonkeyOCR, CycleReward), internal benchmarks (dots.ocr), or model comparisons (Image Regeneration). We propose it as a **deployable quality metric** that works with any OCR engine outputting text + bounding boxes, without ground truth or model internals.

### Acknowledged Limitations

- **Reference quality assumption**: Rethinking SR Eval (#22) shows SSIM/LPIPS can give contradictory results when the reference (original document) is itself degraded. Our metric assumes the original is a reasonable reference, which may not hold for damaged/noisy documents.
- **Text area underweighting**: GLYPH-SR (#13) shows text occupies <1% of image area, making pixel-level metrics insensitive to text errors. Our LM perplexity component partially addresses this but the visual metrics remain vulnerable.
- **Bbox extraction dependency**: Our approach requires text + bounding boxes from the OCR engine. If the OCR produces no bboxes (50% failure rate in our experiments), the visual reconstruction is meaningless — only LM perplexity remains usable.

---

## Frequency-Domain Analysis (2026-03-25)

Research into frequency transforms, wavelet methods, watermarking, and related techniques for potential integration into the comparison pipeline.

### 31. Focal Frequency Loss (ICCV 2021)
**Title**: Focal Frequency Loss for Image Reconstruction and Synthesis
**Authors**: Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy
**Venue**: ICCV 2021
**Link**: [arXiv:2012.12821](https://arxiv.org/abs/2012.12821) | [GitHub](https://github.com/EndlessSora/focal-frequency-loss)
**Summary**: Uses 2D FFT to transform images to frequency domain, then applies adaptive weighting that focuses on hard-to-synthesize frequency components. Down-weights easy frequencies while emphasizing difficult ones. Addresses the known spectral bias of neural networks toward low-frequency content.
**Relevance**: Directly applicable to render-and-compare. Text edges and fine strokes are high-frequency content — a focal frequency comparison between original and reconstruction could detect OCR errors that spatial metrics (SSIM) miss due to text occupying <1% of pixel area.

### 32. HaarPSI — Haar Wavelet Perceptual Similarity (Signal Processing 2018)
**Title**: A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment
**Authors**: R. Reisenhofer, S. Bosse, G. Kutyniok, T. Wiegand
**Venue**: Signal Processing: Image Communication, Vol. 61, 2018 (actively benchmarked 2024-2025)
**Link**: [arXiv:1607.06140](https://arxiv.org/abs/1607.06140) | [GitHub](https://github.com/rgcda/haarpsi)
**Summary**: Uses 6 discrete 2D Haar wavelet filters for local similarity, weighting importance by low-frequency coefficients. Mimics primary visual cortex orientation and spatial frequency selectivity. Computationally cheaper than SSIM with higher correlation to human perception on standard benchmarks.
**Relevance**: Potential drop-in replacement or complement to SSIM. Haar wavelets respond strongly to horizontal/vertical edges — directly relevant for text character edges. pip-installable (`haarpsi`).

### 33. CW-SSIM — Complex Wavelet SSIM (IEEE TIP 2009)
**Title**: Complex Wavelet Structural Similarity: A New Image Similarity Index
**Authors**: M.P. Sampat, Z. Wang, et al.
**Venue**: IEEE Transactions on Image Processing, 2009
**Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/5109651/)
**Summary**: Computes SSIM in complex steerable pyramid domain (6-scale, 16-orientation). Robust to small rotations and translations via phase consistency — consistent phase shifts from geometric distortions are ignored while structural changes are detected.
**Relevance**: Useful when bbox extraction has slight alignment errors. Standard SSIM is sensitive to sub-pixel shifts between original and reconstruction; CW-SSIM tolerates this while still capturing structural differences.

### 34. WGSR — Wavelet-Domain Losses for SR (CVPR 2024)
**Title**: Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts
**Authors**: Cansu Korkmaz, A. Murat Tekalp, Zafer Dogan
**Venue**: CVPR 2024
**Link**: [arXiv:2402.19215](https://arxiv.org/abs/2402.19215) | [GitHub](https://github.com/mandalinadagi/WGSR)
**Summary**: Trains discriminator only on high-frequency wavelet subbands (LH, HL, HH), not RGB. Uses fidelity loss over wavelet subbands with per-subband weighting. Separates genuine HF details from artifacts better than RGB or Fourier losses.
**Relevance**: Directly applicable as a wavelet-domain comparison metric. Text edges appear primarily in HF subbands (LH=horizontal edges, HL=vertical edges, HH=diagonal/fine detail). Comparing per-subband similarity between original and reconstruction with higher weight on detail subbands could capture text quality that spatial metrics miss.

### 35. Fourier Spectrum Discrepancies in Generated Images (NeurIPS 2020)
**Title**: Fourier Spectrum Discrepancies in Deep Network Generated Images
**Authors**: Tarik Dzanic et al.
**Venue**: NeurIPS 2020
**Link**: [arXiv:1911.06465](https://arxiv.org/abs/1911.06465)
**Summary**: Shows systematic high-frequency deficiencies in neural network-generated images detectable via Fourier analysis. The spectral signature of generated images differs predictably from real images in HF modes.
**Relevance**: If OCR reconstruction uses neural rendering or if we compare against neural-generated reconstructions, spectral analysis can detect artifacts. Also provides theoretical grounding for why HF comparison matters — text fidelity lives in the high-frequency domain.

### 36. Phase Congruency for IQA (EURASIP 2023)
**Title**: Phase Congruency Based on Derivatives of Circular Symmetric Gaussian Function for Image Quality Assessment
**Venue**: EURASIP Journal on Image and Video Processing, 2023
**Link**: [SpringerOpen](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-023-00611-2)
**Summary**: Phase congruency computed via log-Gabor filter banks across multiple scales/orientations. Phase carries more structural information than amplitude. Phase congruency is illumination-invariant, contrast-invariant, and dimensionless (0-1 range).
**Relevance**: Could provide a contrast-invariant structural comparison between original and reconstruction. Particularly useful when original document has varying brightness/contrast that affects SSIM — phase congruency focuses purely on structural edges regardless of rendering differences.

### 37. Spatial-Frequency Document Forgery Detection (JVCIR 2025)
**Title**: Document Forgery Detection Based on Spatial-Frequency and Multi-Scale Feature Network
**Venue**: Journal of Visual Communication and Image Representation, 2025
**Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1047320325000070)
**Summary**: Fuses spatial and frequency domain features with HRNet + attention for tampered text detection. Combining both domains improves F1 by 5.73% over spatial-only. Can localize AND classify tampering type.
**Relevance**: The spatial-frequency fusion concept directly transfers — detecting "reconstruction errors" is analogous to detecting "tampering." The multi-scale attention handles different text sizes, relevant for documents with mixed heading/body text.

### 38. Tampered Text Detection via Frequency (ECCV 2024)
**Title**: Enhancing Tampered Text Detection Through Frequency Feature Fusion and Decomposition
**Venue**: ECCV 2024
**Link**: [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04834.pdf)
**Summary**: Uses DCT for spectral features + Wavelet-like Frequency Enhancement (WFE) for preserving high-frequency detail during downsampling. Exploits Block Artifact Grid (BAG) discontinuities in DCT domain to detect text tampering.
**Relevance**: Text-specific frequency analysis. The WFE module preserving HF details during downsampling could detect font substitution, spacing errors, or stroke quality differences in OCR reconstruction.

### 39. FreqMark — Frequency Watermarking (NeurIPS 2024)
**Title**: FreqMark: Invisible Image Watermarking via Frequency Based Optimization in Latent Space
**Venue**: NeurIPS 2024
**Link**: [arXiv:2410.20824](https://arxiv.org/abs/2410.20824)
**Summary**: Encodes watermarks in latent frequency space (VAE + Fourier transform). Achieves >90% bit accuracy on 48-bit messages even under regeneration attacks. Dual-domain (latent + frequency) approach provides robustness to image transformations.
**Relevance**: The dual-domain concept (spatial + frequency) is transferable. Comparing frequency signatures in both pixel space and learned latent space could provide a more robust quality metric.

### 40. DINOHash — Adversarially Robust Perceptual Hashing (2025)
**Title**: DINOHash: Learning Adversarially Robust Perceptual Hashes
**Venue**: OpenReview, 2025
**Link**: [OpenReview](https://openreview.net/pdf?id=HrGa8Mq2NE)
**Summary**: DINOv2 features + PCA whitening + binarization, adversarially fine-tuned. Outperforms classical DCT-DWT perceptual hashing under heavy crops, compression, and adversarial attacks (~83% bit accuracy).
**Relevance**: A fast perceptual hash for document similarity — comparing hashes of original vs reconstructed gives a single robust similarity score. Could serve as a lightweight complementary metric.

### 41. Wav-KAN — Wavelet Kolmogorov-Arnold Networks (2024)
**Title**: Wav-KAN: Wavelet Kolmogorov-Arnold Networks
**Authors**: Zavareh Bozorgasl, Hao Chen
**Venue**: arXiv:2405.12832, May 2024
**Link**: [arXiv:2405.12832](https://arxiv.org/abs/2405.12832) | [GitHub](https://github.com/zavareh1/Wav-KAN)
**Summary**: Integrates wavelet functions (Mexican hat, Morlet, DOG, Shannon) as learnable activation functions in the Kolmogorov-Arnold Network framework. Captures both HF and LF components efficiently. DOG and Mexican hat outperform spline-based KAN.
**Relevance**: Novel architecture for learning wavelet-based perceptual features. Could potentially be trained as a learned document quality metric that operates in the wavelet domain — a "wavelet LPIPS" tuned for documents.

### 42. DIQA Survey — Frequency Gap (ACM Computing Surveys, 2023)
**Note**: Same survey as #25 but with additional frequency-specific findings.
**Key Finding**: OCR-based metrics dominated 1991-2000; perception-based metrics dominant 2016-2022. Frequency-domain methods are notably underrepresented in DIQA — identified as a gap in the literature. Most DIQA methods operate purely in spatial domain.
**Relevance**: Validates frequency-domain comparison as an unexplored direction for document quality assessment.

---

## Geometric & Curve-Based Modeling (2026-03-25)

Research into Bezier curves, elliptic Fourier descriptors, and geometric shape comparison for potential glyph-level and layout-level quality assessment.

### 43. TIQA — Text-in-Image Quality Assessment (arXiv 2025)
**Title**: TIQA: Text-in-Image Quality Assessment
**Venue**: arXiv, 2025
**Link**: [arXiv:2603.07119](https://arxiv.org/html/2603.07119)
**Summary**: Predicts scalar scores for text rendering fidelity independent of semantic correctness. Specifically targets malformed glyphs, broken strokes, and inconsistent thickness. Notes that pixel metrics under-penalize perceptual defects humans rate poorly.
**Relevance**: Directly addresses OCR rendering quality assessment — the most relevant single paper for our use case. Could serve as an additional metric component evaluating whether reconstructed text "looks right" at the glyph level.

### 44. DeepVecFont-v2 (CVPR 2023)
**Title**: DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts
**Authors**: Yizhi Wang, Zhouhui Lian
**Venue**: CVPR 2023
**Link**: [arXiv:2303.14585](https://arxiv.org/abs/2303.14585)
**Summary**: Dual-modality font learning using both Bezier curve sequences and raster images. Samples auxiliary points along curves to align generated vs target Bezier curves. Uses transformer architecture for sequence prediction.
**Relevance**: Demonstrates Bezier curve alignment for comparing glyph shapes — could enable comparing character shapes in curve space rather than pixel space, making comparison robust to anti-aliasing and rendering differences.

### 45. Bezier Splatting (arXiv, March 2025)
**Title**: Bezier Splatting for Fast Differentiable Rendering
**Authors**: Liu et al. (Clemson/Adobe)
**Venue**: arXiv, March 2025
**Link**: [arXiv:2503.16424](https://arxiv.org/html/2503.16424v3)
**Summary**: Samples 2D Gaussians along Bezier curves for differentiable rendering. 31x faster forward, 149x faster backward than DiffVG. Enables practical gradient-based optimization of curve parameters.
**Relevance**: Fast differentiable rendering could enable gradient-based optimization comparing glyph curves between original and reconstruction. The speed improvement makes curve-level comparison practical for full documents.

### 46. StarVector (CVPR 2025)
**Title**: StarVector: Foundation Model for SVG Generation
**Authors**: Rodriguez et al. (ServiceNow)
**Venue**: CVPR 2025
**Link**: [arXiv:2312.11556](https://arxiv.org/abs/2312.11556)
**Summary**: 8B parameter foundation model for converting images to SVG code. Outputs SVG primitives including text elements. Could vectorize document images for curve-level comparison.
**Relevance**: Could vectorize both original and reconstructed document images into SVG/Bezier curves, then compare curve parameters directly — resolution-independent and robust to rendering differences.

### 47. ElliShape — Elliptic Fourier Descriptors (arXiv, December 2024)
**Title**: Reliable and Superior Elliptic Fourier Descriptor Normalization and ElliShape
**Authors**: Hui Wu et al., Chinese Academy of Sciences
**Venue**: arXiv:2412.10795, December 2024
**Link**: [arXiv:2412.10795](https://arxiv.org/abs/2412.10795)
**Summary**: Reformulated EFD calculation with "true normalization" invariant to translation, rotation, scale, and starting point. Eliminates manual alignment. Captures coarse shape with few numeric values; detail tunable by adding descriptor elements.
**Relevance**: Could compare character/glyph contours between original and reconstructed images. Robust to font substitution if overall character shape preserved. Would detect structural deformations (squashed letters, wrong aspect ratios). Python: `pyefd` library.

### 48. LTSim — Layout Transportation Similarity (arXiv, July 2024)
**Title**: LTSim: Layout Transportation-based Similarity Measure
**Venue**: arXiv:2407.12356, July 2024
**Link**: [arXiv:2407.12356](https://arxiv.org/html/2407.12356v1)
**Summary**: Optimal transport problem minimizing cost of "moving" elements between layouts. Cost function combines Generalized IoU (position) + category matching. Allows flexible many-to-many and cross-category matching. Retrieves semantically similar layouts (Kendall's τ ≥ 0.97).
**Relevance**: Directly applicable — compare bounding box layouts between original detection and reconstruction. Works even if element count differs slightly. Handles the layout comparison component more rigorously than our current pixel-level comparison.

### 49. DeepSSIM — Structural Similarity in Deep Features (arXiv, December 2024)
**Title**: DeepSSIM: Structural Similarity in Deep Features
**Authors**: Keke Zhang, Weiling Chen, Tiesong Zhao, Zhou Wang
**Venue**: arXiv:2412.19553, December 2024
**Link**: [arXiv:2412.19553](https://arxiv.org/html/2412.19553v1)
**Summary**: Deep structure representation via self-correlation of deep features. Handles Geometrically-Disparate-Reference (GDR) IQA without requiring pixel alignment. Non-training-based with attention calibration.
**Relevance**: Directly addresses comparing non-pixel-aligned images — traditional SSIM fails when images aren't perfectly aligned. Could replace standard SSIM when there are slight positioning differences between original and reconstruction.

### 50. Fréchet Distance for Curve Comparison
**Title**: Various (foundational metric)
**Summary**: The Fréchet distance ("dog-walking" metric) respects curve ordering and flow, unlike Hausdorff distance. Better suited for comparing parametric curves like Bezier glyph outlines. Used alongside Hausdorff distance (used in Stroke2Font for Chinese character curve fitting).
**Relevance**: If we adopt curve-based glyph comparison (vectorize → compare), Fréchet distance is the preferred metric for comparing stroke/curve quality. Captures stroke order and flow which matters for CJK character fidelity.

### 51. Stroke2Font (MDPI Algorithms, March 2025)
**Title**: Stroke2Font: Decomposing Chinese Characters into Stroke Elements
**Venue**: MDPI Algorithms, March 2025
**Link**: [MDPI](https://www.mdpi.com/1999-4893/19/3/231)
**Summary**: Decomposes Chinese characters into stroke elements parameterized by Bezier curves with control vectors. Uses Hausdorff distance minimization for curve fitting.
**Relevance**: Directly relevant to our CJK document handling. Bezier stroke decomposition could enable per-stroke quality comparison for Chinese/Japanese characters.

### 52. VecFusion — Vector Font Generation with Diffusion (CVPR 2024)
**Title**: VecFusion: Vector Font Generation with Diffusion
**Authors**: Thamizharasan et al.
**Venue**: CVPR 2024
**Link**: [arXiv:2312.10540](https://arxiv.org/abs/2312.10540)
**Summary**: Generates vector fonts as ordered sequences of cubic Bezier control points via cascaded diffusion (raster → vector). Demonstrates that font quality can be evaluated in curve parameter space.
**Relevance**: Supports the concept of evaluating text quality in Bezier space. If both original and reconstructed glyphs are vectorized, comparing control point sequences gives a resolution-independent quality signal.

### 53. GNN Document Layout Analysis (arXiv, May 2025)
**Title**: Benchmarking GNNs for Document Layout Analysis in Public Affairs
**Venue**: arXiv:2505.14699 (CVPR 2025 submission)
**Link**: [arXiv:2505.14699](https://arxiv.org/abs/2505.14699)
**Summary**: Documents modeled as graphs (text blocks = nodes, spatial relationships = edges). GraphSAGE with dual-branch multimodal architecture. Captures relational structure invariant to exact pixel positions.
**Relevance**: Could model document layout as a graph and compare graph structure between original and reconstruction — detecting if paragraphs/tables maintain correct spatial relationships independent of pixel-level rendering.

---

## Synthesis: New Research Directions (2026-03-25)

### Frequency-Domain Opportunities

The frequency-domain literature reveals a clear gap in document quality assessment (confirmed by the DIQA Survey #25/#42). Key opportunities:

1. **Wavelet subband comparison**: Compare LH/HL/HH subbands between original and reconstruction — text edges live in these HF bands. Weight detail subbands higher since text quality is fundamentally a high-frequency phenomenon.

2. **HaarPSI as SSIM replacement**: Drop-in replacement with better perceptual correlation, especially for edge-heavy content like text. Already pip-installable.

3. **Phase congruency**: Contrast/illumination-invariant structural comparison — addresses the limitation that original documents may have varying brightness/scan quality.

4. **Focal frequency weighting**: Adaptively weight frequency components based on reconstruction difficulty — similar to FFL but applied as a comparison metric rather than training loss.

### Geometric/Curve Opportunities

1. **Layout comparison via optimal transport (LTSim)**: Compare bounding box layouts without requiring pixel alignment. Most immediately practical addition to our pipeline.

2. **EFD character contour comparison**: Low-dimensional, transformation-invariant shape comparison for glyph-level quality assessment. Particularly powerful for detecting structural character errors (e.g., "rn" vs "m").

3. **TIQA integration**: A dedicated text rendering quality assessor as an additional metric component.

### Proposed Metric Extensions (Priority Order)

| Priority | Extension | Effort | Expected Gain |
|----------|-----------|--------|---------------|
| 1 | HaarPSI (replace/supplement SSIM) | Low | Better edge sensitivity |
| 2 | DWT subband comparison | Medium | Text-specific frequency metric |
| 3 | LTSim layout comparison | Medium | Layout fidelity signal |
| 4 | Phase congruency map comparison | Medium | Contrast-invariant structure |
| 5 | EFD glyph shape comparison | High | Per-character quality |
| 6 | TIQA text rendering quality | Medium | Glyph-level assessment |
| 7 | Focal frequency loss as metric | Low | Adaptive HF weighting |

---

## Structured Output Formats & Document Parsing (2026-03-25)

Research into OCR output representation formats and how major document parsing libraries structure their output — informing format choice for reconstruction-based evaluation.

### 54. hOCR Standard (ICDAR 2007, actively maintained)
**Title**: The hOCR Microformat for OCR Workflow and Results
**Authors**: Thomas Breuel
**Venue**: ICDAR 2007
**Link**: [hOCR 1.2 Spec](http://kba.github.io/hocr-spec/1.2/) | [GitHub](https://github.com/kba/hocr-spec)
**Summary**: HTML-based OCR output standard. Encodes text, bounding boxes (`bbox x0 y0 x1 y1`), confidence (`x_wconf`), font info (`x_font`, `x_fsize`), baselines, and reading order in a hierarchical DOM: `ocr_page → ocr_carea → ocr_par → ocr_line → ocrx_word`. Produced by Tesseract, ABBYY, OCRopus. Tools: hocr2pdf, ocrmypdf, hocrjs.
**Relevance**: The richest widely-supported OCR format. Compared to our raw HTML+data-bbox: hOCR adds font name/size, word confidence, baselines, and character-level confidence. These signals could improve reconstruction fidelity and enable confidence-weighted metrics.

### 55. ALTO XML (Library of Congress standard)
**Title**: Analyzed Layout and Text Object
**Maintainer**: Library of Congress
**Link**: [LoC ALTO](https://www.loc.gov/standards/alto/) | [Schema v4](https://github.com/altoxml/schema)
**Summary**: XML format for digitized text. Encodes `TextStyle` (FONTFAMILY, FONTSIZE, FONTSTYLE), paragraph alignment, per-word/character confidence (WC 0.0-1.0, CC 0-9), margins, illustrations, and graphical elements. Structure: `Layout → Page → PrintSpace → TextBlock → TextLine → String`. Used by Library of Congress (Chronicling America), British Library, Europeana.
**Relevance**: Richest style metadata of any standard format — explicit font family, size, style (bold/italic), and paragraph alignment. If we could consume ALTO input, we'd get ground-truth typography rather than estimating via DocumentAnalyzer.

### 56. PAGE XML (PRImA Research Lab)
**Title**: PAGE (Page Analysis and Ground Truth Elements) Format
**Maintainer**: PRImA Research Lab
**Link**: [GitHub](https://github.com/PRImA-Research-Lab/PAGE-XML)
**Summary**: XML format supporting **polygon coordinates** (not just bounding boxes), explicit `<ReadingOrder>` element, baseline polylines, and typed regions (TextRegion, ImageRegion, TableRegion, GraphicRegion). Official format for ICDAR Document Layout Analysis competitions. Tools: PRImA Aletheia editor, pagexml-tools (Python).
**Relevance**: Polygon coordinates handle rotated/skewed text better than rectangular bboxes. The explicit reading order representation could improve text flow evaluation. ICDAR competition format means compatibility with standard benchmarks.

### 57. Docling / DoclingDocument (IBM, AAAI 2025)
**Title**: Docling: An Efficient Document Parsing System
**Authors**: IBM Research
**Venue**: AAAI 2025
**Link**: [arXiv:2501.17887](https://arxiv.org/html/2501.17887v1) | [GitHub](https://github.com/docling-project/docling)
**Summary**: Pydantic-based JSON schema (DoclingDocument) with provenance tracking. Stores texts, tables, pictures, groups in a tree with `RefItem` pointers. BoundingBox with `l, t, r, b` + `coord_origin`. Exports to Markdown, HTML, DocTags (LLM-friendly XML-like format with `<loc>` tags normalized to 500×500). Benchmarked on 89 PDFs, 4,008 pages. Also introduced SmolDocling (256M parameter model) producing DocTags directly.
**Relevance**: DoclingDocument is the most complete modern structured format — preserves layout, hierarchy, tables with spans, and reading order. Their DocTags format (compact XML with normalized locations) is interesting as an LLM-friendly alternative to HTML.

### 58. Nougat (Meta, 2023)
**Title**: Nougat: Neural Optical Understanding for Academic Documents
**Authors**: Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic
**Venue**: arXiv:2308.13418, 2023
**Link**: [arXiv:2308.13418](https://arxiv.org/abs/2308.13418) | [GitHub](https://github.com/facebookresearch/nougat)
**Summary**: End-to-end Swin Transformer encoder + mBART decoder. Outputs Mathpix Markdown (.mmd) with native LaTeX math and tabular support. Trained on 1.7M arXiv papers. **No bounding boxes or position information** — content-only extraction. Evaluated with Edit Distance, BLEU, METEOR against LaTeX source.
**Relevance**: Demonstrates that Markdown-only output sacrifices all spatial information — **cannot be used for reconstruction-based evaluation**. However, their evaluation metrics (Edit Distance, BLEU) complement our visual metrics for text content accuracy.

### 59. GOT-OCR 2.0 (2024)
**Title**: General OCR Theory: Towards OCR-2.0 via a Unified End-to-End Model
**Venue**: arXiv:2409.01704, 2024
**Link**: [arXiv:2409.01704](https://arxiv.org/abs/2409.01704) | [GitHub](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
**Summary**: 580M parameter model (VitDet encoder + Qwen-0.5B decoder). Multiple output modes: plain text, Markdown, TikZ (diagrams), SMILES (chemistry). Fine-grained mode provides bounding boxes normalized to 0-1000. Supports 8000 output tokens.
**Relevance**: The fine-grained mode with bboxes is compatible with our pipeline. Standard mode (Markdown) loses positions — confirms that format choice determines whether reconstruction-based evaluation is possible.

### 60. MinerU (OpenDataLab, 2024-2026)
**Title**: MinerU: An Open-Source Solution for Precise Document Content Extraction
**Link**: [GitHub](https://github.com/opendatalab/MinerU)
**Summary**: Pipeline approach: DocLayout-YOLO for layout → specialized models for tables (HTML), formulas (LaTeX) → Markdown + JSON output. JSON includes `content_list.json` (flat blocks with `bbox [x0,y0,x1,y1]` normalized 0-1000, `text_level`, `type`) and `model.json` (raw predictions). Evaluates on OmniDocBench with Edit Distance, BLEU, METEOR, TEDS.
**Relevance**: Their JSON output (`content_list.json`) is directly consumable by our pipeline — bboxes + text + element types. OmniDocBench evaluation methodology is the current standard we should benchmark against.

### 61. Marker (Datalab, 2024-2026)
**Title**: Marker: Fast and Accurate Document Parser
**Link**: [GitHub](https://github.com/datalab-to/marker)
**Summary**: Outputs Markdown (primary) and JSON (with bounding polygons). 23+ block types including `Line`, `Span`, `FigureGroup`, `TableGroup`, `Equation`, `SectionHeader`, etc. JSON provides tree structure with `children` field and 4-corner polygon coordinates. Claims 10x faster than Nougat with better accuracy.
**Relevance**: Their JSON output format with polygons and hierarchy is among the richest of modern parsers. Compatible with our pipeline if we add polygon-to-bbox conversion. The tree structure preserves document hierarchy we currently lose.

### 62. OmniDocBench Evaluation Framework (CVPR 2025)
**Title**: OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations
**Venue**: CVPR 2025
**Link**: [arXiv:2412.07626](https://arxiv.org/abs/2412.07626) | [GitHub](https://github.com/opendatalab/OmniDocBench)
**Summary**: 1,355 PDF pages, 9 document types, 20,000+ block-level + 80,000+ span-level annotations. Evaluates text (Edit Distance, BLEU, METEOR), tables (TEDS), formulas (CDM), reading order (NED). Uses Adjacency Search Match algorithm for aligning predictions to ground truth. Key finding: pipeline tools (MinerU, Marker) outperform general VLMs on standard documents.
**Relevance**: The current standard benchmark for document parsing. Our reconstruction-based metric is complementary — OmniDocBench requires ground truth while ours doesn't. We already have an OmniDocBench bridge module for correlation analysis.

### 63. Image2Struct / Render-Based Evaluation
**Title**: Image2Struct: Benchmarking via Visual Reconstruction
**Summary**: Emerging evaluation paradigm: VLM generates code (LaTeX, HTML, LilyPond) from document screenshot → code rendered to image → rendered image compared to original. Metrics: rendering success rate + visual similarity. Used for tables, music scores, charts.
**Relevance**: Validates our reconstruction-based approach as a growing evaluation paradigm. Their observation that different code formats (HTML vs LaTeX) yield different rendering fidelity supports format-aware metric design.

---

## Synthesis: Output Format Landscape (2026-03-25)

### Format Suitability for Reconstruction-Based Evaluation

| Format | Position | Style | Hierarchy | Confidence | Render-Back | Used By |
|--------|----------|-------|-----------|------------|-------------|---------|
| **HTML+data-bbox** (ours) | bbox (0-999) | None | Tags only | None | Good | Qwen3-VL |
| **hOCR** | bbox (pixels) | Font name/size | Full DOM | Word+char | Excellent | Tesseract, ABBYY |
| **ALTO XML** | bbox (abs) | Full (family, size, bold) | Full | Word+char+glyph | Excellent | Libraries/archives |
| **PAGE XML** | Polygons | Limited | Full + ReadingOrder | Yes | Good | ICDAR competitions |
| **DoclingDocument** | bbox + origin | None | Full tree | None | Good | Docling/SmolDocling |
| **Marker JSON** | 4-corner polygon | None | Tree + types | None | Good | Marker |
| **MinerU JSON** | bbox (0-1000) | None | text_level | None | Good | MinerU |
| **SVG** | Vector coords | Full | Layers | None | Excellent | dots.mocr |
| **Markdown** | **None** | Headers only | Implicit | None | **Impossible** | Nougat, GOT-OCR |
| **LaTeX** | **None** | Semantic | Sections | None | **Impossible** | Nougat, Mathpix |

### Key Insights

1. **Markdown cannot support reconstruction-based evaluation** — it discards all spatial information. Systems outputting only Markdown (Nougat, Mathpix, standard GOT-OCR mode) are incompatible with our approach. This is a fundamental limitation of the format, not the parser.

2. **Our HTML+bbox format is good but not the richest** — hOCR and ALTO provide font info and confidence that we currently estimate via DocumentAnalyzer. If we consumed hOCR input (e.g., from Tesseract), reconstruction would be more faithful.

3. **Confidence scores are an untapped signal** — most standard formats (hOCR, ALTO, PAGE, cloud APIs) provide per-word confidence. Low-confidence words could be downweighted in our visual comparison, distinguishing "OCR was uncertain" from "OCR was wrong."

4. **Polygon coordinates handle rotated text** — PAGE XML and Marker support polygon bounding regions rather than axis-aligned rectangles. Our current bbox approach fails for rotated or skewed text.

5. **The industry is converging on JSON + Markdown dual output** — Docling, Marker, MinerU all provide structured JSON (with positions) alongside Markdown (for readability). This pattern suggests our pipeline should accept JSON input as well as HTML.

6. **Format richness directly impacts reconstruction fidelity** — the more the format preserves (style, confidence, hierarchy, reading order), the better the reconstruction, and the more meaningful the visual comparison.

### Compatibility Recommendations

| Parser | Our Pipeline Compatibility | Conversion Needed |
|--------|---------------------------|-------------------|
| Qwen3-VL (HTML+bbox) | Native | None |
| Tesseract (hOCR) | High | Parse `title` attrs → TextElement |
| Docling (JSON) | High | Extract `prov.bbox` from items |
| Marker (JSON) | High | Polygon → bbox, flatten tree |
| MinerU (JSON) | High | Direct bbox mapping |
| Cloud APIs (JSON) | High | Normalize coordinates |
| GOT-OCR (fine-grained) | Medium | Parse normalized coords |
| Nougat (Markdown) | **None** | No positions available |
| Mathpix (Markdown) | **None** | No positions available |

### Proposed Format Enhancements for Our TextElement

Current:
```python
@dataclass
class TextElement:
    text: str
    bbox: tuple[int, int, int, int]
    tag: str
```

Potential enrichment from standard formats:
- `confidence: float | None` — from hOCR/ALTO/cloud APIs (enable confidence-weighted metrics)
- `font_family: str | None` — from hOCR/ALTO (reduce reliance on CV-based detection)
- `font_size: float | None` — from hOCR/ALTO (complement DPI estimation)
- `baseline: float | None` — from hOCR/PAGE (improve vertical positioning)

---

## Object Detection & Segmentation for Documents (2026-03-25)

Research into detection and segmentation models relevant to document layout analysis, text region extraction, and region-aware quality assessment.

### 64. DocLayout-YOLO (arXiv, October 2024)
**Title**: DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception
**Authors**: Zhiyuan Zhao, Hengrui Kang, Bin Wang, Conghui He (OpenDataLab/Shanghai AI Lab)
**Venue**: arXiv:2410.12628 (submitted to ICLR 2025)
**Link**: [arXiv:2410.12628](https://arxiv.org/abs/2410.12628) | [GitHub](https://github.com/opendatalab/DocLayout-YOLO)
**Summary**: YOLOv10-based document layout detector with Global-to-Local Controllable Receptive Module (GL-CRM) for handling multi-scale elements (tiny text alongside large tables). Pre-trained on DocSynth-300K synthetic dataset. Detects 10 element types: Title, Plain Text, Figure, Table, Formula, Caption, etc. 79.7 mAP on DocLayNet at 85.5 FPS.
**Relevance**: Could provide independent text region detection to compare against OCR-reported bboxes — a "completeness" signal. If DocLayout-YOLO finds 15 text regions but OCR reports 12, we know OCR missed content. pip-installable (`doclayout-yolo`).

### 65. Hi-SAM — Hierarchical Text Segmentation (IEEE TPAMI 2025)
**Title**: Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation
**Authors**: Maoyuan Ye, Jing Zhang, Juhua Liu, et al.
**Venue**: IEEE TPAMI, Vol. 47, pp. 1431-1447, 2025
**Link**: [arXiv:2401.17904](https://arxiv.org/abs/2401.17904)
**Summary**: Adapts SAM with frozen encoder + trainable adapters for 4-level text segmentation: (1) stroke/pixel level — binary text masks, (2) word level — individual word instances, (3) text-line level, (4) paragraph level. S-Decoder produces 1024×1024 stroke masks; H-Decoder generates instance masks from point prompts. 88.96% fgIOU on TextSeg.
**Relevance**: **Directly solves our <1% text area problem.** Stroke-level masks define exact text pixel coverage — we can compute SSIM/LPIPS only on text pixels, ignoring background entirely. Also enables word-level alignment between original and reconstruction for per-word quality scores.

### 66. DBNet++ — Differentiable Binarization (TPAMI 2022)
**Title**: Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion
**Venue**: TPAMI 2022
**Link**: [arXiv:2202.10304](https://arxiv.org/abs/2202.10304) | [GitHub](https://github.com/MhLiao/DB)
**Summary**: Segmentation-based text detector with differentiable binarization module that learns adaptive thresholds per pixel. Adaptive Scale Fusion (ASF) handles multi-scale text. Detects arbitrary-shape text at word and line level in real-time.
**Relevance**: Fast, robust text region detection. Could create binary text masks for region-focused metric computation. 2024 extension AC-DBNet adds attention for small text detection (92.1% accuracy).

### 67. CRAFT — Character Region Awareness for Text Detection
**Title**: Character Region Awareness for Text Detection
**Authors**: Clova AI (NAVER)
**Link**: [GitHub](https://github.com/clovaai/CRAFT-pytorch)
**Summary**: FCN (VGG-16 backbone) detecting individual character regions + affinity between characters, then linking characters into text instances. Character-level granularity enables fine-grained bbox verification.
**Relevance**: Character-level detection provides the finest granularity for comparing OCR output. Could verify that OCR-reported character positions match independently detected character regions. Also used as the text feature extractor in OCR-VQGAN's OCR perceptual loss (#14).

### 68. RT-DETR for Document Layout (Docling, 2025)
**Title**: Advanced Layout Analysis Models for Docling
**Venue**: arXiv:2509.11720, 2025
**Link**: [arXiv:2509.11720](https://arxiv.org/abs/2509.11720)
**Summary**: RT-DETRv2 with ResNet-101 backbone trained on 150K heterogeneous documents. Best model "heron-101" achieves 78% mAP with 28ms/image inference on A100. Production-ready transformer-based detection integrated into IBM's Docling pipeline.
**Relevance**: Production-grade alternative to YOLO-based detection, integrated with Docling's rich output format (#57).

### 69. Masked SSIM / Region-Weighted Metrics
**Title**: Various (established technique)
**Summary**: Extends SSIM with spatially adaptive masking to weight error contributions by region importance. Three-component variant: 0.5 weight for edges, 0.25 for textures, 0.25 for smooth regions. Text regions classified as "edge-heavy" receive highest weight.
**Relevance**: Direct solution to our background-dominated metric problem. Implementation: `masked_ssim = SSIM(original * text_mask, reconstructed * text_mask)`. Combined with Hi-SAM (#65) stroke masks, this isolates text quality from background noise.

### 70. OCR-Based Perceptual Loss (from OCR-VQGAN, WACV 2023)
**Title**: OCR-Based Perceptual Loss Using Pre-trained Text Detector Features
**Summary**: Key finding from OCR-VQGAN (#14): "Perceptual losses pre-trained on ImageNet are inadequate to measure distances in text generation since text recognition is not an objective of ImageNet." Uses CRAFT text detector features as perceptual loss — text-specific features substantially outperform generic LPIPS for measuring text fidelity.
**Relevance**: Strongest argument for replacing our ImageNet-trained LPIPS with OCR-trained perceptual features. Using CRAFT intermediate features to compare original vs reconstruction would be directly sensitive to text quality rather than general image similarity.

### 71. RSFIQA — Region-aware Semantic Fine-grained IQA (arXiv, August 2025)
**Title**: RSFIQA: Region-aware Semantic Fine-grained Image Quality Assessment
**Venue**: arXiv:2508.07818, August 2025
**Link**: [arXiv:2508.07818](https://arxiv.org/html/2508.07818)
**Summary**: Uses SAM to partition images into non-overlapping semantic regions, then applies Qwen2.5-VL to rate each region across 5 quality dimensions (color, noise, artifact, blur, overall). Region-Aware Semantic Attention (RSA) constrains attention to individual regions.
**Relevance**: The SAM + per-region scoring paradigm could be adapted for documents — segment text blocks, figures, tables, then compute separate quality scores per region type. Weight text regions higher.

### 72. Detect Any Text (DAT) — Unified Multi-Granularity (May 2024)
**Title**: Towards Unified Multi-granularity Text Detection with Interactive Attention
**Venue**: arXiv, May 2024
**Link**: [arXiv:2405.19765](https://arxiv.org/html/2405.19765v1)
**Summary**: Single end-to-end model with across-granularity interactive attention that unifies scene text detection, layout analysis, and page detection. Handles word, line, paragraph, and page granularities simultaneously.
**Relevance**: Could replace multiple specialized models (text detector + layout detector) with a single unified detector, simplifying our pipeline for independent region verification.

### 73. RoDLA — Robustness Benchmark for Document Layout Analysis (CVPR 2024)
**Title**: RoDLA: Benchmarking the Robustness of Document Layout Analysis Models
**Venue**: CVPR 2024
**Link**: [CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_RoDLA_Benchmarking_the_Robustness_of_Document_Layout_Analysis_Models_CVPR_2024_paper.pdf)
**Summary**: 450K+ documents with 36 perturbation types. Metrics: Mean Perturbation Effect (mPE), Mean Robustness Degradation (mRD). Tests 10+ DLA methods on PubLayNet-P, DocLayNet-P, M6Doc-P.
**Relevance**: Perturbation taxonomy could define quality degradation types for our metric. Understanding how detection models degrade under perturbations informs which detection signals are reliable.

### 74. Small Object Detection Survey (arXiv, March 2025)
**Title**: Survey on Small Object Detection
**Venue**: arXiv:2503.20516, March 2025
**Link**: [arXiv:2503.20516](https://arxiv.org/abs/2503.20516)
**Summary**: Defines small objects as occupying <1% of image area — exactly matching our text region problem. Key challenges: limited spatial/contextual information, minor shifts creating large proportional errors. Reviews multi-scale feature fusion, super-resolution-based, and attention-based approaches.
**Relevance**: Text regions in documents face identical challenges to small object detection. Their multi-scale and attention-based solutions could inform text-region-focused metric design.

---

## Synthesis: Detection & Segmentation for OCR Evaluation (2026-03-25)

### The Core Problem: Text is <1% of Image Area

GLYPH-SR (#13) and the small object detection survey (#74) confirm: text pixels occupy <1% of a document image. This means SSIM/LPIPS scores are 99%+ determined by background similarity, not text quality. Detection and segmentation models offer three solutions:

### Solution 1: Text-Masked Metrics (Highest Priority)

Use Hi-SAM (#65) or DBNet++ (#66) to create binary text masks, then compute metrics only on text pixels:

```
text_mask = HiSAM.stroke_segment(image)  # Binary text pixel mask
masked_ssim = SSIM(original * text_mask, reconstructed * text_mask)
masked_lpips = LPIPS(original * text_mask, reconstructed * text_mask)
```

This directly addresses the <1% problem — background pixels contribute zero to the metric.

### Solution 2: Detection Completeness Signal

Use DocLayout-YOLO (#64) independently on the original image to detect text regions, then compare against OCR-reported bboxes:

```
completeness = |OCR_regions ∩ detected_regions| / |detected_regions|
coverage = area(OCR_bboxes) / area(detected_bboxes)
```

This catches the 50% bbox extraction failure rate we observe — if the detector finds text that OCR missed, we know the reconstruction is incomplete.

### Solution 3: OCR-Trained Perceptual Features

Replace ImageNet-trained LPIPS with CRAFT text detector features (#70):
- CRAFT intermediate features are trained to recognize text
- Comparing these features between original and reconstruction is directly sensitive to text quality
- OCR-VQGAN (#14) demonstrated this substantially outperforms generic LPIPS for text fidelity

### Proposed Integration (Priority Order)

| Priority | Extension | Effort | Impact |
|----------|-----------|--------|--------|
| 1 | Hi-SAM text masks → masked SSIM/LPIPS | Medium | Solves <1% text area problem |
| 2 | DocLayout-YOLO completeness signal | Low | Catches missing text regions |
| 3 | CRAFT features as OCR perceptual loss | Medium | Text-specific perceptual metric |
| 4 | Region-cropped per-bbox metrics | Low | Per-element quality scores |
| 5 | Detection confidence as quality proxy | Low | Independent quality signal |

---

## Font Recognition & Typography Analysis (2026-03-25)

Research into automated font identification, font similarity metrics, and typographic feature extraction for improving reconstruction fidelity and measuring typography match.

### 75. FontCLIP (Eurographics 2024)
**Title**: FontCLIP: A Semantic Typography Visual-Language Model for Multilingual Font Applications
**Authors**: Tatsukawa, Shen, Qi, Koyama, Igarashi, Shamir
**Venue**: Eurographics 2024
**Link**: [arXiv:2403.06453](https://arxiv.org/abs/2403.06453)
**Summary**: Integrates typography-specific knowledge into pretrained CLIP through compound descriptive prompts. Produces font embeddings for similarity measurement via cosine distance. Trained on Roman alphabet but generalizes to CJK characters. Enables multilingual font retrieval and cross-lingual font matching.
**Relevance**: Could measure typography similarity between original and reconstruction — compute FontCLIP cosine similarity on text crops from each. Provides a learned font similarity metric complementing our CV-based style detection.

### 76. VLMs Fail at Font Recognition (COLM 2025)
**Title**: Texture or Semantics? VLMs Get Lost in Font Recognition
**Venue**: COLM 2025
**Link**: [arXiv:2503.23768](https://arxiv.org/abs/2503.23768)
**Summary**: Created Font Recognition Benchmark (FRB) with 15 fonts. Tested GPT-4o, Claude-3.5-Sonnet, Qwen2-VL, Llama-3.2-Vision — all perform poorly. Chain-of-thought and few-shot provide minimal improvement. VLMs fail to attend to character edges (the distinctive font features).
**Relevance**: Validates that our VLM-based approach (Qwen3-VL) cannot be relied on for font detection. Justifies our CV-based DocumentAnalyzer approach (stroke width, edge analysis) and motivates dedicated font recognition models.

### 77. Serif Classification for Typography Analysis (JDMDH 2024)
**Title**: Toward Automatic Typography Analysis: Serif Classification and Font Similarities
**Venue**: Journal of Data Mining & Digital Humanities, 2024
**Link**: [JDMDH](https://jdmdh.episciences.org/13008)
**Summary**: Classifies fonts into sans-serif, linear-serif, slab-serif, triangular-serif using TransFG (fine-grained classification). Dataset: 126,666 training images. Provides trained models for serif type detection.
**Relevance**: Could replace our stroke-width-variance heuristic for serif detection with a trained classifier. More robust across document types and font sizes.

### 78. Font-Agent (CVPR 2025)
**Title**: Font-Agent: Edge-Aware Traces for Font Recognition
**Venue**: CVPR 2025
**Summary**: Proposes Edge-Aware Traces (EAT) module to capture stroke details that VLMs miss. Uses frequency-domain attention for high-frequency font features. Addresses the fundamental limitation identified in #76.
**Relevance**: Confirms that edge/stroke features are the critical signal for font recognition — aligning with our DocumentAnalyzer's use of stroke width analysis. Their frequency-domain attention could inform a combined frequency+typography metric.

### 79. DeepFont (Adobe, 2015)
**Title**: DeepFont: Identify Your Font from An Image
**Authors**: Zhangyang Wang et al. (Adobe)
**Venue**: ACM MM 2015
**Link**: [arXiv:1507.03196](https://arxiv.org/abs/1507.03196)
**Summary**: First large-scale visual font recognition system. CNN with domain adaptation (SCAE). 80%+ top-5 accuracy on AdobeVFR dataset. Produces font similarity embeddings (Euclidean distance in feature space).
**Relevance**: Foundational work. Font embeddings could measure typography similarity between original and reconstruction, though newer models (FontCLIP #75) are more capable.

---

## Self-Supervised & Contrastive Document Representations (2026-03-25)

Research into learned document representations that could serve as document-specific similarity metrics.

### 80. ColPali — Document Retrieval via Vision-Language (2024)
**Title**: ColPali: Efficient Document Retrieval with Vision Language Models
**Authors**: Manuel Faysse et al. (Illuin Technology)
**Venue**: arXiv:2407.01449, 2024
**Link**: [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)
**Summary**: PaliGemma-3B backbone producing multi-vector patch embeddings for document pages. Uses late interaction matching (ColBERT-style) for fine-grained patch-level similarity. Captures text, figures, layout, tables, fonts. Achieves 81.3 nDCG@5 on ViDoRe benchmark (vs 65-75 for text-based methods). No OCR dependency.
**Relevance**: **Most promising approach for a "document LPIPS."** Could directly compare original vs reconstruction at patch level — late interaction computes fine-grained similarity without requiring pixel alignment. Captures text appearance, layout, and typography implicitly through visual patches.

### 81. DiT — Document Image Transformer (ACM MM 2022)
**Title**: DiT: Self-supervised Pre-training for Document Image Transformer
**Authors**: Junlong Li et al. (Microsoft Research Asia)
**Venue**: ACM MM 2022
**Link**: [arXiv:2203.02378](https://arxiv.org/abs/2203.02378)
**Summary**: ViT pre-trained on 42M document images (IIT-CDIP) via Masked Image Modeling (BEiT-style). Achieves 94.9 mAP on layout analysis, 96.55 AP on table detection. Patch embeddings learn document structure without labels.
**Relevance**: Document-specific visual encoder — unlike ImageNet-trained LPIPS backbone, DiT features are trained on documents. Extracting DiT patch embeddings and comparing with cosine similarity would give a document-aware perceptual metric. Could serve as backbone for a learned "Document-LPIPS."

### 82. LayoutLMv3 (ACM MM 2022)
**Title**: LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
**Authors**: Yupan Huang, Tengchao Lv, et al. (Microsoft)
**Venue**: ACM MM 2022
**Link**: [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)
**Summary**: Tri-modal Transformer (text + image + layout). Pre-trained with MLM, MIM, and Word-Patch Alignment (WPA) — predicting whether a text word and image patch are aligned. 95.1% mAP on PubLayNet.
**Relevance**: The WPA objective explicitly models text-visual correspondence — exactly what our metric needs to measure. If text in the reconstruction doesn't match the visual appearance, WPA-trained features would detect it. Requires OCR input alongside images.

### 83. SelfDocSeg — Self-Supervised Document Segmentation (ICDAR 2023)
**Title**: SelfDocSeg: A Self-Supervised Vision-based Approach towards Document Segmentation
**Venue**: ICDAR 2023 (Oral)
**Link**: [arXiv:2305.00795](https://arxiv.org/abs/2305.00795)
**Summary**: BYOL contrastive learning framework with layout prediction module for document segmentation. Pure vision approach — no text/OCR dependency. Language-independent. Pre-trained on DocLayNet.
**Relevance**: Language-independent document representation learning. Could provide embeddings for comparing documents across languages (relevant for our CJK support).

### 84. Research Gap: No "Document LPIPS" Exists
**Key Finding**: No published work trains a perceptual similarity metric specifically on document image pairs with human quality judgments. LPIPS is trained on ImageNet (natural images). This represents a **novel contribution opportunity**.
**Proposed approach**: Use ColPali (#80) or DiT (#81) as backbone, train similarity head using OCR accuracy as proxy supervision, following the LPIPS methodology (learned linear weights on intermediate features).

---

## Information-Theoretic Quality Metrics (2026-03-25)

Research into information-theoretic approaches for measuring information preservation between original and reconstructed document images.

### 85. VIF — Visual Information Fidelity (IEEE TIP 2006)
**Title**: Image Information and Visual Quality
**Authors**: Sheikh & Bovik
**Venue**: IEEE Transactions on Image Processing, 2006 (4,861+ citations)
**Link**: [PyTorch-Metrics](https://lightning.ai/docs/torchmetrics/stable/image/visual_information_fidelity.html)
**Summary**: Models HVS as information channel. Steerable pyramid decomposition (4 scales) → Gaussian Scale Mixture modeling → computes mutual information in each subband. Final score: ratio of distorted-image MI to reference-image MI. SROCC ~0.96 on LIVE database. Used in Netflix VMAF.
**Relevance**: Theoretically grounded information preservation metric. Measures how much visual information from the original document survives in the reconstruction. Complements SSIM (structural) with an information-theoretic perspective. pip-installable via torchmetrics.

### 86. CMMD — CLIP Maximum Mean Discrepancy (CVPR 2024)
**Title**: Rethinking FID: Towards a Better Evaluation Metric for Image Generation
**Authors**: Jayasumana, Ramalingam, Veit, et al. (Google)
**Venue**: CVPR 2024
**Link**: [arXiv:2401.09603](https://arxiv.org/abs/2401.09603) | [GitHub](https://github.com/google-research/google-research/tree/master/cmmd)
**Summary**: Replaces FID's Gaussian assumption with MMD (Maximum Mean Discrepancy) on CLIP embeddings. Unbiased estimator, no normality assumption, better sample efficiency. Better correlation with human judgment on distortion detection than FID.
**Relevance**: Could compare CLIP embedding distributions of text crops from original vs reconstruction. No pixel alignment needed — distributional comparison. CLIP's text-image training may capture document semantics better than Inception features.

### 87. DISTS — Deep Image Structure and Texture Similarity (TPAMI 2022)
**Title**: Image Quality Assessment: Unifying Structure and Texture Similarity
**Authors**: Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli
**Venue**: IEEE TPAMI, 2022
**Link**: [GitHub](https://github.com/dingkeyan93/DISTS)
**Summary**: Combines structure similarity (geometric, sensitive to spatial layout) and texture similarity (statistical, tolerant to geometric variation) using VGG features. Explicitly separates these two quality dimensions.
**Relevance**: Documents have both structural elements (layout, positioning) and textural elements (font rendering, anti-aliasing). DISTS' explicit separation could provide more interpretable quality scores than LPIPS. Structure component measures layout fidelity; texture component measures rendering quality.

### 88. SSEQ — Spatial-Spectral Entropy Quality (2014, actively used)
**Title**: No-Reference Image Quality Assessment Based on Spatial and Spectral Entropies
**Authors**: Lixiong Liu, Bao Liu, Hua Huang, Alan C. Bovik
**Venue**: Signal Processing: Image Communication, 2014
**Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0923596514000927)
**Summary**: No-reference metric using local spatial entropy + spectral entropy from DCT coefficients. 2-stage SVM (distortion classification → quality prediction). Low computational cost.
**Relevance**: Could independently assess reconstruction quality without the original image — a sanity check that our rendered reconstruction looks like a valid document. Low entropy in text regions might indicate rendering issues.

### 89. FID and Document-Specific Variants
**Title**: Various (FID: Heusel et al. 2017; KID: Bińkowski et al. 2018)
**Summary**: FID computes Fréchet distance between Gaussian-fitted Inception feature distributions. KID uses polynomial kernel MMD. Both compare feature distributions between image sets. Limitations: FID assumes Gaussianity and is biased; KID is unbiased but sensitive to kernel choice.
**Relevance**: A "Document-FID" using DiT (#81) features instead of Inception features could compare document image distributions. Useful for evaluating our metric across a dataset rather than single images.

---

## Synthesis: Font, Representation Learning & Information Theory (2026-03-25)

### Font Recognition Insights

1. **VLMs cannot be trusted for font recognition** (#76) — validates our CV-based approach
2. **FontCLIP** (#75) provides the best available font similarity metric — could measure typography match between original and reconstruction
3. **Edge/stroke features are the key signal** (#78) — our DocumentAnalyzer's focus on stroke width analysis is well-motivated
4. **Proposed addition**: FontCLIP similarity as a typography-specific metric component

### The "Document LPIPS" Opportunity

**No document-specific perceptual metric exists** (#84). This is a research gap we could fill:
- Use **ColPali** (#80) or **DiT** (#81) as backbone (pre-trained on documents, not ImageNet)
- Train learned linear weights on intermediate features (like LPIPS)
- Supervise with OCR accuracy as proxy for perceptual quality
- This would be a genuine novel contribution

### Information-Theoretic Additions

| Metric | Type | Key Advantage | Effort |
|--------|------|---------------|--------|
| **VIF** (#85) | Full-reference | Information preservation, used in VMAF | Low (torchmetrics) |
| **CMMD** (#86) | Distribution | No pixel alignment, unbiased | Low (pip) |
| **DISTS** (#87) | Full-reference | Separates structure vs texture quality | Low (pip) |
| **SSEQ** (#88) | No-reference | Independent reconstruction validation | Medium |

## Image-to-Image Reconstruction for Documents (2026-03-25)

This section surveys neural approaches for reconstructing document images from OCR-extracted text + layout, potentially conditioned on the original image for style preservation. The goal: replace or augment deterministic Pillow-based rendering with models that better capture visual style.

### 90. Glyph-ByT5 / Glyph-ByT5-v2 (ECCV 2024)
**Title**: A Customized Text Encoder for Accurate Visual Text Rendering
**Authors**: Liu et al. (Microsoft Research Asia / Tsinghua)
**Link**: https://github.com/AIGText/Glyph-ByT5
**Summary**: Fine-tunes character-aware ByT5 encoder on 1M+ glyph-text pairs, integrates with SDXL. Improves text rendering accuracy from ~20% to ~90% on design benchmarks. v2 extends to 10 languages (French, Spanish, Chinese, Japanese, Korean) with 10M graphic design pairs.
**Key capability**: Paragraph-level text rendering (tens to hundreds of characters).
**Relevance**: Currently best text rendering accuracy in diffusion models. Could generate document pages with accurate text, but does not preserve original document style (fonts/colors/textures).

### 91. AnyText / AnyText2 (Alibaba, ICLR 2024 / arXiv 2024)
**Title**: Multilingual Visual Text Generation and Editing
**Authors**: Tuo et al. (Alibaba Tongyi Lab)
**Link**: https://arxiv.org/abs/2311.03054
**Summary**: Text Embedding Module (glyph + position + font + color encoders) integrated with ControlNet-like network. AnyText2 introduces WriteNet + AttnX architecture that decouples text rendering from image generation. English +9.3%, Chinese +3.3% accuracy over AnyText. Only 0.34% parameter overhead on SD.
**Relevance**: Most practical for our use case — can render OCR text at specified bbox positions. However, uses Arial Unicode for glyph rendering, not original document fonts.

### 92. TextDiffuser-2 (Microsoft, ECCV 2024)
**Title**: Unleashing the Power of Language Models for Text Rendering
**Authors**: Chen et al. (Microsoft Research)
**Link**: https://jingyechen.github.io/textdiffuser2/
**Summary**: Two-stage: LLM-based layout planning + line-level conditioned diffusion. Automates layout planning via chat. 6.51% F-measure improvement with GlyphControl. Line-level encoding more flexible than character masks.
**Relevance**: Good automated layout, but focuses on legibility over style preservation. Not suitable for matching original document appearance.

### 93. UDiffText (Peking University, ECCV 2024)
**Title**: A Unified Framework for High-quality Text Synthesis in Arbitrary Images
**Authors**: Zhao & Lian
**Link**: https://arxiv.org/abs/2312.04884
**Summary**: Character-level text encoder (replaces CLIP) + local attention control with character-level segmentation maps + inference-stage refinement. Built on SD 2.0 inpainting. Supports text generation, scene text editing, and T2I with precise text.
**Relevance**: Inpainting variant makes it suitable for editing text in existing document images — could replace text regions with OCR text while preserving background.

### 94. TextCtrl (NeurIPS 2024)
**Title**: Diffusion-based Scene Text Editing with Prior Guidance Control
**Link**: https://github.com/weichaozeng/TextCtrl
**Summary**: Explicit text style disentanglement + glyph structure guidance + adaptive inference. Outperforms MOSTEL (GAN), DiffSTE, TextDiffuser, AnyText on style preservation. Addresses the critical style deviation problem in diffusion-based text editing.
**Relevance**: **Most relevant for OCR-conditioned reconstruction** — can edit text content while preserving the original document's visual style (font, color, weight).

### 95. TextPixs (2025)
**Title**: OCR-in-the-Loop Fine-Tuning for Accurate Text Rendering
**Summary**: Achieves SOTA text rendering with CER 0.08 through OCR-in-the-loop training. Uses OCR recognition loss during diffusion fine-tuning to ensure generated text is actually readable.
**Relevance**: Demonstrates that OCR-guided training dramatically improves text fidelity. Principle could be applied to document reconstruction training.

### 96. DocDiff (ACM Multimedia 2023)
**Title**: Document Enhancement via Residual Diffusion Models
**Authors**: Zhang et al.
**Link**: https://arxiv.org/abs/2305.03892
**Summary**: Coarse Predictor + High-Frequency Residual Refinement (HRR). Only 4.17M parameters in HRR module (12GB VRAM at 128x128). Tasks: deblurring, denoising, watermark/seal removal.
**Relevance**: Could enhance deterministic renderings to look more realistic. The residual approach adds texture/detail without changing text content.

### 97. Uni-DocDiff (ACM Multimedia 2025)
**Title**: A Unified Document Restoration Model Based on Diffusion
**Link**: https://arxiv.org/html/2508.04055
**Summary**: Multi-task document restoration: deblurring, deshadowing, illumination correction, binarization, handwriting removal, dewarping. Uses learnable task prompts for scalability.
**Relevance**: Unified restoration framework. Could add a "style enhancement" task to make deterministic renderings more document-like.

### 98. TextStyleBrush (Meta AI, TPAMI 2023)
**Title**: Transfer of Text Aesthetics from a Single Example
**Summary**: One-shot text style transfer from a single word image. StyleGAN2-based generator with disentangled style/content. Self-supervised training with OCR + typeface classifier losses.
**Relevance**: Excellent style matching per word region. Could transfer style from each text region in the original document to OCR text. Limited to word-level, not full pages.

### 99. ControlNet for Documents (ICCV 2023 / ICDAR 2024)
**Title**: Adding Conditional Control to Text-to-Image Diffusion Models
**Authors**: Zhang et al. (Stanford)
**Link**: https://github.com/lllyasviel/ControlNet
**Summary**: Copies SD encoder weights into trainable branch with zero-convolution injection. Conditioning: edge maps, depth, segmentation. ICDAR 2024 extension: ControlNet for Chinese text layout with controllable arrangement direction and curvature.
**Relevance**: Could condition on document layout masks, but text rendering would be illegible without text-specific conditioning (Glyph-ByT5 or similar).

### 100. DocSynthv2 (Adobe, CVPR Workshop 2024)
**Title**: A Practical Autoregressive Modeling for Document Generation
**Authors**: Biswas et al.
**Link**: https://arxiv.org/abs/2406.08354
**Summary**: Autoregressive GPT-2 Transformer decoder. Input tokens: layout category + position + font style + text content. Generates structured document data (not images). Spotlight (Oral) at CVPR 2024 GDUG Workshop.
**Relevance**: Generates document structure combining layout + text, but no visual rendering. Could be combined with an image synthesis model.

### 101. InstanceDiffusion (Meta/UC Berkeley, CVPR 2024)
**Title**: Instance-level Control for Image Generation
**Link**: https://arxiv.org/abs/2402.03290
**Summary**: UniFusion + ScaleU + Multi-instance Sampler. Supports points, scribbles, bounding boxes, instance masks + per-instance text. 2x higher AP50 than GLIGEN for box inputs.
**Relevance**: Best current layout-to-image option for precise control, but text rendering still limited by base SD model.

### 102. GSDM — Text Image Inpainting (AAAI 2024)
**Title**: Text Image Inpainting via Global Structure-Guided Diffusion Models
**Summary**: Structure Prediction Module + Reconstruction Module. Restores corrupted text while preserving original style. Uses global structure priors.
**Relevance**: Could be adapted for a two-stage approach: (1) remove original text with text erasure, (2) inpaint OCR text using style guidance.

### 103. VQ-Font (ICCV 2023) & FontDiffuser (2024)
**Title**: Few-Shot Font Generation via Similarity-Guided Global/Local Styles
**Summary**: VQ-Font: generates complete character sets from few reference characters. FontDiffuser: diffusion-based font generation with multi-scale content aggregation.
**Relevance**: Could generate document-specific fonts from a few character samples, then use those fonts for improved deterministic rendering. Adds complexity but improves style fidelity.

### 104. DCDM — Diffusion-Conditioned-Diffusion Model (ECCV 2024)
**Title**: Scene Text Image Super-Resolution via Dual Diffusion
**Summary**: Combines diffusion-based text prior generation as conditioning for image diffusion model. Specifically for scene text image super-resolution (STISR).
**Relevance**: The dual-conditioning architecture (text prior + image) is conceptually similar to our OCR text + original image → reconstruction approach.

## Synthesis: Image-to-Image Reconstruction for OCR Evaluation (2026-03-25)

### The Core Challenge

No existing model directly implements our exact workflow: **original document image + OCR-extracted text → reconstructed document image preserving original style**. This represents a genuine research gap.

### Three Viable Approaches

**Approach 1: Text Inpainting (Most Promising for Our Use Case)**
1. Mask text regions in original image (using Hi-SAM or CRAFT)
2. Re-render OCR text into masked regions using TextCtrl or UDiffText
3. Preserves background, layout, and non-text elements perfectly
4. TextCtrl achieves best style preservation; UDiffText built on inpainting SD
- **Pros**: Style preservation, original background intact
- **Cons**: Requires accurate text masks, region-by-region processing

**Approach 2: Hybrid Deterministic + Neural Enhancement**
1. Render text deterministically (current Pillow approach, 100% text accuracy)
2. Apply DocDiff or Uni-DocDiff for texture/realism enhancement
3. Optional: Use few-shot font generation (VQ-Font) for better font matching
- **Pros**: Guaranteed text accuracy, existing pipeline compatible
- **Cons**: Enhancement may not fully match original style

**Approach 3: Full Neural Generation**
1. Extract style features from original (IP-Adapter or CLIP)
2. Render text with Glyph-ByT5-v2 (~90% accuracy) or AnyText2
3. Condition on layout via ControlNet/InstanceDiffusion
- **Pros**: End-to-end neural, potentially most realistic
- **Cons**: ~70-90% text accuracy introduces noise in OCR evaluation

### Critical Trade-Off: Text Accuracy vs. Style Fidelity

For an OCR **evaluation metric**, text accuracy must be prioritized:
- Neural text rendering at ~70-90% accuracy would corrupt the evaluation signal
- A reconstruction that looks different but has correct text is more useful than one that looks similar but has rendering errors
- **Recommendation**: Use Approach 2 (deterministic rendering + neural enhancement) as default
- **Experiment**: Test Approach 1 (text inpainting) as an advanced option for better visual metrics

### Text Rendering Accuracy Landscape (2024-2025)

| Model | Text Accuracy | Style Control | Speed | Best For |
|-------|---------------|---------------|-------|----------|
| Glyph-ByT5-v2 | ~90% | Line-level | Slow | Paragraph rendering |
| AnyText2 | ~75% | Position/font/color | Medium | Multilingual |
| TextPixs | CER 0.08 | Limited | Slow | Maximum accuracy |
| TextDiffuser-2 | ~70% | Line-level | Medium | Auto-layout |
| UDiffText | High | Character masks | Medium | Inpainting/editing |
| TextCtrl | High | Style-preserving | Medium | Style transfer |
| Pillow (ours) | 100% | Font/size/color | Fast | Guaranteed accuracy |

## Embedding Scaling & DeepSeek Engram (2026-04-01)

This section evaluates the emerging embedding scaling paradigm for potential relevance to reference-free OCR evaluation.

### 105. Engram — Conditional Memory via Scalable Lookup (DeepSeek/PKU, January 2026)
**Title**: Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
**Authors**: Cheng et al. (Peking University, DeepSeek-AI)
**Link**: https://arxiv.org/abs/2601.07372
**Summary**: Introduces "conditional memory" — hash-based n-gram embedding lookup tables as a complement to MoE expert routing. LLMs perform two sub-tasks: compositional reasoning (needs deep computation) and knowledge retrieval (static lookup). Engram provides O(1) hash-based retrieval for static patterns, freeing Transformer depth for reasoning. Discovers a U-shaped scaling law: ~75-80% sparse params to MoE experts + ~20-25% to Engram memory is optimal. Engram-27B beats iso-parameter MoE-27B on MMLU (+3.0), BBH (+5.0), HumanEval (+3.0).
**Relevance**: **None for our project.** Embedding scaling is an LLM internal architecture technique for knowledge storage efficiency. Our pipeline operates on image-level and text-level comparisons, not LLM internals.

### 106. QualiCLIP — No-Reference Image Quality via CLIP (2024-2025)
**Title**: Quality-Aware Image-Text Alignment for No-Reference Image Quality Assessment
**Authors**: Agnolucci et al.
**Link**: https://arxiv.org/abs/2403.11176
**Summary**: Fine-tunes CLIP with quality-aware text prompts to produce a no-reference image quality score in [0,1]. SOTA on UHD-IQA benchmarks. Available in IQA-PyTorch toolbox (`pip install pyiqa`).
**Relevance**: **High.** A no-reference quality metric that could complement our reconstruction-based approach. Could assess both original and reconstructed image quality independently without needing the comparison between them.

### 107. SCONE — Scalable Contextualized Offloaded N-gram Embedding (Google, February 2025)
**Title**: Scaling Embedding Layers in Language Models
**Authors**: Yu et al. (Google)
**Link**: https://arxiv.org/abs/2502.01637
**Summary**: 1B model + 1B f-gram embeddings beats 1.9B baseline at half the FLOPs. Demonstrates embedding scaling outperforms expert scaling at high sparsity ratios.
**Relevance**: **None for our project.** Same category as Engram — LLM architecture innovation.

### Synthesis: Embedding Scaling Assessment

The embedding scaling field (Engram, SCONE, SuperBPE, BLT, LongCat-Flash-Lite) is a significant LLM architecture trend but has **no direct application** to our reference-free OCR metric. These techniques optimize how language models store and retrieve knowledge internally.

The one actionable discovery from this research: **QualiCLIP** (#106) provides a no-reference CLIP-based quality metric that could serve as an independent signal alongside our reconstruction-based metrics.

## Diffusion-Based Document OCR Paradigm (2026-04-01)

A major paradigm shift: replacing autoregressive (AR) token-by-token OCR decoding with parallel discrete diffusion. The key insight is that OCR is deterministic (image strictly dictates output), making it uniquely suited for masked diffusion models that decode all tokens in parallel.

### 108. MinerU-Diffusion — OCR as Inverse Rendering (Shanghai AI Lab, March 2026)
**Title**: Rethinking Document OCR as Inverse Rendering via Diffusion Decoding
**Authors**: Dong, Niu, Wang, Zeng, Zhang, He (Shanghai AI Lab, Peking University)
**Link**: https://arxiv.org/abs/2603.22458 | https://github.com/opendatalab/MinerU-Diffusion
**Model**: 2.5B parameters (Vision Encoder from Qwen2-VL-7B + SDAR-1.7B diffusion decoder)
**Summary**: Reconceptualizes OCR as *inverse rendering* — recovering structured text that was rendered into images. Uses block-wise discrete diffusion: bidirectional attention within blocks, causal across blocks, reducing complexity from O(L^2) to near-linear. Achieves 93.37 on OmniDocBench v1.5 (matching AR SOTA) at **3.2x speedup** (165 TPS vs 52 TPS). Training uses uncertainty-driven curriculum learning: stochastic multi-pass inference identifies hard cases for targeted training.
**Key innovation — Semantic Shuffle Benchmark**: Shuffles words in documents and re-renders them. AR models degrade 60-80% on nonsense text (relying on linguistic priors). MinerU-Diffusion remains nearly constant — proving it truly reads visual signal rather than guessing.
**Native output**: Layout bboxes + element categories + rotation + text + LaTeX formulas + OTSL tables.
**Relevance**: **Very High.**
- Native layout+bbox output could solve our 50% bbox extraction failure
- Diffusion decoding reduces hallucination (less linguistic prior dependence)
- "Inverse rendering" framing is conceptually complementary to our "forward rendering" reconstruction
- Semantic Shuffle diagnostic directly applicable to validating our metric
- Per-token confidence (tau threshold) could serve as OCR quality signal

### 109. DODO — Discrete OCR Diffusion Models (February 2026)
**Title**: DODO: Discrete OCR Diffusion Models
**Authors**: Man, Ganz, Ronen, Tsiper, Mazor, Nayman
**Link**: https://arxiv.org/abs/2602.16872
**Summary**: Independent parallel effort to MinerU-Diffusion. Uses Qwen2.5-VL-3B backbone with block size 256. DODO-fast achieves ~66 TPS (~3x AR baseline). NED 0.066 on OmniDocBench (290 English docs). Does not handle layout detection, tables, or formulas (text transcription only).
**Relevance**: Medium. Validates the diffusion-for-OCR paradigm but less complete than MinerU-Diffusion for our full-pipeline needs.

### 110. MinerU2.5 — Decoupled VLM for Document Parsing (September 2025)
**Title**: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing
**Authors**: OpenDataLab
**Link**: https://arxiv.org/abs/2509.22186
**Summary**: 1.2B parameter AR model. Coarse-to-fine two-stage: layout on downsampled images, then content recognition on native-resolution crops. Atomic Decomposition & Recombination for long formulas. Outperforms GPT-4o and Gemini-2.5 Pro on OmniDocBench despite only 1.2B params.
**Relevance**: Medium. The predecessor to MinerU-Diffusion; MinerU-Diffusion directly replaces its AR decoder with diffusion.

### 111. LLaDA — Large Language Diffusion with mAsking (NeurIPS 2025 Oral)
**Title**: Large Language Diffusion Models
**Link**: https://arxiv.org/abs/2502.09992
**Summary**: 8B parameter masked diffusion language model, pre-trained from scratch on 2.3T tokens. Competitive with LLaMA3 8B. Foundation for the discrete diffusion LLM paradigm. LLaDA-V extension (May 2025) noted as "struggling with OCR tasks" — the limitation that DODO and MinerU-Diffusion specifically address.
**Relevance**: Low (foundational work). Important context for understanding why diffusion suits OCR.

### 112. SDAR — Synergy of Diffusion and AutoRegression (October 2025)
**Title**: SDAR: Lightweight Paradigm Conversion from AR to Diffusion
**Link**: https://arxiv.org/abs/2510.06303
**Summary**: Converts pretrained AR models into block-wise diffusion models through brief data-efficient adaptation. MinerU-Diffusion's decoder is initialized from SDAR-1.7B-Chat (block size 32). Models from 1.7B to 30B. Any AR decoder-only model can be adapted.
**Relevance**: Medium. The enabling technology for MinerU-Diffusion's decoder. Could theoretically convert other AR OCR models to diffusion.

### 113. SmolDocling / Granite-Docling (IBM, ICCV 2025)
**Title**: SmolDocling: Ultra-Compact VLM for Document Conversion
**Link**: https://arxiv.org/abs/2503.11576
**Summary**: 256M parameters, introduces DocTags format separating content from structure. 0.35 seconds per page on consumer GPU, under 500MB VRAM. Granite-Docling-258M is the production evolution (Apache 2.0). End-to-end: bboxes, reading order, hierarchical linking.
**Relevance**: Medium. Ultra-compact alternative OCR option. DocTags format interesting for structured output.

### 114. Layout-Aware OCR Unsupervised Evaluation (ACM FAccT 2026)
**Title**: Layout-Aware OCR in Black Digital Archives: An Unsupervised Evaluation Approach
**Authors**: Beyene & Dancy
**Link**: https://arxiv.org/abs/2509.13236
**Summary**: Three reference-free metrics: Semantic Coherence Score (SCS, fraction of valid dictionary words), Region Entropy Divergence (RED, information diversity across regions), Textual Redundancy Score (TRS, repetitive text artifacts). Purely text-based, no visual reconstruction.
**Relevance**: **High.** Closest published work to our reference-free evaluation goal. Their text-only approach is complementary to our visual reconstruction method. Combining both could yield a more robust reference-free metric.

## Synthesis: Diffusion OCR & Evaluation Implications (2026-04-01)

### The Inverse Rendering Loop

MinerU-Diffusion frames OCR as *inverse rendering*. Our project does *forward rendering*. Together they form a conceptual loop:

```
Original Image → [Inverse Rendering: OCR] → Structured Text+Layout
Structured Text+Layout → [Forward Rendering: Reconstruction] → Reconstructed Image
Reconstructed Image vs Original → [Visual Metrics] → Quality Score
```

This framing strengthens our theoretical foundation: the quality score measures the fidelity loss through the inverse→forward rendering cycle.

### Why Diffusion OCR Matters for Our Metric

1. **Reduced hallucination**: AR models guess from linguistic priors; diffusion models read visual signal. For an OCR *evaluation* metric, the OCR engine itself should minimize hallucination — otherwise our metric penalizes the engine for being too creative rather than too inaccurate.

2. **Native layout output**: MinerU-Diffusion outputs bboxes + categories natively, potentially solving our 50% bbox extraction failure rate without any prompt engineering.

3. **Semantic Shuffle as metric validation**: If we shuffle words in test documents and our metric score changes dramatically, it means our pipeline is sensitive to semantics rather than visual fidelity. A good reconstruction metric should be invariant to semantic content.

4. **Per-token confidence as quality signal**: The diffusion threshold tau provides per-token confidence. Low-confidence tokens indicate ambiguous visual regions — exactly where OCR errors concentrate.

### MinerU-Diffusion vs Qwen3-VL for Our Pipeline

| Aspect | Qwen3-VL (current) | MinerU-Diffusion |
|--------|-------------------|------------------|
| Architecture | AR (autoregressive) | Block-wise diffusion |
| Layout output | Via prompt engineering (50% failure) | Native (bboxes + categories) |
| Speed | ~52 TPS (comparable) | ~165 TPS (3.2x faster) |
| Hallucination | Higher (linguistic priors) | Lower (visual grounding) |
| Table/Formula | Text only | OTSL tables + LaTeX formulas |
| Model size | 7B | 2.5B |
| Deployment | vLLM | SGLang / Nano-vLLM |
| License | Apache 2.0 | MIT |

## Advanced Mathematical Spaces for Document Comparison (2026-04-06)

Explores complex numbers, Hilbert spaces, optimal transport, Riemannian manifolds, and other mathematical frameworks for measuring document image similarity beyond standard pixel-level metrics.

### 115. DeepWSD — Wasserstein Distance in Deep Feature Space (ACM MM 2022)
**Authors**: Liao, Chen, Zhu, Wang, Zhou, Kwong
**Link**: https://arxiv.org/abs/2208.03323
**Summary**: Measures image quality as 1D Wasserstein distance between VGG deep feature distributions of reference and distorted images. Unlike pixel-wise SSIM or learned LPIPS, DeepWSD captures distributional shifts at multiple VGG stages. Satisfies true metric definition (triangle inequality). Works as both quality predictor and perceptual loss. Open-source code available.
**Relevance**: **Very High.** Fundamentally different signal from SSIM/LPIPS. Low implementation effort (VGG-based, pip-installable). Could detect distributional differences in text rendering quality that pixel-level metrics miss.

### 116. MS-SWD — Multiscale Sliced Wasserstein Distance (ECCV 2024)
**Authors**: He, Wang, Wang, Liu, Fang, Sun, Ma
**Link**: https://arxiv.org/abs/2407.10181
**Summary**: Perceptual metric based on multiscale sliced Wasserstein distance in CIELAB space. Training-free, O(M log M) via 1D sorting. **Uniquely robust to image misalignment** — enables non-local patch comparison unlike co-located pixel methods (SSIM). Available in IQA-PyTorch.
**Relevance**: **Very High.** Addresses a critical weakness of SSIM: our reconstructed documents inevitably have slight positional differences. MS-SWD compares patch distributions across the image, tolerating small shifts. Very low effort to integrate.

### 117. GLIPS — Global-Local Image Perceptual Score (2024)
**Authors**: Danish et al.
**Link**: https://arxiv.org/abs/2405.09426
**Summary**: Combines transformer attention-weighted local patch similarity with MMD for global distributional comparison. Interpolative Binning Scale aligns with human judgments. Outperforms FID, SSIM, MS-SSIM.
**Relevance**: **High.** Local+global comparison architecture matches our needs. Attention-based weighting could focus on text-heavy regions.

### 118. Dual-Stream Complex-Valued CNN for IQA (IEEE TIP 2024)
**Authors**: Guan, Li, Zheng, Wu, Bovik
**Link**: https://ieeexplore.ieee.org/document/10375348/
**Summary**: First complex-valued CNN applied to image quality assessment. Dual-stream: distortion-sensitive (RGB) + domain-aware. Phase captures structural relationships that magnitude-only metrics miss. Outperforms SOTA NR-IQA methods.
**Relevance**: **Medium-High.** Phase information could capture text edge coherence and alignment consistency. Novel for document IQA.

### 119. OT for Sub-Letter Orthographic Processing (Cognitive Science, 2025)
**Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12534030/
**Summary**: Wasserstein distance for character shape comparison aligns with human perceptual judgments of letter similarity. Transformation-invariant (translation, rescaling). Bridges computational vision and cognitive science of reading.
**Relevance**: **High.** Character-level Wasserstein metric for comparing individual glyphs between original and reconstruction. Novel signal complementing text-level perplexity and image-level SSIM.

### 120. HSIC for Feature Dependence Measurement (ICCV 2023)
**Link**: https://openaccess.thecvf.com/content/ICCV2023/html/Guo_Automatic_Network_Pruning_via_Hilbert-Schmidt_Independence_Criterion_Lasso_under_Information_ICCV_2023_paper.html
**Summary**: HSIC measures statistical dependence between feature sets in RKHS without density estimation. Applied to network pruning but the metric itself is general-purpose.
**Relevance**: **Medium.** HSIC between original and reconstruction feature maps would measure information-theoretic fidelity — a fundamentally different signal from perceptual similarity.

### 121. HypStructure — Hyperbolic Embeddings for Hierarchy (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2412.01023
**Summary**: Regularizes visual representations using hyperbolic geometry to capture label hierarchies. Embeds hierarchical structure naturally via exponential volume growth of hyperbolic space.
**Relevance**: **Medium.** Documents have natural hierarchy (page > section > paragraph > line > character). Hyperbolic document comparison could capture structural fidelity at multiple levels. Novel research direction.

### 122. CliffordNet — Geometric Algebra Vision Backbone (2026)
**Authors**: Ji
**Link**: https://arxiv.org/abs/2601.06793
**Summary**: Uses Clifford Geometric Product (uv = u·v + u∧v) to simultaneously capture coherence (inner product) and variation (wedge product). 8x fewer parameters than ResNet-18 at equal accuracy. Linear O(N) complexity.
**Relevance**: **Medium.** Geometric product naturally decomposes comparison into "what matches" and "what differs" — conceptually ideal for original vs. reconstruction comparison. Very novel.

### 123. SPD Manifold Deep Metric Learning (IEEE TNNLS 2024)
**Link**: https://ieeexplore.ieee.org/document/10467142/
**Summary**: Deep metric learning on SPD (Symmetric Positive Definite) manifold. Represents image sets as covariance matrices with Riemannian distance. Captures second-order statistics.
**Relevance**: **Medium.** Document region covariance matrices on SPD manifold could capture texture/structure statistics that first-order metrics cannot.

## RL with Reconstruction-as-Reward for OCR Training (2026-04-06)

Explores using visual similarity between original and reconstructed document images as a reward signal for reinforcement learning to train OCR models. This is a novel architecture where OCR quality improves through a render-and-compare cycle.

### 124. CycleCap — Cycle Consistency GRPO for VLM Captioning (March 2026)
**Authors**: Krestenitis et al.
**Link**: https://arxiv.org/abs/2603.18282
**Summary**: **Exact analog of our proposed architecture applied to captioning.** VLM generates caption → text-to-image model reconstructs image → DreamSim similarity between original and reconstruction becomes GRPO reward. No ground-truth labels needed. Applied to 4 VLMs (1B-7B), consistent improvements on captioning and hallucination benchmarks. Surpasses supervised CycleReward.
**Relevance**: **Critical.** This IS our proposed OCR architecture applied to captioning. Replace "captioning model" with "OCR model," replace "text-to-image model" with "our HTML renderer." Validates the core idea and provides a working GRPO recipe.

### 125. olmOCR 2 — Unit Test Rewards for Document OCR (October 2025)
**Authors**: Poznanski, Soldaini, Lo et al.
**Link**: https://arxiv.org/abs/2510.19817
**Summary**: 7B VLM trained with GRPO using binary unit test rewards. Renders clean HTML from ground-truth, generates verifiable unit tests as reward signals. +14.2 points on olmOCR-Bench. Largest gains in math formulas, tables, multi-column layouts.
**Relevance**: **Critical.** Closest existing OCR+RL system. Uses text-based unit tests as rewards; we propose visual similarity as reward. The two approaches are complementary.

### 126. Infinity-Parser — Layout-Aware RL for Document Parsing (June 2025)
**Authors**: Wang, Wu et al.
**Link**: https://arxiv.org/abs/2506.03197
**Summary**: layoutRL: GRPO with composite reward of normalized edit distance + paragraph count accuracy + reading order preservation. Fine-tunes Qwen2.5-VL-7B. SOTA on OmniDocBench, olmOCR-Bench. Constructs Infinity-Doc-400K dataset.
**Relevance**: **High.** Demonstrates multi-aspect reward design for OCR with GRPO. Their composite reward approach maps to our multi-metric (SSIM + LPIPS + CLIP) reward.

### 127. Perception-R1 — Visual Perception Reward for MLLMs (2025)
**Links**: https://arxiv.org/abs/2506.07218, https://arxiv.org/abs/2504.07954
**Summary**: Two papers applying GRPO to visual perception including OCR. Achieves +4.2% F1 on PageOCR with Qwen2-VL-2B. Demonstrates that RL with perception rewards dramatically improves visual understanding.
**Relevance**: **High.** Directly validates that GRPO improves OCR on VLMs. Shows even 2B models benefit substantially.

### 128. DanceGRPO — Multi-Reward GRPO for Visual Generation (May 2025)
**Authors**: ByteDance/HKU
**Link**: https://arxiv.org/abs/2505.07818
**Summary**: First unified GRPO for visual generation. Uses 5 reward models simultaneously. Key insight: sum group-normalized advantages rather than raw scores for multi-reward stability. Up to 181% improvement.
**Relevance**: **High.** Directly applicable multi-reward normalization technique for combining our SSIM + LPIPS + CLIP rewards within GRPO.

### 129. RLVR-World — LPIPS/SSIM as GRPO Rewards (NeurIPS 2025)
**Authors**: Tsinghua University
**Link**: https://arxiv.org/abs/2505.13934
**Summary**: Uses GRPO with LPIPS, SSIM, PSNR, MSE as verifiable rewards for video world models. Only hundreds of gradient steps needed (vs. hundreds of thousands for MLE). Reduces artifacts from 48.6% to 9.9%.
**Relevance**: **High.** Directly demonstrates our exact metrics (LPIPS, SSIM) working as GRPO rewards. Validates sample efficiency.

### 130. SPO — Segment Policy Optimization (NeurIPS 2025)
**Link**: https://arxiv.org/abs/2505.23564
**Summary**: Segment-level advantage estimation — middle ground between token-level (PPO) and trajectory-level (GRPO). Partitions output into segments with per-segment rewards. 6-12pp improvement over GRPO.
**Relevance**: **High.** Solves our credit assignment problem. Segment OCR output by document regions → compute per-region SSIM/LPIPS → provide localized feedback instead of single page-level score.

### 131. Format Decoupled RL for Document OCR (December 2025)
**Authors**: Zhong et al.
**Link**: https://arxiv.org/abs/2601.08834
**Summary**: Decouples format (layout/structure) and content (text accuracy) when applying GRPO to OCR. Separate reward signals for each. Improves both aspects.
**Relevance**: **High.** Our visual similarity metric conflates layout and text accuracy. This paper argues for decomposition — directly applicable to our reward design.

### 132. SCST — Self-Critical Sequence Training (IBM, CVPR 2017)
**Authors**: Rennie et al.
**Link**: https://arxiv.org/abs/1612.00563
**Summary**: Foundational work using REINFORCE to optimize non-differentiable metrics for image captioning. Uses model's greedy decode as variance-reduction baseline. CIDEr improved from 104.9 to 114.7.
**Relevance**: **Foundational.** Direct ancestor of our approach. Replace CIDEr with visual similarity as reward.

### 133. VLM-R1 — Open-Source GRPO for VLMs (April 2025)
**Link**: https://arxiv.org/abs/2504.07615
**Summary**: Open-source reproduction of DeepSeek-R1 for Qwen2.5-VL. Provides training recipes and codebase for applying GRPO to VLMs. RL model generalizes to OOD data while SFT deteriorates.
**Relevance**: **High.** Directly reusable infrastructure for applying GRPO to our Qwen-based OCR pipeline.

## Differentiable OCR-to-Rendering Pipeline (2026-04-06)

Investigates making the entire OCR → render → compare pipeline differentiable for end-to-end backpropagation. The key challenge: text tokens are discrete, but gradients need to flow through them to the rendering step.

### 134. DiffVG — Differentiable Vector Graphics Rasterization (SIGGRAPH Asia 2020)
**Authors**: Li, Lukac, Gharbi, Ragan-Kelley
**Link**: https://people.csail.mit.edu/tzumao/diffvg/
**Summary**: Foundation for differentiable 2D rendering. Computes unbiased gradients for each pixel w.r.t. Bezier curve control points, fill colors, stroke parameters. Text glyphs are Bezier curves, so DiffVG can render them differentiably.
**Relevance**: **High.** Core building block for differentiable text rendering in our pipeline.

### 135. Differentiable Variable Fonts (U.Toronto/Adobe/NVIDIA, 2025)
**Authors**: Parikh, Kaufman, Levin, Jacobson
**Link**: https://arxiv.org/abs/2510.07638
**Summary**: Fully differentiable formulation of variable font interpolation. Optimizes font axis weights (weight, width, slant) against image loss via gradient descent + DiffVG. Demonstrates recovering best-matching font instance from raster target by backpropagating through rasterization.
**Relevance**: **High.** Demonstrates the exact primitive we need: optimizing text appearance by comparing rendered output against target image via gradients.

### 136. GRADE — Gumbel-Softmax Replacing Policy Gradients (December 2025)
**Authors**: Nel
**Link**: https://arxiv.org/abs/2601.11574
**Summary**: Replaces REINFORCE/PPO with direct backpropagation through discrete token generation via Gumbel-Softmax + STE. Forward: discrete tokens. Backward: continuous gradients. 14x lower gradient variance, 50% better alignment than PPO.
**Relevance**: **High.** Directly solves the discrete bottleneck. If OCR outputs token logits, GRADE flows gradients from image comparison loss backward through discrete selection to OCR parameters.

### 137. Decoupled Straight-Through Gumbel-Softmax (2024)
**Authors**: Shah, Yan, Mozer, Liu
**Link**: https://arxiv.org/abs/2410.13331
**Summary**: Standard ST-GS is sensitive to temperature. Proposes decoupled temperatures for forward (discrete quality) and backward (gradient accuracy) passes. Significantly improves ST-GS across tasks.
**Relevance**: **High.** Practical improvement for implementing Gumbel-Softmax in our pipeline. Balances token quality with gradient signal quality.

### 138. Soft-DiMO — Soft Embeddings for Discrete Generation (2025)
**Link**: https://arxiv.org/abs/2509.22925
**Summary**: Replaces hard one-hot token selections with soft embeddings via teacher model's embedding layer. Enables differentiable reward fine-tuning on discrete generation models. SOTA one-step performance.
**Relevance**: **High.** Instead of rendering discrete OCR tokens, render "soft token mixtures" and backpropagate through the weighted embedding lookup. Avoids Gumbel-Softmax entirely.

### 139. TRACE — Differentiable Stroke Recovery for Handwriting (ICDAR 2021)
**Authors**: Archibald, Poggemann, Chan, Martinez
**Link**: https://arxiv.org/abs/2105.11559
**Summary**: CRNN infers temporal stroke trajectory from offline handwriting images, with DTW alignment. The recovered trajectory is differentiable end-to-end. First system trained on entire text lines of arbitrary width.
**Relevance**: **High.** Closest existing "inverse rendering" system for text. TRACE solves image→strokes differentiably; we need image→text→render→compare.

### 140. Stochastic Gradient Estimation for Rasterization (I3D/SIGGRAPH 2024)
**Authors**: Deliot, Heitz, Belcour (Intel)
**Link**: https://arxiv.org/abs/2404.09758
**Summary**: Makes any existing rasterizer differentiable with minimal effort via stochastic gradient estimation. Randomly perturbs parameters, estimates gradients from pixel changes. Successfully optimized 1M+ parameter scenes on consumer GPU.
**Relevance**: **High.** Pragmatic path: wrap our existing Pillow/HTML renderer in stochastic gradient estimation to make it approximately differentiable without rewriting from scratch.

### 141. Applicability Limitations of Differentiable IQA Metrics (IEEE 2023)
**Link**: https://ieeexplore.ieee.org/document/10125387/
**Summary**: Shows differentiable IQA metrics (SSIM, LPIPS, DISTS, HaarPSI, VIF) can be adversarially manipulated — neural preprocessing can increase LPIPS by 36.8% while quality drops. Metrics designed for natural images, not text legibility.
**Relevance**: **Critical warning.** If we optimize OCR by backpropagating through SSIM/LPIPS, the model may exploit metric weaknesses. Motivates using OCR-specific losses (CRAFT features) rather than generic metrics as training signal.

### 142. Bezier Splatting — Fast Differentiable Vector Graphics (2025)
**Link**: https://arxiv.org/abs/2503.16424
**Summary**: 150x faster backward computation than DiffVG via Gaussian splatting along Bezier curves. Direct positional gradients at boundaries. Preserves fine-grained texture.
**Relevance**: **High.** Makes differentiable rendering at document resolution practically feasible. Already in registry (#45) for geometric modeling but not evaluated for differentiable training.

## Synthesis: Three Novel Research Architectures (2026-04-06)

### Architecture 1: GRPO with Visual Reconstruction Reward (Most Practical)

Validated by CycleCap (#124): OCR model generates text → render to image → visual similarity as GRPO reward. No ground truth needed.

```
OCR Model → [sample N outputs] → Render each → Compare with original
                                                      ↓
GRPO: reward = α·SSIM + β·LPIPS + γ·CLIP ← per-region via SPO (#130)
                                                      ↓
                                              Update OCR model
```

**Key design decisions** (from literature):
- Use SPO (#130) for per-region credit assignment, not whole-page reward
- Decompose reward into format + content components (#131)
- Normalize multi-metric rewards via DanceGRPO's group normalization (#128)
- Use preference ranking over direct regression (CycleReward insight)
- Combine with text-based unit tests from olmOCR 2 (#125) for hybrid reward

**Novelty**: No published system uses visual reconstruction similarity as OCR reward. CycleCap does this for captioning; olmOCR 2 uses text rewards. Our visual approach is genuinely novel.

### Architecture 2: Fully Differentiable Pipeline (Most Ambitious)

All building blocks exist but nobody has assembled them for OCR:

```
OCR Model → logits → [Gumbel-Softmax #136] → soft tokens
                                                    ↓
              [Bezier lookup] → glyph Beziers → [DiffVG/Bezier Splatting #142]
                                                    ↓
                                          Rendered image
                                                    ↓
                            [Differentiable SSIM/LPIPS + OCR-CRAFT loss]
                                                    ↓
                                          Backprop to OCR model
```

**Three options for the discrete bottleneck**:
1. **Gumbel-Softmax + STE** (#136, #137): Forward discrete, backward continuous. GRADE (#136) shows 14x lower variance than REINFORCE.
2. **Soft embeddings** (#138): Avoid discretization entirely. Render weighted mixtures of glyph embeddings.
3. **Stochastic gradient estimation** (#140): Wrap existing renderer, estimate gradients by perturbation. Simplest but noisiest.

**Critical warning** (#141): Optimizing against SSIM/LPIPS can be gamed. Use OCR-specific perceptual loss (CRAFT features from OCR-VQGAN) as primary signal, SSIM/LPIPS as auxiliary.

### Architecture 3: Wasserstein/Hilbert Space Metrics (Novel Comparison Signal)

Replace or complement SSIM/LPIPS with mathematically richer metrics:

| Metric | Space | What it captures | Effort |
|--------|-------|------------------|--------|
| DeepWSD (#115) | Wasserstein on VGG | Distributional shifts in features | Low |
| MS-SWD (#116) | Sliced Wasserstein | Misalignment-robust comparison | Very Low |
| HSIC (#120) | RKHS | Information-theoretic dependence | Medium |
| OT character shapes (#119) | Wasserstein on glyphs | Per-character shape fidelity | Medium |
| Complex-valued IQA (#118) | Complex CNN | Phase-encoded structural coherence | High |

**Immediate wins**: DeepWSD and MS-SWD can be added alongside existing metrics with minimal effort (both in IQA-PyTorch). MS-SWD specifically addresses SSIM's known weakness to positional misalignment.

## Cross-Technique Synergies (2026-04-06)

Analysis of combinations where 2+ techniques from our 142-paper corpus amplify each other beyond their individual contributions.

### Synergy 1: Hi-SAM Masks + MS-SWD/DeepWSD (#1 + #25 + #24)
**Papers**: #65 Hi-SAM, #116 MS-SWD, #115 DeepWSD
**Insight**: Hi-SAM produces stroke-level binary masks isolating text pixels. Applying MS-SWD or DeepWSD only on masked text regions combines the <1% area fix with misalignment-robust distributional comparison. Standard masked SSIM still suffers from pixel alignment issues; masked Wasserstein metrics do not.
**Why synergistic**: Each technique solves a different failure mode — Hi-SAM solves "background drowns signal," MS-SWD solves "pixel misalignment penalizes correct OCR." Together they give a text-focused, shift-tolerant metric.
**Effort**: Medium (Hi-SAM masks + IQA-PyTorch metric) | **Impact**: Very High

### Synergy 2: CRAFT Perceptual Loss + Differentiable Rendering (#4 + #28)
**Papers**: #14 OCR-VQGAN, #67 CRAFT, #134 DiffVG, #142 Bezier Splatting, #141 IQA limitations
**Insight**: Paper #141 warns that backpropagating through generic SSIM/LPIPS gets gamed. OCR-VQGAN (#14) showed CRAFT features outperform ImageNet LPIPS for text. Combining CRAFT perceptual loss as the objective function with Bezier Splatting (#142) as the differentiable renderer avoids both the gaming problem and the text-insensitivity problem.
**Why synergistic**: Differentiable rendering needs a loss function; CRAFT provides a text-aware one that resists adversarial exploitation. Neither is sufficient alone — DiffVG + LPIPS would be gamed; CRAFT without differentiable rendering has no gradients.
**Effort**: High | **Impact**: Very High — enables safe end-to-end OCR training

### Synergy 3: SPO + Hi-SAM Region Segmentation (#27 + #1)
**Papers**: #130 SPO, #65 Hi-SAM, #64 DocLayout-YOLO
**Insight**: SPO (#130) needs segments for per-region rewards but doesn't specify how to segment. Hi-SAM (#65) provides hierarchical text segmentation (word/line/paragraph) and DocLayout-YOLO (#64) provides layout element detection. Using these detectors to define SPO segments gives semantically meaningful reward regions aligned with document structure.
**Why synergistic**: SPO provides the RL mechanism; Hi-SAM/DocLayout-YOLO provide the segmentation. Without meaningful segments, SPO falls back to arbitrary chunking. Without SPO, per-region metrics are just evaluation — not a training signal.
**Effort**: High | **Impact**: Very High — dense, structured reward for OCR RL

### Synergy 4: MinerU-Diffusion + CycleCap GRPO (#19 + #26)
**Papers**: #108 MinerU-Diffusion, #124 CycleCap, #129 RLVR-World
**Insight**: MinerU-Diffusion provides native layout+bbox output (solving our 50% extraction failure) and per-token confidence. CycleCap provides the GRPO training recipe using visual reconstruction reward. Combining them: use MinerU-Diffusion as the OCR backbone, render its native bbox output, compute visual reward, fine-tune with GRPO. The per-token confidence from diffusion decoding can weight the reward signal.
**Why synergistic**: MinerU-Diffusion solves the OCR quality + structured output problem; CycleCap's GRPO recipe makes it self-improving. Neither alone closes the loop — MinerU-Diffusion is fixed after training; CycleCap needs an OCR model + renderer.
**Effort**: High | **Impact**: Very High — self-improving OCR without ground truth

### Synergy 5: ColPali Patch Embeddings + Hyperbolic Hierarchy (#10 + #121)
**Papers**: #80 ColPali, #121 HypStructure
**Insight**: ColPali produces multi-vector patch embeddings for document pages. Hyperbolic geometry naturally captures hierarchy. Embedding ColPali patches into hyperbolic space and comparing at multiple hierarchical levels (page → region → line → word) gives a hierarchy-aware document similarity metric that respects document structure.
**Why synergistic**: ColPali provides the raw patch embeddings but treats all patches equally. Hyperbolic geometry adds structural awareness — nearby patches in the document hierarchy are closer in the embedding space. Flat ColPali comparison misses structural relationships; hyperbolic embedding without good features has nothing to embed.
**Effort**: Very High (research) | **Impact**: High — genuinely novel "Document LPIPS"

### Synergy 6: Wavelet Subbands + Focal Frequency + Phase Congruency (#7 + #31 + #11)
**Papers**: #34 WGSR, #31 Focal Frequency Loss, #36 Phase Congruency
**Insight**: Decompose document images into wavelet subbands, apply focal frequency weighting (up-weight hard-to-match frequencies), and use phase congruency for contrast-invariant comparison. The wavelet decomposition separates text edges (HF) from background (LF); focal weighting emphasizes the hardest frequencies; phase congruency ignores brightness/contrast variations.
**Why synergistic**: Each frequency technique captures a different aspect — wavelets separate scales, focal frequency adapts importance, phase congruency normalizes contrast. Individually they're modest improvements; together they form a comprehensive frequency-domain document comparison pipeline that addresses SSIM's known blindspots.
**Effort**: Medium | **Impact**: High — multi-scale frequency comparison

### Synergy 7: FontCLIP + VQ-Font + TextCtrl (#9 + #17 + #16)
**Papers**: #75 FontCLIP, #103 VQ-Font, #94 TextCtrl
**Insight**: FontCLIP identifies the font in the original document. VQ-Font generates that font from a few character samples. TextCtrl applies style-preserving text editing. Pipeline: FontCLIP detects font → VQ-Font generates matching font → TextCtrl renders OCR text in that font with original style → comparison is more meaningful because fonts match.
**Why synergistic**: Each solves one link in the font-matching chain. FontCLIP without VQ-Font can identify but not reproduce fonts. VQ-Font without FontCLIP doesn't know which font to generate. TextCtrl without matching fonts produces style-mismatched reconstructions.
**Effort**: Very High | **Impact**: High — style-faithful reconstruction

### Synergy 8: Gumbel-Softmax + Bezier Splatting + CRAFT Loss (#136 + #142 + #4)
**Papers**: #136 GRADE, #137 Decoupled ST-GS, #142 Bezier Splatting, #67 CRAFT, #14 OCR-VQGAN
**Insight**: The complete differentiable OCR training pipeline: OCR logits → Gumbel-Softmax (#136) with decoupled temperatures (#137) → token-to-Bezier lookup → Bezier Splatting (#142) for fast differentiable rendering → CRAFT perceptual loss (#67/#14) for text-aware comparison. This is the full Architecture 2 instantiation with specific component choices optimized for text.
**Why synergistic**: This is the minimal viable fully-differentiable OCR training pipeline. Remove any component and it breaks: without Gumbel-Softmax, no gradients through discrete tokens; without Bezier Splatting, rendering is too slow for training; without CRAFT loss, optimization exploits metric weaknesses.
**Effort**: Very High | **Impact**: Very High — genuinely novel end-to-end system

### Synergy 9: Semantic Shuffle + MinerU-Diffusion + Our Metric (#21 + #19 + current)
**Papers**: #108 MinerU-Diffusion (Semantic Shuffle benchmark), our reconstruction metric
**Insight**: Use MinerU-Diffusion's Semantic Shuffle methodology to validate our metric. Shuffle words in test documents → re-render → run both MinerU-Diffusion and Qwen3-VL → reconstruct → compute our metric. If our metric scores differ between shuffled and unshuffled for the same OCR accuracy, it means we're measuring linguistic plausibility rather than visual fidelity.
**Why synergistic**: Semantic Shuffle is a diagnostic; our metric is the thing being diagnosed. Together they validate whether reconstruction-based evaluation truly measures OCR accuracy or gets fooled by language model priors.
**Effort**: Low | **Impact**: High — critical validation of our core methodology

### Synergy 10: DeepWSD + DISTS + Format Decoupling (#24 + #6 + #131)
**Papers**: #115 DeepWSD, #87 DISTS, #131 Format Decoupled RL
**Insight**: DISTS separates structure similarity from texture similarity. DeepWSD measures distributional feature shifts. Format Decoupled RL (#131) argues for separating layout from content rewards. Combine: use DISTS structure component for layout fidelity reward + DeepWSD for content/text fidelity reward — decomposed visual reward that aligns with the format decoupling principle.
**Why synergistic**: Format decoupling tells us *what* to decompose; DISTS provides the structure/texture split; DeepWSD provides the distributional text signal. Each paper's insight strengthens the others' application.
**Effort**: Medium | **Impact**: High — principled decomposed visual metric

## Flow Matching for Document Reconstruction and Comparison (2026-04-08)

Explores Flow Matching (FM) as a generative modeling framework for document image reconstruction, as a learned perceptual metric via transport cost, and as a differentiable backbone for end-to-end OCR training. FM is a simulation-free approach for training Continuous Normalizing Flows that has rapidly become the dominant paradigm, powering SD3, FLUX, and state-of-the-art layout/text generation.

### Core Flow Matching Theory

### 143. Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)
**Link**: https://arxiv.org/abs/2210.02747
**Summary**: Foundational paper introducing Flow Matching (FM), a simulation-free approach for training Continuous Normalizing Flows by regressing vector fields of fixed conditional probability paths. Compatible with general Gaussian paths including Optimal Transport displacement interpolation, which produces straighter trajectories and faster inference. SOTA on ImageNet in both NLL and FID. Published concurrently with Rectified Flow (#144) and Stochastic Interpolants (#146) — all three independently discovered simulation-free continuous flow training.
**Relevance**: **Foundational.** Establishes the theoretical framework for all subsequent FM work. OT displacement interpolation connects FM to transport cost as a potential quality metric.

### 144. Rectified Flow: Flow Straight and Fast (Liu et al., ICLR 2023 Spotlight)
**Link**: https://arxiv.org/abs/2209.03003
**Summary**: Learns ODEs following straight paths between source and target distributions. The "reflow" procedure iteratively straightens trajectories, enabling one-step generation (FID 4.85 on CIFAR-10). Also demonstrated for unpaired image-to-image translation without cycle-consistency losses. Provably non-increasing convex transport costs through reflow.
**Relevance**: **High.** Transport cost minimization is directly relevant to using OT cost as a metric. The domain transfer capability applies to learning a mapping from reconstructed to original document images.

### 145. OT-CFM: Conditional Flow Matching with Minibatch OT (Tong et al., TMLR 2024)
**Link**: https://arxiv.org/abs/2302.00482
**Summary**: Generalizes FM by allowing arbitrary source distributions (not just Gaussian). Uses minibatch OT plans for straighter, more stable flows. When the true OT plan is available, OT-CFM approximates dynamic optimal transport. TorchCFM library provides practical implementation.
**Relevance**: **High.** Non-Gaussian source enables conditioning on reconstructed images directly. Minibatch OT approximation is computationally tractable for measuring transport cost between original and reconstructed documents.

### 146. Stochastic Interpolants: A Unifying Framework (Albergo et al., JMLR 2025)
**Link**: https://arxiv.org/abs/2303.08797
**Summary**: Unifying class bridging flow-based and diffusion-based methods. Interpolants combine data from two distributions with tunable diffusion, yielding both deterministic (ODE) and stochastic (SDE) models. Recovers Schrödinger bridge when optimized. Can connect arbitrary distributions, not just noise-to-data.
**Relevance**: **High.** Ability to bridge arbitrary distributions is critical for image-to-image translation. Could learn a flow from original to reconstructed documents, with transport cost measuring OCR quality.

### 147. Building Normalizing Flows with Stochastic Interpolants (Albergo, ICLR 2023)
**Link**: https://arxiv.org/abs/2209.15571
**Summary**: Foundational paper proposing continuous-time normalizing flows between arbitrary base and target densities via stochastic interpolants. Uses simple quadratic loss. Flow can be optimized to minimize path length for optimal transport maps.
**Relevance**: Medium. Establishes that simulation-free CNF training with simple losses is viable and scalable.

### 148. Flow Matching Guide and Code (Lipman et al., NeurIPS Tutorial 2024)
**Link**: https://arxiv.org/abs/2412.06264
**Summary**: Comprehensive tutorial covering FM foundations, design choices, and extensions. Accompanied by PyTorch library (facebookresearch/flow_matching) with practical examples for both continuous and discrete FM.
**Relevance**: **Essential reference** for implementation. Canonical codebase and design guidance.

### Advanced FM Architectures

### 149. SiT: Scalable Interpolant Transformers (Ma et al., ECCV 2024)
**Link**: https://arxiv.org/abs/2401.08740
**Summary**: Applies stochastic interpolant framework to DiT architecture. Systematic exploration shows continuous-time velocity prediction with linear interpolants and SDE sampling outperforms standard diffusion transformers. FID 2.06 on ImageNet 256x256.
**Relevance**: Medium. Provides empirically validated design choices if we use a transformer-based FM model for document reconstruction.

### 150. Scaling Rectified Flow Transformers / SD3 (Esser et al., ICML 2024)
**Link**: https://arxiv.org/abs/2403.03206
**Summary**: Technical foundation of Stable Diffusion 3 and FLUX. Rectified flow with MMDiT architecture (separate text/image weights with bidirectional flow). Perceptually-biased noise sampling. **Superior text rendering** compared to prior diffusion formulations.
**Relevance**: **High.** Industrial validation of FM at scale. SD3/FLUX's superior text rendering is directly relevant — demonstrates FM excels at the exact task (accurate text in generated images) our reconstruction pipeline needs.

### 151. Improving Training of Rectified Flows (Lee et al., NeurIPS 2024)
**Link**: https://arxiv.org/abs/2405.20320
**Summary**: U-shaped timestep distribution focusing on boundary timesteps + LPIPS-Huber premetric loss. Single Reflow iteration suffices. Improves FID by up to 75% in 1-NFE setting. Rectified flows enable inversion by backward ODE integration with lower reconstruction error than EDM.
**Relevance**: **High.** LPIPS-Huber loss directly connects to our perceptual metrics. One-step generation with superior reconstruction error is exactly what we need. Inversion property enables cycle-consistency checks.

### 152. Consistency Models (Song et al., ICML 2023)
**Link**: https://arxiv.org/abs/2303.01469
**Summary**: Directly maps noise to data in a single step by enforcing self-consistency along the probability flow ODE. One-step FID 3.55 on CIFAR-10. Can be combined with flow matching (see #153).
**Relevance**: Medium-High. One-step generation is attractive for fast metric evaluation. Combinable with FM via Flow Map Matching.

### 153. Flow Map Matching: Math Framework for Consistency Models (2024)
**Link**: https://arxiv.org/abs/2406.07507
**Summary**: Connects consistency models to flow matching rigorously. Both losses provably control Wasserstein distance between teacher and student models.
**Relevance**: Medium. Formalizes FM + consistency model combination for potential one-step document reconstruction.

### 154. Rectified Diffusion: Straightness Is Not Your Need (ICLR 2025)
**Link**: https://arxiv.org/abs/2410.07303
**Summary**: Shows rectification's key benefit is matched noise-sample pairs, not trajectory straightness. Phased rectification divides the ODE into segments for reduced training cost. Outperforms InstaFlow and PeRFlow.
**Relevance**: Medium. Practical nuance — phased approach could let different phases handle different document aspects (layout, text, fine detail).

### 155. Metric Flow Matching for Smooth Interpolations (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2405.14780
**Summary**: Interpolants are approximate geodesics of a data-induced Riemannian metric, staying on the data manifold instead of crossing through low-density regions. OT-MFM variant combines with optimal transport. Measured with FID and LPIPS on unpaired image translation.
**Relevance**: **Very High.** Most directly relevant to our metric question. Geodesic distance on the document image manifold between original and reconstruction would be a perceptually meaningful metric — richer than SSIM/LPIPS because it accounts for manifold geometry.

### 156. Optimal Flow Matching: Straight Trajectories in One Step (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2403.13117
**Summary**: Recovers the exact OT displacement for quadratic transport in one FM step via Input Convex Neural Networks. Directly solves the Wasserstein-2 problem.
**Relevance**: Medium. Theoretically interesting for exact OT distances but ICNN scalability is a concern at document resolution.

### Discrete Flow Matching & Text

### 157. Discrete Flow Matching (Gat et al., NeurIPS 2024 Spotlight)
**Link**: https://arxiv.org/abs/2407.15595
**Summary**: Extends FM to discrete data (text tokens). Uses probability paths interpolating between source and target on discrete state spaces. Supports denoiser and noise-prediction sampling. Scales to 1.7B parameters, closing the gap between AR and non-AR text generation.
**Relevance**: **High.** Directly applicable to OCR token generation as a non-autoregressive alternative. Competes with DODO (#109) and MinerU-Diffusion (#108) but uses flow matching instead of masked diffusion.

### 158. Fisher Flow Matching over Discrete Data (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2405.14664
**Summary**: Geometric approach treating categorical distributions as points on the d-hypersphere with Fisher-Rao metric. Flows via closed-form geodesics on the statistical manifold with Riemannian OT.
**Relevance**: Medium. The Fisher-Rao metric on the probability simplex could provide a principled distance for comparing OCR output distributions rather than hard token sequences.

### 159. MDLM: Simple Masked Diffusion Language Models (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2406.07524
**Summary**: Shows masked discrete diffusion is more performant than previously thought. Training objective is a weighted average of masked LM losses. 27.04 PPL on LM1B. Powers ByteDance's Seed Diffusion and NVIDIA's Genmol.
**Relevance**: **High comparison point.** MDLM is the masked diffusion baseline that MinerU-Diffusion (#108) and DODO (#109) build upon. Understanding MDLM vs DFM (#157) is essential for choosing the best non-AR paradigm for OCR.

### 160. FS-DFM: Few-Step Discrete Flow Matching (Apple, ICLR 2026)
**Link**: https://arxiv.org/abs/2509.20624
**Summary**: Makes DFM practical with 8 function evaluations instead of 1024 (128x speedup). Step-aware training with shortcut teacher distillation. Outperforms LLaDA-8B and Dream-7B at much smaller model sizes.
**Relevance**: **High.** Solves the practical problem of DFM being too slow for OCR inference. If we adopt DFM for OCR, FS-DFM's few-step recipe makes it viable for real-time document processing.

### 161. Edit Flows: Flow Matching with Edit Operations (Meta, NeurIPS 2025)
**Link**: https://arxiv.org/abs/2506.09018
**Summary**: Generalizes DFM to variable-length sequences via edit operations (insert, delete, substitute) modeled as CTMC. No padding needed. 138% improvement over mask models on code generation.
**Relevance**: **Critical.** OCR output is variable-length by nature. Standard DFM requires fixed-length sequences. Edit Flows handles this naturally — the most theoretically suitable FM framework for OCR text generation.

### 162. AR vs Masked Diffusion: A Controlled Comparison (2026)
**Link**: https://arxiv.org/abs/2603.22075
**Summary**: First controlled comparison (same data, compute, hardware). AR converges faster but overfits sooner. MDLM produces dramatically more diverse outputs (93.4% unique openings vs AR's 3.3%).
**Relevance**: **High.** OCR is deterministic (one correct answer), so MDLM's diversity advantage is irrelevant or harmful. AR's faster convergence and higher fluency may be preferable.

### 163. FlowTok: Flowing Across Text and Image Tokens (ByteDance, 2025)
**Link**: https://arxiv.org/abs/2503.10772
**Summary**: Directly evolves between text and image modalities through flow matching by encoding images into compact 1D tokens. No conditioning mechanisms needed — pure flow between text tokens and image tokens. Supports bidirectional generation.
**Relevance**: **High.** FlowTok's text-to-image flow is conceptually identical to our OCR reconstruction: given text tokens, generate the corresponding image. If adapted for documents, could serve as a learned reconstruction model replacing our HTML renderer.

### FM for Documents, Layout & Text Rendering

### 164. LayoutFlow: Flow Matching for Layout Generation (ECCV 2024)
**Link**: https://arxiv.org/abs/2403.18187
**Summary**: Applies FM to document and UI layout generation on PubLayNet (360K document layouts) and RICO (66K UI layouts). SOTA FID while significantly faster than diffusion-based methods. 4-layer Transformer, 15M params. Single model handles unconditional, conditional, completion, and refinement.
**Relevance**: **Very High.** Most directly relevant document-domain paper. Validates FM for document layouts with strong quality and fast inference. The layout generation task is closely related to our reconstruction.

### 165. FonTS: Text Rendering with Typography Controls (ICCV 2025)
**Link**: https://arxiv.org/abs/2412.00136
**Summary**: Two-stage DiT on rectified flow (SD3/FLUX architecture). Typography control fine-tuning with enclosing tokens for word-level font, bold, italic, underline control. **Uses HTML rendering to generate training data** — mirrors our pipeline's HTML-based reconstruction.
**Relevance**: **High.** Demonstrates controllable text rendering with rectified flow. Their HTML rendering for training data mirrors our approach. Typography control tokens are analogous to our bbox + font metadata extraction.

### 166. TextFlux: OCR-Free DiT for Multilingual Text Synthesis (2025)
**Link**: https://arxiv.org/abs/2505.17778
**Summary**: OCR-free text synthesis on FLUX (rectified flow DiT). Spatially concatenates glyph-rendered text with original image. Achieves strong multilingual results with only 1% of competing methods' training data. Zero-shot generalization to unseen characters.
**Relevance**: **High.** Validates that rectified flow models handle multilingual text without explicit OCR modules. Glyph concatenation approach could inform our reconstruction pipeline.

### 167. SLayR: Scene Layout Generation with Rectified Flow (2024)
**Link**: https://arxiv.org/abs/2412.05003
**Summary**: Extends rectified flow to scene layout generation with global prompt conditioning. Confirms generality of FM for layout tasks beyond documents.
**Relevance**: Low-Medium. Supplementary evidence for FM's suitability for layout tasks.

### FM for Image-to-Image Translation

### 168. InstaFlow: One-Step Stable Diffusion with Rectified Flow (ICLR 2024)
**Link**: https://arxiv.org/abs/2309.06380
**Summary**: Applies rectified flow's reflow to Stable Diffusion, creating the first one-step diffusion-based text-to-image generator at SD quality. Compatible with LoRAs and ControlNets.
**Relevance**: Medium. ControlNet compatibility could enable document-conditioned fast reconstruction.

### 169. FlowEdit: Inversion-Free Text-Based Editing (ICCV 2025 Oral)
**Link**: https://arxiv.org/abs/2412.08629
**Summary**: Constructs an ODE that directly maps between source and target distributions, achieving lower transport cost than inversion-based approaches. Key insight: direct path between distributions in image space instead of inverting to noise. Works with SD3 and FLUX.
**Relevance**: **High.** The "direct path with lower transport cost" concept is precisely what we want. Instead of pixel-level comparison, we could measure the transport cost of the direct flow from original to reconstructed document — a "flow-based image distance."

### 170. LBM: Latent Bridge Matching for Fast I2I Translation (ICCV 2025 Highlight)
**Link**: https://arxiv.org/abs/2503.07535
**Summary**: Brownian Bridge matching in latent space for I2I translation achieving SOTA with a single inference step. Source and target encoded via VAE, connected by Brownian Bridge, U-Net (from SDXL) predicts drift. Uses LPIPS during training. Outperforms both diffusion and FM multi-step methods.
**Relevance**: **Very High.** Directly applicable: train LBM on original-reconstructed document pairs. The bridge's transport cost becomes the quality metric. Single-step inference is critical for practical use as a metric.

### 171. SANA-I2I: Text-Free Flow Matching for Paired I2I Translation (2026)
**Link**: https://arxiv.org/abs/2604.00298
**Summary**: Text-free, image-conditioned flow matching in latent space using Deep Compression Autoencoder. Learns conditional velocity field driven solely by input images.
**Relevance**: Medium. Clean formulation for our use case (no text prompts needed, just image pairs).

### FM for Image Restoration

### 172. PnP-Flow: Plug-and-Play Image Restoration with Flow Matching (ICLR 2025)
**Link**: https://arxiv.org/abs/2410.02423
**Summary**: Combines PnP framework with FM. Time-dependent denoiser from pre-trained FM model. Alternates between data fidelity gradient descent, FM path reprojection, and denoising. No backprop through ODEs needed. Superior on denoising, SR, deblurring, inpainting.
**Relevance**: Medium-High. Could enhance degraded document images before OCR, or enhance reconstructed images before metric comparison.

### 173. Restora-Flow: Mask-Guided Image Restoration with Flow Matching (2025)
**Link**: https://arxiv.org/abs/2511.20152
**Summary**: Training-free method using degradation masks to guide FM sampling. Trajectory correction for consistency with degraded inputs.
**Relevance**: Medium. Mask-guided approach could selectively restore text regions using Hi-SAM/CRAFT masks before metric evaluation.

### 174. FlowIE: Efficient Image Enhancement via Rectified Flow (CVPR 2024)
**Link**: https://arxiv.org/abs/2406.00508
**Summary**: Conditioned rectified flow for image enhancement with novel mean-value sampling. ~10x faster than DiffBIR. Path estimator fine-tuned from SD 2.1 via LoRA.
**Relevance**: Medium. Document image enhancement via rectified flow could improve reconstructions.

### 175. FlowSR: Fast Image SR via Consistency Rectified Flow (ICCV 2025)
**Link**: https://arxiv.org/abs/2412.00899
**Summary**: Reformulates SR as rectified flow from LR to HR images (not noise-to-HR). Combines rectified flow with consistency learning for single-step inference. HR regularization + fast-slow scheduling.
**Relevance**: **High.** Our pipeline could reformulate comparison as a flow from reconstructed to original. Single-step capability is critical for practical metric deployment at scale.

### FM as Metrics & Differentiable Pipelines

### 176. DiffSim: Taming Diffusion Models for Visual Similarity (ICCV 2025)
**Link**: https://arxiv.org/abs/2412.14580
**Summary**: First paper showing pretrained diffusion model features measure visual similarity effectively. Aligns U-Net self-attention and cross-attention features. Key insight: shallower layers + higher timesteps capture low-level/style similarity; deeper layers + lower timesteps capture semantic similarity. SOTA on visual coherence benchmarks. Generalizable via "Attention-Aligned Similarity" (AAS).
**Relevance**: **Critical.** Most directly applicable paper for our metric question. We could extract features from a pretrained FM model (FLUX/SD3) at appropriate layers/timesteps and compute attention-aligned similarity between original and reconstructed documents. This "FlowSim" would be a flow-matching-based perceptual metric. The timestep/layer insight means we can tune for text fidelity (low-level, high timestep) vs layout similarity (high-level, low timestep).

### 177. D-Flow: Differentiating through Flows for Controlled Generation (ICML 2024)
**Link**: https://arxiv.org/abs/2402.14017
**Summary**: Framework for controlling generation by differentiating through the entire flow, optimizing initial noise. For FM with Gaussian paths, differentiating implicitly projects gradients onto the data manifold. SOTA on linear and non-linear inverse problems. Generalizes to arbitrary differentiable cost functions.
**Relevance**: **Critical for differentiable pipeline.** Directly answers whether we can backpropagate through FM generation — yes. If we reconstruct documents via FM, D-Flow enables optimizing OCR parameters by differentiating visual quality loss through the flow-based reconstruction.

### 178. AdjointDPM: Adjoint Sensitivity for Gradient Backprop (ICLR 2024)
**Link**: https://arxiv.org/abs/2307.10711
**Summary**: Solves the memory problem of backpropagating through multi-step diffusion/flow sampling. Generates samples via probability-flow ODE, backpropagates via adjoint sensitivity (augmented backward ODE). **Constant memory** regardless of sampling steps.
**Relevance**: **Critical for differentiable pipeline.** Without this, storing intermediate states of a 50-step ODE at document resolution would be prohibitive. AdjointDPM enables practical end-to-end backprop through FM-based reconstruction.

### 179. DP-IQA: Utilizing Diffusion Prior for Blind IQA (2024)
**Link**: https://arxiv.org/abs/2405.19996
**Summary**: First method applying pretrained diffusion priors (Stable Diffusion) to blind/no-reference IQA. Multi-level U-Net features guided by learnable quality-aware text prompts. Validates that generative model features transfer to quality assessment.
**Relevance**: **High.** For our no-reference OCR quality metric, FM features could assess document image quality directly (is this image clean enough for good OCR?) as a complementary signal.

### 180. Multimarginal Generative Modeling with Stochastic Interpolants (ICLR 2024)
**Link**: https://arxiv.org/abs/2303.16048
**Summary**: Generalizes stochastic interpolants to connect multiple distributions (not just two). Supports data-dependent couplings and more general paths than score-based diffusion.
**Relevance**: **High.** Could model the full OCR pipeline as a sequence of distributions: original image → OCR text distribution → rendered image distribution. Total transport cost through the chain measures cumulative quality degradation — a novel theoretical framing for our end-to-end metric.

### Synthesis: Flow Matching for Reference-Free OCR Metrics (2026-04-08)

**Key Finding 1: FM transport cost as a document comparison metric is unexplored but promising.**

Evidence:
- Metric FM (#155) shows Riemannian geodesic distance on the data manifold is more perceptually meaningful than Euclidean distance
- FlowEdit (#169) demonstrates direct flow paths have lower transport cost, suggesting cost correlates with perceptual change
- DiffSim (#176) proves generative model features ARE effective similarity metrics
- DeepWSD (#115, prior research) already shows Wasserstein distance in feature space outperforms SSIM

**Concrete proposal — "FlowSim" for documents:**
1. Extract attention features from pretrained FLUX/SD3 at multiple layers/timesteps for original and reconstructed images (following DiffSim's recipe)
2. Compute attention-aligned similarity as document comparison metric
3. Alternatively, learn OT-CFM flow from original to reconstructed, use path length as metric

**Key Finding 2: All building blocks exist for differentiable FM-based OCR training.**

Three approaches (increasing ambition):

| Approach | Method | Memory | Speed | Novelty |
|----------|--------|--------|-------|---------|
| A: D-Flow optimization (#177) | Differentiate quality loss through FM | Constant via AdjointDPM (#178) | Medium | Low |
| B: LBM bridge metric (#170) | Train bridge, use transport cost | Low (single step) | Fast | Medium |
| C: FlowSim as GRPO reward | FM features as reward in GRPO (#124) | N/A (no backprop) | Fast | Medium |

**Key Finding 3: Discrete FM could replace masked diffusion for OCR.**

- DFM (#157) competes with MDLM (#159) for non-AR text generation
- Edit Flows (#161) solves the variable-length problem critical for OCR
- FS-DFM (#160) makes inference practical (128x speedup)
- But AR models still converge faster for deterministic tasks like OCR (#162)

**Key Finding 4: FM is validated for document-domain tasks.**

- LayoutFlow (#164) achieves SOTA on PubLayNet with faster inference
- FonTS (#165) uses HTML rendering (mirrors our pipeline) for typography-controlled text
- SD3/FLUX (#150) demonstrate superior text rendering
- TextFlux (#166) shows multilingual text synthesis without explicit OCR

**Research gaps identified:**
1. No "FlowSim" for documents — nobody has built DiffSim's equivalent for FM models on document comparison
2. No flow-based document IQA — DP-IQA (#179) exists for diffusion but not FM
3. No transport cost as OCR quality metric — despite extensive OT-CFM theory
4. No FM for document restoration — PnP-Flow (#172) handles natural images only
5. No multimarginal flow for OCR pipeline evaluation — theoretical framework exists (#180) but is unapplied

**Comparison with existing metrics:**

| Metric | Type | Differentiable | Text-Aware | Alignment-Robust |
|--------|------|---------------|------------|-------------------|
| SSIM | Pixel-level | Yes | No | No |
| LPIPS | Feature-level | Yes | No | No |
| CLIP-Score | Embedding | Yes | Partial | Yes |
| DeepWSD (#115) | Wasserstein on features | Yes | No | Partial |
| MS-SWD (#116) | Sliced Wasserstein | Yes | No | Yes |
| DiffSim (#176) | Diffusion attention | Yes | No | Yes |
| **FlowSim (proposed)** | **Flow attention** | **Yes** | **Tunable** | **Yes** |
| **LBM cost (proposed)** | **Bridge transport** | **Yes** | **Learned** | **Yes** |

## World Models as Reward Signals and Imagination-Based RL Training (2026-04-09)

This section surveys world models research relevant to the Reference-Free OCR Metric project. The central question: can we train a small world model that learns "what documents should look like" and use its prediction quality as an OCR reward signal, replacing our current multi-metric comparison (SSIM+LPIPS+CLIP) with a single learned world model reward?

### 181. World Models (Ha & Schmidhuber, NeurIPS 2018)
**Summary**: The seminal paper that established the modern world model paradigm. Proposes a three-component architecture: a VAE (Vision) compresses each image frame into a latent vector z, an MDN-RNN (Memory) predicts the distribution of future latent states, and a compact Controller selects actions. The key insight is that an agent can be trained entirely inside its own hallucinated dream generated by the world model, and the resulting policy transfers back to the actual environment. Demonstrated on CarRacing-v0 and VizDoom. The MDN-RNN models stochastic outcomes as a mixture density, making it harder for the controller to exploit model deficiencies.
**Relevance**: Establishes the foundational architecture for our proposed approach. A document world model would learn p(z_{t+1} | z_t, OCR_output) -- predicting what a document reconstruction should look like given OCR text. The prediction quality becomes the reward signal.

### 182. Dream to Control: DreamerV1 (Hafner et al., ICLR 2020)
**Summary**: Introduces the Recurrent State-Space Model (RSSM) that combines deterministic and stochastic latent states for richer dynamics modeling. DreamerV1 trains an actor-critic entirely in imagination by propagating analytic gradients through imagined trajectories in the learned latent space. On 20 visual control tasks, it exceeds existing approaches in data-efficiency, computation time, and final performance. The representation model encodes observations and actions into continuous vector-valued model states with Markovian transitions.
**Relevance**: The RSSM architecture is the backbone of modern world models. For document reconstruction, an RSSM-like model could maintain a latent representation of document structure (layout, fonts, spacing) that evolves as OCR text elements are processed sequentially.

### 183. Mastering Atari with Discrete World Models: DreamerV2 (Hafner et al., ICLR 2022)
**Summary**: The key innovation is replacing Gaussian latent variables (V1) with categorical random variables, making the latent representation a set of 32 one-hot vectors sampled from 32 categorical distributions. The authors argue this provides more expressivity. DreamerV2 achieves human-level performance on the Atari 55-game benchmark, the first model-based agent to do so. Also applicable to continuous action spaces (humanoid robot walking from pixels). Introduces KL balancing to prevent posterior collapse.
**Relevance**: Discrete latent representations are directly applicable to document world models, where document elements (characters, layout blocks, font styles) are inherently discrete. A categorical world model aligns naturally with tokenized document representations.

### 184. DreamerPro: Reconstruction-Free MBRL with Prototypical Representations (Deng et al., ICML 2022)
**Summary**: Addresses the fundamental limitation of reconstruction-based world models: they waste capacity on task-irrelevant visual details and are fragile to visual distractions. DreamerPro replaces the image reconstruction loss with prototypical representations learned from recurrent states, distilling temporal structures from past observations and actions into a fixed set of prototypes. Achieves better robustness to complex background distractions than Dreamer while maintaining performance in standard tasks. However, requires data augmentation to prevent representation collapse.
**Relevance**: Highly relevant design consideration. For document world models, we face a similar choice: should the model reconstruct pixel-level document images (expensive, may focus on irrelevant details like background texture) or learn abstract prototypical representations of document structure? DreamerPro suggests the latter may be more robust.

### 185. DreamerV3: Mastering Diverse Domains through World Models (Hafner et al., Nature 2025)
**Summary**: The most important paper in the Dreamer series. Achieves state-of-the-art across 150+ diverse tasks with a single fixed hyperparameter configuration, eliminating the need for per-task tuning. Key innovations: symlog transformation for reward normalization, unimix categoricals to prevent mode collapse, percentile return normalization, and symexp two-hot loss. A critical ablation finding: performance rests predominantly on the unsupervised reconstruction loss of the world model, not on task-specific reward signals. This means the world model's generic understanding of the environment is more important than reward prediction. All training runs on a single A100 GPU. First algorithm to collect diamonds in Minecraft from scratch.
**Relevance**: The ablation finding that unsupervised reconstruction loss matters more than task-specific reward is directly relevant to our approach. A document world model trained with reconstruction loss (predicting what documents look like) would develop representations useful for downstream OCR quality assessment, even without explicit quality labels. The single-GPU training is also encouraging for our compute constraints.

### 186. Dreamer 4: Training Agents Inside of Scalable World Models (Hafner & Yan, 2025)
**Summary**: A 2B-parameter world model that uses an efficient transformer architecture with a novel "shortcut forcing" objective (building on flow matching but predicting the final clean state instead of update vectors). First agent to obtain diamonds in Minecraft from a fixed offline dataset without environment interaction. Learns general action conditioning from small amounts of labeled data while extracting most knowledge from diverse unlabeled videos. Achieves real-time inference (21 FPS) on a single GPU. Outperforms OpenAI's VPT offline agent while using 100x less data. Also demonstrates accurate physics and counterfactual interactions on robotics data.
**Relevance**: The data efficiency story is compelling for our use case. Dreamer 4 learns from mostly unlabeled videos plus small amounts of action-labeled data. Similarly, a document world model could learn "what documents look like" from large unlabeled document corpora, then be fine-tuned with small amounts of OCR quality data. The shortcut forcing objective (flow-matching variant) connects to our existing flow matching research (papers #143-180).

### 187. R2-Dreamer: Redundancy-Reduced World Models without Decoders (ICLR 2026)
**Summary**: Proposes a decoder-free MBRL framework that uses a Barlow Twins-inspired redundancy-reduction objective as an internal regularizer, preventing representation collapse without requiring data augmentation or image decoders. Replaces the computationally expensive image decoder with a lightweight linear projector that maps latent states to the feature space of image embeddings. Trains 1.59x faster than DreamerV3 and 2.36x faster than DreamerPro. Competitive with DreamerV3 and TD-MPC2 on standard benchmarks, with substantial gains on tasks with tiny task-relevant objects (DMC-Subtle).
**Relevance**: The decoder-free approach with redundancy reduction is directly applicable. For a document world model reward, we do not need to reconstruct full document images -- we need the latent representation to capture document structure well enough that prediction error in latent space serves as a quality signal. R2-Dreamer shows this is feasible and significantly faster.

### 188. IRIS: Transformers are Sample-Efficient World Models (Micheli et al., ICLR 2023)
**Summary**: Introduces a world model composed of a discrete autoencoder and an autoregressive GPT-like Transformer. Observations are tokenized by the autoencoder, and the Transformer autoregressively predicts future frame tokens plus reward and termination signals. Policy is trained purely on imagined trajectories. Achieves mean human-normalized score of 1.046 on Atari 100k (just 2 hours of gameplay), outperforming humans on 10/26 games. The discrete tokenization gives the Transformer sub-observation attention resolution.
**Relevance**: The autoregressive token prediction architecture maps directly to language model architectures already used for OCR (Qwen-VL, etc.). A document world model built as an autoregressive Transformer over discrete document tokens would share architectural components with existing OCR models, enabling parameter sharing or distillation. The follow-up Delta-IRIS encodes stochastic deltas between timesteps, which could model how OCR errors change document reconstructions.

### 189. TD-MPC2: Scalable, Robust World Models for Continuous Control (Hansen et al., ICLR 2024)
**Summary**: An implicit (decoder-free) world model that learns latent dynamics and performs local trajectory optimization via model-predictive control. Trains a single 317M parameter agent to perform 80 tasks across multiple domains, embodiments, and action spaces with one set of hyperparameters. Shows that agent capabilities increase consistently with model and data size (favorable scaling properties). Open-sources 324 model checkpoints. Compares favorably to DreamerV3 on 104 continuous control tasks.
**Relevance**: The decoder-free, implicit world model approach is relevant for document quality assessment. TD-MPC2's latent trajectory optimization could inspire a document world model that plans in latent space to find the best reconstruction, with the planning cost serving as a quality metric (harder-to-reach reconstructions indicate lower OCR quality).

### 190. DreamSmooth: Model-Based RL via Reward Smoothing (Lee et al., ICLR 2024)
**Summary**: Addresses a critical bottleneck in model-based RL: reward prediction for sparse rewards is challenging and ambiguous. Proposes temporally smoothing rewards before adding them to the replay buffer, making reward model learning easier. Achieves state-of-the-art on long-horizon sparse-reward tasks. However, follow-up work notes that temporal smoothing merely attenuates sparse signals rather than constructing informative gradients in fully sparse environments.
**Relevance**: Directly relevant to the reward design for OCR training. If using world model prediction error as OCR reward, the reward may be sparse (most OCR outputs are either clearly good or clearly bad). DreamSmooth's approach of temporal smoothing could help by spreading quality signals across the OCR generation process rather than assigning a single final score.

### 191. InDRiVE: Reward-Free World-Model Pretraining via Latent Disagreement (Khanzada & Kwon, 2025)
**Summary**: A DreamerV3-style agent that performs reward-free pretraining using only intrinsic motivation from latent ensemble disagreement. An ensemble of world models is trained, and disagreement between their predictions serves as a proxy for epistemic uncertainty, driving exploration toward under-explored states. The pretrained world model transfers effectively to new environments with zero-shot or few-shot adaptation. Compares disagreement vs. ICM vs. RND as intrinsic objectives under identical conditions.
**Relevance**: The ensemble disagreement approach could serve as a document quality signal without any ground truth. Train an ensemble of document world models, then measure their disagreement on a given OCR output's reconstruction. High disagreement indicates the OCR output is unusual or likely erroneous -- serving as a reference-free quality metric. This avoids the need for any labeled quality data.

### 192. Understanding World or Predicting Future? Comprehensive Survey of World Models (Ding et al., ACM CSUR 2025)
**Summary**: The most comprehensive world model survey to date, covering 300+ papers. Categorizes world models into two functions: (1) constructing internal representations to understand world mechanisms, and (2) predicting future states to simulate and guide decision-making. Covers applications in generative games, autonomous driving, robotics, and social simulacra. Reviews training approaches including variational inference, maximum likelihood estimation, and auxiliary predictive/reconstruction losses. Published in ACM Computing Surveys, Volume 58.
**Relevance**: Essential reference for positioning our document world model work within the broader landscape. The taxonomy of "understanding vs. prediction" maps to our two potential approaches: a document understanding model (learns what valid documents look like) vs. a document prediction model (predicts what reconstruction should look like given OCR output).

### 193. V-JEPA 2: Self-Supervised Video Models for Understanding and Planning (Meta, 2025)
**Summary**: Extends the JEPA (Joint Embedding Predictive Architecture) paradigm to video, pre-training on 1M+ hours of internet video without labels. V-JEPA 2 predicts missing parts of video in abstract representation space (not pixel space), achieving strong motion understanding (77.3% on Something-Something v2) and state-of-the-art action anticipation. Post-trained as a latent action-conditioned world model (V-JEPA 2-AC) using only 62 hours of unlabeled robot videos. Deployed zero-shot on robots for pick-and-place tasks without task-specific training or reward. Shows higher "surprise" (prediction error) for physically impossible events.
**Relevance**: The JEPA approach of predicting in abstract representation space (not pixel space) is particularly relevant. A document-JEPA model could learn to predict masked document regions in embedding space, and prediction error would indicate document anomalies. The fact that V-JEPA 2 shows calibrated "surprise" for impossible events suggests a document-JEPA could similarly detect impossible/erroneous OCR outputs. The data efficiency (62 hours of robot data for world model fine-tuning) is encouraging.

### 194. Genie 2: A Large-Scale Foundation World Model (DeepMind, Dec 2024)
**Summary**: A foundation world model that generates rich, interactive 3D environments from a single image prompt. Trained on large-scale video data, it demonstrates emergent capabilities including object interactions, complex character animation, physics simulation, and agent behavior modeling. Uses a video autoencoder with an autoregressive transformer in latent space, sampled frame-by-frame conditioned on actions. Generates consistent worlds for up to 60 seconds. Represents the frontier of large-scale world models but remains computationally intensive and not publicly available.
**Relevance**: While Genie 2 operates at a much larger scale than needed for document understanding, it demonstrates that world models can learn complex visual generation from data alone. A document-specific variant at much smaller scale could learn to generate plausible document images, with generation quality serving as an OCR reward signal.

### 195. Smaller World Models for Reinforcement Learning (Robine et al., 2023)
**Summary**: Addresses the computational burden of large world models by proposing a compact architecture based on VQ-VAE encoding and convolutional LSTM prediction. Achieves competitive performance with significantly fewer parameters. Trains in approximately 12 hours on a single A100 GPU (vs. 500 hours for SimPLe on P100). The VQ-VAE discrete tokenization of observations enables efficient world model learning with lower memory usage and faster training. A model-free PPO agent is trained purely on simulated experience.
**Relevance**: Directly addresses our compute constraint concern. Demonstrates that effective world models do not require massive compute -- a VQ-VAE + conv-LSTM architecture could be trained on a single GPU to learn document dynamics. For our 4x RTX 6000 Ada setup, this is easily feasible. The VQ-VAE tokenization of document images is a natural fit for document world models.

### 196. ImageReward: Human Preferences for Text-to-Image Generation (Xu et al., NeurIPS 2023)
**Summary**: The first general-purpose text-to-image human preference reward model, trained on 137K expert comparisons using rating and ranking. Outperforms CLIP (by 38.6%), Aesthetic Score (by 39.6%), and BLIP (by 31.6%) in predicting human preferences. Proposes Reward Feedback Learning (ReFL) to directly tune diffusion models against the reward scorer. The reward model takes both image and text as input, producing a single quality score.
**Relevance**: ImageReward demonstrates that a learned reward model can replace hand-crafted metrics (like our SSIM+LPIPS+CLIP combination) with a single learned score. A "DocReward" model trained on document reconstruction quality preferences could serve the same role. However, ImageReward requires human preference data, while our world model approach would be self-supervised.

### 197. VisionReward: Multi-Dimensional Human Preference for Image/Video (AAAI 2026)
**Summary**: A fine-grained, multi-dimensional reward model that decomposes subjective quality judgments into interpretable dimensions with weighted scoring. Excels at video quality prediction by analyzing dynamic features. Provides precise and comprehensive evaluations that capture multiple aspects of visual quality simultaneously. Sets a new benchmark for video quality assessment.
**Relevance**: The multi-dimensional decomposition is relevant to our multi-metric approach. Rather than collapsing SSIM+LPIPS+CLIP into a single score, VisionReward shows that maintaining interpretable sub-dimensions is valuable. A document world model reward could similarly decompose into layout quality, text accuracy, and visual fidelity dimensions.

### 198. DocThinker: Explainable MLLM with Rule-Based RL for Documents (2025)
**Summary**: Leverages rule-based reinforcement learning (GRPO) to train multimodal LLMs for document understanding, generating explainable intermediate steps including reasoning traces, rephrased questions, regions of interest (RoI), and final answers. Uses bounding box annotations to compute an RoI IoU reward during RL training, improving the model's spatial focus. Achieves greater explainability and generalization in multimodal document understanding through structured RL.
**Relevance**: Shows that RL with spatial rewards (bounding box IoU) can improve document understanding models. Our proposed approach of using world model prediction as reward is analogous -- both use a geometric/visual signal rather than text-only feedback to train document processing models.

### 199. DianJin-OCR-R1: Reasoning-and-Tool Interleaved VLM for OCR (2025)
**Summary**: Applies GRPO with two reward signals (format reward for structured outputs, accuracy reward for correctness) to fine-tune Qwen2.5-VL-7B for document OCR. Proposes a reasoning-and-tool interleaved framework that balances VLM reasoning with expert OCR models. Consistently outperforms non-reasoning counterparts on benchmarks. Trained via SFT + Reinforcement Fine-Tuning (RFT). Extracts structured content (text, tables, formulas) from document images.
**Relevance**: Direct evidence that GRPO-based RL training improves OCR models. Our proposed world model reward would provide an additional, self-supervised reward signal alongside DianJin-OCR-R1's accuracy reward. The key difference is that our approach does not require ground truth for the accuracy reward -- the world model provides it.

## Synthesis: World Models for OCR Reward Signals (2026-04-09)

### Key Architectural Options for a Document World Model

Based on this literature survey, three viable architectures emerge for training a document world model as an OCR reward signal:

**Option A: RSSM-based Document World Model (Dreamer-style)**
- Architecture: RSSM with categorical latent states (like DreamerV2/V3) encoding document structure
- Training: Reconstruction loss on document images (predict next document region given current state + OCR tokens)
- Reward: Reconstruction prediction error = OCR quality score
- Pros: Well-proven architecture, single-GPU trainable, strong theoretical foundations
- Cons: Reconstruction loss may focus on irrelevant visual details (DreamerPro concern)
- Estimated effort: Medium-High (need to adapt RSSM to sequential document parsing)

**Option B: Decoder-Free Document World Model (R2-Dreamer/DreamerPro-style)**
- Architecture: Latent dynamics model with redundancy-reduction objective (no pixel decoder)
- Training: Barlow Twins-style cross-correlation between document embeddings and latent states
- Reward: Latent prediction error or latent distance serves as quality signal
- Pros: 1.6x faster training, avoids pixel-level distraction, more robust
- Cons: Harder to interpret, requires careful design to prevent representation collapse
- Estimated effort: Medium (simpler architecture, but less precedent for document domain)

**Option C: Ensemble Disagreement Reward (InDRiVE-style)**
- Architecture: Ensemble of small world models (3-5 models, could be RSSM or Transformer-based)
- Training: Each model trained on document data with different random seeds
- Reward: Disagreement between ensemble predictions on given OCR output = uncertainty = quality signal
- Pros: Completely reference-free, no ground truth needed, calibrated uncertainty estimate
- Cons: Higher compute cost (multiple models), may conflate OCR errors with inherent document ambiguity
- Estimated effort: High (need to train and manage ensemble)

**Option D: Autoregressive Token World Model (IRIS/Dreamer 4-style)**
- Architecture: Discrete autoencoder (VQ-VAE) + autoregressive Transformer over document tokens
- Training: Next-token prediction on tokenized document representations
- Reward: Token prediction probability = OCR quality (high probability = expected/good OCR, low = anomalous)
- Pros: Shares architecture with OCR models (Qwen-VL), enables parameter sharing, aligns with RLVR-World framework
- Cons: Discrete tokenization may lose fine-grained visual quality information
- Estimated effort: Medium (can leverage existing VLM infrastructure)

### Recommendation

**Option D (Autoregressive Token World Model)** is the most promising for our specific use case because:
1. It directly leverages the RLVR-World framework (#129) which already showed 30.7% accuracy improvements on language-based world models
2. It shares the autoregressive Transformer architecture with our existing Qwen-VL OCR model, enabling knowledge transfer
3. It naturally integrates with our CycleCap-style (#124) GRPO training pipeline
4. The discrete token representation aligns with OCR's inherent discrete output (text + bbox tokens)
5. The VQ-VAE component can be trained efficiently on document images (per Smaller World Models, #195, approximately 12 hours on single GPU)

The recommended implementation path:
1. Train a VQ-VAE on document images to learn discrete document tokens
2. Train an autoregressive Transformer to predict document tokens given OCR text output
3. Use token prediction probability (or reconstruction quality after decoding) as GRPO reward
4. This replaces our current SSIM+LPIPS+CLIP multi-metric with a single learned reward

### Connection to Existing Project Architecture

| Current Pipeline Component | World Model Replacement |
|---|---|
| SSIM metric | Pixel-space prediction error (Option A) or token probability (Option D) |
| LPIPS metric | Latent space prediction distance (Option B) or latent disagreement (Option C) |
| CLIP metric | Embedding similarity in world model latent space |
| LM Perplexity | Token prediction perplexity of document world model |
| Multi-metric weighted sum | Single world model reward score |

### Key Insight from DreamerV3 Ablation

The DreamerV3 Nature paper's finding that "performance rests predominantly on the unsupervised reconstruction loss" is the strongest theoretical support for our approach. It means:
- A document world model trained on unsupervised reconstruction will develop representations that capture document quality -- even without explicit quality labels
- The reconstruction objective alone provides a sufficient learning signal
- Task-specific reward signals (OCR accuracy) are secondary to the world model's generic understanding

### Gaps Identified

1. **No existing document-specific world models**: All surveyed world models target games, robotics, or driving. Document understanding is an unexplored application domain for world model architectures.
2. **No direct precedent for "world model as quality metric"**: While world model prediction error is used for intrinsic reward (exploration), using it as an external quality assessment metric is novel.
3. **Discrete vs. continuous reward tradeoff**: Using token prediction probability gives a discrete quality signal, while latent distance gives a continuous one. The optimal choice for GRPO training is unclear.
4. **Scale of document world model**: Dreamer 4 is 2B parameters, but Smaller World Models shows competitive results with much fewer. The minimum viable scale for document understanding is unknown.

### Additional Papers: World Model Ecosystem (2026-04-09)

### 200. I-JEPA: Self-Supervised Learning from Images with Joint-Embedding Predictive Architecture (CVPR 2023)
**Summary**: Predicts representations of masked image regions from a single context block, learning semantic features without pixel-level reconstruction or contrastive pairs. Uses Vision Transformers for scalable self-supervised learning. A ViT-Huge/14 trains on ImageNet in under 72 hours on 16 A100 GPUs. The non-generative approach avoids the pixel-reconstruction bias that plagues VAE-based world models.
**Relevance**: A document I-JEPA could learn to predict masked document regions in embedding space. Prediction error on document patches would indicate document quality, as a high-quality OCR output should produce a document where all regions are mutually predictable.

### 212. DIAMOND: Diffusion for World Modeling (NeurIPS 2024 Spotlight)
**Summary**: Uses diffusion models instead of autoregressive token prediction for world modeling, showing that visual details matter for sample-efficient RL in Atari. Achieves state-of-the-art Atari 100k results by leveraging the fine-grained visual modeling capability of diffusion models. Demonstrates that higher-fidelity world models lead to better policies.
**Relevance**: Connects our existing diffusion/flow matching research with world models. A diffusion-based document world model could generate high-fidelity document reconstructions, with denoising score as a quality metric. Aligns with our flow matching papers (#143-180).

### 219. Evaluating the World Model Implicit in a Generative Model (NeurIPS 2024 Spotlight)
**Summary**: Proposes methods to evaluate whether a generative model has learned an accurate internal world model. Tests whether generative models show calibrated "surprise" at impossible events, maintain consistent internal state, and can answer counterfactual queries. Finds that some video models learn genuine world models while others merely pattern-match.
**Relevance**: Directly relevant methodology for evaluating our proposed document world model. We could test whether a trained document world model shows appropriate "surprise" at OCR errors (impossible document structures), validating that it has learned document structure rather than surface patterns.

### 220. World Models for Anomaly Detection during MBRL Inference (Domberg & Schildbach, 2025)
**Summary**: Uses world model prediction error as an anomaly detection signal during inference. When the world model encounters observations that diverge significantly from its predictions, this signals out-of-distribution inputs or environment changes. Proposes threshold-based anomaly detection using prediction error magnitude.
**Relevance**: Directly applicable to OCR error detection. A document world model that predicts expected document appearance would show high prediction error when OCR errors distort the reconstruction, functioning as an anomaly detector for OCR quality without needing ground truth.

## JEPA Family and Latent-Space Perceptual Metrics for Document Comparison (2026-04-09)

This section provides deep analysis of JEPA architectures and latent-space perceptual metrics, focusing on whether these representations could replace or augment SSIM/LPIPS/CLIP for document image comparison in our reference-free OCR pipeline.

### 203. A Path Towards Autonomous Machine Intelligence (LeCun, 2022)
**Link**: https://openreview.net/pdf?id=BZ5a1r-kVsf
**Summary**: LeCun's position paper proposes a six-module architecture for autonomous AI, with JEPA at its core. Two encoding branches map observations x and targets y into representations s_x and s_y, with a predictor mapping s_x to predicted s_y. Predictions are in abstract representation space, discarding irrelevant details. JEPA is an Energy-Based Model: energy (prediction error in embedding space) is low when representations match, high when they mismatch. Proposes Hierarchical JEPA (H-JEPA) for multi-scale representations. Training uses non-contrastive methods (VICReg, Barlow Twins) without requiring negative examples or labels.
**Relevance**: **Critical conceptual foundation.** The JEPA energy function is itself a distance metric in embedding space. For OCR quality: encode original document image and OCR-reconstructed image with JEPA encoders, measure prediction energy between their embeddings. High energy = poor OCR (large discrepancy), low energy = good OCR (consistent visual content). Non-contrastive training aligns with our reference-free goal -- no human quality labels needed.

### 201. VL-JEPA: Joint Embedding Predictive Architecture for Vision-language (Dec 2025)
**Link**: https://arxiv.org/abs/2512.10942
**Summary**: Extends JEPA to vision-language by predicting continuous text embeddings instead of autoregressive token generation. Uses V-JEPA 2 as vision encoder, Llama 3 layers for prediction, lightweight text decoder invoked only when needed. 50% fewer trainable parameters than token-space VLMs with stronger performance. Supports selective decoding (2.85x fewer operations). Naturally supports open-vocabulary classification, retrieval, and discriminative VQA without architecture changes. Surpasses CLIP, SigLIP2, and Perception Encoder on 16 video benchmarks with only 1.6B parameters.
**Relevance**: **Very High for our project.** VL-JEPA directly bridges visual and textual embeddings, making it the ideal architecture for OCR quality assessment. Proposed approach: (1) encode original document with VL-JEPA's vision encoder, (2) encode OCR-extracted text with text encoder, (3) measure JEPA energy (embedding prediction error) as OCR quality score. This requires NO reconstruction step -- a fundamental simplification of our pipeline. The model already surpasses CLIP on retrieval benchmarks, suggesting it would outperform CLIPScore for document-text matching.

### 202. C-JEPA: Connecting JEPA with Contrastive Self-supervised Learning (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2410.19560
**Summary**: Identifies two limitations of I-JEPA: EMA fails to prevent representational collapse, and the predictor inadequately learns the mean of patch representations. Integrates JEPA with VICReg (Variance-Invariance-Covariance Regularization) to maintain embedding variance, minimize feature redundancy, and align representation means. On ImageNet-1K ViT-B/16: +0.8% linear probing, +1.0% fine-tuning vs I-JEPA. Improves COCO detection +0.8% AP, ADE20K segmentation +1.1% mIoU, DAVIS-2017 video segmentation +1.7%.
**Relevance**: If we build a document-JEPA metric, stability is critical -- the metric must give consistent scores for the same document pair. C-JEPA's VICReg integration prevents the embedding space from degenerating, which would cause the metric to give arbitrary scores. This paper shows that vanilla JEPA representations can be unstable, making C-JEPA's regularization a requirement for any JEPA-based document metric.

### 204. Genie: Generative Interactive Environments (DeepMind, ICML 2024)
**Link**: https://arxiv.org/abs/2402.15391
**Summary**: First generative interactive environment trained unsupervised from unlabeled Internet videos. 11B parameters. Three components: spatiotemporal video tokenizer, Latent Action Model (VQ-VAE that discovers interpretable latent actions from frame pairs without labels), and MaskGIT-based Dynamics Model. Learns fine-grained controls from Internet video without any action labels. The latent action codebook is kept small by design to encourage interpretable actions. Enables training agents in generated environments.
**Relevance**: Genie's Latent Action Model is conceptually interesting for documents: it learns what "transformation" connects two visual states without labels. Applied to original-vs-reconstructed document pairs, a similar model could learn the latent transformation induced by the OCR process. The magnitude/complexity of this latent action encodes how much the OCR changed the document -- a direct quality metric. Works without labels, matching our reference-free goal.

### 205. Cosmos World Foundation Model Platform for Physical AI (NVIDIA, Jan 2025)
**Link**: https://arxiv.org/abs/2501.03575
**Summary**: Suite of world foundation models including diffusion-based (latent diffusion in learned video tokenizer space) and autoregressive (Llama3-style GPT) variants. Video tokenizers compress to continuous (latent vectors) and discrete (integers) tokens. Trained on 9000 trillion tokens from 20M hours of data. Cosmos-Predict2 generates physics-aware images/videos. Open-weight, permissive license. Known limitations include temporal inconsistency and unrealistic physical interactions.
**Relevance**: Cosmos's dual continuous/discrete tokenization is relevant: continuous latents capture fine visual quality, discrete tokens capture structural content. For document comparison, continuous latent distance could measure rendering fidelity while discrete token comparison measures content accuracy. However, Cosmos is trained on physical-world video, so domain gap to documents is substantial.

### 206. iVideoGPT: Interactive VideoGPTs are Scalable World Models (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2405.15223
**Summary**: Autoregressive transformer integrating visual observations, actions, and rewards into unified token sequence. Novel compressive tokenization: conditional VQGAN with dual encoder-decoder pairs tokenizes future frames conditionally on earlier frames, achieving up to 16x token reduction by exploiting temporal redundancy. Pre-trained on millions of trajectories. Extended by RLVR-World (May 2025) with reinforcement learning from verifiable rewards.
**Relevance**: The conditional VQGAN tokenizer is most relevant: it encodes the "difference from expected" rather than absolute content. For document comparison, a conditional tokenizer trained on original-reconstruction pairs would encode specifically the OCR-induced changes, producing a compact error representation. Token count or code complexity of this conditional encoding serves as a quality metric -- simple codes mean minimal change (good OCR), complex codes mean substantial change (poor OCR).

### 207. How Far is Video Generation from World Model: A Physical Law Perspective (Nov 2024)
**Link**: https://arxiv.org/abs/2411.02385
**Summary**: Evaluates whether VAE-DiT video generation models (like Sora) can learn physical laws from visual data alone. Uses 2D simulation testbed with known physics. Tests in-distribution, out-of-distribution, and combinatorial generalization. Key finding: models approximate but do not truly learn underlying rules -- they struggle with extrapolation and law discovery. Provides quantitative velocity error metrics for evaluating physical plausibility.
**Relevance**: Important cautionary finding. A world-model-based document metric might capture visual similarity well (pattern matching) but may not truly understand text content correctness. The quantitative evaluation framework (measuring error against known ground truth) is instructive: we should similarly evaluate any world-model metric against known OCR error rates, not just visual quality scores.

### 208. DreamSim: Learning New Dimensions of Human Visual Similarity (NeurIPS 2023 Spotlight)
**Link**: https://arxiv.org/abs/2306.09344
**Summary**: Bridges "low-level" metrics (LPIPS, SSIM) and "high-level" measures (CLIP) by capturing mid-level similarities in layout, object pose, and semantic content. Concatenates CLIP, OpenCLIP, and DINO embeddings, fine-tunes with LoRA on ~20k image triplets of human judgments generated by diffusion models. Achieves 96.16% human agreement. Sensitive to foreground objects, semantic content, color, and layout. Generalizes from synthetic to real images. Follow-up (NeurIPS 2024) shows perceptually-aligned representations improve segmentation, depth estimation, and retrieval.
**Relevance**: **Very High.** DreamSim provides the recipe for building "DocSim": (1) concatenate embeddings from document-aware models (ColPali #80, DiT #81, LayoutLMv3 #82), (2) fine-tune with LoRA on OCR quality judgments from our pipeline. DreamSim's mid-level sensitivity to layout is exactly what document comparison needs -- it would distinguish layout-preserving OCR from layout-destroying OCR. The synthetic data pipeline (generate variations, collect judgments) maps directly to our OCR-reconstruct pipeline.

### 209. MILO: A Lightweight Perceptual Quality Metric for Image and Latent-Space Optimization (ACM TOG 2025)
**Link**: https://arxiv.org/abs/2509.01411
**Summary**: Lightweight, multiscale FR-IQA metric operating in both image space (MILO_I) and VAE latent space (MILO_L using Stable Diffusion's encoder). Trained via pseudo-MOS supervision (reproducible distortions scored by metric ensemble). Spatial masking models perceptual importance of different regions. Curriculum learning strategy: optimize perceptually less relevant regions first, then shift to more distorted areas. Outperforms existing FR-IQA metrics with compact architecture and fast inference. Published by MPI Informatics in ACM TOG 2025.
**Relevance**: **Directly applicable and immediately actionable.** MILO explicitly bridges pixel-space and latent-space quality metrics. For our pipeline: encode original and reconstructed documents with SD's VAE, apply MILO_L in latent space. The spatial masking is particularly relevant -- text regions in documents are perceptually more important than backgrounds, and MILO's masking could learn to weight them appropriately. This is the closest existing work to a "latent-space document quality metric" and could be integrated in 1-2 days.

### 210. PIM: An Unsupervised Information-Theoretic Perceptual Quality Metric (NeurIPS 2020)
**Link**: https://arxiv.org/abs/2006.06752
**Summary**: Combines information-theoretic objectives with human visual system-inspired architecture for unsupervised IQA. Multi-scale linear filtering decomposes images into pyramid representations; learns latent distributions via multivariate mutual information optimization. Quality measured as symmetrized KL divergence between latent distributions, which collapses to squared distance between means -- a "perceptually uniform space" like CIE Delta-E. Guided by efficient coding and slowness principles from neuroscience. Competitive with supervised LPIPS on BAPPS benchmark without any human labels. Code at google-research/perceptual-quality.
**Relevance**: **Very High and directly relevant to "reference-free" goal.** PIM shows that an unsupervised latent space can serve as a perceptual distance metric competitive with supervised LPIPS. For documents: (1) efficient coding principle aligns with OCR information extraction, (2) "perceptually uniform space" property means equal latent distances = equal perceptual differences, (3) unsupervised training requires no document-specific labels. The squared-mean-distance formula is computationally trivial, enabling real-time scoring. Could replace LPIPS in our pipeline with a more principled, unsupervised alternative.

### 211. Assessing Sample Quality via the Latent Space of Generative Models (ECCV 2024)
**Link**: https://arxiv.org/abs/2407.15171
**Summary**: Proposes examining a trained generative model's latent space to infer sample quality without external feature extractors. Key insight: quality relates to training data density in the latent space. High-density regions = well-represented data = good quality; low-density regions = rare/anomalous data = potentially lower quality. Latent density score correlates highly with quality for VAEs, GANs, and Latent Diffusion Models. Advantages: pre-generation quality estimation, domain-agnostic, applicable to latent editing. Code at cvlab-stonybrook/LS-sample-quality.
**Relevance**: **Conceptually novel for our project.** Instead of comparing original vs. reconstructed, we could assess OCR quality by examining where the reconstruction falls in a document generative model's latent space. High-density = looks like a typical document = good OCR. Low-density = anomalous appearance = likely OCR errors. This is a genuinely reference-free metric -- no original image comparison needed, just assessment of reconstruction plausibility.

### Synthesis: JEPA and Latent-Space Metrics for Reference-Free OCR (2026-04-09)

**Key Finding 1: JEPA energy is a theoretically ideal OCR quality metric.**

The chain of evidence:
- LeCun (#203) defines JEPA energy = prediction error in embedding space, a native distance metric
- I-JEPA (#200) learns semantic representations capturing layout and structure without augmentations
- VL-JEPA (#201) bridges visual and textual embeddings -- the ideal vision-text alignment for OCR
- C-JEPA (#202) adds stability guarantees critical for reliable metrics
- V-JEPA 2 (#193) shows that prediction error is calibrated ("surprise" correlates with physical impossibility)

**Concrete proposal: "VL-JEPA OCR Score"**
1. Encode original document image with VL-JEPA vision encoder
2. Encode OCR-extracted text with VL-JEPA text encoder
3. Measure embedding prediction error as OCR quality
4. Eliminates the reconstruction step entirely from our pipeline
5. Single forward pass through VL-JEPA replaces SSIM + LPIPS + CLIP + reconstruction

**Key Finding 2: Mature latent-space metrics exist for immediate integration.**

| Metric | Space | Supervised | Integration Effort | Document Suitability |
|--------|-------|-----------|-------------------|---------------------|
| PIM (#210) | Learned latent | No | 1 day (pip install) | Good -- principled, unsupervised |
| MILO_L (#209) | VAE latent | Pseudo-MOS | 1-2 days | Very Good -- spatial masking for text |
| DreamSim (#208) | Foundation ensemble | Triplets | 1-2 weeks | Excellent -- layout-sensitive |
| Latent Density (#211) | Generative latent | No | 1 week (needs gen model) | Novel -- truly reference-free |
| DiffSim (#176) | Diffusion attention | No | 2-3 days | Good -- attention-aligned |
| VL-JEPA (#201) | JEPA embedding | No | 2-4 weeks (model setup) | Best -- vision-text native |

**Key Finding 3: World model latent spaces offer distinct advantages over LPIPS/CLIP for documents.**

Why world model representations are better than classification/contrastive features:
- LPIPS: features learned for ImageNet classification (objects, scenes -- not documents)
- CLIP: features learned for image-text matching (web images, captions -- not document layouts)
- JEPA/world models: features learned to PREDICT future/missing visual content -- capturing causal structure

For documents specifically, this means a world-model metric could:
- Distinguish content-preserving changes (font swap) from content-altering changes (text error)
- Weight text regions by information content (title misspelling worse than watermark error)
- Capture hierarchical document structure (section > paragraph > line > word > character)

**Key Finding 4: Self-supervised world models directly parallel our "reference-free" requirement.**

| World Model Concept | OCR Metric Analogue |
|--------------------|--------------------|
| Learn dynamics without action labels | Assess quality without reference text |
| Predict next frame from current frame | Predict document quality from reconstruction |
| Latent prediction error as intrinsic reward | Visual/embedding error as quality signal |
| Unsupervised pre-training on video | Unsupervised pre-training on documents |
| Zero-shot transfer to new environments | Zero-shot transfer to unseen document types |

**Three actionable directions ranked by feasibility:**

1. **Immediate (1-2 days):** Integrate PIM (#210) or MILO_L (#209) as latent-space alternatives to LPIPS. Off-the-shelf, no document-specific training needed.

2. **Near-term (1-2 weeks):** Build "DocSim" following DreamSim (#208) recipe -- concatenate ColPali + DiT + LayoutLMv3 embeddings, fine-tune with LoRA on OCR quality judgments from our pipeline.

3. **Research frontier (1-3 months):** Apply VL-JEPA (#201) for direct vision-text embedding comparison, eliminating reconstruction entirely. Or train document-JEPA using I-JEPA (#200) approach on large-scale document images, using prediction energy as quality metric.

### Additional World Model Papers: Interactive and Diffusion-Based Systems (2026-04-09)

### 213. GameNGen: Diffusion Models Are Real-Time Game Engines (Google, Aug 2024)
**Authors**: Dani Valevski, Yaniv Leviathan, Moab Arar, Shlomi Fruchter
**Link**: https://arxiv.org/abs/2408.14837
**Summary**: First neural model to run a complex game (DOOM) interactively in real time. Built on augmented Stable Diffusion v1.4, runs at 20 FPS on a single TPU with only 4 denoising steps. Achieves PSNR 29.4 (comparable to JPEG) and LPIPS 0.249. Human raters cannot reliably distinguish short clips from real gameplay. Two-phase training: RL agent plays to generate data, then diffusion model trains on frame+action sequences. Handles complex state persistence (health, ammo, enemies, doors) over multi-minute sessions.
**Relevance**: **High.** Demonstrates that a single diffusion model can learn a complete visual environment simulator with complex state tracking. The PSNR/LPIPS evaluation metrics are the same ones in our pipeline. The two-phase training (collect data, then train generative model) maps to our approach. The 4-denoising-step inference is fast enough for real-time quality assessment.

### 214. STORM: Efficient Stochastic Transformer-based World Models (NeurIPS 2023)
**Authors**: Weipu Zhang, Gang Wang, Jian Sun, Yetian Yuan, Gao Huang
**Link**: https://arxiv.org/abs/2310.09615
**Summary**: Combines GPT-like Transformer sequence modeling with stochastic categorical latent variables (32 categories x 32 classes). Achieves 126.7% mean human performance on Atari 100k. Trains an agent with 1.85 hours of real-time interaction on a single RTX 3090 in only 4.3 hours. Stochastic latents prevent overfitting to specific trajectories and help bridge model-reality gaps. Uses causal-masked self-attention on latent categorical distributions produced by CNN encoder/decoder.
**Relevance**: **Very High for feasibility.** The single RTX 3090 / 4.3 hour training benchmark is the strongest evidence that lightweight world models are trainable on consumer hardware. Our 4x RTX 6000 Ada GPUs are substantially more powerful, so a STORM-style document world model could train in about 1 hour. The stochastic component addresses visual variance in documents (font rendering, anti-aliasing) for identical text+layout inputs.

### 215. Delta-IRIS: Efficient World Models with Context-Aware Tokenization (ICML 2024)
**Authors**: Vincent Micheli, Eloi Alonso, Francois Fleuret
**Link**: https://arxiv.org/abs/2406.19320
**Summary**: Addresses IRIS scalability by encoding stochastic deltas (changes) between time steps rather than full frames. Uses continuous tokens to summarize the current world state as context. An order of magnitude faster to train than previous attention-based approaches. Achieves SOTA on Crafter benchmark at multiple frame budgets. The delta encoding reduces the tokenizer burden by focusing only on what changed, while continuous context tokens provide a compact state summary.
**Relevance**: **High.** Delta encoding maps directly to document comparison: the "change" between an original document image and its OCR reconstruction is exactly what we want to measure. A Delta-IRIS-style model could learn to predict visual changes from specific OCR text+layout configurations. The magnitude of predicted deltas would serve as a quality metric -- small deltas indicate OCR output closely matches expected rendering.

### 216. Oasis: A Universe in a Transformer (Decart, Oct 2024)
**Authors**: Decart (Julian Quevedo, Quinn McIntyre, et al.)
**Link**: https://oasis-model.github.io/
**Summary**: First real-time, open-world interactive world model for Minecraft-like experiences, built on a diffusion transformer (DiT). Takes keyboard/mouse input, generates frames autoregressively. Open-sourced a 500M parameter model runnable locally. Achieves 47ms inference per frame at 360p on H100. Learns complex mechanics (building, physics, inventory) from gameplay video alone without game code. Pioneered GPU-efficient DiT inference techniques achieving 150ms per training iteration.
**Relevance**: **Medium-High.** The open-source 500M model and real-time inference demonstrate practical accessibility. The DiT architecture combining diffusion with transformers could generate document images conditioned on OCR tokens. The 47ms/frame inference suggests quality assessment could run at interactive speed during document processing.

### 217. MineWorld: Real-Time Open-Source Interactive World Model on Minecraft (Microsoft, Apr 2025)
**Authors**: Junliang Guo, Yang Ye, Tianyu He, et al. (Microsoft)
**Link**: https://arxiv.org/abs/2504.08388
**Summary**: Visual-action autoregressive Transformer interleaving image tokens with action tokens, trained via next-token prediction. Novel parallel decoding predicts spatially redundant tokens simultaneously, enabling 4-7 fps real-time interaction. Proposes new evaluation metrics assessing both visual quality and action-following capacity. Outperforms Oasis across all metrics while using fewer resources. First fully open-source Minecraft world model with code and weights released.
**Relevance**: **Medium-High.** The interleaved image-token + action-token architecture fits document modeling naturally: interleave document patch tokens with OCR text tokens. Separate metrics for visual quality and action-following parallel our need to evaluate visual fidelity and OCR accuracy independently. Parallel decoding for spatially redundant tokens is relevant for documents where most area is whitespace/background.

### 218. Sora: Video Generation Models as World Simulators (OpenAI, Feb 2024)
**Link**: https://openai.com/index/video-generation-models-as-world-simulators/
**Summary**: OpenAI's technical report framing text-to-video generation as world simulation. Uses a Diffusion Transformer (DiT) operating on visual patches in compressed latent space. Trained jointly on videos and images of variable durations, resolutions, and aspect ratios. Demonstrates emergent physical simulation: object permanence, 3D consistency, state tracking. Not a true interactive world model (no action conditioning), but shows scaling video generation leads to implicit world modeling.
**Relevance**: **Medium.** Establishes that scaling generative models on visual data produces emergent understanding of environmental dynamics. The DiT architecture on visual patches is shared by DIAMOND and other efficient world models. However, Sora's compute requirements (far beyond our budget) limit direct applicability. The hypothesis that sufficiently powerful generators implicitly learn world dynamics motivates document world model research.

### 221. Is Sora a World Simulator? Comprehensive Survey on General World Models (2024)
**Link**: https://arxiv.org/abs/2405.03520
**Summary**: Curates 250+ studies tracing evolution from text-to-video generation to world simulation. Categorizes models by spatial intelligence, action intelligence, and strategic intelligence. Argues current video generation models (including Sora) are NOT true world simulators -- they generate plausible outputs but lack grounded causal understanding. Identifies gaps: lack of physical consistency, limited long-horizon coherence, inability to handle novel concept compositions.
**Relevance**: **Medium.** The distinction between "plausible generation" and "grounded simulation" is directly relevant to our metric design. A reconstruction that looks plausible but contains systematic OCR errors is the exact failure mode we need to detect. Motivates moving beyond pixel-level metrics to world-model-based assessment that tests causal understanding.

### 222. LeWorldModel: Stable End-to-End JEPA from Pixels (Maes et al., March 2026)
**Link**: https://arxiv.org/abs/2603.19312
**Summary**: A lightweight end-to-end JEPA world model (15M params: 5M ViT-tiny encoder + 10M transformer predictor) that learns environment dynamics from raw pixels without pre-trained encoders, auxiliary rewards, or complex training heuristics. Key innovation: **SIGReg regularization** — uses the Cramér–Wold theorem to enforce Gaussian-distributed embeddings via statistical normality testing on random projections, reducing tunable hyperparameters from 6+ (PLDM) to just 1. Achieves 48x faster planning than foundation model baselines (DINO-WM) with 200x fewer latent tokens. Trains on a single GPU in a few hours with smooth, monotonic convergence. Demonstrates violation-of-expectation detection (calibrated surprise at teleportation/impossible events) and emergent temporal path straightening (increasing cosine similarity between consecutive velocity vectors) without explicit regularization. Linear probes recover physical quantities (agent location, object position) from latent space.
**Relevance**: **Very High.** The most practical JEPA variant for our project:
- **15M params, single GPU, hours to train** — fits our compute constraints perfectly
- **SIGReg** provides a principled, single-hyperparameter solution to the collapse problem that C-JEPA (#202) solves via VICReg (more complex)
- **Violation-of-expectation** directly maps to OCR error detection: a document JEPA showing high "surprise" at a reconstruction indicates OCR errors distorted expected document structure
- **Emergent path straightening** connects to Rectified Flow (#144) — the latent dynamics naturally produce straight interpolation paths, useful for document comparison in latent space
- **End-to-end from pixels** means no need for pre-trained document encoders — the model learns document representations jointly with dynamics
- A "LeDocModel" could train on document image patches with OCR tokens as actions, learning to predict what document regions should look like. Prediction error = OCR quality metric. Total training: ~15M params, single RTX 6000, a few hours.

## FFT-Based Frequency Fingerprinting for Character-Level Image Matching (2026-04-10)

This section covers the idea of using FFT to extract frequency fingerprints from character/glyph images in documents and matching against a pre-built database -- analogous to how Shazam matches audio via spectrogram peak constellation maps. Papers span the Shazam algorithm itself, FFT-based image fingerprinting, Fourier descriptors for character recognition, phase-based image comparison, template matching in the frequency domain, perceptual hashing, and spectral methods for document analysis.

### 223. An Industrial-Strength Audio Search Algorithm (Wang, ISMIR 2003)
**Link**: https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf
**Summary**: The foundational Shazam paper by Avery Li-Chun Wang. Describes a noise-robust, scalable audio search engine that identifies songs from short phone-microphone recordings against a database of millions of tracks. The key insight: extract high-energy spectral peaks from the Short-Time Fourier Transform to create a sparse "constellation map," then pair peaks combinatorially into compact hash tuples [f1, f2, delta_t] for database lookup. Matching is performed by finding clusters of time-aligned hash tokens in scatterplots. The system achieves identification with as few as 1-2% hash survival from heavily corrupted audio, running in under 10ms per query.
**Relevance**: **The direct inspiration for the user's proposed approach.** The constellation-map + combinatorial-hashing paradigm could be adapted to 2D image spectra: extract high-magnitude peaks from the 2D FFT of character images, pair them into hash tuples encoding spatial-frequency relationships, and match against a database of known character frequency fingerprints. Key differences from audio: images are 2D (not 1D+time), characters need rotation/scale invariance, and the "database" would be per-font rather than per-song. No published work applies this exact paradigm to image matching -- this is an open research direction.

### 224. SpectroMap: Peak Detection Algorithm for Audio Fingerprinting (2022)
**Link**: https://arxiv.org/abs/2211.00982
**Summary**: Proposes a refined peak detection algorithm for audio fingerprinting that extracts topological prominences from spectrograms via time-frequency bands. Improves upon the basic Shazam peak-picking by using band-specific adaptive thresholds, reducing false peaks while retaining true landmarks. The algorithm is designed for embedded/mobile contexts where computational efficiency matters.
**Relevance**: Provides a more modern, refined version of Shazam's peak detection. If adapting constellation maps to 2D image FFTs, SpectroMap's band-adaptive thresholding translates to frequency-ring-adaptive thresholding in 2D Fourier space -- pick peaks separately in low/mid/high frequency rings to ensure representation across all spatial scales.

### 225. Fast OCR through Glyph Hashing for Document Conversion (IEEE 2005)
**Link**: https://ieeexplore.ieee.org/document/1575661/
**Summary**: Proposes a hash-based approach to OCR where character glyphs are hashed for rapid lookup against a pre-built hash table. Tested on 68,987 PDF documents containing 1.15 billion characters, yielding only 3.2 million unique hashes. Recognizes 80% of unique glyphs and over 96% of all character instances from unseen documents. Processes over 100,000 characters per second. The hash captures not just character identity but also font size, style (bold, italic), and font name. Scales to hundreds of fonts and thousands of characters per font.
**Relevance**: **Directly validates the "character fingerprint database" concept.** This paper proves that character glyphs from rendered documents have highly repeatable visual signatures -- 1.15 billion characters mapped to only 3.2 million unique hashes. The key difference from the FFT approach: this uses pixel-domain hashing rather than frequency-domain fingerprints. An FFT-based fingerprint could provide additional robustness to minor rendering differences (anti-aliasing, sub-pixel shifts) that pixel hashing would miss.

### 226. WaveOCR: FFT-Based Optical Character Recognition (AGH Krakow, 2024)
**Link**: https://github.com/AleGrz/WaveOCR
**Summary**: A proof-of-concept C++ implementation (with Python bindings) of OCR using FFT-based template matching. Generates a character alphabet from a provided TrueType font, then uses FFT cross-correlation to match input text regions against the stored character templates. Employs non-maximum suppression with IoU and coefficient norming as post-processing heuristics. Can recognize black, white, and colored text that is slightly rotated and slightly noisy. Uses OpenCV for FFT, Hough Transform, padding, and resizing. Not competitive with ML-based OCR (Tesseract) but demonstrates the feasibility of frequency-domain character matching.
**Relevance**: **The closest existing implementation to the proposed approach.** WaveOCR proves that FFT cross-correlation can match characters against font-specific templates in the frequency domain. For the reference-free OCR metric, the same principle applies in reverse: given the OCR output (which specifies expected characters), generate frequency-domain templates from the expected font, then cross-correlate with the actual image regions to verify each character is correctly rendered. Low correlation scores indicate OCR errors or rendering distortions.

### 227. An FFT-Based Technique for Translation, Rotation, and Scale-Invariant Image Registration (Reddy & Chatterji, IEEE TIP 1996)
**Link**: https://ieeexplore.ieee.org/document/506761/
**Summary**: Extends the phase correlation technique to handle translation, rotation, AND scaling between two images. The Fourier-Mellin transform converts the magnitude spectrum to log-polar coordinates, where rotation becomes a horizontal shift and scaling becomes a vertical shift. Phase correlation then recovers rotation and scale as translations in log-polar space. After derotating and rescaling one image, a final phase correlation recovers the translational offset. Requires only 3 FFTs and 3 IFFTs total. Shows excellent robustness against random noise.
**Relevance**: **Essential for handling character variations.** Characters in scanned documents may be slightly rotated, scaled differently, or shifted relative to expected positions. The Fourier-Mellin transform provides a mathematically elegant way to achieve RST-invariant matching of character frequency fingerprints against database templates. This removes the need to enumerate all possible rotations/scales when comparing, reducing O(N*R*S) to O(N) matching complexity. Multiple open-source implementations exist (imreg_dft on PyPI).

### 228. Shape Discrimination Using Fourier Descriptors (Persoon & Fu, IEEE SMC 1977)
**Link**: https://ieeexplore.ieee.org/document/4309681/
**Summary**: Seminal paper on using Fourier descriptors to encode object boundaries for shape recognition. Transforms the boundary contour into the frequency domain, where low-frequency coefficients capture gross shape and high-frequency coefficients capture fine detail. Proposes a distance measure between shapes in Fourier descriptor space. Demonstrates experimental results on character recognition (letters, digits) and machine parts recognition. The Fourier descriptor magnitude is invariant to translation, rotation, scaling, and contour starting point.
**Relevance**: **Foundational work for character shape fingerprinting.** Fourier descriptors of character contours provide a frequency-domain representation that is naturally invariant to geometric transformations. For the proposed approach, rather than computing the full 2D FFT of character images, one could extract character contours and compute 1D Fourier descriptors, using only the top K (typically 16) coefficients as a compact fingerprint. The distance measure between Fourier descriptors directly quantifies shape dissimilarity between expected and actual characters.

### 229. Fast Normalized Cross-Correlation (Lewis, Vision Interface 1995)
**Link**: https://scribblethink.org/Work/nvisionInterface/nip.pdf
**Summary**: Describes how to efficiently compute normalized cross-correlation (NCC) for template matching using precomputed integral images. While the cross-correlation numerator can be computed in the frequency domain via FFT (exploiting the convolution theorem), the normalization denominator does not have a simple frequency-domain expression. Lewis's key insight: precompute running sums of the image and image-squared using integral tables, then normalize efficiently in the spatial domain. The overall complexity is O(MN log MN) dominated by the FFT. This became the standard algorithm used in OpenCV's matchTemplate (TM_CCOEFF_NORMED) and MATLAB's normxcorr2.
**Relevance**: **The practical workhorse algorithm for character-template matching.** If building a frequency fingerprint matching system, NCC via FFT is the go-to method for comparing each document region against character templates. Lewis's method provides illumination-invariant matching (via normalization) with FFT-accelerated computation, processing entire images against a template in a single FFT operation rather than sliding a window pixel-by-pixel.

### 230. Computation of the Normalized Cross-Correlation by FFT (Kaso, PLOS ONE 2018)
**Link**: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203434
**Summary**: Develops a scheme for computing the full normalized cross-correlation entirely in the frequency domain using FFT, handling complex-valued signals. Benchmarks against Lewis's hybrid approach and finds that the fully-frequency-domain computation enables parallelization opportunities but does not always outperform the optimized hybrid approach (FFT numerator + spatial denominator). The method supports template matching in facial recognition, motion tracking, and medical image registration.
**Relevance**: Provides a modern, formal treatment of FFT-based NCC with implementation details. For batch character fingerprint matching (comparing many character regions simultaneously), the fully-FFT approach may enable better GPU parallelization than the hybrid method, which is relevant when processing full document pages with hundreds of characters.

### 231. Image Features from Phase Congruency (Kovesi, Videre/MIT 1999)
**Link**: https://www.cs.rochester.edu/u/brown/Videre/001/articles/v1n3001.pdf
**Summary**: Introduces phase congruency as a dimensionless, illumination-invariant feature detector based on the observation that perceptually significant features (edges, lines, ridges) occur at points where Fourier components are maximally in phase. Uses Log-Gabor filters for multi-scale decomposition. Unlike gradient-based edge detectors, phase congruency detects features at all phase angles (not just steps). The measure is contrast-invariant -- it depends only on the alignment of phase components, not their amplitude.
**Relevance**: **Highly relevant for document comparison.** Phase carries structural information (where edges and features are), while magnitude carries contrast/intensity information. For comparing original vs. reconstructed document images, phase congruency maps would highlight where structural features (character edges, stroke junctions) exist in both images. Comparing phase congruency maps between original and reconstruction directly measures structural fidelity -- a character with distorted strokes would show different phase congruency patterns even if overall intensity/contrast is similar.

### 232. Biological Basis and Applications of Image Phase Congruency: Comprehensive Survey (PMC 2024)
**Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11274423/
**Summary**: The most comprehensive survey on image phase congruency (IPC) to date. Covers the biological basis (human visual cortex processes phase information for feature detection), mathematical foundations (Morrone et al. 1986, Kovesi extensions), computational methods (Log-Gabor filters, monogenic filters, quaternion extensions), and 20+ application areas including image quality assessment, denoising, segmentation, registration, fusion, and object detection. Reviews how IPC is robust to illumination/contrast changes and provides absolute (not relative) feature measures.
**Relevance**: Validates phase congruency as a mature, well-understood framework with direct IQA applications. The survey catalogs multiple IQA metrics built on phase congruency (FSIM, GSM, etc.) that could serve as alternatives or supplements to SSIM/LPIPS for our document comparison pipeline. The biological basis (human vision system uses phase for feature detection) provides theoretical grounding for using phase-based metrics in document quality assessment.

### 233. Spectral Hashing (Weiss, Torralba & Fergus, NeurIPS 2008)
**Link**: https://people.csail.mit.edu/torralba/publications/spectralhashing.pdf
**Summary**: Proposes a data-dependent hashing method that maps images to compact binary codes for similarity search. Uses a spectral graph partitioning approach where the optimal hash function solutions are thresholded eigenvectors of the graph Laplacian. The resulting binary codes preserve semantic similarity: nearby images in feature space get similar hash codes. Outperforms Locality-Sensitive Hashing (LSH) and Restricted Boltzmann Machines for nearest-neighbor retrieval tasks. The "spectral" name refers to the graph spectral decomposition used in learning the hash functions, not to Fourier spectral analysis.
**Relevance**: While not directly frequency-domain fingerprinting, spectral hashing provides the theoretical framework for learning compact binary codes that preserve similarity. For the character fingerprint database, one could learn hash functions (via spectral hashing or its successors) that map character frequency features to compact binary codes, enabling sub-millisecond lookup of the nearest character match in a database of thousands of character templates.

### 234. State of the Art: Image Hashing (arXiv:2108.11794, 2021)
**Link**: https://arxiv.org/abs/2108.11794
**Summary**: Comprehensive survey of perceptual image hashing covering classical methods (Average Hash, Difference Hash, pHash/DCT-based, Wavelet-based, SVD-based) and deep hashing methods (autoencoders, GANs, contrastive learning). Catalogs applications: image retrieval, copy detection, authentication, watermarking, forensics, and reduced-reference image quality assessment. The DCT-based pHash is identified as the most widely deployed classical method, computing the DCT of a downscaled grayscale image and thresholding the low-frequency coefficients into a binary hash. Deep hashing methods learn task-specific hash functions that outperform hand-crafted approaches but require training data.
**Relevance**: Provides the complete landscape of image hashing methods, several of which are directly applicable to character fingerprinting. The DCT-based pHash approach (compute DCT of character image, threshold top-K coefficients) is the simplest possible "frequency fingerprint" for characters. The survey's coverage of deep hashing suggests that learning character-specific hash functions from font rendering data could outperform fixed DCT/FFT approaches.

### 235. A Survey of Perceptual Hashing for Multimedia (ACM TOMM 2025)
**Link**: https://dl.acm.org/doi/10.1145/3727880
**Summary**: The most recent and comprehensive survey on perceptual hashing, covering image, video, and neural network model hashing. Systematically categorizes hash generation methods by transform domain: DFT, DCT, DWT, Radon transform, log-polar transform, quaternion transform, and SVD. Analyzes robustness of each transform against specific distortions (rotation, scaling, compression, noise). Reviews evaluation metrics including normalized Hamming distance, BER, ROC curves, and discusses the fundamental tradeoff between robustness (tolerance to acceptable modifications) and discrimination (sensitivity to meaningful differences).
**Relevance**: **Key reference for designing the character frequency fingerprint system.** The survey's systematic comparison of transforms reveals that DCT concentrates energy in low-frequency coefficients (good for noise robustness), DWT provides multi-scale analysis (good for capturing both coarse shape and fine detail), and log-polar DFT provides rotation/scale invariance. For character fingerprinting, the optimal approach may combine multiple transforms -- e.g., log-polar DFT for invariance + DCT for compactness.

### 236. Fourier Descriptors for Broken Shapes (EURASIP JASP 2013)
**Link**: https://link.springer.com/article/10.1186/1687-6180-2013-161
**Summary**: Extends classical Fourier descriptors to handle broken/fragmented shapes that have multiple disconnected contours. Uses the convex hull of the shape and distance to the closest actual contour point along the convex hull to define a single continuous function amenable to Fourier analysis. Evaluated on glyph recognition (musical neumes) where print quality causes random fragmentation into up to 8 pieces. Recognition rates are comparable to standard Fourier descriptors on connected shapes while also handling the broken case. Includes scale/rotation normalization schemes.
**Relevance**: **Directly relevant to OCR character fingerprinting.** In real document images, character strokes may be broken due to poor scanning, compression artifacts, or font rendering issues. Standard Fourier descriptors fail on disconnected contours, but this method handles exactly the case we face: matching character glyphs that may be fragmented in the original scan against clean character templates in the database. The glyph recognition application (musical notation) is closely analogous to character recognition.

### 237. Template Matching Advances and Applications in Image Analysis (arXiv:1610.07231, 2016)
**Link**: https://arxiv.org/abs/1610.07231
**Summary**: Comprehensive survey of template matching methods covering both spatial-domain (pixel correlation, NCC, SSD) and frequency-domain (FFT cross-correlation, phase correlation) approaches, as well as feature-based methods (SIFT, SURF). Reviews applications in satellite imagery, medical imaging, face recognition, and OCR. Key finding: cross-correlation methods in the frequency domain are more efficient for large templates, while spatial methods are faster for small templates. The paper explicitly discusses template matching for OCR (license plate recognition) and notes that the limited variability of printed text makes template matching viable for character recognition.
**Relevance**: Provides the theoretical and practical grounding for choosing between spatial and frequency-domain template matching for character fingerprinting. For our use case (matching character regions ~20-50px against templates), the survey suggests that spatial NCC may actually be faster than FFT-based matching due to the small template size. However, FFT matching becomes advantageous when scanning the entire page for multiple characters simultaneously.

### 238. High Accuracy Character Recognition Using Fourier and Topological Descriptors (Pattern Recognition 1984)
**Link**: https://www.sciencedirect.com/science/article/abs/pii/0031320384900499
**Summary**: Achieves 98%+ character recognition accuracy using a two-stage approach: first classify using Fourier descriptors of character boundaries, then resolve ambiguities using topological descriptors (holes, endpoints, junctions). The Fourier descriptors' rotational invariance causes confusion between characters related by rotation (2/5, 6/9), which the topological stage resolves. The combined approach is fast and highly accurate for printed characters.
**Relevance**: Demonstrates that Fourier descriptors alone achieve near-perfect character discrimination, with a small set of confusable pairs (2/5, 6/9, rn/m, cl/d) that require additional topological features. For the character frequency fingerprint database, this suggests a two-level approach: first match by FFT/Fourier descriptor similarity, then verify ambiguous matches using topological features (hole count, stroke endpoint positions). The 98% accuracy from 1984 is notable given the simplicity of the method.

### 239. Advancing Audio Fingerprinting Accuracy: Addressing Background Noise (arXiv:2402.13957, 2024)
**Link**: https://arxiv.org/abs/2402.13957
**Summary**: Proposes an AI/ML-enhanced audio fingerprinting algorithm building on the Shazam/Dejavu framework. Signal processing includes FFT, spectrogram computation, and spectral peak extraction, with the constellation-map + hash pairing approach from Shazam. The ML component adds noise-robust feature extraction that improves identification accuracy in high-noise environments. Evaluates on real-world noisy recordings and demonstrates improved precision over the baseline Dejavu implementation.
**Relevance**: Shows that the Shazam constellation-map paradigm can be augmented with learned features for improved robustness. If adapting the approach to character images, a similar hybrid could work: use FFT peak extraction as the base fingerprint, then augment with learned features (e.g., from a small CNN) to handle degraded/noisy document scans. The noise robustness improvements are directly relevant to scanned documents with compression artifacts.

### 240. Scalable Hash-Based Character Recognition (US Patent US20060171588, 2006)
**Link**: https://patents.google.com/patent/US20060171588
**Summary**: Microsoft patent describing a scalable character glyph hash table system for OCR. Glyphs are hashed based on their visual appearance, and the hash table allows quick lookup of character metadata (Unicode value, font name, size, style). The system can be trained for specific environments, users, languages, and document types. Traditional OCR serves as a fallback for unknown glyphs, and successful OCR results are fed back to update the hash table, creating a continuously improving system. The patent notes that traditional OCR is slow (~1000 chars/sec) and does not scale to many fonts.
**Relevance**: Provides a practical system architecture for the character fingerprint database concept. The key architectural ideas are: (1) hash-based primary lookup with OCR fallback, (2) continuous learning from OCR results to expand the hash table, (3) environment-specific training (we would train per-font). For the FFT approach, replace pixel hashing with frequency-domain fingerprinting for better robustness to rendering variations.

### 241. Digital Fingerprinting on Multimedia: A Survey (arXiv:2408.14155, 2024)
**Link**: https://arxiv.org/abs/2408.14155
**Summary**: Comprehensive survey covering digital fingerprinting across audio, image, video, and text modalities. For images, catalogs fingerprinting methods including DCT-based, wavelet-based, and deep learning-based approaches. For audio, details the FFT-based spectrogram peak extraction (Shazam-style) as the dominant paradigm. Discusses cross-modal fingerprinting and the use of perceptual hashing as compact content summaries for efficient retrieval. Notes that fingerprints are highly compressed representations containing only the most relevant features, drastically reducing the search space.
**Relevance**: Provides the cross-modal perspective linking audio fingerprinting (FFT spectral peaks) with image fingerprinting (DCT/wavelet features). The survey implicitly supports the user's idea: if FFT peak extraction works for audio content identification, the same principle applied to 2D image spectra could work for character identification. The survey's coverage of deep fingerprinting methods also suggests that learned frequency features could outperform hand-crafted FFT peaks.

### 242. OCR-Based Document Image Quality Assessment (Frontiers in Signal Processing, 2026)
**Link**: https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2026.1779355/full
**Summary**: Proposes a document image quality assessment model that predicts OCR accuracy without running OCR. Extracts twelve features capturing sharpness, focus, edge clarity, and structural distortion. Key frequency-domain features include: Variance of Laplacian (high-frequency content indicator), Gaussian-Laplacian combination (edge detection + high-frequency detail), and spectral analysis of edge energy distribution. The model correlates predicted quality with actual OCR accuracy, demonstrating that frequency-domain features (particularly high-frequency energy) are strong predictors of OCR performance.
**Relevance**: **Directly validates using frequency-domain analysis for OCR quality assessment.** The paper shows that high-frequency energy (edges, fine details) measured via Laplacian/spectral analysis strongly predicts OCR accuracy. For our approach, this suggests that comparing high-frequency energy between original and reconstructed document images could serve as a quality metric -- OCR errors that distort character edges would reduce high-frequency energy in the reconstruction relative to the original.

### 243. DCT vs. FFT Frequency Features for Deepfake Detection (IEEE 2025)
**Link**: https://ieeexplore.ieee.org/document/11368196/
**Summary**: First systematic comparison of DCT versus FFT features for image authenticity detection across 12 JPEG quality levels. A dual-branch CNN achieves 99.47% accuracy on clean data. Critical finding: DCT features degrade 24.7% less than FFT features under JPEG compression (72.67% vs 62.67% accuracy at Q=10). This is because JPEG operates in the DCT domain, so DCT features are naturally aligned with compression artifacts, while FFT features are disrupted by the block-based DCT compression.
**Relevance**: **Important practical consideration for choosing DCT vs FFT for character fingerprints.** Since scanned/compressed documents undergo JPEG compression (DCT-based), character fingerprints based on DCT coefficients would be more robust to compression artifacts than FFT-based fingerprints. This suggests using DCT rather than FFT for the frequency fingerprint, or using FFT only on high-quality uncompressed source images while using DCT for compressed document scans.

## Perceptual Hashing and Image Hashing for Document Comparison — 2026-04-10

Research into whether perceptual hashing (pHash, dHash, wHash, etc.) and related hashing techniques can serve as fast, lightweight alternatives or complements to SSIM/LPIPS for comparing original vs. reconstructed document images.

### 244. PHASER: Perceptual Hashing Algorithms Evaluation and Results (Forensic Sci Int: Digital Investigation, 2024)
**Authors:** McKeown, Buchanan, et al. (Edinburgh Napier University)
**Link:** https://www.sciencedirect.com/science/article/pii/S2666281723001993
**Summary:** PHASER is an open-source forensic framework for rigorous evaluation of perceptual hashing algorithms. It provides a modular pipeline where users specify a perceptual hash (pHash, PDQ, NeuralHash, etc.), image transform (crop, rotate, compress, etc.), and distance metric (Hamming, cosine), then evaluates performance via ROC curves, EER plots, and confusion matrices. The framework revealed that bit-weighting in Hamming comparisons can improve matching performance for DCT-based hashes, and that different hashes have very different robustness profiles across transforms. PHASER is available as a Python library with seaborn-based plotting. Presented at DFRWS EU 2024.
**Relevance:** Directly applicable to our project. We could use PHASER to systematically benchmark pHash, dHash, wHash, and PDQ on our original-vs-reconstructed document image pairs, measuring which hash algorithm and distance metric best correlates with OCR quality. The modular framework saves significant implementation effort.

### 245. Beyond Hamming Distance: Spatial Encoding in Perceptual Hashes (Forensic Sci Int: Digital Investigation, 2025)
**Authors:** McKeown & Buchanan (Edinburgh Napier University)
**Link:** https://www.sciencedirect.com/science/article/pii/S2666281725000174
**Summary:** This paper demonstrates that perceptual hash bits encode spatial/positional information about the image -- localized modifications produce observable patterns in hash strings. Standard Hamming distance is a global measure that discards this positional information. The authors prototype three alternative distance metrics: Normalised Convolution Distance, Hatched Matrix Distance, and 2-D Ngram Cosine Distance. Key finding: the worst-case image mirroring transform for DCT-based hashes (pHash, PDQ) can be completely mitigated by using spatially-aware distance metrics without changing the hash generation algorithm. Hatched Matrix Distance with numba compilation is only 1.5x slower than Hamming distance.
**Relevance:** Highly relevant. Our reconstructed documents have positional differences from originals (shifted text blocks, misaligned regions). A spatially-aware hash distance metric could distinguish "globally different layout" from "locally shifted but content-correct" -- exactly the kind of nuance needed for OCR quality assessment. The 2-D Ngram approach could capture block-level spatial patterns in document layout changes.

### 246. Distance Distributions and Runtime Analysis of Perceptual Hashing Algorithms (JVCIR, 2024)
**Authors:** Sharma, S.
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S1047320324002669
**Summary:** Large-scale benchmarking study evaluating perceptual hashing algorithms on two million images. Measures both hash computation time and Hamming distance distributions across nine image variants (compression, rotation, scaling, etc.). Key finding: there is a fundamental tradeoff between computation time and robustness -- faster algorithms (dHash, aHash) are less robust to transformations, while more robust algorithms (pHash) are slower. The paper provides the most comprehensive runtime analysis available, showing that perceptual hashing remains orders of magnitude faster than neural metrics while providing reasonable similarity detection.
**Relevance:** Directly addresses our core question of speed vs. sensitivity tradeoffs. The 2M-image benchmark provides confidence intervals for thresholding decisions. For our pipeline, this data helps choose which hash to use: dHash for pre-filtering (sub-millisecond), pHash for quality scoring (still under 120ms), vs. LPIPS (~50-200ms with GPU). The runtime analysis confirms hashing as a viable lightweight complement to our existing metrics.

### 247. Hamming Distributions of Popular Perceptual Hashing Techniques (Forensic Sci Int: Digital Investigation, 2023)
**Authors:** McKeown & Buchanan
**Link:** https://arxiv.org/abs/2212.08035
**Summary:** Million-image-scale evaluation of perceptual hashing archetypes including Facebook PDQ, Apple NeuralHash, pHash, dHash, aHash, and wHash. Evaluates against seven image variants and analyzes Hamming distance distributions for both matching and non-matching pairs. Key finding: pHash and wHash provide the best overall performance, with wHash showing particular strength for compression, scaling, and thumbnail transformations. PDQ performed poorly across most transform types. NeuralHash showed different distributional characteristics from classical hashes. The paper provides threshold guidance: for 64-bit hashes, typical matching thresholds are Hamming distance <= 10 for high confidence.
**Relevance:** Critical for choosing which hash to implement. The finding that wHash (wavelet-based) matches or exceeds pHash for most transforms is notable -- wavelet decomposition may be more sensitive to text/edge features in documents. The threshold guidance helps us convert hash distances to quality scores. The million-image statistics give us false-positive/false-negative rate estimates.

### 248. PDQ & TMK+PDQF: A Test Drive of Facebook's Perceptual Hashing (arXiv, 2019)
**Authors:** Dalins, Wilson, Carman (Monash University / Australian Federal Police)
**Link:** https://arxiv.org/abs/1912.07745
**Summary:** Independent evaluation of Facebook/Meta's PDQ perceptual hashing algorithm (256-bit, DCT-based, evolution of pHash) and its video counterpart TMK+PDQF. PDQ generates a 256-bit hash via 16x16 subimage averaging followed by quantized DCT, stored as a 64-character hex string. Comparison reduces to Hamming distance, with distance <= 30 (of 256) recommended for confident matching. PDQ hash generation takes ~0.08 seconds vs. ~0.02 for MD5. Key advantage: rotation and mirror hashes can be inferred by manipulating the DCT matrix without re-computing. Open-sourced by Meta with C++, Python, and Rust implementations.
**Relevance:** PDQ's 256-bit hash (vs. pHash's 64-bit) provides finer granularity for quality scoring. The 4x more bits could help distinguish minor OCR quality differences. The rotation/mirror invariance via DCT manipulation is useful since our reconstructed documents may have slight orientation differences. PDQ is production-proven at Facebook scale and has pip-installable Python bindings, making integration trivial.

### 249. Robustness of Practical Perceptual Hashing Algorithms to Hash-Evasion and Hash-Inversion Attacks (NeurIPS AdvML Workshop, 2024)
**Authors:** Madden, Bhavsar, Dorje, Li (Binghamton University)
**Link:** https://arxiv.org/abs/2406.00918
**Summary:** Assesses the adversarial security of three widely-deployed perceptual hashing algorithms -- PhotoDNA (Microsoft, 1152-bit), PDQ (Meta, 256-bit), and NeuralHash (Apple) -- against blackbox hash-evasion and hash-inversion attacks. Contrary to prior literature claiming fragility, the authors find that these PHAs demonstrate significant robustness in blackbox settings, partially due to random hash variations inherent in perceptual hashing. They also propose a defense that introduces intentional perturbations into hashes to further enhance security.
**Relevance:** The robustness finding is encouraging for our use case. If PHAs are robust even against adversarial attacks, they should be robust to the non-adversarial differences between original and reconstructed documents. The comparison of PhotoDNA (1152-bit), PDQ (256-bit), and NeuralHash provides a spectrum of hash sizes to consider for different quality granularity needs.

### 250. Implementation and Benchmarking of Perceptual Image Hash Functions (Master's Thesis, 2010)
**Authors:** Zauner, C. (Upper Austria University of Applied Sciences)
**Link:** https://www.phash.org/docs/pubs/thesis_zauner.pdf
**Summary:** The foundational benchmarking work that introduced the Rihamark framework for evaluating perceptual image hash functions. Benchmarks four algorithms: DCT-based (pHash), Marr-Hildreth operator-based, radial variance-based, and block mean value-based hashing. Key findings: block mean value hashing is fastest; DCT-based (pHash) is slowest but most robust; Marr-Hildreth offers the best discriminative ability regardless of image content. The thesis provided the basis for the pHash open-source library (phash.org) which remains widely used. The Rihamark benchmark applies horizontal flipping, rotation, resizing, and other transforms to evaluate robustness.
**Relevance:** Essential reference for understanding the design space of perceptual hashing. The finding that different hash functions excel at different properties (speed vs. robustness vs. discrimination) directly informs our multi-metric strategy: use fast hashes for coarse screening and robust hashes for fine-grained quality scoring. The Marr-Hildreth hash's content-independent discrimination is interesting for documents where text content varies dramatically.

### 251. Effective Near-Duplicate Image Detection using Perceptual Hashing and Deep Learning (Information Processing & Management, 2025)
**Authors:** Jakhar, Y. & Borah, M.D. (NIT Silchar)
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0306457325000287
**Summary:** Proposes a hybrid framework integrating three complementary technologies: pHash for fast initial filtering, a Siamese network for learning fine-grained image differences, and a Vision Transformer (ViT) for capturing global context via self-attention. The pipeline first applies pHash to quickly filter obviously dissimilar images, then uses the Siamese-ViT architecture for more nuanced comparison. The approach handles diverse transformations including arbitrary rotations (0-360 degrees), scaling, cropping, color variations, and intensity changes.
**Relevance:** The cascaded pHash-then-deep-learning architecture is directly applicable to our pipeline. We could use pHash as a fast pre-filter (if hash distance is very large, the reconstruction is clearly bad -- skip expensive LPIPS/CLIP), then apply our existing metrics only when the hash indicates potential quality. This could reduce LPIPS computation by 50-80% for documents with obviously failed OCR/reconstruction.

### 252. Sub-Region Localized Hashing for Fine-Grained Image Retrieval (IEEE Trans. Image Processing, 2022)
**Authors:** Xiang, Zhang, Jin, Li, Tang (Nanjing University of Science and Technology)
**Link:** https://ieeexplore.ieee.org/document/9638382/
**Summary:** Addresses the challenge of fine-grained image hashing by capturing discriminative local information. Proposes sRLH (sub-Region Localized Hashing) which learns hash codes containing diverse subtle local information. A sub-region localization module locates peaks of non-overlapping sub-regions in the feature map, guiding the network to attend to dispersive local regions. To mitigate intra-class variations, hash codes of the same class approach a common binary center via Gram-Schmidt orthogonalization. Demonstrates superiority on four fine-grained retrieval datasets. Code available on GitHub.
**Relevance:** The sub-region hashing concept maps directly to per-region OCR quality assessment. Instead of a single global hash for the whole document, sRLH-style sub-region hashes could independently hash text blocks, figures, and tables, then compare each region's hash to the reconstructed counterpart. This would enable localized quality scores ("header OCR is good, table OCR is bad") rather than just a single global score.

### 253. Robust Image Hashing for Content Identification through Contrastive Self-Supervised Learning (Neural Networks, 2022)
**Authors:** Fonseca-Bustos, Ramirez-Gutierrez, Feregrino-Uribe (INAOE, Mexico)
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0893608022003781
**Summary:** Proposes learning perceptual hash functions automatically via contrastive self-supervised learning, rather than hand-designing them. The model learns an invariant feature space from unlabeled data by solving a metric learning pretext task that enforces robust hashing properties. Key advantage: when a new manipulation type breaks the existing hash, the model can be retrained on augmented data incorporating that manipulation, rather than redesigning the hash from scratch. Achieves excellent robustness even against difficult transforms like horizontal flip and rotation, and maintains high discriminative power.
**Relevance:** This approach could be used to learn a document-specific perceptual hash. By training on pairs of (original document, OCR-reconstructed document) with contrastive objectives, we could learn a hash that is invariant to acceptable reconstruction differences (font changes, minor spacing) but sensitive to quality-indicating differences (missing text, wrong characters). This is essentially "learning the right invariances for OCR quality."

### 254. Image Alignment Based Perceptual Image Hash for Content Authentication (Signal Processing: Image Communication, 2019)
**Authors:** Various (University of Electronic Science and Technology of China)
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0923596518304909
**Summary:** Introduces an image alignment process into perceptual hashing to handle geometric distortions. SIFT feature points estimate an affine transformation matrix to correct geometric distortion before hashing. The hash is generated from hybrid perceptual features: global and local Zernike moments (rotation-invariant) combined with DCT-based statistical features. Can detect various image forgeries and localize tampered regions. Achieves broad-spectrum robustness including tolerance to JPEG compression, noise, filtering, and geometric distortions.
**Relevance:** Our reconstructed documents inevitably have positional differences from originals -- shifted text blocks, slightly different margins. The SIFT-based alignment step before hashing addresses exactly this problem. We could align original and reconstructed images before computing any hash or metric, improving all downstream comparisons. The Zernike moment features provide rotation invariance useful for slightly rotated text blocks.

### 255. Robust Hashing via Global and Local Invariant Features for Image Copy Detection (ACM TOMM, 2023)
**Authors:** Liang, Tang, Li, Yu, Zhang, Zhang (Guangxi Normal University)
**Link:** https://dl.acm.org/doi/10.1145/3600234
**Summary:** Addresses the fundamental challenge of balancing discrimination and robustness in image hashing. Global features are computed from gray-level co-occurrence moments extracted from a saliency map derived via the phase spectrum of quaternion Fourier transform. Local features are computed via Kernel PCA and vector distances, which maintain geometric relationships within the image. Features are encrypted for security, then converted to a compact hash via ordinal measures. Demonstrates superior classification for detecting image copies with multiple simultaneous distortions.
**Relevance:** The dual global+local feature approach is well-suited for documents. Global features (saliency-based) capture overall document layout quality, while local features (KPCA-based, geometry-preserving) capture per-region text quality. The ordinal hash generation (comparing feature magnitudes rather than absolute values) provides robustness to the absolute intensity differences between original and reconstructed images.

### 256. A Survey on Deep Hashing Methods (ACM Trans. Knowledge Discovery from Data, 2023)
**Authors:** Luo et al.
**Link:** https://dl.acm.org/doi/10.1145/3532624
**Summary:** Comprehensive survey of deep hashing methods for image retrieval, covering supervised (point-wise, pairwise, list-wise), unsupervised, and semi-supervised approaches. Traces the evolution from shallow hashing (hand-crafted features + hash functions) to deep hashing (end-to-end learned features + hash codes). Identifies key challenges: quantization error from continuous-to-binary conversion, imbalanced training data, and the gap between Hamming space structure and semantic structure. Reviews loss functions (contrastive, triplet, cross-entropy, quantization), architectures (CNN-based, attention-based), and training strategies (continuation methods like HashNet's tanh annealing).
**Relevance:** Provides the theoretical foundation for understanding what deep hashing can and cannot capture. The survey's analysis of quantization error is particularly relevant -- when converting image features to binary hash codes, information is inevitably lost. For our OCR quality use case, this quantization means hash-based metrics will always be less sensitive than continuous metrics (SSIM, LPIPS). Helps decide where hashing adds value (speed) vs. where continuous metrics are needed (fine-grained quality).

### 257. HybridHash: Hybrid Convolutional and Self-Attention Deep Hashing for Image Retrieval (arXiv, 2024)
**Authors:** Various
**Link:** https://arxiv.org/abs/2405.07524
**Summary:** Proposes combining CNN and Transformer architectures for deep hashing. The CNN branch captures local texture features while the self-attention (Transformer) branch captures global context and long-range dependencies. Hash codes are generated from the fused representation. Demonstrates superior performance on CIFAR-10, NUS-WIDE, and ImageNet benchmarks compared to CNN-only or Transformer-only deep hashing methods.
**Relevance:** The hybrid CNN+Transformer architecture aligns well with document comparison needs: CNN features capture local text rendering quality (character shapes, line quality) while Transformer attention captures global document structure (layout, reading order, section hierarchy). If we were to train a document-specific hash, this hybrid architecture would be a strong starting point, though training requires labeled data and weeks of effort.

### 258. Block Mean Value Based Image Perceptual Hashing (IIH-MSP, 2006)
**Authors:** Yang, B., Gu, F., Niu, X. (Harbin Institute of Technology)
**Link:** https://ieeexplore.ieee.org/abstract/document/4041692/
**Summary:** Foundational paper proposing block-based perceptual hashing where the image is divided into non-overlapping blocks, the mean value of each block is computed, and each block is classified as black (below global mean) or white (above global mean) to form the hash. Four normalized variants are proposed with different block sizes and overlap strategies. Overlapped blocking and rotation operations enhance robustness to geometric distortions. The resulting "BlockHash" algorithm is one of the simplest and fastest perceptual hashing methods, later implemented in the imagehash Python library and in OpenCV's img_hash module.
**Relevance:** BlockHash is the fastest perceptual hash available and has a natural connection to document comparison: each block corresponds to a spatial region, so comparing block hashes between original and reconstructed images gives a localized similarity map. For documents, blocks containing text will have very different mean values from blocks containing whitespace, making BlockHash surprisingly discriminative for document layout comparison. Extremely fast (<0.5ms) and trivially implementable.

### Synthesis: Perceptual Hashing for Document Image Comparison (2026-04-10)

**Key Finding 1: Perceptual hashing is 100-1000x faster than LPIPS and complements rather than replaces it.**

Speed hierarchy from the literature:
| Method | Speed per image | Precision | GPU Required |
|--------|----------------|-----------|--------------|
| dHash/aHash/BlockHash | <0.5ms | Low (coarse similarity) | No |
| pHash (DCT-64bit) | ~113ms | Medium | No |
| PDQ (DCT-256bit) | ~80ms | Medium-High | No |
| wHash (wavelet) | ~50-100ms | Medium | No |
| SSIM | Low ms | Medium-High (structural) | No |
| LPIPS (AlexNet) | 10-50ms | High (perceptual) | Recommended |
| LPIPS (VGG) | 50-200ms | Very High | Recommended |

The 100-1000x speed advantage of hashing over LPIPS makes it viable for: (a) pre-filtering obviously bad reconstructions before expensive metrics, (b) real-time quality monitoring during batch OCR processing, (c) quality scoring on CPU-only environments.

**Key Finding 2: Spatially-aware hash distance metrics improve upon naive Hamming distance.**

The "Beyond Hamming Distance" paper (#245) shows that positional information IS encoded in perceptual hashes but is discarded by standard Hamming distance. For document comparison, this means:
- Hatched Matrix Distance (1.5x slower than Hamming) preserves spatial patterns
- 2-D Ngram Cosine Distance captures block-level spatial correlations
- These metrics could distinguish "text shifted to wrong position" from "text completely missing"

**Key Finding 3: Sub-region and block-level hashing enables per-region quality scores.**

Sub-Region Localized Hashing (#252) and Block Mean Value hashing (#258) both enable localized comparison. For our pipeline:
1. Divide both original and reconstructed images into blocks (matching document layout regions)
2. Compute per-block hashes
3. Compare per-block hash distances
4. Aggregate into regional quality scores: header quality, body text quality, table quality, figure quality

**Key Finding 4: Learned/self-supervised hashes can capture document-specific invariances.**

Contrastive self-supervised hashing (#253) and DINOHash (#40) show that learned hashes outperform classical methods. For documents:
- Train on (original, reconstruction) pairs with OCR quality labels
- The hash learns to be invariant to acceptable differences (font changes, minor spacing)
- The hash learns to be sensitive to quality-indicating differences (missing text, garbled characters)
- Medium-term investment (weeks of training) but would yield a purpose-built metric

**Key Finding 5: The optimal strategy is a cascaded hash pipeline.**

Based on Jakhar & Borah (#251) and the speed data above:
1. **Fast pre-filter (dHash/BlockHash):** <0.5ms, eliminate obviously failed reconstructions (hash distance > 30/64 bits)
2. **Medium filter (pHash/PDQ):** ~100ms, score remaining images for coarse quality
3. **Fine-grained scoring (SSIM/LPIPS):** only on images that pass hash pre-filter, saving 50-80% of GPU compute
4. **Optional: learned document hash:** trained on our data, replaces steps 1-2 with a single purpose-built metric

**Key Finding 6: wHash (wavelet-based) may be best suited for documents.**

The Hamming distributions study (#247) found wHash matches or exceeds pHash for most transforms. Wavelet decomposition naturally captures multi-scale edge and texture features -- the dominant visual features in document images (text edges, line boundaries). The Haar wavelet default in wHash is particularly good at capturing horizontal and vertical edges, exactly what text lines produce. This makes wHash theoretically better suited for document comparison than DCT-based pHash.

**Recommended Implementation Path:**

| Phase | Action | Effort | Impact |
|-------|--------|--------|--------|
| 1 (Immediate) | Add BlockHash + pHash + wHash as metrics via imagehash library | 1 day | Fast quality pre-screening, CPU-only option |
| 2 (Short-term) | Implement PHASER framework for systematic metric evaluation | 2-3 days | Rigorous comparison of all hash variants on our data |
| 3 (Short-term) | Add PDQ via pdqhash pip package (256-bit for finer granularity) | 0.5 day | Better sensitivity than 64-bit hashes |
| 4 (Medium-term) | Implement spatially-aware distance metrics from #245 | 1-2 days | Better localized quality assessment |
| 5 (Medium-term) | Train contrastive self-supervised document hash (#253 approach) | 2-3 weeks | Purpose-built document quality hash |
| 6 (Long-term) | Sub-region hashing with document layout detection (#252) | 2-4 weeks | Per-region quality scores |


## Fourier Descriptors for Shape Matching and Character Recognition (2026-04-10)

Research into Fourier-based shape descriptors (contour-based and region-based), their application to character/glyph recognition, and their potential as a reference-free OCR quality signal by comparing character contour Fourier signatures between original and reconstructed document images. Covers the mathematical foundations (Zahn-Roskies descriptors, Granlund descriptors, Elliptic Fourier Descriptors), character recognition applications, perceptual validation, and practical implementation via `pyefd` and OpenCV.

### 228. Shape Discrimination Using Fourier Descriptors (Persoon & Fu, IEEE SMC 1977)
**Title**: Shape Discrimination Using Fourier Descriptors
**Authors**: Eric Persoon, King-Sun Fu
**Venue**: IEEE Transactions on Systems, Man and Cybernetics, Vol. 7(3), pp. 170-179, 1977
**Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/4309681/) | [PDF](https://www.math.ucdavis.edu/~saito/data/morphometrics/persoon-fu_fourier-descr-shape-discr.pdf)
**Summary**: Foundational paper on using Fourier descriptors (FDs) for shape discrimination. Presents a critical review of two types of FDs and proposes a distance measure between boundary curves expressed in terms of Fourier coefficients. Shows how FDs can extract skeletons of objects and demonstrates experimental results on character recognition and machine parts recognition. The distance metric between two shapes is computed as the L2 norm between their FD vectors, providing a direct similarity score.
**Relevance**: Directly applicable to our OCR quality metric. The FD distance measure between original and reconstructed character contours could serve as a per-character quality signal. Characters that OCR missed or misidentified would have significantly different FD vectors. The character recognition experiments validate this approach for glyph comparison.

### 236. Fourier Descriptors for Broken Shapes (Dalitz et al., EURASIP JASP 2013)
**Title**: Fourier Descriptors for Broken Shapes
**Authors**: Christoph Dalitz, Christian Brandt, Steffen Goebbels, David Kolanus
**Venue**: EURASIP Journal on Advances in Signal Processing, 2013, Article 161 (Open Access)
**Link**: [Springer](https://link.springer.com/article/10.1186/1687-6180-2013-161)
**Summary**: Extends Fourier descriptors to handle broken shapes (shapes with multiple disconnected contour fragments). The method uses the convex hull of the shape and measures distance to the closest actual contour point. Tested on a new dataset of broken glyphs from 19th-century music prints (Byzantine neumes), where print degradation causes characters to fragment into up to 8 pieces. Recognition rates are comparable to standard FDs while additionally handling broken contours. Different normalization schemes for scale/rotation invariance are evaluated. Open-source toolkit released for the Gamera framework.
**Relevance**: Highly relevant for our OCR quality assessment. Document images often have degraded or broken characters (low DPI, compression artifacts, aging). The convex-hull-based approach for handling broken contours means our metric could remain robust when character shapes are fragmented. The normalization schemes for invariance under scale and rotation are directly useful for comparing characters rendered at different sizes between original and reconstruction.

### 238. High Accuracy Character Recognition Using Fourier and Topological Descriptors (Shridhar & Badreldin, PR 1984)
**Title**: High Accuracy Character Recognition Algorithm Using Fourier and Topological Descriptors
**Authors**: M. Shridhar, A. Badreldin
**Venue**: Pattern Recognition, Vol. 17(5), pp. 515-524, 1984
**Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0031320384900499)
**Summary**: Presents a two-stage character recognition algorithm achieving over 98% accuracy. The first stage performs primary classification using Fourier descriptors derived from Fourier analysis of digitized character boundaries. The second stage uses topological descriptors for fine discrimination between confusable character classes. The combination of FD-based coarse classification with topological refinement demonstrates that FDs alone capture the bulk of character shape information but need supplementary features for near-identical characters (e.g., "O" vs "0").
**Relevance**: Validates Fourier descriptors as strong features for character shape discrimination. The two-stage approach suggests that for our OCR quality metric, FD comparison would catch most structural character errors (wrong character, missing character), while topological features could detect subtler errors (e.g., "rn" vs "m"). The 98%+ accuracy indicates FDs are discriminative enough for character-level quality signals.

### 259. Fourier Descriptors for Plane Closed Curves (Zahn & Roskies, IEEE TC 1972)
**Title**: Fourier Descriptors for Plane Closed Curves
**Authors**: Charles T. Zahn, Ralph Z. Roskies
**Venue**: IEEE Transactions on Computers, Vol. C-21, pp. 269-281, 1972
**Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/5008949/) | [PDF (JHU)](https://www.cs.jhu.edu/~misha/Papers/Zahn72.pdf)
**Summary**: Seminal paper (2000+ citations) developing the analysis and synthesis of closed curves using Fourier descriptors based on Cosgriff's method. Represents a curve parametrically as accumulated change in direction as a function of arc length, then expands this function in a Fourier series. Key contributions: (1) proves that Fourier amplitudes are pure form invariants (invariant to translation, rotation, scale, starting point); (2) shows rotational and axial symmetry correspond to simple FD properties; (3) establishes that the Fourier series expansion is optimal and unique for obtaining coefficients insensitive to starting point. The "ZR descriptors" (Zahn-Roskies) became one of two foundational FD formulations alongside Granlund's.
**Relevance**: The theoretical foundation for our proposed approach. The cumulative angular bend representation is particularly suitable for character contours since it captures the turning pattern of strokes. The proven invariance properties mean we can compare characters regardless of their position, size, or orientation in the image. For our pipeline: extract contours from both original and reconstructed images, compute ZR descriptors, compare amplitudes.

### 260. Fourier Preprocessing for Handprint Character Recognition (Granlund, IEEE TC 1972)
**Title**: Fourier Preprocessing for Hand Print Character Recognition
**Authors**: G.H. Granlund
**Venue**: IEEE Transactions on Computers, Vol. 21(2), pp. 195-201, 1972
**Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/5008926/)
**Summary**: Describes a pattern recognition method using Fourier transformations to extract shape-invariant features for character recognition. From ordinary Fourier coefficients (which contain size, rotation, and phase factors), derives "property constants" that separate genuine shape constants from size, location, and orientation parameters. Tested on 175 samples of handprinted letters (7 sets of A-Z), achieving 98% correct recognition with a simple non-optimized decision method. The "Granlund descriptors" (G descriptors) became one of two canonical FD formulations.
**Relevance**: Demonstrates that Fourier-derived shape constants achieve high accuracy for character recognition even with simple classifiers. The property constants (separating intrinsic shape from extrinsic pose) are exactly what we need: compare the shape of a character in the original vs reconstruction independent of where it appears. 98% accuracy on handprinted letters with a simple classifier validates that FDs carry rich character-discriminative information.

### 261. Elliptic Fourier Features of a Closed Contour (Kuhl & Giardina, CGIP 1982)
**Title**: Elliptic Fourier Features of a Closed Contour
**Authors**: Frank P. Kuhl, Charles R. Giardina
**Venue**: Computer Graphics and Image Processing, Vol. 18(3), pp. 236-258, 1982
**Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0146664X8290034X) | [PDF (Utah)](https://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Kuhl-Giardina-CGIP1982.pdf)
**Summary**: Foundational paper (2500+ citations) on Elliptic Fourier Descriptors (EFDs). Presents a direct procedure for obtaining Fourier coefficients from chain-encoded contours without requiring integration or FFT. Each harmonic n yields four coefficients (a_n, b_n, c_n, d_n) that define an ellipse; the contour is approximated as the sum of these rotating ellipses (epicycles). Shows that elliptic properties of the coefficients enable intuitive normalization invariant to translation, rotation, and scale. The number of harmonics controls the level of detail: few harmonics capture coarse shape, more harmonics add fine detail. This is the paper that established EFDs as the standard for contour-based morphometric analysis.
**Relevance**: This is the direct mathematical basis for our proposed "Fourier epicycle comparison" metric. Each character contour can be decomposed into a series of rotating ellipses (epicycles). Comparing the (a_n, b_n, c_n, d_n) coefficients between original and reconstructed characters gives a multi-scale shape similarity measure: low harmonics detect gross shape errors (wrong character), high harmonics detect subtle deformations (font substitution, rendering artifacts). Python implementation available via `pyefd` library with direct OpenCV integration.

### 262. Review of Shape Representation and Description Techniques (Zhang & Lu, PR 2004)
**Title**: Review of Shape Representation and Description Techniques
**Authors**: Dengsheng Zhang, Guojun Lu
**Venue**: Pattern Recognition, Vol. 37(1), pp. 1-19, 2004
**Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320303002759) | [PDF (Temple)](https://cis.temple.edu/~latecki/Courses/CIS601-04/ProjectPapers/shapeRepPR04.pdf)
**Summary**: Comprehensive survey classifying shape description techniques into contour-based and region-based methods. For Fourier descriptors, identifies four key advantages: (1) simple to compute; (2) each descriptor has specific physical meaning; (3) simple normalization makes matching straightforward; (4) captures both global and local features depending on number of coefficients. Evaluates methods against MPEG-7 criteria: retrieval accuracy, compactness, generality, computational complexity, robustness, and hierarchical representation. Concludes that FDs and Zernike moments are among the strongest shape descriptors overall.
**Relevance**: Authoritative reference for understanding the landscape of shape descriptors. Confirms FDs as strong candidates for our application. The MPEG-7 evaluation criteria (compactness, robustness, hierarchical representation) align well with our needs: we want a compact per-character quality signal that is robust to minor rendering differences and can operate at multiple scales (character-level, word-level, region-level). The survey also identifies alternatives (Zernike moments, CSS descriptors) we could consider.

### 263. Generic Fourier Descriptor for Shape-Based Image Retrieval (Zhang & Lu, SPI:C 2002)
**Title**: Generic Fourier Descriptor for Shape-Based Image Retrieval
**Authors**: Dengsheng Zhang, Guojun Lu
**Venue**: Signal Processing: Image Communication, Vol. 17(10), pp. 825-848, 2002
**Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S092359650200084X)
**Summary**: Proposes the Generic Fourier Descriptor (GFD), derived from 2D Fourier transform on polar-raster sampled shape images. Unlike contour-based FDs that require boundary extraction, GFD works directly on shape regions, capturing both boundary and interior features. GFD outperforms Zernike Moment Descriptors (the MPEG-7 shape descriptor) on standard benchmarks. Satisfies all six MPEG-7 requirements: good retrieval accuracy, compact features, general application, low computation complexity, robust retrieval performance, and hierarchical coarse-to-fine representation. Translation, scale, and rotation invariant.
**Relevance**: GFD offers a region-based alternative to contour-based EFDs. For our OCR quality metric, GFD could be more robust than contour-based methods when character contours are difficult to extract cleanly (e.g., touching characters, complex backgrounds). The polar-raster sampling approach naturally handles the character comparison task. Could be used as a complementary signal alongside contour-based EFD comparison -- contour FDs for isolated characters, GFD for characters in complex layouts.

### 264. Classification of Digital Typefaces Using Spectral Signatures (Morris, PR 1992)
**Title**: Classification of Digital Typefaces Using Spectral Signatures
**Authors**: Robert A. Morris
**Venue**: Pattern Recognition, Vol. 25(8), pp. 869-876, 1992
**Link**: [PDF (TUG)](http://tug.tug.org/interviews/interview-files/morris/morris-classification-spectral.pdf)
**Summary**: Applies Fourier spectral analysis to automatic typeface identification. Mean Fourier amplitudes through a bank of bandpass filters (11 intervals on each axis) provide a feature vector for typeface classification. Using a piecewise quadratic classifier on 17-dimensional PCA features derived from 38 quadratic discriminant functions, achieves 96% correct classification across 9 fonts with 100 samples each. The spectral signature captures the frequency characteristics that distinguish typefaces (e.g., serif vs sans-serif detected in specific frequency bands).
**Relevance**: Directly demonstrates that Fourier spectral analysis can differentiate typefaces -- the reverse of our problem. If OCR reconstruction uses a different font than the original, the spectral signatures will differ, providing a font-sensitivity signal. This could detect font substitution errors in our reconstruction pipeline. The bandpass filter approach could be adapted: compute spectral signatures for image patches in original and reconstruction, compare to detect text rendering differences.

### 265. Perceptual Similarity of Shapes Generated from Fourier Descriptors (Cortese & Dyre, JEP:HPP 1996)
**Title**: Perceptual Similarity of Shapes Generated from Fourier Descriptors
**Authors**: James M. Cortese, Brian P. Dyre
**Venue**: Journal of Experimental Psychology: Human Perception and Performance, Vol. 22(1), pp. 133-143, 1996
**Link**: [PubMed](https://pubmed.ncbi.nlm.nih.gov/8742257/)
**Summary**: Investigates whether Fourier descriptors align with human perceptual shape similarity. Three experiments used multidimensional scaling on similarity judgments for shapes generated by Fourier synthesis. Results show that particular Fourier components (amplitude and phase of specific frequencies) independently predict perceptual similarity dimensions. Variations in amplitude of one frequency and phase of another produce independent perceptual effects. Concludes that a Fourier representation is consistent with human perceptual similarity for shapes, at least for relatively low-dimensional Fourier shapes.
**Relevance**: Validates that FD-based shape comparison aligns with human perception. This is critical for our OCR quality metric: if FD distance between original and reconstructed characters correlates with human perception of character similarity, then FD comparison is a perceptually meaningful quality signal. The finding that low-order Fourier components dominate perceptual similarity means we can use relatively few coefficients (reducing computation) while still capturing perceptually important shape differences.

### 266. Fourier Descriptors and Handwritten Digit Recognition (Lu et al., MVA 1993)
**Title**: Fourier Descriptors and Handwritten Digit Recognition
**Authors**: Yi Lu, Steven Schlosser, Michael Janeczko
**Venue**: Machine Vision and Applications, Vol. 6(1), pp. 25-34, 1993
**Link**: [Springer](https://link.springer.com/article/10.1007/BF01212429)
**Summary**: Comparative study of five distinct Fourier descriptor representations for handwritten digit recognition on 14,000 test digits. Discusses characteristics of each FD formulation and illustrates ambiguous digit classes inherent to each representation. Concludes that FDs alone are practically effective only within an intelligent system capable of reasoning about digit hypotheses; a hypothesis-generating algorithm based on FDs allows associating multiple digit classes with each input. Some FD formulations perform significantly better than others for handwritten digits.
**Relevance**: Important reality check for our approach. FDs alone may not perfectly discriminate all characters (especially handwritten or degraded ones), but they generate strong hypotheses. For our metric, we don't need perfect recognition -- we need a distance measure. Even if FDs can't distinguish "1" from "l" perfectly, the FD distance between a correctly rendered character and a completely wrong one will be large. The comparison of five FD formulations helps us choose the best variant.

### 267. Arabic Character Recognition Using Fourier Descriptors and Contour Encoding (Mahmoud, PR 1994)
**Title**: Arabic Character Recognition Using Fourier Descriptors and Character Contour Encoding
**Authors**: Sabri A. Mahmoud
**Venue**: Pattern Recognition, Vol. 27(6), pp. 815-824, 1994
**Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/003132039490166X)
**Summary**: Applies normalized Fourier descriptors (scale/translation/rotation invariant) combined with curvature features of character contours to Arabic character recognition. Uses a distance measure between feature vectors for classification: the model with minimum distance is the predicted class. Achieves 100% recognition on model classes, dropping to 98% in the post-recognition phase (identifying specific characters including dots and holes). Errors primarily come from corrupted data. The combination of FDs with curvature features and dot/hole detection is shown to be powerful for cursive script.
**Relevance**: Demonstrates FD-based distance measures work for complex scripts (Arabic, with its cursive connections and diacritical marks). For our multilingual OCR quality metric, this validates that FD comparison can handle non-Latin scripts. The distance-measure approach (L2 distance between FD vectors) directly maps to our use case: compute FD vectors for character patches in original and reconstruction, measure distance as quality signal.

### 268. Automated Classification of Cell Shapes: Comparative Evaluation of Shape Descriptors (arXiv 2024)
**Title**: Automated Classification of Cell Shapes: A Comparative Evaluation of Shape Descriptors
**Authors**: (Various, arXiv:2411.00561)
**Venue**: arXiv, November 2024
**Link**: [arXiv](https://arxiv.org/abs/2411.00561)
**Summary**: Comprehensive modern evaluation of shape descriptors for classifying noisy contours from cell segmentation. Compares Elliptical Fourier Descriptors, curvature features, PCA-based representations, wavelet descriptors, and scalar summary statistics. Key results: PCA on raw contours achieves 99.0% accuracy; Fourier descriptors at order 10 ("Fourier 10 raw") achieve 98.7%; wavelet raw reaches 98.9%. Stats-based summary descriptors underperform their raw counterparts, indicating that raw Fourier/wavelet representations retain more discriminative information. Provides a rigorous experimental framework on synthetic data with controlled noise levels.
**Relevance**: Most recent and rigorous benchmark of shape descriptors including EFDs. The finding that "Fourier 10 raw" (10 harmonics, unnormalized) achieves 98.7% accuracy on noisy contours is very encouraging for our application -- document characters are typically cleaner than biological cell contours. Confirms that approximately 10 Fourier harmonics are sufficient for high-accuracy shape discrimination. The noise robustness evaluation is directly relevant since our original images may have compression artifacts or scanning noise.

### 269. Fourier and Wavelet Descriptors for Shape Recognition Using Neural Networks (Osowski et al., PR 2002)
**Title**: Fourier and Wavelet Descriptors for Shape Recognition Using Neural Networks -- A Comparative Study
**Authors**: Stanislaw Osowski, Bartlomiej Stodolski, Tomasz Markiewicz
**Venue**: Pattern Recognition, Vol. 35(9), 2002
**Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320301001534)
**Summary**: Compares Fourier and wavelet-based shape descriptors as inputs to three neural network architectures (MLP, Kohonen SOM, and a hybrid SOM+MLP cascade) for 2D pattern recognition. Features are extracted from Fourier and wavelet transformations of shape boundaries. Different neural network structures combined with different preprocessing methods yield varying accuracy; the hybrid SOM+MLP with Fourier features achieves the best recognition rates. Demonstrates that Fourier features can be effectively combined with learned classifiers, and that the choice of descriptor representation significantly impacts downstream neural network performance.
**Relevance**: Shows that Fourier descriptors pair well with neural network classifiers -- a hybrid classical+learned approach. For our OCR metric, we could potentially learn a small neural network that takes FD vectors from original and reconstructed characters and outputs a quality score, rather than just computing raw Euclidean distance. The wavelet comparison is also useful: wavelet descriptors provide localized frequency information that FDs lack, suggesting we might want to use both.

---

### Summary: Fourier Descriptors for OCR Quality Assessment

The literature on Fourier descriptors for shape matching and character recognition spans 50+ years and provides a strong theoretical and empirical foundation for our proposed approach.

**Theoretical Foundation:**
- Zahn & Roskies (1972) and Granlund (1972) established that Fourier amplitudes are pure shape invariants (translation, rotation, scale, starting-point invariant)
- Kuhl & Giardina (1982) showed that closed contours decompose into rotating ellipses (epicycles), with each harmonic adding detail
- Cortese & Dyre (1996) proved that FD-based shape comparison aligns with human perceptual similarity

**Character Recognition Performance:**
- Granlund (1972): 98% on handprinted letters with simple classifier
- Shridhar & Badreldin (1984): 98%+ with FD + topological features
- Mahmoud (1994): 100% on model classes for Arabic characters
- Lu et al. (1993): FDs effective as hypothesis generators for handwritten digits

**Practical Considerations:**
- 10-20 Fourier harmonics sufficient for character-level discrimination (confirmed by 2024 cell shape study)
- EFDs via `pyefd` library integrate directly with OpenCV contour extraction
- Broken/degraded characters handled by Dalitz et al. (2013) convex-hull method
- GFD (Zhang & Lu 2002) provides a region-based alternative when contour extraction is noisy
- Morris (1992) shows spectral signatures can even distinguish typefaces

**Proposed Implementation Pipeline:**
1. Extract character contours from original and reconstructed images using OpenCV `findContours`
2. Compute EFD coefficients using `pyefd.elliptic_fourier_descriptors(contour, order=15, normalize=True)`
3. For each matched character pair, compute L2 distance between normalized EFD vectors
4. Aggregate per-character FD distances into a document-level quality score
5. Use Dalitz's broken-shape method for degraded characters
6. Consider GFD as fallback for characters where contour extraction fails

**Priority Ranking for Implementation:**

| # | Approach | Effort | Impact |
|---|---------|--------|--------|
| 1 | EFD character contour comparison via pyefd (#261) | Low | High -- per-character quality signal |
| 2 | Normalized FD distance measure (#228) | Low | High -- direct shape similarity metric |
| 3 | Broken-shape FD for degraded characters (#236) | Medium | Medium -- handles fragmented characters |
| 4 | GFD region-based fallback (#263) | Medium | Medium -- works when contours fail |
| 5 | Spectral typeface signature comparison (#264) | Medium | Medium -- detects font substitution |
| 6 | Neural network on FD features (#269) | High | High -- learned quality predictor |


## Font Family Detection for Document Reconstruction (2026-04-17)

Research into training-free, feature-based methods for identifying font families from grayscale document image crops using OpenCV and NumPy. Covers character aspect ratio analysis (condensed vs. expanded), monospace detection via connected component width variance, handwriting detection via stroke width analysis and baseline irregularity, display/decorative font identification, and serif detection refinements. All methods are heuristic and parameter-free — no training data required.

---

## Font Family Detection

### Why Font Categories Matter for Reconstruction Quality

Not all font distinctions improve rendered output equally. The most impactful categories for reconstruction fidelity are:

1. **Monospace vs. proportional** — affects letter-spacing in CSS/HTML rendering; wrong choice produces severe horizontal overflow or gaps
2. **Handwriting vs. printed** — dictates font family selection (cursive families vs. sans/serif families); also signals to avoid fixed-width reconstruction
3. **Serif vs. sans-serif** — affects baseline rendering style and perceived legibility match
4. **Condensed vs. expanded** — affects line wrapping and column width estimates in layout reconstruction
5. **Display/decorative** — signals large-size title text; incorrect assignment at body-text size produces illegible output

Categories that matter less for reconstruction (but more for aesthetics): old-style vs. transitional serif, exact weight within a family.

---

### 1. Aspect Ratio Analysis (Condensed vs. Expanded)

**Concept**: Character width-to-height ratio is the most direct proxy for font "stretch". Condensed fonts have narrow characters (ratio < 0.45); expanded fonts have wide characters (ratio > 0.75). Measuring mean aspect ratio across all connected components in a bbox crop gives a stable aggregate estimate.

**Algorithm**:
```python
import cv2
import numpy as np

def estimate_aspect_ratio(gray_crop: np.ndarray) -> float:
    """Returns mean width/height ratio of character blobs in a grayscale bbox crop."""
    _, binarized = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binarized, connectivity=8)
    
    ratios = []
    h_crop, w_crop = gray_crop.shape
    min_area = 0.001 * h_crop * w_crop  # ignore dust
    max_area = 0.8 * h_crop * w_crop    # ignore full-crop blobs
    
    for i in range(1, n_labels):  # skip background (label 0)
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area and h > 3:
            ratios.append(w / h)
    
    return float(np.median(ratios)) if ratios else 0.5

def classify_stretch(ratio: float) -> str:
    if ratio < 0.45:
        return "condensed"
    elif ratio > 0.75:
        return "expanded"
    else:
        return "normal"
```

**Thresholds** (from typography industry standards, confirmed by numberanalytics.com):
- Condensed: ratio 0.3–0.45
- Normal: ratio 0.45–0.75
- Expanded: ratio 0.75–1.0+

**Limitations**: 
- Punctuation and symbols skew the ratio — filter by aspect ratio outliers (drop components where h < 3px or w/h > 3)
- Works best on crops containing 5+ characters; unreliable on single-character crops
- Italicized text produces slightly lower ratios due to horizontal compression of rendered width
- Binarization quality affects connected component extraction at small font sizes

---

### 2. Monospace Detection (Connected Component Width Variance)

**Concept**: In monospace fonts, every character occupies an identical horizontal cell. Connected component widths extracted from a text line will have very low variance relative to their mean. The Coefficient of Variation (CV = std/mean) of component widths discriminates monospace from proportional fonts reliably.

**Algorithm**:
```python
def detect_monospace(gray_crop: np.ndarray) -> dict:
    """
    Returns dict with keys: is_monospace (bool), cv_width (float), verdict (str).
    CV of connected component widths < 0.15 → monospace.
    """
    _, binarized = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binarized, connectivity=8)
    
    h_crop, w_crop = gray_crop.shape
    min_area = 0.001 * h_crop * w_crop
    max_area = 0.5 * h_crop * w_crop
    
    widths = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if min_area < area < max_area and h > 3:
            widths.append(w)
    
    if len(widths) < 4:
        return {"is_monospace": False, "cv_width": None, "verdict": "insufficient_data"}
    
    widths = np.array(widths, dtype=float)
    cv = widths.std() / widths.mean() if widths.mean() > 0 else 1.0
    
    return {
        "is_monospace": cv < 0.15,
        "cv_width": round(float(cv), 3),
        "verdict": "monospace" if cv < 0.15 else ("likely_monospace" if cv < 0.22 else "proportional")
    }
```

**Thresholds**:
- CV < 0.15 → monospace (very strong signal)
- CV 0.15–0.22 → borderline (could be monospace with punctuation interference)
- CV > 0.22 → proportional

**Limitations**:
- Connected characters (touching letters) merge into one wide component, increasing apparent variance — morphological separation helps but risks splitting ligatures
- Word spacing gaps can corrupt the measurement if the crop contains inter-word spaces; prefer crops that are single words or single lines without large gaps
- Very short crops (< 4 characters) are unreliable

---

### 3. Handwriting Detection

**Concept**: Handwritten text differs from printed text along three independent dimensions measurable from a grayscale crop: (1) stroke width variation (SWT coefficient of variation), (2) baseline deviation (vertical position scatter of component centroids), and (3) component connectivity ratio (connected strokes produce fewer but wider blobs than equivalent printed text). Combining all three gives a robust signal.

**Stroke Width Transform (SWT) Approach** — from Epshtein et al. CVPR 2010 (#270):
```python
def compute_swt_cv(gray_crop: np.ndarray) -> float:
    """
    Approximates Stroke Width Transform coefficient of variation.
    High CV (> 0.5) → handwriting. Low CV (< 0.3) → printed text.
    Uses distance transform on binarized image as SWT proxy.
    """
    _, binarized = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Distance transform gives approximate local stroke half-width at each foreground pixel
    dist = cv2.distanceTransform(binarized, cv2.DIST_L2, 5)
    stroke_pixels = dist[binarized > 0]
    if len(stroke_pixels) < 50:
        return 0.0
    # Stroke width ~ 2 * distance-transform value at each pixel
    stroke_widths = stroke_pixels * 2
    cv = stroke_widths.std() / stroke_widths.mean() if stroke_widths.mean() > 0 else 0.0
    return float(cv)

def compute_baseline_deviation(gray_crop: np.ndarray) -> float:
    """
    Measures vertical scatter of character centroids relative to crop height.
    High deviation (> 0.08) → handwriting baseline irregularity.
    """
    _, binarized = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binarized, connectivity=8)
    
    h_crop, w_crop = gray_crop.shape
    min_area = 0.001 * h_crop * w_crop
    max_area = 0.5 * h_crop * w_crop
    
    cy_values = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            cy_values.append(centroids[i, 1] / h_crop)  # normalized centroid y
    
    if len(cy_values) < 4:
        return 0.0
    return float(np.std(cy_values))

def detect_handwriting(gray_crop: np.ndarray) -> dict:
    swt_cv = compute_swt_cv(gray_crop)
    baseline_dev = compute_baseline_deviation(gray_crop)
    
    # Weighted score — both features contribute
    handwriting_score = 0.6 * min(swt_cv / 0.6, 1.0) + 0.4 * min(baseline_dev / 0.12, 1.0)
    is_handwriting = handwriting_score > 0.5
    
    return {
        "is_handwriting": is_handwriting,
        "handwriting_score": round(handwriting_score, 3),
        "swt_cv": round(swt_cv, 3),
        "baseline_deviation": round(baseline_dev, 3),
    }
```

**Thresholds** (derived from SWT literature):
- SWT CV < 0.3 → printed text (consistent stroke width)
- SWT CV 0.3–0.5 → ambiguous (could be bold+regular mixture, or light handwriting)
- SWT CV > 0.5 → strong handwriting signal
- Baseline deviation (normalized) > 0.08 → irregular baseline

**Key insight from literature**: The SWT paper (Epshtein et al.) notes that in valid text regions "the variance of stroke widths within a component must not be too big" — specifically, pixels are grouped as text only if their SWT ratio lies between 0.33 and 3.0. Handwriting violates this constraint widely due to pen pressure variation.

**Limitations**:
- Very regular handwriting (calligraphy, print-style) may score low
- Italic printed fonts can produce slightly elevated SWT CV due to angled strokes; not usually enough to cross the handwriting threshold
- Low-resolution crops (< 20px height) produce unreliable distance transforms

---

### 4. Display/Decorative Font Detection

**Concept**: Display fonts have two distinguishing visual properties: (1) unusually high stroke width variation (high-contrast fonts like Bodoni have thick verticals and hairline horizontals), and (2) unusual character proportions (either very tall/thin or very wide). The stroke contrast ratio — max stroke width / min stroke width within a single glyph — is the cleanest discriminator.

**Algorithm**:
```python
def detect_display_font(gray_crop: np.ndarray) -> dict:
    """
    Display/decorative fonts show high intra-image stroke contrast.
    Stroke contrast ratio > 4.0 → high-contrast display font (Bodoni/Didot style).
    Extreme aspect ratios also signal display use.
    """
    _, binarized = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binarized, cv2.DIST_L2, 5)
    stroke_pixels = dist[binarized > 0]
    
    if len(stroke_pixels) < 50:
        return {"is_display": False, "stroke_contrast_ratio": None}
    
    stroke_widths = stroke_pixels * 2
    p5 = np.percentile(stroke_widths, 5)    # thin strokes (hairlines)
    p95 = np.percentile(stroke_widths, 95)  # thick strokes
    
    contrast_ratio = (p95 / p5) if p5 > 0.5 else float('inf')
    
    # Also check aspect ratio for "display" proportions
    aspect = estimate_aspect_ratio(gray_crop)
    unusual_proportions = aspect < 0.25 or aspect > 1.2  # extremely condensed or wide
    
    is_display = contrast_ratio > 4.0 or unusual_proportions
    
    return {
        "is_display": is_display,
        "stroke_contrast_ratio": round(float(contrast_ratio), 2) if np.isfinite(contrast_ratio) else None,
        "unusual_proportions": unusual_proportions,
    }
```

**Thresholds**:
- Stroke contrast ratio > 4.0 → high-contrast display (Bodoni, Didot, Playfair style)
- Stroke contrast ratio 2.0–4.0 → moderate contrast (transitional serif: Times, Georgia)
- Stroke contrast ratio < 2.0 → low contrast (sans-serif, slab-serif, or monospace)
- Aspect ratio < 0.25 → ultra-condensed display
- Aspect ratio > 1.2 → ultra-wide display

**Typography grounding**: The "Typeface Classifications with More Contrasting Strokes" discussion notes that the contrast ratio in Didone fonts (Bodoni/Didot) is extreme — thick verticals can be 10x wider than hairline horizontals. Times New Roman sits around 2.5x. Helvetica is near 1.0.

**Limitations**:
- Very small font sizes (< 16px height) may not have enough pixels to resolve hairline strokes, compressing the apparent contrast ratio toward 1.0
- Drop caps and decorative initials produce outlier single-glyph crops — aspect ratio detection helps catch these
- Some display fonts have uniform stroke width but unusual letter shapes (stencil, outline fonts) — these are missed by SWT-based detection alone

---

### 5. Serif Detection Refinements

**Current approach (CV ≥ 0.35)**: Uses coefficient of variation of stroke widths as a serif proxy. This conflates serif with high stroke contrast (Bodoni) and misses slab serifs (uniform stroke, rectangular serifs like Rockwell).

**Better approach — horizontal projection profile serif feet detection**:

```python
def detect_serif_feet(gray_crop: np.ndarray) -> dict:
    """
    Detects serif feet via horizontal projection profile analysis.
    Serif fonts show characteristic horizontal ink spread at baseline and cap-line.
    
    Method:
    1. Binarize and compute row-wise pixel counts (horizontal projection)
    2. Find baseline and cap-line rows (peaks in lower/upper thirds)
    3. Measure ink spread width at those rows vs. midline rows
    4. High spread ratio → serifs present
    """
    _, binarized = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = binarized.shape
    # Row-wise ink counts
    row_counts = np.sum(binarized > 0, axis=1).astype(float)
    
    if row_counts.sum() < 50:
        return {"has_serif": False, "serif_spread_ratio": None}
    
    # Thirds of image height
    top_third = row_counts[:h//3]
    mid_third = row_counts[h//3: 2*h//3]
    bot_third = row_counts[2*h//3:]
    
    # Peak ink density at top/bottom vs middle
    top_peak = np.percentile(top_third, 90) if len(top_third) > 0 else 0
    bot_peak = np.percentile(bot_third, 90) if len(bot_third) > 0 else 0
    mid_mean = np.mean(mid_third) if len(mid_third) > 0 else 1
    
    # Spread ratio: how much wider is ink at baseline/cap-line than at midline
    spread_ratio = (top_peak + bot_peak) / (2 * mid_mean + 1e-6)
    
    # Supplementary: stroke width CV as before
    dist = cv2.distanceTransform(binarized, cv2.DIST_L2, 5)
    stroke_pixels = dist[binarized > 0]
    swt_cv = float(stroke_pixels.std() / stroke_pixels.mean()) if len(stroke_pixels) > 50 and stroke_pixels.mean() > 0 else 0.0
    
    # Combined decision
    has_serif = spread_ratio > 1.35 or swt_cv > 0.35
    
    return {
        "has_serif": has_serif,
        "serif_spread_ratio": round(float(spread_ratio), 3),
        "swt_cv": round(swt_cv, 3),
    }
```

**Thresholds for refined serif detection**:
- Spread ratio > 1.35 → serifs detected (ink bulge at baseline/cap-line from serif feet)
- Spread ratio 1.1–1.35 → ambiguous (could be slab serif or bold sans)
- SWT CV > 0.35 → high stroke contrast (triangular serif, old-style, or transitional)
- SWT CV < 0.2 → low stroke contrast (slab serif or sans-serif)

**Improved serif subtype classification**:
```python
def classify_serif_subtype(spread_ratio: float, swt_cv: float) -> str:
    """Classify into serif subtype based on two features."""
    if swt_cv > 0.5 and spread_ratio > 1.4:
        return "triangular_serif"   # Bodoni, Didot — high contrast + spread
    elif swt_cv > 0.3 and spread_ratio > 1.2:
        return "transitional_serif" # Times, Georgia — moderate contrast + spread
    elif swt_cv < 0.2 and spread_ratio > 1.2:
        return "slab_serif"         # Rockwell, Courier — low contrast + spread
    elif swt_cv < 0.2 and spread_ratio < 1.15:
        return "sans_serif"         # Helvetica, Arial — low contrast + no spread
    else:
        return "linear_serif"       # Old style, ambiguous
```

**Why the current CV ≥ 0.35 threshold is not wrong but incomplete**:
- It correctly catches high-contrast serifs (triangular/transitional)
- It misses slab serifs entirely (they have CV < 0.2 due to uniform stroke width)
- The horizontal spread ratio catches slab serifs that CV misses
- Using both features reduces false-negative rate substantially

**Limitations**:
- Very short crops (single line, < 30px height) don't have enough vertical resolution to separate thirds reliably; need at least 2 lines of text or 40+ px height
- All-caps text (no descenders, no ascenders) has reduced serif foot visibility
- Dense leading (tight line spacing) can blur the baseline/cap-line peaks together

---

### 6. Complete Font Family Classifier

**Integration** — combining all detectors into a priority-ordered pipeline:

```python
def classify_font_family(gray_crop: np.ndarray) -> dict:
    """
    Returns a dict with font_family (str), confidence (str), and all sub-scores.
    Priority order: handwriting > monospace > display > serif subtype
    """
    hw = detect_handwriting(gray_crop)
    if hw["is_handwriting"]:
        return {"font_family": "handwriting", "confidence": "high", **hw}
    
    mono = detect_monospace(gray_crop)
    if mono["is_monospace"]:
        return {"font_family": "monospace", "confidence": "high", **mono}
    
    display = detect_display_font(gray_crop)
    aspect = estimate_aspect_ratio(gray_crop)
    
    serif = detect_serif_feet(gray_crop)
    stretch = classify_stretch(aspect)
    
    subtype = classify_serif_subtype(
        serif.get("serif_spread_ratio", 1.0) or 1.0,
        serif.get("swt_cv", 0.0) or 0.0
    )
    
    # Compose family label
    parts = []
    if display["is_display"]:
        parts.append("display")
    if stretch != "normal":
        parts.append(stretch)
    parts.append(subtype)
    
    font_family = "_".join(parts) if parts else subtype
    
    return {
        "font_family": font_family,
        "confidence": "medium",
        "aspect_ratio": round(aspect, 3),
        "stretch": stretch,
        "is_display": display["is_display"],
        "stroke_contrast_ratio": display.get("stroke_contrast_ratio"),
        "has_serif": serif["has_serif"],
        "serif_spread_ratio": serif.get("serif_spread_ratio"),
        "swt_cv": serif.get("swt_cv"),
    }
```

**Example outputs**:
- `"handwriting"` — for cursive or print-style handwritten crops
- `"monospace"` — for code blocks, terminal output
- `"display_triangular_serif"` — for Bodoni-style title text
- `"condensed_sans_serif"` — for condensed grotesque fonts
- `"slab_serif"` — for Rockwell, Courier (non-monospace)
- `"transitional_serif"` — for Times, Georgia

---

### Key Papers Supporting These Methods

### #270. Detecting Text in Natural Scenes with Stroke Width Transform (Epshtein et al., CVPR 2010)
**Title**: Detecting Text in Natural Scenes with Stroke Width Transform
**Authors**: Boris Epshtein, Eyal Ofek, Yonatan Wexler
**Venue**: IEEE CVPR 2010
**Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/5540041/) | [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/201020CVPR20TextDetection.pdf)
**Summary**: Introduces the Stroke Width Transform — a local, data-dependent image operator that computes stroke width for every pixel without multi-scale scanning. Groups pixels into connected components where SWT ratio ≤ 3.0. Discriminates text from non-text using: (a) SWT variance within components, (b) aspect ratio bounds, (c) diameter-to-median-SWT ratio. Achieves 79.04% word recall on ICDAR. Handles fancy fonts, perspective distortion, and blur.
**Relevance**: The SWT is the core primitive for stroke-width-based font analysis. Low CV of SWT values within a crop → printed text. High CV → handwriting or high-contrast display font. The paper's grouping criteria (SWT ratio bounds, variance thresholds) directly map to our font classification thresholds. The distance transform approximation (implemented above) gives a CPU-efficient SWT proxy without full ray-casting implementation.

### #271. Font Clustering and Classification in Document Images (ApOFIS System, 2000)
**Title**: Font Clustering and Classification in Document Images
**Venue**: ResearchGate, 2000
**Link**: [ResearchGate](https://www.researchgate.net/publication/228990634_Font_clustering_and_classification_in_document_images)
**Summary**: The ApOFIS system performs font identification *a priori* — without needing to recognize individual characters. Uses geometric properties of connected components (bounding box dimensions, height histograms) and heuristic thresholds for block classification. Separates dominant characters from title/section characters using character size ratio factors. Classifies document blocks using heuristic parameters over connected component statistics.
**Relevance**: Validates the connected-component-bounding-box approach to font classification without OCR or training data. The character size ratio factor is analogous to our aspect ratio CV approach. Confirms that heuristic geometry-based methods were the standard before deep learning and remain useful for training-free inference.

### #272. Distinction between Handwritten and Machine-Printed Text Based on Bag of Visual Words (Zagoris et al., PRImA 2013)
**Title**: Distinction between Handwritten and Machine-Printed Text
**Authors**: K. Zagoris, I. Pratikakis, A. Antonacopoulos, B. Gatos, N. Papamarkos
**Venue**: PRImA Research Lab, 2013
**Link**: [PDF](https://www.primaresearch.org/www/assets/papers/PR2013_Zagoris_BagOfVisualWords.pdf)
**Summary**: Uses Bag of Visual Words (BoVW) to distinguish handwritten from machine-printed text in noisy document images. Extracts connected components, merges them into word blocks, then extracts Gabor filter features and crossing features. Demonstrates that connected component spatial patterns and texture features are highly discriminative for the handwritten/printed distinction, even in degraded documents.
**Relevance**: Confirms that connected-component analysis is sufficient for the handwriting/printed distinction — no deep learning needed. The Gabor/crossing feature approach is more robust than SWT alone but adds complexity. For our DocumentAnalyzer, the SWT CV + baseline deviation combination (implemented above) captures the same discriminative signal with less code.

---

**Recommended Implementation Path for DocumentAnalyzer:**

| Priority | Method | Effort | Impact on Reconstruction |
|----------|--------|--------|--------------------------|
| 1 | Monospace detection (CC width CV) | 20 lines | Critical — fixes code block rendering |
| 2 | Handwriting detection (SWT CV + baseline dev) | 40 lines | High — avoids serif/monospace for handwritten regions |
| 3 | Aspect ratio / stretch classification | 15 lines | Medium — improves condensed heading reconstruction |
| 4 | Serif spread ratio (horizontal projection) | 30 lines | Medium — catches slab serifs missed by CV threshold |
| 5 | Display font stroke contrast | 20 lines | Low-medium — useful for title/header detection |

---

## Shift-Tolerant & Probabilistic Perceptual Metrics (2026-04-18)

### #273. ST-LPIPS: Shift-Tolerant Perceptual Similarity Metric (Ghildyal & Liu, ECCV 2022)
**Authors**: Abhijay Ghildyal, Feng Liu
**Venue**: ECCV 2022
**Link**: https://arxiv.org/abs/2207.13686 | https://github.com/abhijay9/ShiftTolerant-LPIPS
**Summary**: Investigates how small pixel-level shifts between images cause standard LPIPS to produce scores misaligned with human perception. Proposes ST-LPIPS by systematically studying anti-aliasing filtering, pooling, striding, padding, and skip connections in the backbone CNN, producing a metric that is robust to imperceptible spatial misalignments while retaining sensitivity to genuine perceptual differences. Achieves higher agreement with human 2AFC judgments than standard LPIPS on misaligned image pairs. Integrated into PyTorch IQA toolbox in 2023.
**Handles positional shifts**: Yes — core contribution, explicitly designed for this.
**Captures character-level errors**: Moderate — inherits LPIPS sensitivity but adds anti-aliasing; character-specific sensitivity depends on backbone features, which are not text-trained.
**Implementation complexity**: Low — drop-in replacement for LPIPS via pip package.
**Relevance**: **High and directly actionable.** This is exactly the metric profile we need: tolerant to the small positional shifts caused by line-wrapping or bbox misplacement in our reconstructed documents, yet still sensitive to genuine visual differences (missing/wrong characters should register as genuine perceptual differences). Drop-in replacement for LPIPS in our current pipeline. Likely the single highest-impact low-effort improvement available.

### #274. Attacking Perceptual Similarity Metrics (Ghildyal & Liu, TMLR 2023)
**Authors**: Abhijay Ghildyal, Feng Liu
**Venue**: Transactions on Machine Learning Research (TMLR), 2023 (Featured Certification)
**Link**: https://arxiv.org/abs/2305.02274
**Summary**: Follow-up to ST-LPIPS: demonstrates that all perceptual similarity metrics — including ST-LPIPS, LPIPS, SSIM, and DreamSim — are vulnerable to adversarial attacks that produce imperceptible perturbations causing large metric fluctuations. Conducts systematic analysis of adversarial robustness properties across metric families. Key finding: no existing metric is simultaneously shift-tolerant, adversarially robust, and aligned with human perception.
**Handles positional shifts**: Analyzes shift robustness systematically.
**Captures character-level errors**: Not targeted for character detection; this is an adversarial robustness analysis.
**Implementation complexity**: Low — primarily a research analysis paper.
**Relevance**: **Medium — important for understanding limitations.** For our OCR evaluation pipeline, adversarial attacks are not a concern (we control both images), but this paper confirms that ST-LPIPS is the current best-practice for shift-tolerant perceptual comparison. The finding that all metrics have fundamental robustness tradeoffs motivates our multi-metric ensemble approach.

### #275. LipSim: A Provably Robust Perceptual Similarity Metric (Ghazanfari et al., ICLR 2024)
**Authors**: Sara Ghazanfari, Alexandre Araujo, Prashanth Krishnamurthy, Farshad Khorrami, Siddharth Garg
**Venue**: ICLR 2024
**Link**: https://arxiv.org/abs/2310.18274 | https://github.com/SaraGhazanfari/lipsim
**Summary**: Addresses the adversarial vulnerability identified in #274 by proposing the first perceptual similarity metric with provable robustness guarantees. Uses a student-teacher distillation approach where a 1-Lipschitz neural network student learns to mimic DreamSim embeddings. The Lipschitz constraint provides certified robustness: for any perturbation within an ℓ₂ ball of radius ε, the metric change is bounded. Against L2-APGD attacks, DreamSim drops to 0.93% 2AFC accuracy while LipSim maintains 72.20%.
**Handles positional shifts**: Inherits DreamSim's tolerance (moderate); Lipschitz constraint adds robustness to arbitrary perturbations including shifts.
**Captures character-level errors**: Moderate — relies on DreamSim embeddings trained on human perceptual judgments for natural images.
**Implementation complexity**: Medium — requires 1-Lipschitz backbone, but code is available.
**Relevance**: **Medium.** While adversarial robustness is not critical for our use case, the Lipschitz property means LipSim produces more stable, calibrated similarity scores — useful for threshold-based quality gates in our pipeline. The same author group produced UniSim-Bench (#282), suggesting this line of work will continue.

### #276. SUSS: Structured Uncertainty Similarity Score (Seidler et al., Dec 2024)
**Authors**: Paula Seidler et al.
**Venue**: arXiv preprint (Dec 2024)
**Link**: https://arxiv.org/abs/2512.03701
**Summary**: Proposes a generative probabilistic perceptual metric that models each image via a set of perceptual components, each represented as a structured multivariate Normal distribution (via SUPN — Structured Uncertainty Prediction Networks). Similarity is measured as a weighted sum of log-Normal densities (Mahalanobis distance), learned from human perceptual datasets. Provides interpretable pixelwise maps highlighting only perceivable differences. Multi-scale structural (Y) and color (Cb, Cr) similarity are modeled separately. Competitive with LPIPS and SSIM as a training loss, with stable convex optimization.
**Handles positional shifts**: Moderate — the probabilistic model captures uncertainty around each pixel position, giving natural tolerance to minor shifts.
**Captures character-level errors**: Potentially high — the pixelwise Mahalanobis distance map would highlight individual mismatched characters as perceivable differences, unlike global pooling in LPIPS.
**Implementation complexity**: High — requires SUPN probabilistic network; no public implementation confirmed yet.
**Relevance**: **Medium-High.** The interpretable pixelwise map aligns well with our desire to localize OCR errors. The probabilistic framework is theoretically well-suited to document comparison where character-level differences should have high local salience. Worth monitoring for a public release.

### #277. SPIPS: Scene Perceived Image Perceptual Score (2025)
**Authors**: Not specified in search results
**Venue**: IEEE, April 2025 — arXiv:2504.17234
**Link**: https://arxiv.org/abs/2504.17234 | https://ieeexplore.ieee.org/document/11050661/
**Summary**: Combines traditional IQA (PSNR, SSIM, MS-SSIM), deep CNN perceptual features (low-level from early AlexNet/VGG layers, high-level from late layers), and a learned MLP scoring function. Key innovation: the traditional IQA module generates spatially-aware quality maps (same resolution as input) rather than scalar scores, enabling localized quality assessment. Shows superior correlation with human judgments over LPIPS and DISTS on CNN sub-datasets of BAPPS.
**Handles positional shifts**: Moderate — traditional IQA module is pixel-aligned, but deep feature module has pooling-induced tolerance.
**Captures character-level errors**: Moderate — the spatially-aware quality maps would highlight local differences including individual character mismatches.
**Implementation complexity**: Medium — requires pre-trained CNN + MLP fitting; trained on BAPPS.
**Relevance**: **Medium.** The spatially-aware quality map output format is useful for diagnostics in our pipeline (which bbox regions have lowest quality?). The combination of traditional and deep features is a sensible engineering approach, though not specifically designed for document images.

---

## Dense ViT / DINO Features for Patch-Level Similarity (2026-04-18)

### #278. DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., TMLR 2024)
**Authors**: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafransky, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. (Meta AI Research)
**Venue**: Transactions on Machine Learning Research (TMLR), published January 2024
**Link**: https://arxiv.org/abs/2304.07193 | https://github.com/facebookresearch/dinov2
**Summary**: Trains ViT models via self-supervised discriminative learning (DINO + iBOT objectives + curated data pipeline) to produce all-purpose visual features competitive with the best weakly-supervised models without any finetuning. Key property for our use case: patch-level tokens exhibit strong spatial coherence — semantically similar regions produce clustered, consistent patch embeddings. Dense cosine similarity between patch tokens of two images provides a natural local feature matching score. Layer selection matters: blocks 3-6 capture mid-level texture/structure; block 12 captures global semantics.
**Handles positional shifts**: High — patch-level cosine similarity is intrinsically shift-tolerant because the best-matching patch from image B can be found for each patch in image A (nearest-neighbor matching, not pixel-aligned comparison).
**Captures character-level errors**: High potential — patch size (stride 14px) is small enough to isolate individual characters in standard-resolution document scans (e.g., 14px patch covers roughly 1-3 characters at 150 DPI). Missing or wrong characters produce distinctly different patch features.
**Implementation complexity**: Very Low — pretrained models available on HuggingFace; cosine similarity computation is trivial. The DINO perceptual loss formula `1 − mean(cosine_sim(patches_A, patches_B))` is 5 lines of code.
**Relevance**: **Very High.** DINOv2 patch cosine similarity is likely our best candidate for a shift-tolerant, character-sensitive visual metric. Unlike LPIPS (ImageNet-trained, object-centric) or CLIP (contrastive, global), DINOv2 features are self-supervised on diverse images including text/documents, and the patch tokens provide spatial granularity. Critically: the patch cosine alignment formula is training-free, fast, and can be applied directly as `DINOSim(A, B) = 1 − mean_p(cosine_sim(f_p(A), f_p(B)))` where f_p extracts patch tokens. This directly replaces or supplements LPIPS with a more spatially-aware alternative.

### #279. DINOv2 Meets Text: Unified Framework for Image- and Pixel-Level Vision-Language Alignment (CVPR 2025)
**Authors**: Jose et al.
**Venue**: CVPR 2025
**Link**: https://arxiv.org/abs/2412.16334 | https://openaccess.thecvf.com/content/CVPR2025/papers/Jose_DINOv2_Meets_Text_A_Unified_Framework_for_Image-_and_Pixel-Level_CVPR_2025_paper.pdf
**Summary**: Extends DINOv2 with language alignment (dino.txt) by freezing the DINOv2 vision encoder and training a text encoder to align with it, using LiT-style training with improvements for dense tasks (concatenating CLS token with patch average). Achieves CLIP-like zero-shot classification performance while preserving DINOv2's dense patch-level spatial sensitivity. Both image-level and pixel-level alignment are handled within a single framework.
**Handles positional shifts**: High — inherits DINOv2's patch-level spatial tolerance, adds text-grounded spatial querying.
**Captures character-level errors**: High — the pixel-level alignment enables text-region-specific similarity queries, which could directly highlight individual character discrepancies.
**Implementation complexity**: Medium — requires DINOv2 + trained text encoder (dino.txt model, available via GitHub).
**Relevance**: **High.** For our pipeline, dino.txt enables a new query mode: embed the expected text as a query, compute dense cosine similarity with the document image patches, and measure how well the expected text localizes to the correct regions. A character that is wrong or missing would fail to activate the expected text-image correspondence. This bridges the gap between pure visual comparison and character-aware evaluation.

### #280. Perception Encoder: The Best Visual Embeddings Are Not at the Output of the Network (Meta, NeurIPS 2025)
**Authors**: Daniel Bolya et al. (Meta FAIR)
**Venue**: NeurIPS 2025 (Oral)
**Link**: https://arxiv.org/abs/2504.13181 | https://github.com/facebookresearch/perception_models
**Summary**: Discovers that intermediate layers of a CLIP-trained ViT encoder contain embeddings that outperform final-layer embeddings for dense spatial tasks — matching or exceeding DINOv2 on depth estimation and semantic segmentation despite being trained with a global contrastive loss. Introduces two alignment methods to extract these hidden embeddings: language alignment (PE-lang) and spatial alignment (PE-spatial). For semantic correspondence: extracts dense feature maps, bilinearly upsamples to image resolution, computes patch cosine similarity maps, finds best-matching location. Sets COCO detection SOTA (66.0 mAP) with PE-spatial.
**Handles positional shifts**: High — PE-spatial uses patch cosine similarity for dense correspondence, inherently shift-tolerant.
**Captures character-level errors**: High — the spatial alignment heads produce per-patch features that can be used to find fine-grained correspondences between image regions, including individual character glyphs.
**Implementation complexity**: Low-Medium — pretrained models open-sourced; dense feature extraction follows standard ViT patch extraction pattern.
**Relevance**: **High.** PE-spatial provides potentially the best dense patch features for document comparison: trained on internet-scale data (like CLIP) with spatial sensitivity (like DINOv2). The intermediate-layer insight suggests we should evaluate features from multiple layers rather than only the final pooled embedding. PE also provides a unified model covering document VQA (94.6 DocVQA), which means the same encoder could both understand document content and compute patch-level similarity.

### #281. DINOv3: High-Resolution Dense Visual Features (Meta, 2025)
**Authors**: Meta FAIR
**Venue**: arXiv 2025
**Link**: https://arxiv.org/abs/2508.10104 | https://github.com/facebookresearch/dinov3
**Summary**: Extends DINOv2 with Gram anchoring (a loss that stabilizes patch-wise feature similarity structure during long training, preventing patch features from collapsing toward the global CLS representation) and a high-resolution training stage. Produces cosine similarity maps that remain spatially sharp even at high resolution. Key fix: in long DINOv2 training runs, patch features become too similar to the CLS token (losing spatial specificity); Gram anchoring prevents this. DINOv3 features retain local specificity at scales relevant for character-level analysis.
**Handles positional shifts**: High — improved patch feature quality means more reliable local matching.
**Captures character-level errors**: High — high-resolution dense features are especially suitable for character-scale spatial discrimination.
**Implementation complexity**: Low — drop-in DINOv2 replacement; models available on GitHub.
**Relevance**: **High.** If using DINOv2 patch similarity (#278) in our pipeline, DINOv3 is a direct upgrade that specifically fixes the patch-feature-degradation problem in large models. For documents, where we need fine-grained patch-level discrimination at character scale, this matters. Check DINOv3 first before committing to DINOv2-ViT-L.

---

## Multi-Modal Perceptual Benchmarks & Patch-Level CLIP Metrics (2026-04-18)

### #282. UniSim-Bench: Unified Benchmark and Models for Multi-Modal Perceptual Metrics (Ghazanfari et al., CVPR Workshop 2025)
**Authors**: Sara Ghazanfari et al.
**Venue**: CVPR Workshop 2025
**Link**: https://arxiv.org/abs/2412.10594 | https://github.com/SaraGhazanfari/UniSim
**Summary**: First benchmark tracking perceptual similarity metrics across 7 multi-modal tasks (25 datasets total), including both 2AFC core tasks and out-of-distribution generalization tests. Key finding: general-purpose VLMs (CLIP, LLaVA-NeXT) perform reasonably on average but lag behind task-specific metrics on individual tasks. Proposed UniSim family: fine-tune CLIP and LLaVA-NeXT on multi-task perceptual data, achieving the highest average performance. Encoder-based VLMs show better generalization than generative models as perceptual metrics.
**Handles positional shifts**: Partial — 2AFC task set includes distortions but not systematically shift-augmented pairs.
**Captures character-level errors**: Not specifically targeted; benchmark covers natural image distortions.
**Implementation complexity**: Low — fine-tuned models available on GitHub.
**Relevance**: **Medium.** UniSim-Bench provides a principled framework for evaluating our metric ensemble against human perceptual judgments. The finding that encoder-based VLMs generalize better motivates our CLIP/DINOv2-based approaches over generative metrics (DiffSim). The same group produced LipSim (#275), suggesting a coherent research agenda around robust, generalizable perceptual metrics.

### #283. TokenCLIP: Token-wise Prompt Learning for Zero-shot Anomaly Detection via Optimal Transport (2025)
**Authors**: Not specified in search results
**Venue**: arXiv 2025
**Link**: https://arxiv.org/abs/2510.21171
**Summary**: Reformulates CLIP patch-token alignment as an optimal transport problem. Each visual patch token is transported to the most semantically relevant textual subspace; the transport constraints ensure coverage across subspaces and encourage focus on different semantics. Solves OT to get a transport plan that adaptively assigns each image patch to semantically relevant text tokens. Addresses the key limitation of existing methods: relying on a single global textual embedding to simultaneously align with all diverse visual patch tokens.
**Handles positional shifts**: High — OT-based matching finds the globally optimal assignment between image patches and text tokens, independent of fixed spatial layout.
**Captures character-level errors**: High — each image patch is independently matched to relevant text tokens; a patch containing a wrong character would fail to match its expected text token, increasing transport cost.
**Implementation complexity**: Medium — requires OT solver (e.g., POT library) in addition to CLIP backbone.
**Relevance**: **High.** TokenCLIP provides a principled framework for character-sensitive patch-to-text matching that is simultaneously shift-tolerant and fine-grained. For our pipeline: encode expected text tokens with CLIP text encoder, encode document image patches with CLIP vision encoder, compute OT transport plan cost as OCR quality score. Higher transport cost = worse match = lower OCR quality. This is the most theoretically grounded approach to our exact problem: character-level patch matching with positional flexibility.

### #284. A Survey of OCR Evaluation Methods and Metrics and the Invisibility of Historical Documents (2025)
**Authors**: Multiple authors
**Venue**: arXiv, 2025
**Link**: https://arxiv.org/abs/2603.25761
**Summary**: Comprehensive survey arguing that dominant OCR metrics (CER, WER) reduce document fidelity to string-edit distance, missing structural, spatial, and semantic dimensions. Reviews newer benchmarks: OCRBench v2 (11,500 photos, 31 contexts), OmniDocBench (1,355 annotated PDFs with degradation attributes), olmOCR-Bench (7,010 layout-fidelity unit tests). Argues that VLMs/transformers can now extract spatial relationships and semantic structure, but evaluation methods have not kept pace. Calls for visual-grounding evaluation approaches.
**Handles positional shifts**: N/A — survey paper.
**Captures character-level errors**: N/A — survey paper; discusses both CER-based and emerging approaches.
**Implementation complexity**: N/A — survey.
**Relevance**: **High — validates our research direction.** The survey explicitly identifies the gap that our project addresses: evaluation methods have not kept up with VLM-based OCR. Our render-and-compare visual approach is in the category the survey calls for. Useful for positioning our work in the literature and citing as motivation for pursuing visual metrics over CER/WER.

### #285. Spatially-Grounded Document Retrieval via Patch-to-Region Relevance Propagation (Dec 2024)
**Authors**: Not specified in search results
**Venue**: arXiv, December 2024
**Link**: https://arxiv.org/abs/2512.02660
**Summary**: Extends ColPali's late-interaction patch similarity framework with spatially-grounded patch-to-region relevance propagation. ColPali represents each document page as ~1,024 patch embeddings (32×32 grid); this paper adds mechanisms to propagate relevance scores from individual patches back to semantic document regions (paragraphs, tables, figures). Identifies that visual document retrieval effectiveness is significantly correlated with text coverage in the image, and that ColPali encourages abstract lexical matching rather than precise local matching.
**Handles positional shifts**: High — late-interaction patch similarity inherently handles spatial flexibility; the 32×32 patch grid covers document layout at paragraph/word granularity.
**Captures character-level errors**: Moderate — patch size in ColPali is coarse (~14×14 image patches over 448×448 input = ~14px per patch), likely too coarse for individual character discrimination, but good for word/phrase level.
**Implementation complexity**: Medium — builds on ColPali; requires spatial relevance propagation layer.
**Relevance**: **Medium-High.** The spatially-grounded extension makes patch similarity interpretable at the document region level — which regions of the document contribute to the similarity score? For our OCR quality metric, this would enable region-level quality reporting (e.g., "the table in the upper-right has low fidelity"). The finding about text coverage correlation is important: our reconstructed images will have text coverage proportional to OCR completeness, making patch similarity scores sensitive to missed text blocks.


## OCR-Error-Only Metric Research — Text-Content Extraction & No-Reference Document IQA (2026-04-25)

### #286. Char-SAM: Turning SAM into a Scene-Text Segmentation Annotator with Character-level Visual Prompts
**Summary:** Adapts SAM into a training-free, character-level scene-text segmentation annotator. Two modules — Character Bounding-box Refinement (CBR) splits coarse word boxes into per-character prompts, and Character Glyph Refinement (CGR) injects glyph-shape priors to fix SAM's over-/under-segmentation on thin strokes. Produces high-fidelity stroke-level masks on TextSeg, COCO-Text, and MLT17 with no fine-tuning. Code/weights public.
**Relevance:** Drop-in stroke-mask producer that does not need training data — exactly the kind of mask we need to multiply onto both rendered and OCR-reconstructed images so a metric only compares ink-on-ink pixels and ignores font softness, paper colour, and subpixel rendering differences.

### #287. EAFormer: Scene Text Segmentation with Edge-Aware Transformers (ECCV 2024)
**Summary:** Argues prior text segmenters bleed at glyph edges and proposes a text-edge extractor that filters non-text edges, an edge-guided encoder, and an MLP decoder for crisp pixel-level text masks. Re-annotates COCO_TS and MLT_S because old labels were too coarse. Outperforms TextFormer/TextSeg baselines on Total-Text, TextSeg, BTS. Public code and weights.
**Relevance:** Edge-tight masks matter when comparing two renderings of the same character — soft masks drag background pixels into the metric and dilute the OCR-fidelity signal. EAFormer's sharp edges make stroke-level masked similarity behave more like an "ink-only" metric.

### #288. CHSAM: Efficient Scene Text Segmentation via SAM with Convolutional Adapters and Hierarchical Decoding (ICDAR 2025)
**Summary:** Critiques Hi-SAM (#65) as expensive and dependent on heavy pixel-level supervision; plugs lightweight convolutional adapters into a frozen SAM backbone with a hierarchical decoder that predicts stroke / word / line / paragraph masks jointly. Reaches Hi-SAM-class fgIoU on Total-Text and TextSeg with a fraction of training cost.
**Relevance:** Cheaper alternative to Hi-SAM (#65) for the same job — generating stroke masks for masked LPIPS/SSIM. Useful if Hi-SAM inference latency becomes a bottleneck on the 4 RTX 6000 Ada when scoring many page pairs.

### #289. VQualA 2025 Document Image Quality Assessment Challenge (ICCV W 2025)
**Summary:** Workshop report documenting the DIQA-5000 challenge: 16 teams, 97 final submissions. Top entries combine large VLMs (LLaMA2-7B, Qwen2.5-VL-32B, SigLIP2-NaFlex) fine-tuned on document MOS data, often ensembled. Provides a thorough leaderboard of current state-of-the-art no-reference document IQA approaches with their architectures and tricks.
**Relevance:** A buffet of state-of-the-art no-reference document quality scorers we can pull off-the-shelf. The top-3 methods give us several concrete drop-in NR-IQA baselines to compare against our masked-similarity metric.

### #290. A Fair Evaluation of Various Deep Learning-based Document Image Binarization (Jan 2024)
**Summary:** Side-by-side benchmark of modern DL binarizers (DeepOtsu, DP-LinkNet, SauvolaNet, Robin, U-Net variants, transformer–CNN hybrids) under unified preprocessing and DIBCO evaluation. Identifies which architectures generalise across DIBCO 2009–2019 and which collapse on heavy degradation. Code references for each baseline.
**Relevance:** Picks the right text-foreground extractor for our use case. Binarization gives a coarser but cheaper foreground mask than Hi-SAM/EAFormer; this paper says which binarizer to actually use as the cheap path in a "Sauvola+CC then masked SSIM" pipeline.

### #291. TATSR: Scene Text Image Super-Resolution via Content Perceptual Loss and Criss-Cross Transformer Blocks
**Summary:** Introduces a Content Perceptual (CP) loss using multi-scale features from a pretrained text recognizer to supervise reconstruction of LR text. Unlike VGG-perceptual or CRAFT-perceptual, CP loss focuses on content semantics (which character) rather than appearance, paired with criss-cross transformer blocks that propagate horizontal/vertical text context. Demonstrates that recognition-feature distance is a typography-tolerant, character-fidelity-sensitive signal.
**Relevance:** Canonical "OCR-perceptual loss that ignores typography but penalises wrong glyphs" design. The idea — distance in recognition-network feature space — is directly usable as a reference-free metric between render and OCR-reconstruction.

### #292. Real-Time Text Detection with Similar Mask in Traffic, Industrial, and Natural Scenes (Nov 2024)
**Summary:** Diverges from DBNet's shrink-mask paradigm and proposes a "similar mask" representation that preserves boundary fidelity better while keeping inference real-time. Reports gains over DBNet++ (#66) on shape-irregular text and tighter character boundaries on industrial documents.
**Relevance:** A 2024-fresh, real-time detector that gives tighter foreground masks than DBNet++. For our pipeline this means cheaper, more accurate text-region masks per page when stroke-level detail is overkill — useful for region-level (not stroke-level) masked metrics.


## OCR-Error-Only Metric Research — Font / Typography Disentanglement (2026-04-25)

### #293. Total Disentanglement of Font Images into Style and Character Class Features (Pattern Recognition, 2024 — revised Sep 2025)
**Summary:** Proposes a network that nonlinearly and completely decomposes a font glyph image into a style code (which font) and a content code (which character class). Unlike most font-generation works that bake style/content into a generator, this paper is explicit that the disentangled features work for three downstream tasks: font recognition, character recognition, and one-shot font image generation — confirming each branch is task-pure. Validates orthogonality with classifier probes on each branch.
**Relevance:** Closest match to what we need: a publicly described content branch certified as character-identifying and font-suppressing. We can render GT and OCR text in any font, push both through the content branch, and use cosine distance as a font-invariant glyph-identity metric.

### #294. HyGDL: Disentangling Content from Style to Overcome Shortcut Learning (2025)
**Summary:** SSL framework with a single ViT/MAE encoder where style is analytically defined as the component of the representation orthogonal to a learned style-invariant content direction. Uses (1) self-distillation to learn the content direction, (2) closed-form vector projection to split feature into orthogonal content/style, (3) style-conditioned reconstruction as end-to-end supervision. Public code.
**Relevance:** Off-the-shelf recipe for taking any pretrained vision encoder (including a font classifier or DINOv2) and producing a content-only embedding via orthogonal projection — exactly the "font-classifier-as-feature-extractor with content-orthogonal projection" idea. Could be applied without retraining a font model.

### #295. DA-Font: Few-Shot Font Generation via Dual-Attention Hybrid Integration (Sep 2025)
**Summary:** Few-shot font model with a Dual-Attention Hybrid Module: component-attention blocks pull radical/component structure from the content image and spatial-attention blocks transfer fine style from references. Explicit two-branch encoder; outperforms VQ-Font, FsFont, FontDiffuser on Chinese few-shot benchmarks.
**Relevance:** Component-attention content branch is structure-focused (radicals, strokes), so it is plausibly font-invariant for CJK glyphs — useful given our document corpus contains Japanese. The content encoder is the candidate descriptor extractor.

### #296. EdgeFont: Multi-Scale Edge Self-Supervision for Few-Shot Font Generation (ESwA 2024)
**Summary:** Adds a Multi-Scale Edge Extraction module as auxiliary self-supervision so the content encoder learns shape-edges (a font-invariant skeleton-like signal) while the style encoder absorbs everything else. Reports improvements on PSNR/SSIM/LPIPS/FID over MX-Font, DG-Font, FsFont. Edge supervision is unsupervised (Sobel/Canny-style), no manual labeling.
**Relevance:** Edge maps are nearly font-style-invariant for the same character at the same scale; using EdgeFont's content encoder (or just its multi-scale edge representation) gives a cheap font-agnostic descriptor that fits our render-and-compare pipeline directly.

### #297. MSD-Font: Multi-Stage Diffusion Font Generation (CVPR 2024)
**Summary:** Multi-stage diffusion font generator that explicitly mimics the human font-design process: stage 1 transfers global skeleton/shape, stage 2 transfers stroke style. The first stage builds a skeleton-only intermediate which is by construction a font-invariant rendering of the target character. Released code via CVPR.
**Relevance:** The intermediate skeleton output is exactly the "render-glyph-without-style" canonical form we want for comparing GT vs. OCR strings in different fonts. We can extract just the stage-1 skeleton predictor as a content-only descriptor.

### #298. DG-Font: Deformable Generative Networks for Unsupervised Font Generation (CVPR 2021)
**Summary:** Seminal unsupervised font generator with a clean two-encoder split: a content encoder with deformable skip connections (FDSC) plus a VGG-11-based style encoder. Trained without paired data; explicitly preserves "domain-invariant" content. Public PyTorch implementation.
**Relevance:** Well-tested, lightweight baseline for extracting a content embedding. Smaller and faster than diffusion-based competitors, ideal for a first integration test.

### #299. MX-Font: Multi-Localized-Expert Few-Shot Font Generation (ICCV 2021)
**Summary:** Few-shot font generation with multiple localized experts; content-style adversarial loss + independence loss explicitly enforce content-style disentanglement. Each of K experts specializes in a different local concept, weakly supervised by component labels. Cross-lingual transfer demonstrated, suggesting the content branch generalizes beyond a fixed character set. Public code from Clova AI.
**Relevance:** Content-style adversarial loss is the cleanest formal training objective for what we want; resulting content features are explicitly shown to be cross-lingual, hinting at strong font-invariance. Encoder weights are public.

### #300. Diff-Font: Diffusion Model for Robust One-Shot Font Generation (IJCV 2024)
**Summary:** One-shot diffusion font generator that conditions on a predefined content embedding token (which character) plus a one-shot style reference. The content side is explicitly a discrete character ID embedding rather than image-derived, so by construction the content code is font-invariant. Robust to large style gaps.
**Relevance:** The image-conditioned style encoder, after subtracting the known content token, gives near-pure style — its complement (or the per-image content-prediction head) is a font-invariant content estimator.

### #301. Patch-Font: Few-Shot Font Generation with Patch-Based Attention and Multitask Encoding (Applied Sciences 2025)
**Summary:** Few-shot font generator combining patch-based attention with a multitask encoder that simultaneously learns content classification and style classification heads on the same backbone — explicitly factorizing the feature space.
**Relevance:** The multitask encoder's content-classification head produces a per-patch character-identity feature that is style-suppressed by training objective. Practical to extract because it is a single-pass CNN/ViT, not a generative model — aligns with our patch-based comparison machinery.

### #302. Simple Disentanglement of Style and Content in Visual Representations (ICML 2023)
**Summary:** Theoretical and practical framework showing that style/content can be disentangled from a frozen pretrained encoder using a simple linear probe + post-hoc orthogonalization, with no retraining. Provides identifiability guarantees under mild assumptions. Validated on style-transfer and OOD generalization benchmarks.
**Relevance:** Lets us turn any of the encoders we already use (DINOv2, font classifier, OCR encoder) into a content-only feature with a tiny labeled subset (rendered glyphs of known character × known font). Very low-effort baseline before training a full font-disentanglement model.


## OCR-Error-Only Metric Research — Shift-Tolerant & Deformation-Aware IQA (2026-04-25)

### #303. CrossScore: Multi-View Image Evaluation and Scoring (ECCV 2024)
**Summary:** Predicts a per-pixel SSIM-like score for a query image given a set of unregistered reference images, using DINOv2 features and a Transformer Decoder cross-attention head between query and references. Cross-attention picks the best-matching reference patch within the candidate pool; residual after attention drives the score. Trained self-supervised from NVS-distorted pairs. Generalizes to indoor/outdoor/360.
**Relevance:** Cross-attention naturally implements a "soft NN-search with learned receptive radius". For our case, render + OCR-render become the two views; the residual after attention is exactly a shift-tolerant similarity signal. Architecturally we already have DINOv2 patches (#278), so adding a cross-attention head is a small delta.

### #304. Active View Selector with Cross-Reference IQA (Jun 2025)
**Summary:** 2025 follow-on to CrossScore for active view selection in NeRF/3DGS. Refines the cross-reference IQA head to be faster and more accurate, including a windowed cross-attention that bounds where in the reference each query patch can attend (a learned spatial prior). Uses the resulting score for view-selection.
**Relevance:** Directly demonstrates the "windowed / radius-bounded" formulation we want — attention is masked to a local neighbourhood instead of the full image. The mask radius is a hyperparameter we can set to ~30-50 px, matching our font-metric slack constraint.

### #305. A-DISTS: Locally Adaptive Structure and Texture Similarity (TPAMI 2023)
**Summary:** Extension of DISTS (#87). Where DISTS uses global spatial averages of feature maps (giving texture invariance), A-DISTS adaptively localises the structure-vs-texture weighting per-region using a dispersion index, then computes spatial-mean similarity in texture-like regions and pointwise similarity in structure-like regions. The locally-pooled means in texture regions give explicit shift tolerance within the pooling window, while edges remain spatially precise.
**Relevance:** Pooling radius IS the displacement cap, expressed naturally. For text comparison, character interiors get treated as "texture" (shift-tolerant) and stroke edges/spaces as "structure" (precise) — and the dispersion index discovers this automatically. Cheap to integrate.

### #306. MiHo: Image Matching Filtering and Refinement by Planes and Beyond (Nov 2024)
**Summary:** Non-deep, modular method that takes sparse correspondences and (i) clusters them by local homography, (ii) reprojects each match into a "Middle Homography" virtual plane to redistribute geometric distortion symmetrically across the two patches before similarity comparison. Effectively a learned per-region warp-and-compare wrapper that absorbs deformation as zero cost up to the cluster's homography.
**Relevance:** Pattern transfers cleanly: cluster character bounding boxes by local affine motion, then compare per-cluster image content after warping. Implements "warp-then-compare" with explicit, interpretable per-region transforms — easier to set a displacement cap (reject clusters with translation > R px).

### #307. Computing Approximate Graph Edit Distance via Optimal Transport (Dec 2024)
**Summary:** Casts approximate GED as an optimal-transport problem on graph node/edge attributes, with an ensemble combining supervised and unsupervised OT-based estimators. Gives sub-quadratic computation with theoretical bounds tying OT cost to true GED.
**Relevance:** Direct fit for "graph/geometric matching for character grids". Detect chars with PP-OCR, build attributed graph (node = char-image-feature + position), compute OT-GED between query-graph and reference-graph. Spatial position becomes a node attribute, so a small position delta contributes small OT cost — tunable displacement penalty by scaling the position component of the cost matrix.

### #308. GEDAN: Learning the Edit Costs for Graph Edit Distance (Aug 2025)
**Summary:** Learns separate insertion and deletion costs for GED rather than fixing them, using a neural surrogate. Provides a more accurate, calibrated approximate GED.
**Relevance:** Pairs with #307: our character-grid GED needs principled costs (a missing character should cost more than a 5-px shift). GEDAN's framework lets us learn these from a small number of human-judged page pairs.

### #309. Sliced Optimal Transport Plans (Pivot Sliced Discrepancy, Aug 2025)
**Summary:** Addresses the limitation that classic Sliced Wasserstein produces no transport plan, hence no way to enforce structural constraints. Introduces a constrained Kantorovich formulation along slicing directions that yields a transport plan, and shows how to add capacity/locality constraints (incl. a bounded-displacement constraint) on top.
**Relevance:** This is the technical primitive we need to build "MS-SWD with a displacement cap". Where MS-SWD (#116) has no spatial constraint, Pivot Sliced Discrepancy gives us back a transport plan we can mask by a radius-R indicator over (x,y) coordinates appended to feature vectors. Effectively converts MS-SWD into a radius-bounded variant.

### #310. DeepDC: Deep Distance Correlation as a Perceptual Image Quality Evaluator (TIP 2024)
**Summary:** Proposes Deep Distance Correlation between feature distributions of reference and distorted images, computed without alignment between feature locations. Distance correlation is invariant to permutation of samples, so spatial reshuffling within feature maps does not change the score. Beats LPIPS/DISTS on geometric-distortion-heavy IQA splits.
**Relevance:** "Distance correlation between feature bags" is essentially a permutation-invariant (i.e., maximally shift-tolerant) similarity. By replacing the global bag with sliding-window local bags of radius R, we directly recover a radius-bounded distance correlation metric — another concrete route to displacement-capped comparison.


## OCR-Error-Only Metric Research — VLM-as-Judge & Decoupling Framings (2026-04-25)

### #311. DOCR-Inspector: Fine-Grained Automated Evaluation of Document Parsing with VLM (Dec 2025)
**Summary:** Proposes a VLM-as-judge specifically trained for reference-free document-parsing quality assessment. The model takes (document image, parsed output) and outputs fine-grained errors mapped to 28 predefined error types covering text, tables, and equations. Built on Qwen2.5-VL-7B, trained on a curated 200K-instance dataset (DOCRcase-200K) with a separate 882-case manually annotated benchmark (DOCRcaseBench). DOCR-Inspector-7B reportedly outperforms Gemini 2.5 Pro and leading open-source VLMs on the benchmark. Code and weights released.
**Relevance:** Closest existing analog to what we want — a VLM judge that produces a calibrated, content-focused quality score for OCR/parsing output without any ground truth, with explicit per-element error categorization. Drop-in candidate for our render-and-compare pipeline as a learned content-similarity scorer.

### #312. Trust but Verify: Programmatic VLM Evaluation in the Wild (PROVE, ICCV 2025)
**Summary:** Proposes a programmatic evaluation paradigm for free-form VLM outputs. Builds a high-fidelity scene graph from a detailed reference description, then auto-generates QA pairs and executable verification programs. Evaluation decomposes into helpfulness and truthfulness sub-scores via the scene-graph backbone, sidestepping the prompt-sensitivity and inflated-score problems of naive VLM judges. Benchmarks 10.5K visually grounded QA pairs.
**Relevance:** Methodological blueprint for programmatic (not free-form) VLM judging that is robust to phrasing artifacts — directly applicable to our "are characters identical, ignoring style" rubric. Replace the scene graph with element-level OCR text and run per-element verification programs; bypasses VLM-judge inflation.

### #313. Reading Between the Lines: Abstaining from VLM-Generated OCR Errors via Latent Representation Probes (Nov 2025)
**Summary:** Trains lightweight probes on VLM hidden states/attention to flag OCR uncertainty in STVQA. Three probe variants (cross-layer concat, attention aggregation over visual tokens, single-layer ensemble). Improves abstention accuracy +7.6% over baselines across 4 image+video benchmarks. Key finding: optimal uncertainty signal lives in intermediate layers, not the final layer.
**Relevance:** Reference-free OCR quality signal extracted directly from a VLM's internals — a complementary mechanism to a render-and-compare metric. Could be used to gate or weight VLM-judge scores when the judge itself is uncertain about character-level reads.

### #314. Towards End-to-end Document Parsing via Decoupled Rendering-from-Content (EMNLP-Findings 2025)
**Summary:** EMNLP-Findings 2025 paper on document parsing that explicitly decouples the rendering structure (layout, formatting tokens) from content extraction during both training and evaluation. Uses two evaluation streams: a content-fidelity score that strips formatting, and a structure-fidelity score that ignores text. Reports gains over coupled baselines on document parsing benchmarks.
**Relevance:** Second framing paper for "decouple OCR fidelity from rendering fidelity" (alongside #131 FD-RL), and from a venue (EMNLP) that the OCR community follows. Provides a worked example of running content-only and rendering-only sub-metrics in parallel — exactly the architecture we want.


## Decoration & Layout Parsing for Document Image Comparison (2026-04-25)

### #315. DocSAM: Unified Document Image Segmentation via Query Decomposition and Heterogeneous Mixed Learning (CVPR 2025)
**Summary:** Transformer-based unified framework that handles document layout analysis, multi-granularity text segmentation, and table-structure recognition as a combination of instance and semantic segmentation. Core novelty is a Sentence-BERT semantic-query head: category names from any dataset are embedded as queries that prompt the mask decoder to segment "the type of region I just named". This lets it be jointly trained on heterogeneous datasets (DocLayNet, M⁶Doc, historical, scene-text) and, at inference, accept *new* class names (e.g. "decoration", "icon", "ornamental banner") without retraining. Code and weights public at xhli-git/DocSAM.
**Relevance:** Single most directly usable tool for our textbook-banner problem — we can issue text prompts like "decorative shape", "icon", "logo", "coloured banner" and receive a pixel mask that we use to blank those regions before computing render-vs-OCR similarity.

### #316. HybriDLA: Hybrid Generation for Document Layout Analysis (Nov 2025)
**Summary:** Unifies diffusion and autoregressive decoding inside a single decoder layer. The diffusion half iteratively refines bounding-box hypotheses; the autoregressive half injects semantics and dynamically decides how many region queries to emit, which handles both sparse scientific pages and dense magazine/textbook layouts. Achieves 83.5 mAP, SOTA on DocLayNet and M⁶Doc.
**Relevance:** M⁶Doc's 74-class taxonomy includes ornamental elements (decorative frames, advertisements, QR codes, barcodes) that plain DocLayNet lacks. HybriDLA's M⁶Doc checkpoint is the closest off-the-shelf "decoration detector" with fine granularity.

### #317. OmniParser for Pure Vision Based GUI Agent (CVPR 2024)
**Summary:** Microsoft's pipeline training two dedicated heads: an interactable-icon detector (fine-tuned YOLOv8 on curated web-icon data) and an icon-description model. Produces structured JSON of every GUI element — icons, buttons, glyphs — with pixel boxes and semantic captions. Open weights and Hugging Face space.
**Relevance:** The icon detector transfers to textbook pages — small coloured glyphs (speech-bubble A/B/C icons, headphone/star task icons, page-number badges) look visually identical to GUI icons. Every detected icon box becomes a region to exclude from the OCR-quality comparison.

### #318. Marten: VQA with Mask Generation for Multi-modal Document Understanding (CVPR 2025)
**Summary:** Introduces VQAMask — a joint VQA + mask-generation training objective that forces a document MLLM to produce spatial masks for the visual text regions it references. Releases MTMask6M (6M image-mask pairs). Beats SOTA on DocVQA/InfoVQA/FUNSD by up to +10 points.
**Relevance:** Gives us the *complement* of what we want — a text-region mask. Pixels **outside** Marten's mask that are not plain white/background are decorative by construction (icons, banners, coloured shapes, page numbers). Directly usable as "1 − text_mask" decoration mask with no category engineering.

### #319. M⁶Doc: Large-Scale Multi-Format/Type/Layout/Language Modern Document Layout Analysis Dataset (CVPR 2023)
**Summary:** Canonical fine-grained benchmark used by 2025-2026 SOTA models. Defines 74 categories across 9,080 manually annotated pages (textbooks, magazines, newspapers, test papers, books). Categories include ornamental/decorative elements — backgrounds, advertisements, QR codes, barcodes, page numbers, ornamental bars — that PubLayNet/DocBank/DocLayNet explicitly ignore. Public at HCIILAB/M6Doc.
**Relevance:** Ground-truth dataset to (a) train/fine-tune a decoration detector for our pipeline, and (b) evaluate whether any pretrained model classifies the icon-class pixels we care about. Textbook subset matches OmniDocBench visually.

### #320. Manga109 Comprehensive Segmentation Annotations (CVPR 2025)
**Summary:** Re-annotates Manga109 with six pixel-level categories — frame, text/dialog, onomatopoeia, character body, character face, balloon — using SAM + LoRA with a correction-finetuning loop. Per-pixel masks distinguishing text from decorative/illustrative content on richly composed pages.
**Relevance:** Textbook pages with cartoon characters, speech bubbles, onomatopoeic decorations look structurally identical to manga pages. Their SAM+LoRA adapter can likely be zero-shot-applied or cheaply fine-tuned to flag "balloon", "character body", "onomatopoeia" regions on our OmniDocBench pages.

### #321. OmniParser V2: Structured-Points-of-Thought for Unified Visual Text Parsing (Feb 2025)
**Summary:** Alibaba's unified text spotting + KIE + table recognition + layout analysis under a single encoder-decoder with Structured-Points-of-Thought (SPOT) prompting — two stages that first emit text-center points plus structural tokens, then, conditioned on each point, predict polygons and content. SOTA on eight datasets across four tasks.
**Relevance:** SPOT stage-1 gives a clean set of "this pixel is semantic document text" points. Anything outside the polygons from stage 2 — but visually non-white — is decorative by elimination. Second-opinion cross-check against Marten's text mask (#318).


## Document Image Inpainting for Decoration Removal (2026-04-25)

### #322. LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions (WACV 2022)
**Summary:** Feed-forward inpainter built on Fast Fourier Convolutions (FFCs), giving every layer an image-wide receptive field — essential for completing periodic structures (text rows). Combines (1) FFC generator, (2) high-receptive-field perceptual loss via pretrained segmenter with dilated convolutions, (3) aggressive wide-mask training. Beats CoModGAN/MADF/RegionWise on Places/CelebA at 256→2048, generalizes to resolutions well beyond training. Public weights at advimman/lama.
**Relevance:** Workhorse for textbook-banner removal. FFC globality lets it hallucinate plausible white background and extend marginal text lines. Feed-forward and CPU-runnable at 2K.

### #323. MAT: Mask-Aware Transformer for Large Hole Image Inpainting (CVPR 2022)
**Summary:** Transformer-CNN hybrid for *large* holes at high resolution. Mask-aware transformer aggregates only from *valid* tokens with a dynamic mask that evolves through layers. Outperforms LaMa/CoModGAN on Places365/CelebA-HQ at 512×512 with 20–60% mask ratios.
**Relevance:** Transformer alternative worth benchmarking against LaMa when banners span >30% of page width.

### #324. ZITS++: Image Inpainting by Improving the Incremental Transformer on Structural Priors (TPAMI 2023)
**Summary:** Decomposes inpainting into structure + texture. Transformer Structure Restorer recovers holistic low-resolution structure (wireframes/edges); Fourier-CNN Texture Restorer fills texture. Tricks: zero-initialized residual addition for stable structural-prior injection; masking positional encoding for irregular large masks.
**Relevance:** Because ZITS explicitly reconstructs line/edge structure before texture, it preserves thin strokes better than pure pixel-space regressors. When a banner intersects a paragraph, structure priors prevent broken text lines from becoming blurry blobs.

### #325. BrushNet: Plug-and-Play Image Inpainting with Decomposed Dual-Branch Diffusion (ECCV 2024)
**Summary:** Dual-branch design — auxiliary branch (copy of pretrained UNet) takes masked image + mask at pixel level and injects hierarchical features into a frozen base UNet. Plug-and-play with any SD1.5/SDXL. A blending blurred-mask option guarantees unmasked region is pixel-identical to input. Beats SD-Inpaint/ControlNet-Inpaint/HD-Painter on 7 metrics.
**Relevance:** The pixel-exact background preservation is killer for our pipeline — text outside the banner mask must not drift. Good fit when LaMa produces visible seams on coloured backgrounds.

### #326. PowerPaint: Task-Prompted Inpainting with Removal Training (ECCV 2024)
**Summary:** Single SD-based inpainting model supporting four tasks — text-guided insertion, object *removal*, shape-guided insertion, outpainting — via learnable task prompts prepended to text conditioning. Removal task-prompt trained on paired data where target is clean background, forcing surround extension rather than hallucination.
**Relevance:** Unlike generic SD-Inpaint (which re-imagines decorative objects), PowerPaint's removal prompt is explicitly trained to produce empty backgrounds — exactly what we want on the banner region.

### #327. Inpaint Anything: Segment Anything Meets Image Inpainting (2023)
**Summary:** Mask-free pipeline: user clicks → SAM produces instance mask → mask dilated → LaMa/SD-Inpaint fills the hole. Three modes: Remove Anything, Fill Anything, Replace Anything. Production recipe combining SAM + off-the-shelf inpainters with mask-dilation tricks for large holes.
**Relevance:** Gives us the automated mask-generation half of the pipeline. SAM2-auto + colour/saliency filtering → LaMa is the "Remove" flow specialized to our domain.

### #328. MorphoMod: Blind Visible Watermark Removal with Morphological Dilation (Feb 2025)
**Summary:** Three-phase blind watermark remover: (1) learned segmenter produces watermark mask; (2) morphological dilation expands mask to cover soft boundaries / anti-aliasing halos; (3) inpaint + restore — inpainter fills dilated region, cleaned pixels composited back with input outside the mask so non-watermark pixels are untouched. Improves removal quality up to 50.8% over prior SOTA on CLWD/LOGO.
**Relevance:** Directly analogous to textbook-banner removal — the red banner is an opaque watermark with sharp edges and anti-aliased red halos. Dilation recipe translates 1-for-1: segment banner (colour threshold or SAM), dilate by N pixels, inpaint, composite back.

### #329. What Shape Is Optimal for Masks in Text Removal? (Nov 2025)
**Summary:** Studies how mask shape — not just the inpainter — affects text-removal quality on dense-text document images. Parameterizes mask profile as a flexible-shape family; uses Bayesian optimization to learn per-instance mask parameters. Releases a benchmark dataset of document-style dense-text images.
**Relevance:** Addresses the hardest subproblem in banner-removal: when the banner touches a line of text, where do we cut? Principled BO approach to choose optimal dilation/erosion per region.


## Foreground-Text-Only Masked Similarity Metrics (2026-04-25)

### #330. Alpha-CLIP: A CLIP Model Focusing on Wherever You Want (CVPR 2024)
**Summary:** Extends the CLIP image encoder with a 4th "alpha" input channel indicating a region of interest (binary or soft mask). Fine-tuned on millions of synthesized RGBA-region/text pairs so the CLIP [CLS] embedding represents the content of the masked region rather than the whole image, while preserving original CLIP zero-shot when alpha is all-ones. +4.1% ImageNet zero-shot over CLIP. Checkpoint public at SunzeY/AlphaCLIP.
**Relevance:** Cleanest answer to "CLIP-style similarity restricted to a mask". Feed the Hi-SAM / Char-SAM text-region mask as alpha and the cosine of the resulting embeddings IS the text-region-only CLIP similarity — no architectural work.

### #331. CLIPSeg: Image Segmentation Using Text and Image Prompts (CVPR 2022)
**Summary:** Lightweight transformer decoder on frozen CLIP that produces dense binary segmentation given either free-text ("text", "characters", "formula") or an image prompt. One model handles referring expression, zero-shot, and one-shot segmentation after training on extended PhraseCut. Small (~1M trainable params), ships in HuggingFace Transformers.
**Relevance:** Ready-made mask source cheaper than Hi-SAM. Text-prompt masks ("printed text", "mathematical symbols", "table grid") in a single forward pass on both render and OCR reconstruction, used as the weighting mask for any downstream masked SSIM / LPIPS / Alpha-CLIP.

### #332. CLIP Surgery: Training-Free CLIP Explainability (PR 2025 / arXiv 2023)
**Summary:** Training-free intervention fixing two CLIP pathologies: self-attention linking opposite regions and noisy background activations. Key tricks: "v-v" self-attention path + feature-surgery subtraction of category-invariant redundancies. No fine-tuning needed. Raises Cityscapes open-vocab mIoU +8.74%.
**Relevance:** Text-region saliency maps for free from vanilla CLIP with no training. Cheap soft-mask generator for "text pixels" to weight SSIM/LPIPS, and a sanity baseline before committing to Hi-SAM.

### #333. Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks (2024)
**Summary:** Chains Grounding DINO (open-vocab detector driven by free-form text) with SAM to produce text-prompted instance masks without task-specific training. 48.7 mAP on SegInW zero-shot with DINO-Base + SAM-Huge. Grounded-SAM-2 variant with SAM2 video tracking.
**Relevance:** Direct competitor/alternative to Hi-SAM for "text character" and "formula" masks from prompt strings. Zero training, supports SAM2, cleaner closed regions than Hi-SAM's hierarchical character masks — worth benchmarking as mask providers for the masked-metric pipeline.

### #334. Osprey: Pixel Understanding with Visual Instruction Tuning (CVPR 2024)
**Summary:** Extends an MLLM with a mask-aware visual extractor built on ConvNeXt-CLIP. Trained on 724K mask-region/text pairs. Ingests pixel-level SAM masks as region tokens; produces captions, classifications, descriptions of mask-defined regions.
**Relevance:** Semantic comparison of masked regions via LLM descriptions of (ref-image, mask) vs (recon-image, mask). Text-space similarity that is region-conditioned. Higher-level than Alpha-CLIP but captures meaning, not just pixels — useful diagnostic second opinion.

### #335. SEAGULL: No-reference IQA for Regions of Interest via VLM Instruction Tuning (Nov 2024)
**Summary:** First large VLM explicitly designed for ROI-level no-reference IQA. Mask-based Feature Extractor pools global and local tokens over a SAM-provided mask. Trained on SEAGULL-100w (~100M synthetic ROIs) and SEAGULL-3k (authentic). Per-ROI quality score + natural-language quality description.
**Relevance:** Closest published analog to what we need: a scalar quality score over a binary mask, with no reference. Feed Hi-SAM text mask → SEAGULL gives a reference-free quality-of-text-region metric that can be correlated directly against edit distance.

### #336. Q-Ground: Image Quality Grounding with Large Multi-modality Models (ACM MM 2024 Oral)
**Summary:** LMM-based system that jointly answers quality questions and segments distortion regions from text prompts ("where is the blur", "where is the jpeg artifact"). Trained on QGround-100K triplets (image, quality-text, distortion-mask). Pixel-level distortion segmentation aligned with textual queries.
**Relevance:** Complements SEAGULL — tells us *where* the degradation is inside a text region. Produces a distortion mask that, ANDed with Hi-SAM text mask, gives a tight "bad text pixel" region for failure localisation.

### #337. MaskCLIP: Extract Free Dense Labels from CLIP (ECCV 2022)
**Summary:** Training-free method to obtain dense, per-pixel CLIP features by removing the final attention pooling and projecting value tokens directly with the text-head weights. Zero-shot semantic segmentation on open vocabularies with no annotations. MaskCLIP+ with self-training jumps PASCAL-VOC unseen-class mIoU from 35.6 to 86.1.
**Relevance:** Dense per-pixel CLIP embeddings → cosine similarity between reference-render and OCR-render feature maps, averaged over the text-mask pixels → mask-restricted CLIP similarity with no training. Cheaper than Alpha-CLIP; trivially combinable with any binary mask source.

### #338. FourBi: Frequency-Domain Binarization for Documents (ICDAR 2024)
**Summary:** Deep-learning document binarization operating in frequency domain, specifically targeting gradient backgrounds and watermarks. Public code at fax004/FourBi. Referenced in our internal `document_reconstruction_techniques.md` as the recommended DL binarizer for degraded documents.
**Relevance:** Drop-in "remove background from the original" model for our A1-style preprocessing pipeline when Gaussian-divide + Otsu underperforms (e.g., saturated red banners, complex textures). GPU inference, near-perfect binarization on DIBCO-style degradations.

### #339. DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks (CVPR 2024)
**Summary:** Unified document-restoration model handling deblur, denoise, binarization, and dewarp in one model. Referenced in our internal docs. Public code at zzzhang-jx/DocRes.
**Relevance:** Alternative to FourBi; additionally handles dewarping which matters for scanned book spreads. Useful as a single preprocessor that normalises the original document before any comparison metric runs.

### #340. SauvolaNet: Learning Adaptive Sauvola Network for Degraded Document Binarization (2021)
**Summary:** CNN that mimics Sauvola thresholding but learns optimal window/k parameters per region. Public code referenced in our internal docs.
**Relevance:** Baseline DL binarizer. Fastest of the three (FourBi / DocRes / SauvolaNet); useful as the cheap tier in a tiered-binarizer pipeline.


## Independent Element Detectors for OCR Cross-Validation (2026-04-25)

### #341. PP-DocLayout: Unified Document Layout Detection Model (Mar 2025)
**Summary:** Family of RT-DETR based document layout detectors covering 23 element classes (Formula, Table, Figure, Text, Title, List, Caption, etc.). PP-DocLayout-L: 90.4% mAP@0.5 @ 13.4 ms/page on T4; Medium and Small variants trade accuracy for speed. Trained on 30K manually annotated pages + DocLayNet + PubLayNet. Weights shipped in PaddleX / PaddleOCR.
**Relevance:** Most direct fit for a *second-opinion* element-class detector. Run PP-DocLayout independently on the page image, get (bbox, class) predictions for formula/table/figure, ask "does PP-DocLayout agree Qwen's region is text, or does it say formula/table?". Disagreement is a strong reference-free Case-2 misclassification signal — the primary ceiling-breaker.

### #342. Advanced Layout Analysis Models for Docling (Heron family, Sep 2025)
**Summary:** IBM tech report on next-gen Docling layout models trained on 150k heterogeneous pages. Best model heron-101 (RT-DETRv2, ResNet-101) reaches 78% mAP on DocLayNet — +23.9% over Docling's prior. 28 ms/image on A100. 17 classes including Formula, Table, Picture, Caption, Code, List-item, Section-header, Page-header/footer, Footnote. Weights on HF at `docling-project/docling-layout-heron`.
**Relevance:** Complementary backbone and training data to PP-DocLayout — combining them yields a true independent ensemble rather than correlated errors. Heron has particularly strong Formula and Table classes, ideal for flagging when primary OCR merges displayed formulas into text blocks or loses tables.

### #343. FormulaDet / DynFormula (Pattern Recognition 2024)
**Summary:** Microsoft reframes MFD as joint formula-entity detection + relation extraction on ArxivFormula (600k pages, 15M inline, 1.9M displayed, 795k formula numbers — largest MFD dataset). DynFormula uses dynamic convolutions to localize inline and displayed formulas with strong separation from surrounding text; RelFormer groups entities into logical formula blocks. SOTA on ArxivFormula, IBEM, FormulaNet, Marmot.
**Relevance:** Most specialised "is this region a math formula" detector available — distinguishes inline from displayed (critical because Qwen OCR tends to merge inline math into text tokens). Cross-check each primary-OCR region against DynFormula's map to flag misclassified formulas.

### #344. FormulaNet: Benchmark Dataset for Mathematical Formula Detection (IEEE Access 2022)
**Summary:** 46,672-page STEM benchmark from arXiv with 13 element classes (display/inline formulas, display references, headers, tables, figures, paragraphs, captions, footnotes, lists, bibliography). Baseline: anchor-free FCOS at mAP 0.754. Weights at `felix-schmitt/FormulaNet`.
**Relevance:** Pretrained standalone formula-detector weights (FCOS) ideal as a fast CPU/GPU cross-check. Inline-vs-display distinction matches exactly the OCR misclassification we want to catch.

### #345. PubTables-1M / Table Transformer (TATR, CVPR 2022)
**Summary:** Microsoft PubTables-1M — 947k annotated tables from PubMed — and two DETR-based models: table detection (table vs. rotated-table) and table structure recognition. Detection model is plain DETR with ResNet-50, strong cross-domain transfer zero-shot. Weights `microsoft/table-transformer-detection` on HF. Established GriTS evaluation metric.
**Relevance:** Dedicated table-only cross-check: pass each primary-OCR "text" region through TATR-detection; if TATR emits a table bbox overlapping the OCR region, flag misclassification. Precision on table-vs-text axis.

### #346. HTTD: Hierarchical Transformer for Accurate Table Detection (Mathematics MDPI 2025)
**Summary:** Swin-L backbone with DAB-DETR dynamic anchor boxes, DN-DETR denoising queries, deformable attention. 96.98% precision on ICDAR-2019 cTDaR, 96.43% on TNCR, 93.14% on TabRecSet — SOTA among dedicated table detectors.
**Relevance:** Third independent table-detector opinion stacked with TATR. Swin-L backbone → failure modes largely uncorrelated with TATR → stronger ensemble cross-check without inheriting the primary detector's blind spots.


## SAM 3 & Open-Vocabulary Text-Prompted Segmentation (2026-04-25)

### #347. SAM 3: Segment Anything with Concepts (Nov 2025)
**Summary:** Meta AI's Nov 2025 release. Unified image+video model for Promptable Concept Segmentation (PCS). Unlike SAM 1/2 which returned a single mask per geometric prompt, SAM 3 accepts short noun-phrase text prompts ("table", "math formula", "caption") and/or image exemplars, returning all instances with unique IDs as pixel masks. Architecture: image-level detector + memory-based video tracker sharing a backbone; dedicated "presence head" decouples recognition from localization. Trained on SA-Co (4M unique concepts, 214K eval phrases — 50× prior benchmarks). Reportedly doubles accuracy of prior systems on PCS. Open-source at `facebookresearch/sam3`.
**Relevance:** *Single most important model for this problem.* Direct text-prompt segmentation of "math formula", "table", "caption", "figure" on the page image — per-element IoU between Qwen-OCR's bbox+type and SAM 3's mask becomes an OCR-independent consistency score. Highest ceiling-breaker potential.

### #348. Grounding DINO 1.5 (Pro & Edge, May 2024)
**Summary:** IDEA Research's successor to Grounding DINO. Pro scales to ViT-L, trained on 20M grounded images, 54.3 AP COCO / 55.7 AP LVIS-minival zero-shot. Edge optimised for real-time. Both accept free-form noun-phrase text. Official pairing with SAM2 in Grounded-SAM for text-to-mask.
**Relevance:** Drop-in text-prompted detector for document-element types. Lighter fallback if SAM 3 weights are heavy/gated; pipe boxes into SAM 2 for pixel masks.

### #349. DINO-X: Unified Vision Model for Open-World Object Detection and Understanding (Nov 2024)
**Summary:** IDEA's Nov 2024 follow-up to Grounding DINO 1.5. Supports text/visual/customised prompts in one encoder-decoder. Trained on Grounding-100M. DINO-X Pro: 56.0 AP COCO, 59.8 AP LVIS-minival, 63.3 AP LVIS rare (+5.8 over prior SOTA). Integrates detection, **segmentation**, pose, object captioning, QA — natively emits masks without a separate SAM pass.
**Relevance:** Stronger and more recent than Grounding DINO 1.5, directly emits segmentation masks from text prompts. Long-tail performance on rare LVIS classes suggests strong handling of uncommon document-specific phrases.

### #350. APE: Aligning and Prompting Everything All at Once (CVPR 2024)
**Summary:** Casts detection, segmentation, grounding as a unified instance-level sentence-object matching task. Single checkpoint achieves SOTA or competitive performance on 160 datasets. APE-L and lightweight APE-Ti (6M backbone) released.
**Relevance:** Alternative unified text-prompted segmenter lighter than SAM 3 / DINO-X Pro. Breadth (160 datasets) → document-adjacent concepts likely in-distribution. Useful as a secondary cross-validation head — disagreement between SAM 3 and APE for the same prompt becomes a stronger quality signal than either alone.

### #351. EVF-SAM: Early Vision-Language Fusion for Text-Prompted SAM (Jun 2024)
**Summary:** Adapts SAM and SAM 2 to accept text prompts via a pretrained BEIT-3 multimodal encoder with early vision-language fusion. 1.32B params — ~82% fewer than prior LMM-based SAM extensions — yet SOTA on RefCOCO/+/g. Supports SAM-2 and video.
**Relevance:** Most parameter-efficient text-to-mask path. If SAM 3 access is blocked, EVF-SAM gives a proven text-prompted SAM wrapper with published weights. Early-fusion architecture vs SAM 3's detector-head design — useful ablation axis.

### #352. Florence-2: Unified Representation for a Variety of Vision Tasks (CVPR 2024)
**Summary:** Microsoft seq-to-seq foundation model (DaViT + BERT + encoder-decoder). Textual task prompt → text + location tokens for captioning, detection, grounding, referring segmentation, dense region captioning. Only 0.2B / 0.7B params, MIT-licensed, trained on FLD-5B (5.4B annotations on 126M images).
**Relevance:** Extremely small and fast relative to SAM 3 — suitable for batch over many pages. Its referring-segmentation and open-vocab-detection task prompts directly support "locate the math formula" queries. Also natively captions regions → could verify Qwen-OCR's element text via cross-captioning (second OCR-independent signal).

### #353. T-Rex2: Generic Object Detection via Text-Visual Prompt Synergy (ECCV 2024)
**Summary:** IDEA Research detector accepting both text prompts and visual prompts (point/box exemplars) in one model. Unified via contrastive learning. Strong zero-shot on long-tail/rare categories where text alone struggles.
**Relevance:** Document pages contain visually-stylised elements (formulas, tables) for which drawing a single exemplar on the first page and reusing as visual prompt across a corpus may outperform text alone. Hybrid fallback.


## VLM Missing-Element Judging & Cost-Effective Judges (2026-04-25)

### #354. OmniDiff: Fine-grained Image Difference Captioning Benchmark (ICCV 2025)
**Summary:** IDC benchmark with 324 scenarios, human-annotated ~60-word captions covering 12 distinct change types (including object addition/removal — i.e. "missing elements"). Proposes M3Diff with Multi-scale Differential Perception (MDP) module at visual-encoder layers 17/20/23/26, 2 stacked transformer layers modelling inter-image deltas. SOTA on Spot-the-Diff, IEdit, CLEVR-Change, CLEVR-DC, OmniDiff. Multi-change enumeration as primary task.
**Relevance:** Closest current analogue of "VLM compares original vs reconstruction and lists what's missing". MDP module is a cheap graft onto Qwen3.5-122B's vision tower; 12-change-type taxonomy is a ready JSON schema for per-formula/element verdicts.

### #355. Deploying Tiny LVLM Judges for Real-World Chart-Model Evaluation (Oct 2025)
**Summary:** Targets "cheap judge" problem: can a 2B-parameter LVLM replace GPT-4V for chart-understanding evaluation? Two strategies: (i) multi-criteria prompting (per-rubric decomposition rather than holistic score), (ii) distillation fine-tuning from GPT-4V + Gemini-1.5-Pro judgment labels onto tiny 2B models. Distilled 2B judge approaches GPT-4V correlation at fraction of cost; multi-criteria prompting alone closes much of the gap even without training. Reports throughput + correlation deltas.
**Relevance:** Directly addresses small-VLM judges and cost. Multi-criteria prompting is plug-and-play: decompose "reconstruction fidelity" into per-element rubric items (formula count, symbol match, table rows, text span) on Qwen3.5-122B. Later distill to Qwen2.5-VL-7B for cheaper local inference.

### #356. ECVL-ROUTER: Scenario-Aware Routing for Vision-Language Models (Oct 2025)
**Summary:** First scenario-aware router for VLMs, dynamically picks small vs large VLM based on query/image features and user reliability budget. >80% of queries go to small model with <10% accuracy drop. Introduces routing-specific evaluation metrics (cost-accuracy Pareto, scenario coverage). Open implementations for several backbone pairs.
**Relevance:** Concrete recipe for two-stage screener + verifier. A small Qwen2-VL-7B screens "is this page worth inspecting at all?"; only flagged pages/elements escalate to Qwen3.5-122B for structured per-formula verdict.

### #357. Detect, Describe, Discriminate (D3) — Beyond VQA for MLLM Evaluation (NeurIPS 2024 D&B)
**Summary:** Benchmark of image pairs differing on one prominent point (six Points of Difference: state, camera, positioning, orientation, scene, clutter). Given a pair, model must (1) Detect the difference, (2) Describe target so that (3) it Discriminates target from distractor. Key finding: current open-source MLLMs fail to beat random on independent discernment of fine-grained visual differences; only GPT-4V and top closed models do substantially better.
**Relevance:** D3's task framing matches our "what's different between original and reconstruction?" use case. Honest finding (open MLLMs struggle) is a warning against naively trusting a single Qwen3.5-122B call on page pairs. Motivates combining Consensus Entropy (#360) + structured rubrics (#355).

### #358. OneDiff: A Generalist Model for Image Difference Captioning (2024)
**Summary:** Siamese image encoder + Visual Delta Module that makes differences explicit before language generation. Dual-phase training: Coupled Sample Training (one pair, one caption) then multi-task over IDC benchmarks. Releases DiffCap training set. Generalises across Spot-the-Diff, IEdit, CLEVR-Change without per-dataset fine-tuning.
**Relevance:** Complements #354 with a practical architectural recipe (Visual Delta Module) replicable as a lightweight diff-encoder on top of Qwen features, feeding only the delta representation to the 122B LM — decouples "find the difference" from "describe the difference", matching our goal of per-formula presence flags rather than free-form captions.

### #359. A Unified Approach to Routing and Cascading for LLMs (ICLR 2025)
**Summary:** Theoretical framing: routing (one-shot model choice) and cascading (sequential escalation with per-step abstention) are special cases of a single cost-optimal decision problem. Derives combined "cascade routing" policy dominating either alone under realistic cost/latency constraints. Concrete algorithms and empirical Pareto curves.
**Relevance:** Provides the principled math for our cheap-screener-plus-big-verifier pipeline (#356 is the VLM-specific instance; this paper is the theory). Useful for deciding when to escalate Qwen-7B → Qwen-122B based on predicted confidence and per-element abstention.


## OCR Self-Uncertainty & Ensemble Disagreement (2026-04-25)

### #360. Consensus Entropy: Multi-VLM Agreement for Self-Verifying and Self-Improving OCR (Apr 2025)
**Summary:** Training-free, model-agnostic reliability score — Shannon entropy over a distribution of output agreements across multiple VLMs (or samples of one VLM). Correct predictions converge in output space; errors diverge. CE-OCR routes adaptively — low CE accepts consensus, high CE routes to stronger VLM. +42.1% F1 over VLM-as-Judge baselines on reliability detection, equal inference cost to plain self-consistency. Reference-free and element-localisable: disagreement can be attributed to specific tokens/regions.
**Relevance:** *Canonical reference for ensemble-disagreement and the strongest single hit for this research wave.* Run Qwen3.5-122B K=3 times (or Qwen-122B + Qwen-7B + InternVL) on the same page pair, compute CE per-formula → reference-free per-element confidence signal feeding the correlation baseline.

### #361. Teaching VLMs to Admit Uncertainty in OCR from Lossy Visual Inputs (ICLR-submit 2025)
**Summary:** Trains a VLM to bracket uncertain transcription spans with explicit `<C>...</C>` delimiters rather than silently hallucinating. Pseudo-labeled cold start → GRPO with multi-objective reward balancing transcription accuracy and uncertainty coverage. Also provides a multi-model voting baseline: four VLMs transcribe; if ≥2 disagree with anchor in a region, flag as uncertain. Final model: uncertainty-tag F1 = 0.685 while preserving transcription accuracy.
**Relevance:** Concrete element-level self-uncertainty formulation for VLM OCR + explicit ensemble-voting baseline with published precision/recall. Voting baseline is directly transferable as a method contract.

### #362. Entropy Heat-Mapping: Localising GPT-Based OCR Errors with Sliding-Window Shannon Analysis (May 2025)
**Summary:** Sliding-window Shannon entropy over GPT-based OCR token distributions → spatial heat-map localising likely recognition errors. Pure information-theoretic, no training, from token logprobs of a single OCR pass. Correlation between high-entropy regions and actual OCR errors on both printed and degraded inputs.
**Relevance:** Ideal CPU-cheap quick-cross-check layer. Only needs logprobs from our existing Qwen3.5-122B vLLM calls — via OpenAI-compatible `logprobs=True` — mapping high-entropy spans back to element bboxes for a reference-free per-element suspect score.

### #363. Seeing is Believing? Mitigating OCR Hallucinations in MLLMs (KIE-HVQA, NeurIPS 2025)
**Summary:** First benchmark specifically for OCR hallucination under visual degradation (motion blur, low contrast) — IDs, receipts, invoices (2,000 train / 400 test). GRPO-based framework with reward for visual-uncertainty self-awareness and refusal-to-answer when evidence insufficient. Qwen2.5-VL-7B trained this way beats GPT-4o by ~22% on hallucination-free accuracy with no drop on clean tasks.
**Relevance:** Rigorous benchmark for evaluating reference-free OCR quality signals + concrete formulation of "visual-uncertainty self-awareness" as a trainable objective. Use as stress-test target: does our coverage/disagreement signal flag the degraded-but-confident outputs where GPT-4o fails?

### #364. Assessing GPT Model Uncertainty in Mathematical OCR via Entropy Analysis (2024)
**Summary:** Measures conditional entropy and mutual information over GPT output-token sequences for image-to-LaTeX mathematical OCR at varying input resolutions. Clean monotonic relationship: higher resolution → lower per-token entropy → higher accuracy. Information-theoretic framing + reproducible code.
**Relevance:** Complements #362 for the formula/math dimension of our multi-dim pipeline (relevant since we integrate CDM for formulas). Shows per-token entropy averaged over output is a useful reference-free quality signal sensitive to exactly the input-quality variations we care about.
