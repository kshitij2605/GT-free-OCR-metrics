# Methods Index

This repository includes **146 method implementations** plus the `baseline`, covering pixel-level, perceptual, learned, text-confidence, content-element, and hybrid approaches.

Each method has a YAML spec in `methods/<method_id>.yaml` and a Python implementation in `scripts/methods/<method_id>.py`.

Run any method with:

```bash
bash scripts/run_method.sh <method_id> [gpu_id]
```

The `Spearman mean` column is the per-variant best slot averaged across all 5 OCR-output variants (paper Eq. 1). `—` means the method did not produce full per-variant correlations (crashed, superseded, or yielded no signal); these are retained for transparency.

---

## Composite  (44)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P2_210_elem_p05_table_text_elemclip` | D210 — stack D209 (P5+table_elemclip) + add elem_clip_p5 to text variant at β=0.05. | 0.4938 |
| `P1_147_elem_p05` | D147 — per-element CLIP percentile sweep: P10→P5. | 0.4938 |
| `P2_209_elem_p05_table_elemclip` | D209 — stack D147 (P5 element CLIP) + add elem_clip_p5 to table variant at β=0.05. | 0.4938 |
| `P2_211_dino_elem_p05_formula` | D211 — D147 base (P5 elem CLIP for all/all_no_mask) + DINOv2 element-level P5 for formula at β=0.05. | 0.4938 |
| `P2_212_dino_elem_p05_formula_fixed` | D212 — Fixes D211 bug: formula _flush_batch did not pass content_elem_score=ces to _fuse. | 0.4938 |
| `P2_213_elem_p05_table_fixed` | D213 — Fixes D209 bug: table _flush_batch + 2 primary/fallback paths did not wire content_elem_score into _fuse. | 0.4938 |
| `P2_214_dinov2_patch_cosine` | D214 — Tests DINOv2 patch-level soft-NN cosine for the formula variant (D83 direction). | 0.4937 |
| `P1_148_elem_p10_table_beta05` | D148 — table variant elem_p10 β=0.05. | 0.4936 |
| `P1_137_content_elem_p10_all` | D137 — New mechanism class: per-content-element CLIP P10 for all/all_no_mask. | 0.4932 |
| `P1_140_elem_p10_allnm_beta15` | D140 — per-element CLIP P10 β sweep for all_no_mask: β=0.10→0.15. | 0.4932 |
| `P1_151_text_ssim_beta20` | D151 — text SSIM β 0.15→0.20. | 0.4932 |
| `P1_150_elem_p15` | D150 — per-element CLIP P15 (vs P10 in D137). | 0.4930 |
| `P1_152_allnm_ssim_beta15` | D152 — all_no_mask SSIM β 0.10→0.15. | 0.4928 |
| `P1_136_ssim_all_beta20` | D136 — all SSIM β=0.15→0.20 diagnostic. | 0.4928 |
| `P1_133_ssim_table_beta15` | D133 — SSIM table β sweep step 3: 0.10→0.15. | 0.4927 |
| `P1_139_elem_p10_all_beta15` | D139 — per-element CLIP P10 β sweep step 2 for all: β=0.10→0.15. | 0.4925 |
| `P1_141_elem_p10_text_beta10` | D141 — Apply per-element CLIP P10 to text variant at β=0.10 (D137 only applied to all/all_no_mask). | 0.4925 |
| `P1_146_ssim768` | D146 — SSIM resolution sweep: 512→768. | 0.4924 |
| `P1_132_ssim_table_beta10` | D132 — SSIM table β sweep step 2: 0.05→0.10. | 0.4924 |
| `P1_134_ssim_allnm_beta15` | D134 — all_no_mask SSIM β sweep 0.10→0.15. | 0.4924 |
| `P1_142_elem_mean_all` | D142 — Test MEAN aggregation instead of P10 for per-element CLIP on all/all_no_mask. | 0.4923 |
| `P1_143_elem_p10_formula_beta05` | D143 — Apply per-element CLIP P10 to formula variant at β=0.05 (D141 added text, D137 added all/all_no_mask). | 0.4922 |
| `P1_138_ssim_all_beta25` | D138 — all SSIM β sweep step 2: 0.20→0.25. | 0.4920 |
| `P1_128_ssim_table_beta05` | D128 — SSIM table supplement at β=0.05. | 0.4918 |
| `P1_131_all_nm_tablecell_beta05` | D131 — all_no_mask table_cell β reduction 0.10→0.05. | 0.4918 |
| `P1_135_ssim_table_beta20` | D135 — SSIM table β sweep step 4: 0.15→0.20. | 0.4913 |
| `P1_149_elem_docsim_p10` | D149 — per-element DocSim P10 vs CLIP P10. | 0.4911 |
| `P1_124_ssim_stack_text15_formula05` | D124 — SSIM combined stack: text β=0.15 (D122 KEEP 0.4908) + formula β=0.05 (D123 KEEP 0.4904). | 0.4908 |
| `P1_122_ssim_text_beta15` | D122 — SSIM text β sweep step 3: text=0.15 (up from D121's 0.10). | 0.4908 |
| `P1_125_ssim_text_beta20` | D125 — SSIM text β sweep step 4: text=0.20 (up from D122's 0.15). | 0.4907 |
| `P1_127_ssim_all_beta20` | D127 — SSIM all/all_no_mask beta sweep: all 0.15→0.20, all_no_mask 0.10→0.15. | 0.4905 |
| `P1_129_ssim_formula_beta10` | D129 — SSIM formula β sweep step 2: 0.05→0.10. | 0.4904 |
| `P1_123_ssim_formula_beta05` | D123 — SSIM supplement extended to formula variant at β=0.05. | 0.4904 |
| `P1_121_ssim_text_beta10` | D121 — SSIM text β sweep step 2: text=0.10 (up from D120's 0.05). | 0.4903 |
| `P1_126_dinov2_patch_text_beta05` | D126 — DINOv2 patch-level mean cosine similarity supplement for text variant at β=0.05. | 0.4901 |
| `P1_120_ssim_text_supplement` | D120 — SSIM page supplement extended to the text variant at β_text=0.05. | 0.4891 |
| `P1_116_ssim_variant_specific_beta` | D116 — Variant-specific SSIM β: all=0.15 (D113 per-variant best), all_no_mask=0.10 (D112 per-variant best). | 0.4877 |
| `P1_113_ssim_page_supplement_15` | D113 — D111c: SSIM β_ssim=0.15 sweep. | 0.4872 |
| `P1_112_ssim_page_supplement_10` | D112 — D111b: SSIM β_ssim=0.10 sweep. | 0.4867 |
| `P1_114_ssim_page_supplement_20` | D114 — D111d: SSIM β_ssim=0.20 sweep. | 0.4864 |
| `P1_119_msssim_variant_specific_beta` | D119 — MS-SSIM (Multi-Scale SSIM via piq.multi_scale_ssim) as structural supplement, replacing single-scale SSIM at the same… | 0.4834 |
| `P1_118_haarpsi_variant_specific_beta` | D118 — HaarPSI (Haar Perceptual Similarity Index via piq.haarpsi) as structural supplement, replacing raw grayscale SSIM at the same… | 0.4790 |
| `P2_206_vlm_page_score` | D206 — VLM page score: replace DocSim prod_cc with Qwen3.5-122B-A10B visual judge for text/all/all_no_mask variants. | — |
| `P2_215_dinov2_patch_bidir` | D215 — Tests bidirectional (symmetric) DINOv2 patch soft-NN cosine for formula variant. | — |

## Deep Feature  (50)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P2_060p_docsim_bbox_table_min` | D60.p — single-line variation of D60.m: aggregate per-table-bbox DocSim cosines via MIN instead of MEAN. | 0.4453 |
| `P2_060t_docsim_bbox_min_filter_small` | D60.t — refine D60.p by dropping bboxes whose shorter side is < 32 px before MIN aggregation. | 0.4453 |
| `P2_060w_docsim_bbox_min_anm_extend` | D60.w — extends D60.p per-table-bbox MIN DocSim aggregation from variant=table to variant=all_no_mask. | 0.4453 |
| `P2_060v_docsim_bbox_table_p10` | D60.v — single-line variation of D60.p: aggregate per-table-bbox DocSim cosines via 10th-percentile (P10) instead of MIN. | 0.4452 |
| `P2_060r_docsim_bbox_min_h61_crop` | D60.r — refine D60.p by applying H6.1 preprocessing (grayscale -> SHARPEN -> autocontrast -> RGB) to per-table-bbox crops before passing… | 0.4446 |
| `P2_060m_docsim_bbox_table` | D60.m — last D60-axis lever in Lead-session scope. | 0.4432 |
| `P2_060u_docsim_bbox_table_median` | D60.u — single-line variation of D60.p: aggregate per-table-bbox DocSim cosines via MEDIAN instead of MIN. | 0.4431 |
| `P2_060s_docsim_bbox_min_padded` | D60.s — refine D60.p by padding each per-table-bbox crop by 24 px on every side (clipped to image bounds) before passing to the DocSim head. | 0.4430 |
| `P2_060y_docsim_paragraph_text` | D60.y — paragraph-level per-bbox MIN DocSim for variant=text only. | 0.4402 |
| `P2_060o_docsim_bbox_table_max` | D60.o — single-line variation of D60.m: aggregate per-table-bbox DocSim cosines via MAX instead of MEAN. | 0.4401 |
| `P2_060q_docsim_bbox_min_all_extend` | D60.q — extend D60.p's per-table-bbox MIN DocSim from variant=table to variant=all (using ocr_table_elements.json which is also present in… | 0.4383 |
| `P2_060n_docsim_bbox_text_table` | D60.n — extend D60.m's per-bbox aggregation from variant=table to variant=text. | 0.4382 |
| `P2_060x_docsim_h4e_5050_formula` | D60.x — 50/50 fusion of H4.e (current formula winner: 50/50 CLIP+DINOv2) with DocSim cc for variant=formula only. | 0.4303 |
| `P2_060l_docsim_h61_preproc_text` | D60.l — extends D60.j by routing variant=text DocSim inputs through H6.1 preprocessing (grayscale -> SHARPEN -> autocontrast -> 3-channel… | 0.4095 |
| `P2_060j_docsim_text_all_anomask` | D60.j — extend D60.b's variant-conditional DocSim by adding variant=text to the DocSim-cc set. | 0.4093 |
| `P2_060k_docsim_mc_4thterm_all` | D60.k — extends D60.j by re-using the DocSim cosine output as a 5th term in the multi_composite formula for variant=all only. | 0.4093 |
| `P2_060e_docsim_h61_preproc` | D60.e — refinement of D60.b. | 0.4052 |
| `P2_060b_docsim_variant_conditional` | D60.b — variant-conditional refinement of D60. | 0.4042 |
| `P2_060f_docsim_table_5050_fusion` | D60.f — variant=table fusion experiment. | 0.4042 |
| `P2_060d_docsim_baseline_7030_fusion` | D60.d — asymmetric refinement of D60.b. | 0.3945 |
| `P2_060c_docsim_baseline_5050_fusion` | D60.c — refinement of D60.b. | 0.3856 |
| `P1_120c3b_all_no_mask_baseline_preproc` | H11.2: REFINE of H6.1. | 0.3704 |
| `P1_120b_adists_variant_conditional` | H5.b: Variant-conditional multi_composite formula. | 0.3701 |
| `P1_120c2_table_4quad_clip` | H6.0 (Branch 6, table-cc-specialist): For variant=table only, replace single full-image CLIP cosine with mean of 5 CLIP cosines computed… | 0.3701 |
| `P1_120c3_table_clip_baseline_preproc` | H6.1: REFINE of H6.0. | 0.3701 |
| `P1_120_adists_additive` | H5: A-DISTS unavailable in pyiqa 0.1.15 (only 'dists' listed). | 0.3686 |
| `P1_084c_dinov2_clip_avg` | H4.c: clip_cosine = 0.5 * DINOv2_vitb14_CLS_cosine + 0.5 * OpenCLIP_ViT-B/32_cosine. | 0.3680 |
| `P1_084e_variant_strategy` | H4.e: Per-variant strategy selection. | 0.3680 |
| `P1_158_gtpoly_content_mask_all_no_mask` | H14.1.diag CEILING DIAGNOSTIC for mask-based preprocessing axis. | 0.3664 |
| `P1_084_dinov3_patch` | REPLACES CLIP encoder for clip_cosine slot; multi_composite (SSIM/MSE/LPIPS) UNCHANGED — additive approach. | 0.3643 |
| `P1_084d_variant_conditional` | H4.d: variant-conditional encoder selection in clip_cosine slot. | 0.3643 |
| `P2_060_docsim_dreamsim_recipe_with_document_embeddings_208` | D60 / DocSim — DreamSim recipe applied to documents. | 0.3643 |
| `P1_025_ms_swd_table_4th_term` | H2 (Branch 2 reactivation): MS-SWD (Multi-scale Sliced Wasserstein Distance via pyiqa msswd) as 4th additive term in multi_composite for… | 0.3640 |
| `P1_084b_dinov2_patchmean` | Refinement of H4 (P1_084_dinov3_patch). | 0.3610 |
| `P1_024_deepwsd_wasserstein_metric` | D24 / H_DeepWSD: Sibling REFINE under H5.b confirmed slot. | 0.3610 |
| `P1_121_dists_baseline` | Replaces the LPIPS slot in multi_composite with DISTS (Deep Image Structure and Texture Similarity, Ding et al. | 0.3292 |
| `P1_030_hsic_feature_dependence_metric` | D30: Replace cosine inside clip_cosine slot with normalized HSIC (Centered Kernel Alignment, CKA, with Gaussian/RBF kernels) for… | — |
| `P1_084f_dinov2_only_all` | H8.1: REFINE of H6.1 (current best, spearman_mean=0.3598). | — |
| `P1_084g_dinov2_only_all_no_mask` | H11.1: REFINE of H6.1 (P1_120c3). | — |
| `P1_110_text_wavelet_detail_4th` | H7.1: Add multi-scale 2D-DWT detail-subband cosine similarity as a 4th additive term in multi_composite ONLY for variant=text. | — |
| `P1_110b_text_dwt_smallweight` | H7.1b: REFINE of H7.1 (refuted). | — |
| `P1_116_linear_probe_orthogonalisation_table_cc` | H15 / D116: PCA-based feature orthogonalisation for variant=table cc only. | — |
| `P1_117_deepssim_all_4th_term` | H14.X / D117: Sibling REFINE under H5.b confirmed slot. | — |
| `P1_120c4_table_clip_large` | H6.2: REFINE of H6.1. | — |
| `P1_120c6_table_clip_hires` | H6.4: REFINE of H6.1. | — |
| `P1_120c_dists_all_and_all_no_mask` | H5.c: Extend H5.b — DISTS-augmented composite (0.3 SSIM/0.2 MSE/0.2 LPIPS/0.3 DISTS) applied to BOTH variant=all and variant=all_no_mask. | — |
| `P1_120d_table_dists_4th_term` | H5.e: Extends H5.b's variant-conditional composite to also cover variant=table. | — |
| `P1_120d_vif_all_5term` | H10.1 (Branch 10 root): VIF (Visual Information Fidelity, Sheikh & Bovik 2006) as a 5th additive term in multi_composite for variant=all… | — |
| `P1_165_maskclip_table_dense` | H9.1: MaskCLIP-style dense per-patch cosine for variant=table only. | — |
| `P1_166_siglip_table_clip` | H12.1: Branch 12 root. | — |

## Self-Uncertainty (OCR log-probs)  (33)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_106h_gamma_formula_50` | D106.h — γ_formula=0.50 boundary push from D106.e γ=0.40 (D106.f best stack). | 0.4814 |
| `P1_106f_combined_formula_40_table_10` | D106.f — combine the two cross-class fusion KEEPs that target different variants. | 0.4812 |
| `P1_106k_gamma_formula_60` | D106.k — γ_formula=0.60 boundary push from D106.h γ=0.50 (new best 0.4814). | 0.4812 |
| `P1_106e_gamma_40_formula` | D106.e — gamma boundary push to 0.4 on variant=formula. | 0.4809 |
| `P1_106d_iq_fusion_table` | D106.d — first fusion attempt on variant=table. | 0.4808 |
| `P1_106c_gamma_30_formula` | D106.c — gamma-sweep step. | 0.4807 |
| `P1_106b_iq_fusion_formula` | D106.b — first cross-class fusion. | 0.4804 |
| `P1_106j_gamma_table_20` | D106.j — γ_table=0.20 sweep step from D106.d γ=0.10. | 0.4802 |
| `P1_106m_iq_fusion_all_05` | D106.m — first IQ-fusion on variant=all and all_no_mask at γ=0.05. | 0.4801 |
| `P1_106g_iq_fusion_text` | D106.g — first IQ-fusion attempt on variant=text. | 0.4793 |
| `P1_224e_token_class_filtered_window_min` | D224.e — generalize D224.c's class-filter signal to variant=all and variant=all_no_mask via TOKEN-LEVEL filtering on the rolling-K window-… | 0.4776 |
| `P1_224d_variant_conditional_source` | D224.d — variant-conditional ENTROPY SOURCE routing. | 0.4772 |
| `P1_215l_shannon_top_logprobs` | D215.l — alternative entropy SOURCE: replaces per-token mean confidence exp(top_logprob) with 1 - H/log(K_top) where H is Shannon entropy… | 0.4742 |
| `P1_215r_per_variant_source` | D215.r — per-variant SOURCE selection at the established α/K budget. | 0.4735 |
| `P1_215n_three_way_fusion` | D215.n — three-way blend that splits each variant's D215.k α-budget evenly between the two confidence SOURCES that independently… | 0.4727 |
| `P1_106n_iq_per_formula_bbox_min` | D106.n — replaces page-level IQ similarity with per-formula-bbox IQ MIN for variant=formula. | 0.4719 |
| `P1_215k_variant_k3_20` | D215.k — variant-conditional alpha + K with K=3 for text/all/all_no_mask (push boundary), K=20 for formula (D215.h peak). | 0.4701 |
| `P1_215m_variant_k2_20` | D215.m — extreme K boundary push: K=2 for text/all/all_no_mask. | 0.4684 |
| `P1_215i_variant_alpha_and_k` | D215.i — combine per-variant alpha + per-variant window_k from sweeps. | 0.4669 |
| `P1_215o_alpha_refinement_at_k3` | D215.o — α refinement at K=3 boundary. | 0.4668 |
| `P1_215e_alpha_variant_conditional` | D215.e — variant-conditional alpha based on D215.b/c/d sweep findings: table: alpha=0.0 (entropy DILUTES per-cell-MIN signal) all:… | 0.4638 |
| `P1_215j_window_k3` | D215.j — variant-conditional alpha (table=0, all=0.3, others=0.4) with uniform WINDOW_K=3. | 0.4614 |
| `P1_215g_window_k5` | D215.g — same variant-conditional alpha as D215.e but with smaller window K=5 (instead of K=10) for the rolling-window MIN of per-token… | 0.4614 |
| `P1_215c_alpha030_fusion` | D215.c — alpha sweep step 1: clip_cosine_new = 0.7 * production_cc + 0.3 * window_min_confidence. | 0.4600 |
| `P1_215b_docsim_p_plus_confidence_min_fusion` | D215.b — fusion test: clip_cosine_new = 0.8 * production_cc + 0.2 * window_min_confidence. | 0.4577 |
| `P1_215h_window_k20` | D215.h — same variant-conditional alpha as D215.e but with larger window K=20 (instead of K=10) for the rolling-window MIN of per-token… | 0.4576 |
| `P1_215d_alpha040_fusion` | D215.d — alpha sweep step 2: clip_cosine_new = 0.6 * production_cc + 0.4 * window_min_confidence. | 0.4561 |
| `P1_224b_per_bbox_shannon_min` | D224.b — replaces the rolling-K-token window-MIN entropy aggregation (current D215.l best 0.4742) with TRUE per-bbox-element MIN… | 0.4519 |
| `P1_224c_class_filtered_per_bbox_max_entropy_min` | D224.c — first test of CLASS-AWARE entropy aggregation, leveraging the newly available `data/ocr_logprobs_per_bbox/` per-bbox stats with… | 0.4481 |
| `P1_215f_alpha050_fusion` | D215.f — alpha=0.5 uniform fusion. | 0.4460 |
| `P1_213_sliding_window_heatmap_area_weighted` | D213 — final entropy-axis probe. | 0.4442 |
| `P1_215_qwen_logprob_confidence_mean` | D215 — Phase-1 carried direction, UNBLOCKED 2026-04-30 by operator's logprob OCR re-run (data/ocr_logprobs/<page>/ocr_logprobs.json now… | 0.3290 |
| `P1_224_qwen_token_logprob_window_min` | D224 — DIAGNOSTIC entropy-axis probe at finer granularity than D215. | 0.2270 |

## Per-Element Structure  (8)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_111_ssim_page_supplement_05` | D111 — adds SSIM page-level structural similarity as β_ssim=0.05 supplement on top of D107b. | 0.4847 |
| `P1_107b_beta_table_cell_20` | D107b — sweeps β=0.20 for per-table-cell MIN DocSim supplement on all/all_no_mask. | 0.4820 |
| `P1_108_formula_bbox_supplement_10` | D108 — adds per-formula-bbox MIN DocSim as β_f=0.10 supplement on top of D107b's table-cell supplement (β_table=0.20). | 0.4819 |
| `P1_107c_beta_table_cell_30` | D107c — sweeps β=0.30 for per-table-cell MIN DocSim supplement on all/all_no_mask. | 0.4819 |
| `P1_107_table_cell_supplement_all_10` | D107 — adds per-table-cell MIN DocSim as β=0.10 supplement for variant=all and all_no_mask. | 0.4818 |
| `P1_109_sobel_edge_supplement_05` | D109 — adds Sobel edge-map cosine similarity as β_edge=0.05 supplement on top of D107b. | 0.4818 |
| `P1_110_multiscale_sobel_supplement_05` | D110 — adds multi-scale Sobel edge-map cosine similarity as β_ms_edge=0.05 supplement on top of D107b. | 0.4813 |
| `P1_068_spatially_aware_hash_grid_min` | D68 — adds 4x4 spatially-aware pHash grid MIN as β_grid=0.05 supplement on top of D107b. | 0.4809 |

## Baseline  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `baseline` | Full-page SSIM/MSE/LPIPS and CLIP comparison between masked_original.png (original page with image/table regions white-filled) and… | 0.3390 |

## Perceptual  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_082_st_lpips` | Replaces standard LPIPS with ST-LPIPS (Ghildyal & Liu, ECCV 2022) in the multi_composite formula: 0.4*SSIM + 0.3*(1-MSE) + 0.3*(1-ST_LPIPS). | 0.3188 |

## Perceptual (Latent)  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P2_059_pieapp_perceptual_metric` | D59 — PieAPP (Perceptual Image-Error Assessment through Pairwise Preference) as proxy for PIM/MILO_L. | 0.1502 |

## Classical NR-IQA  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_106_frontiers_diqa_hand_crafted` | D106 — first PIVOT after entropy axis saturated at D224.e 0.4776. | 0.3653 |

## Classical Signal  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_011_phase_cong_table_cc` | H_phasecong / D11: Mechanism-orthogonal probe of the table_cc_encoder_swap_universal dead_end. | — |

## Frequency  (2)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_002b_haarpsi_all_no_mask_4th_term` | H13.1 (linear-strategy queue, builds on P1_120c3b H11.2 best=0.3600): adds piq.haarpsi as a 4th term in multi_composite for… | — |
| `P1_002c_haarpsi_all_4th_term` | H13.2 (linear-strategy queue, sibling REFINE under H5.b confirmed 4th-term mechanism): HaarPSI replaces DISTS as 4th term in… | — |

## Hash  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_066_hashes` | Replaces the SSIM slot in multi_composite with an average of 3 perceptual hashes (pHash, wHash, dHash) at 16x16 (256-bit) resolution. | 0.2869 |

## Hash Metric  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_067_pdq_256bit_hash` | D67 — Page-level 256-bit pHash as PDQ proxy. | 0.2162 |

## Preprocessing  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P1_155_lama_decoration_removal` | H14.0: FIRST preprocessing-axis experiment after 10-refute slot-interior streak. | 0.3422 |

## OCR-Specialised Perceptual  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P2_104_recognition_feature_distance` | D104 (Phase 2, Priority medium, Expected Impact Very High): Recognition-feature distance / Content-Perceptual loss. | — |

## Layout-Detector  (1)

| Method ID | Description | Spearman mean |
|---|---|---:|
| `P2_193_pp_doclayout_disagreement` | D193 (Phase 2, Priority high): Cross-model layout-classifier disagreement signal. | — |

