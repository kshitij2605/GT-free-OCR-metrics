# Methods Index

This repository includes **131 method implementations**, covering
pixel-level, perceptual, learned, text-confidence, content-element, and hybrid approaches.

Each method has a YAML spec in `methods/` and a Python implementation in `scripts/methods/`.

Run any method with:

```bash
bash scripts/run_method.sh <method_id>
```

---

## Perceptual

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_082_st_lpips` | > | — |

## Frequency

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_002b_haarpsi_all_no_mask_4th_term` | > | — |
| `P1_002c_haarpsi_all_4th_term` | > | — |

## Classical Signal

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_011_phase_cong_table_cc` | > | — |

## Deep Feature

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_024_deepwsd_wasserstein_metric` | > | — |
| `P1_025_ms_swd_table_4th_term` | > | — |
| `P1_030_hsic_feature_dependence_metric` | > | — |
| `P1_084_dinov3_patch` | > | — |
| `P1_084b_dinov2_patchmean` | > | — |
| `P1_084c_dinov2_clip_avg` | > | — |
| `P1_084d_variant_conditional` | > | — |
| `P1_084e_variant_strategy` | > | — |
| `P1_084f_dinov2_only_all` | > | — |
| `P1_084g_dinov2_only_all_no_mask` | > | — |
| `P1_110_text_wavelet_detail_4th` | > | — |
| `P1_110b_text_dwt_smallweight` | > | — |
| `P1_116_linear_probe_orthogonalisation_table_cc` | > | — |
| `P1_117_deepssim_all_4th_term` | > | — |
| `P1_120_adists_additive` | > | — |
| `P1_120b_adists_variant_conditional` | > | — |
| `P1_120c2_table_4quad_clip` | > | — |
| `P1_120c3_table_clip_baseline_preproc` | > | — |
| `P1_120c3b_all_no_mask_baseline_preproc` | > | — |
| `P1_120c4_table_clip_large` | > | — |
| `P1_120c6_table_clip_hires` | > | — |
| `P1_120c_dists_all_and_all_no_mask` | > | — |
| `P1_120d_table_dists_4th_term` | > | — |
| `P1_120d_vif_all_5term` | > | — |
| `P1_121_dists_baseline` | > | — |
| `P1_158_gtpoly_content_mask_all_no_mask` | > | — |
| `P1_165_maskclip_table_dense` | > | — |
| `P1_166_siglip_table_clip` | > | — |
| `P2_060_docsim_dreamsim_recipe_with_document_embeddings_208` | > | — |
| `P2_060b_docsim_variant_conditional` | > | — |
| `P2_060c_docsim_baseline_5050_fusion` | > | — |
| `P2_060d_docsim_baseline_7030_fusion` | > | — |
| `P2_060e_docsim_h61_preproc` | > | — |
| `P2_060f_docsim_table_5050_fusion` | > | — |
| `P2_060j_docsim_text_all_anomask` | > | — |
| `P2_060k_docsim_mc_4thterm_all` | > | — |
| `P2_060l_docsim_h61_preproc_text` | > | — |
| `P2_060m_docsim_bbox_table` | > | — |
| `P2_060n_docsim_bbox_text_table` | > | — |
| `P2_060o_docsim_bbox_table_max` | > | — |
| `P2_060p_docsim_bbox_table_min` | > | — |
| `P2_060q_docsim_bbox_min_all_extend` | > | — |
| `P2_060r_docsim_bbox_min_h61_crop` | > | — |
| `P2_060s_docsim_bbox_min_padded` | > | — |
| `P2_060t_docsim_bbox_min_filter_small` | > | — |
| `P2_060u_docsim_bbox_table_median` | > | — |
| `P2_060v_docsim_bbox_table_p10` | > | — |
| `P2_060w_docsim_bbox_min_anm_extend` | > | — |
| `P2_060x_docsim_h4e_5050_formula` | > | — |
| `P2_060y_docsim_paragraph_text` | > | — |

## Hash

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_066_hashes` | > | — |

## Hash Metric

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_067_pdq_256bit_hash` | > | — |

## Per Element Structure

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_068_spatially_aware_hash_grid_min` | > | — |
| `P1_107_table_cell_supplement_all_10` | > | — |
| `P1_107b_beta_table_cell_20` | > | — |
| `P1_107c_beta_table_cell_30` | > | — |
| `P1_108_formula_bbox_supplement_10` | > | — |
| `P1_109_sobel_edge_supplement_05` | > | — |
| `P1_110_multiscale_sobel_supplement_05` | > | — |
| `P1_111_ssim_page_supplement_05` | > | — |

## Nr Iqa Classical

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_106_frontiers_diqa_hand_crafted` | > | — |

## Self Uncertainty

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_106b_iq_fusion_formula` | > | — |
| `P1_106c_gamma_30_formula` | > | — |
| `P1_106d_iq_fusion_table` | > | — |
| `P1_106e_gamma_40_formula` | > | — |
| `P1_106f_combined_formula_40_table_10` | > | — |
| `P1_106g_iq_fusion_text` | > | — |
| `P1_106h_gamma_formula_50` | > | — |
| `P1_106j_gamma_table_20` | > | — |
| `P1_106k_gamma_formula_60` | > | — |
| `P1_106m_iq_fusion_all_05` | > | — |
| `P1_106n_iq_per_formula_bbox_min` | > | — |
| `P1_213_sliding_window_heatmap_area_weighted` | > | — |
| `P1_215_qwen_logprob_confidence_mean` | > | — |
| `P1_215b_docsim_p_plus_confidence_min_fusion` | > | — |
| `P1_215c_alpha030_fusion` | > | — |
| `P1_215d_alpha040_fusion` | > | — |
| `P1_215e_alpha_variant_conditional` | > | — |
| `P1_215f_alpha050_fusion` | > | — |
| `P1_215g_window_k5` | > | — |
| `P1_215h_window_k20` | > | — |
| `P1_215i_variant_alpha_and_k` | > | — |
| `P1_215j_window_k3` | > | — |
| `P1_215k_variant_k3_20` | > | — |
| `P1_215l_shannon_top_logprobs` | > | — |
| `P1_215m_variant_k2_20` | > | — |
| `P1_215n_three_way_fusion` | > | — |
| `P1_215o_alpha_refinement_at_k3` | > | — |
| `P1_215r_per_variant_source` | > | — |
| `P1_224_qwen_token_logprob_window_min` | > | — |
| `P1_224b_per_bbox_shannon_min` | > | — |
| `P1_224c_class_filtered_per_bbox_max_entropy_min` | > | — |
| `P1_224d_variant_conditional_source` | > | — |
| `P1_224e_token_class_filtered_window_min` | > | — |

## Composite

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_112_ssim_page_supplement_10` | > | — |
| `P1_113_ssim_page_supplement_15` | > | — |
| `P1_114_ssim_page_supplement_20` | > | — |
| `P1_116_ssim_variant_specific_beta` | > | — |
| `P1_118_haarpsi_variant_specific_beta` | > | — |
| `P1_119_msssim_variant_specific_beta` | > | — |
| `P1_120_ssim_text_supplement` | > | — |
| `P1_121_ssim_text_beta10` | > | — |
| `P1_122_ssim_text_beta15` | > | — |
| `P1_123_ssim_formula_beta05` | > | — |
| `P1_124_ssim_stack_text15_formula05` | > | — |
| `P1_125_ssim_text_beta20` | > | — |
| `P1_126_dinov2_patch_text_beta05` | > | — |
| `P1_127_ssim_all_beta20` | > | — |
| `P1_128_ssim_table_beta05` | > | — |
| `P1_129_ssim_formula_beta10` | > | — |
| `P1_131_all_nm_tablecell_beta05` | > | — |
| `P1_132_ssim_table_beta10` | > | — |
| `P1_133_ssim_table_beta15` | > | — |
| `P1_134_ssim_allnm_beta15` | > | — |
| `P1_135_ssim_table_beta20` | > | — |
| `P1_136_ssim_all_beta20` | > | — |
| `P1_137_content_elem_p10_all` | > | — |
| `P1_138_ssim_all_beta25` | > | — |
| `P1_139_elem_p10_all_beta15` | > | — |
| `P1_140_elem_p10_allnm_beta15` | > | — |
| `P1_141_elem_p10_text_beta10` | > | — |
| `P1_142_elem_mean_all` | > | — |

## Preprocessing

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P1_155_lama_decoration_removal` | > | — |

## Perceptual Latent

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P2_059_pieapp_perceptual_metric` | > | — |

## Ocr Perceptual

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P2_104_recognition_feature_distance` | > | — |

## Layout Detector

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `P2_193_pp_doclayout_disagreement` | > | — |

## Baseline

| Method ID | Description | Spearman (`all` variant) |
|---|---|---|
| `baseline` | > | — |

