"""Metrics for reference-free OCR quality evaluation."""

from reference_free_ocr_metric.metrics.base import BaseMetric
from reference_free_ocr_metric.metrics.clip_compare.clip_similarity import CLIPSimilarityMetric
from reference_free_ocr_metric.metrics.lm_perplexity.perplexity_scorer import LMPerplexityMetric
from reference_free_ocr_metric.metrics.multi_metric.visual_reconstruction import VisualReconstructionMetric
from reference_free_ocr_metric.metrics.vlm_compare.vlm_similarity import VLMSimilarityMetric

__all__ = [
    "BaseMetric",
    "CLIPSimilarityMetric",
    "LMPerplexityMetric",
    "VLMSimilarityMetric",
    "VisualReconstructionMetric",
]
