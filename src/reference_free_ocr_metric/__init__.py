"""Reference-free OCR metric for evaluating OCR quality without ground truth."""

__version__ = "0.2.0"

from reference_free_ocr_metric.metrics.base import BaseMetric
from reference_free_ocr_metric.metrics.clip_compare.clip_similarity import CLIPSimilarityMetric
from reference_free_ocr_metric.metrics.lm_perplexity.perplexity_scorer import LMPerplexityMetric
from reference_free_ocr_metric.metrics.multi_metric.visual_reconstruction import VisualReconstructionMetric
from reference_free_ocr_metric.metrics.vlm_compare.vlm_similarity import VLMSimilarityMetric
from reference_free_ocr_metric.ocr.base import BaseOCRClient
from reference_free_ocr_metric.ocr.qwen_client import QwenVLClient

__all__ = [
    "BaseMetric",
    "BaseOCRClient",
    "CLIPSimilarityMetric",    "LMPerplexityMetric",
    "QwenVLClient",
    "VLMSimilarityMetric",
    "VisualReconstructionMetric",
    "__version__",
]
