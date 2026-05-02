"""OCR client modules."""

from reference_free_ocr_metric.ocr.base import BaseOCRClient
from reference_free_ocr_metric.ocr.qwen_client import QwenVLClient

__all__ = ["BaseOCRClient", "QwenVLClient"]
