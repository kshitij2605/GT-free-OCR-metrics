"""Abstract base class for OCR clients."""

from abc import ABC, abstractmethod

from reference_free_ocr_metric.reconstruction.html_parser import ParsedDocument


class BaseOCRClient(ABC):
    """Abstract base class for OCR clients."""

    @abstractmethod
    def ocr(self, image_path: str, **kwargs) -> str:
        """Run OCR on an image and return the raw output string."""

    @abstractmethod
    def parse(self, raw_output: str) -> ParsedDocument:
        """Parse raw OCR output into a ParsedDocument."""
