"""Base class for reference-free OCR metrics."""

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Abstract base class for reference-free OCR quality metrics.

    Subclasses should implement the `compute` method to calculate
    a quality score from OCR output without requiring ground truth.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass

    @abstractmethod
    def compute(self, ocr_output: str, **kwargs: Any) -> float:
        """Compute the metric score for the given OCR output.

        Args:
            ocr_output: The text output from an OCR system.
            **kwargs: Additional arguments specific to the metric.

        Returns:
            A float score representing the quality of the OCR output.
            Higher scores should indicate better quality.
        """
        pass

    def __repr__(self) -> str:
        """Return a class-name string representation of the metric."""
        return f"{self.__class__.__name__}()"
