"""Tests for OCR metrics."""

from reference_free_ocr_metric import BaseMetric, __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.2.0"


def test_all_metrics_importable_from_package():
    """All metric classes are importable from the top-level package."""
    from reference_free_ocr_metric import (
        CLIPSimilarityMetric,
        LMPerplexityMetric,
        QwenVLClient,
        VLMSimilarityMetric,
        VisualReconstructionMetric,
    )
    assert all(cls is not None for cls in [
        CLIPSimilarityMetric,
        LMPerplexityMetric,
        QwenVLClient,
        VLMSimilarityMetric,
        VisualReconstructionMetric,
    ])


def test_all_metrics_importable_from_metrics_package():
    """All metric classes are importable from the metrics sub-package."""
    from reference_free_ocr_metric.metrics import (
        CLIPSimilarityMetric,
        LMPerplexityMetric,
        VLMSimilarityMetric,
        VisualReconstructionMetric,
    )
    for cls in [CLIPSimilarityMetric, LMPerplexityMetric,
                VLMSimilarityMetric, VisualReconstructionMetric]:
        assert issubclass(cls, BaseMetric)


def test_base_metric_is_abstract():
    """Test that BaseMetric cannot be instantiated directly."""
    import pytest

    with pytest.raises(TypeError):
        BaseMetric()


class DummyMetric(BaseMetric):
    """A dummy metric for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def compute(self, ocr_output: str, **kwargs) -> float:
        return len(ocr_output) / 100.0


def test_dummy_metric():
    """Test that a concrete metric can be instantiated and used."""
    metric = DummyMetric()
    assert metric.name == "dummy"
    assert metric.compute("hello world") == 0.11
    assert repr(metric) == "DummyMetric()"
