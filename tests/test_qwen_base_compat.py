"""Tests that QwenVLClient satisfies BaseOCRClient interface."""
from unittest.mock import patch

from reference_free_ocr_metric.ocr.base import BaseOCRClient
from reference_free_ocr_metric.ocr.qwen_client import QwenVLClient
from reference_free_ocr_metric.reconstruction.html_parser import ParsedDocument


def test_qwen_is_base_ocr_client():
    assert issubclass(QwenVLClient, BaseOCRClient)


@patch("reference_free_ocr_metric.ocr.qwen_client.openai.OpenAI")
def test_qwen_has_parse_method(mock_openai):
    client = QwenVLClient(api_base="http://fake", api_key="key", model_name="model")
    sample_html = '<div class="text" data-bbox="100 50 500 80">Hello</div>'
    result = client.parse(sample_html)
    assert isinstance(result, ParsedDocument)
    assert len(result.text_elements) >= 1
    assert result.text_elements[0].text == "Hello"
