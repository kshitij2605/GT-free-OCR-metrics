"""Tests for BaseOCRClient ABC."""
import pytest
from reference_free_ocr_metric.ocr.base import BaseOCRClient
from reference_free_ocr_metric.reconstruction.html_parser import ParsedDocument


def test_base_ocr_client_is_abstract():
    with pytest.raises(TypeError):
        BaseOCRClient()


def test_concrete_subclass_must_implement_ocr():
    class Incomplete(BaseOCRClient):
        def parse(self, raw_output):
            return ParsedDocument([], [], [])
    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_subclass_must_implement_parse():
    class Incomplete(BaseOCRClient):
        def ocr(self, image_path, **kwargs):
            return ""
    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_subclass_works():
    class FakeClient(BaseOCRClient):
        def ocr(self, image_path, **kwargs):
            return "fake output"
        def parse(self, raw_output):
            return ParsedDocument([], [], [])
    client = FakeClient()
    assert client.ocr("test.png") == "fake output"
    result = client.parse("anything")
    assert isinstance(result, ParsedDocument)
