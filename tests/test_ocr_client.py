"""Tests for Qwen VLM client."""
from unittest.mock import MagicMock, patch

from reference_free_ocr_metric.ocr.qwen_client import (
    QwenVLClient,
    _DEFAULT_MAX_PIXELS,
    _DEFAULT_MIN_PIXELS,
    _MAX_RETRIES,
    smart_resize,
)


@patch("reference_free_ocr_metric.ocr.qwen_client.openai.OpenAI")
def test_client_initialization(mock_openai_cls):
    client = QwenVLClient(
        api_base="http://test:9000/v1",
        api_key="test_key",
        model_name="test-model",
    )
    assert client.model_name == "test-model"
    mock_openai_cls.assert_called_once_with(base_url="http://test:9000/v1", api_key="test_key")


@patch("reference_free_ocr_metric.ocr.qwen_client.openai.OpenAI")
def test_ocr_returns_html(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "<html><body><p data-bbox='0 0 100 20'>This is sufficient text content for passing</p></body></html>"
    mock_client.chat.completions.create.return_value = mock_response

    client = QwenVLClient(
        api_base="http://test:9000/v1",
        api_key="test_key",
        model_name="test-model",
    )
    client._encode_image = MagicMock(return_value=("fake_base64", "image/png"))
    result = client.ocr("dummy_path.png")
    assert "<html>" in result


@patch("reference_free_ocr_metric.ocr.qwen_client.openai.OpenAI")
def test_ocr_sends_pixels_in_content_block_and_extra_body(mock_openai_cls):
    """Pixels are sent in both content block (DashScope) and extra_body (vLLM)."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "<p>Sufficient text content here for test</p>"
    mock_client.chat.completions.create.return_value = mock_response

    client = QwenVLClient(
        api_base="http://test:9000/v1",
        api_key="test_key",
        model_name="test-model",
    )
    client._encode_image = MagicMock(return_value=("fake_base64", "image/png"))
    client.ocr("dummy.png")

    call_args = mock_client.chat.completions.create.call_args
    # Check extra_body
    extra_body = call_args.kwargs.get("extra_body")
    assert extra_body is not None
    mm_kwargs = extra_body["mm_processor_kwargs"]
    assert mm_kwargs["min_pixels"] == _DEFAULT_MIN_PIXELS
    assert mm_kwargs["max_pixels"] == _DEFAULT_MAX_PIXELS
    # Check content block
    messages = call_args.kwargs.get("messages")
    image_content = messages[0]["content"][0]
    assert image_content["min_pixels"] == _DEFAULT_MIN_PIXELS
    assert image_content["max_pixels"] == _DEFAULT_MAX_PIXELS
    # Check max_tokens
    assert call_args.kwargs.get("max_tokens") == 8192


@patch("reference_free_ocr_metric.ocr.qwen_client.openai.OpenAI")
def test_ocr_uses_correct_mime_type_in_data_url(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "<p>Real text content here for the test</p>"
    mock_client.chat.completions.create.return_value = mock_response

    client = QwenVLClient(
        api_base="http://test:9000/v1",
        api_key="test_key",
        model_name="test-model",
    )
    # Simulate PNG file
    client._encode_image = MagicMock(return_value=("fake_base64", "image/png"))
    client.ocr("test.png")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages")
    url = messages[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")

    # Simulate JPEG file
    client._encode_image = MagicMock(return_value=("fake_base64", "image/jpeg"))
    client.ocr("test.jpg")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages")
    url = messages[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


@patch("reference_free_ocr_metric.ocr.qwen_client.openai.OpenAI")
def test_ocr_retries_on_image_only_response(mock_openai_cls):
    """When response has no real text, retries up to _MAX_RETRIES times."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    # First two calls return image-only div, third returns text
    image_only = MagicMock()
    image_only.choices = [MagicMock()]
    image_only.choices[0].message.content = '<div class="image"></div>'

    good_response = MagicMock()
    good_response.choices = [MagicMock()]
    good_response.choices[0].message.content = "<p>Module 3 Places and activities with enough text</p>"

    mock_client.chat.completions.create.side_effect = [
        image_only,
        image_only,
        good_response,
    ]

    client = QwenVLClient(
        api_base="http://test:9000/v1",
        api_key="test_key",
        model_name="test-model",
    )
    client._encode_image = MagicMock(return_value=("fake_base64", "image/png"))
    result = client.ocr("test.png")

    assert "Module 3" in result
    assert mock_client.chat.completions.create.call_count == 3


@patch("reference_free_ocr_metric.ocr.qwen_client.openai.OpenAI")
def test_ocr_returns_last_attempt_if_all_retries_fail(mock_openai_cls):
    """If all retries produce image-only output, returns the last attempt."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '<div class="image"></div>'
    mock_client.chat.completions.create.return_value = mock_response

    client = QwenVLClient(
        api_base="http://test:9000/v1",
        api_key="test_key",
        model_name="test-model",
    )
    client._encode_image = MagicMock(return_value=("fake_base64", "image/png"))
    result = client.ocr("test.png")

    assert mock_client.chat.completions.create.call_count == _MAX_RETRIES
    assert "image" in result


@patch("reference_free_ocr_metric.ocr.qwen_client.openai.OpenAI")
def test_ocr_no_retry_when_text_present(mock_openai_cls):
    """No retry when first response has sufficient text."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "<p>This is real OCR text content that is long enough</p>"
    mock_client.chat.completions.create.return_value = mock_response

    client = QwenVLClient(
        api_base="http://test:9000/v1",
        api_key="test_key",
        model_name="test-model",
    )
    client._encode_image = MagicMock(return_value=("fake_base64", "image/png"))
    result = client.ocr("test.png")

    assert "real OCR text" in result
    assert mock_client.chat.completions.create.call_count == 1


def test_smart_resize_within_budget():
    """Image already within min/max pixels stays close to original size."""
    h, w = 1000, 1000  # 1M pixels, within default budget
    new_h, new_w = smart_resize(h, w)
    assert new_h % 32 == 0
    assert new_w % 32 == 0
    assert new_h * new_w <= _DEFAULT_MAX_PIXELS
    assert new_h * new_w >= _DEFAULT_MIN_PIXELS


def test_smart_resize_scales_down_large_image():
    """Large image is scaled down to fit within max_pixels."""
    h, w = 5000, 6000  # 30M pixels
    new_h, new_w = smart_resize(h, w, max_pixels=2097152)
    assert new_h * new_w <= 2097152
    assert new_h % 32 == 0
    assert new_w % 32 == 0


def test_smart_resize_scales_up_small_image():
    """Small image is scaled up to meet min_pixels."""
    h, w = 100, 100  # 10K pixels
    new_h, new_w = smart_resize(h, w, min_pixels=524288)
    assert new_h * new_w >= 524288
    assert new_h % 32 == 0
    assert new_w % 32 == 0


def test_smart_resize_preserves_aspect_ratio():
    """Aspect ratio is approximately preserved after resize."""
    h, w = 4000, 2000  # 2:1 aspect, 8M pixels (above max)
    new_h, new_w = smart_resize(h, w, max_pixels=2097152)
    original_ratio = h / w
    new_ratio = new_h / new_w
    assert abs(original_ratio - new_ratio) < 0.1


def test_default_prompt_contains_bbox_instructions():
    """Default prompt includes qwenvl html trigger and data-bbox instructions."""
    from reference_free_ocr_metric.ocr.qwen_client import _DEFAULT_PROMPT

    assert "qwenvl html" in _DEFAULT_PROMPT
    assert "data-bbox" in _DEFAULT_PROMPT
