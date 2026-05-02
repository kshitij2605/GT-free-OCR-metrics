# OCR Model Configuration: Qwen3.5-122B-A10B

All values are sourced directly from `src/reference_free_ocr_metric/ocr/qwen_client.py` and `scripts/run_experiment.py`.

---

## Endpoint

| Parameter | Value |
|-----------|-------|
| API base | `${OCR_ENDPOINT_URL}/v1` |
| API key | `tensorflow` |
| Model name | `Qwen/Qwen3.5-122B-A10B` |
| Protocol | OpenAI-compatible chat completions (`/v1/chat/completions`) |
| Backend | vLLM |

---

## Inference Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_tokens` | `8192` | Maximum response length in tokens |
| `temperature` | `0.0` | Fully deterministic / greedy decoding |

`temperature=0.0` means the model always picks the highest-probability token. Output is reproducible for the same image and prompt.

---

## Image Pixel Budget

Controls how the vLLM server resizes the image before passing it to the vision encoder. Sent via two channels simultaneously for compatibility with both DashScope and vLLM backends:

1. In the content block (`min_pixels` / `max_pixels` fields alongside `image_url`)
2. In `extra_body.mm_processor_kwargs`

| Parameter | Value | Pixels |
|-----------|-------|--------|
| `min_pixels` | `512 × 32 × 32` | 524,288 |
| `max_pixels` | `2048 × 32 × 32` | 2,097,152 |

The `factor=32` value means all dimensions are rounded to the nearest multiple of 32 (required by the Qwen3-VL vision encoder's patch grid).

### Smart Resize Logic

The client does **no client-side resizing** — images are sent as raw base64. The vLLM server applies `smart_resize` internally using the same logic as the Qwen3-VL cookbook:

```python
# Round to nearest multiple of factor
h_bar = max(factor, round(height / factor) * factor)
w_bar = max(factor, round(width / factor) * factor)

# Scale down if above max_pixels
if h_bar * w_bar > max_pixels:
    beta = sqrt((height * width) / max_pixels)
    h_bar = floor(height / beta / factor) * factor
    w_bar = floor(width / beta / factor) * factor

# Scale up if below min_pixels
elif h_bar * w_bar < min_pixels:
    beta = sqrt(min_pixels / (height * width))
    h_bar = ceil(height * beta / factor) * factor
    w_bar = ceil(width * beta / factor) * factor
```

**Practical implication**: a full-page scan at 300 DPI (~2480×3508 px = 8.7M pixels) is downscaled ~2× to fit the 2.1M pixel budget. Dense pages (newspaper, technical documents) lose fine detail at high DPI. Increasing `max_pixels` to `8192×32×32` (~8.4M, 8192 visual tokens) would improve quality for dense pages at the cost of higher VRAM and latency.

---

## Input Format

Images are base64-encoded and sent inline as a data URI. MIME type is inferred from file extension:

| Extension | MIME type |
|-----------|-----------|
| `.png` | `image/png` |
| `.jpg` / `.jpeg` | `image/jpeg` |
| `.webp` | `image/webp` |
| `.gif` | `image/gif` |
| `.bmp` | `image/bmp` |
| *(other)* | `image/jpeg` (default) |

---

## Prompt

The model is prompted with `"qwenvl html"` as the trigger phrase (activates the model's structured HTML output mode), followed by detailed instructions. Key directives:

- **Output format**: `<div>` elements with `class` and `data-bbox="x1 y1 x2 y2"` (coordinates normalized 0–1000)
- **Element classes**: `text`, `formula`, `image`, `table`
- **Formula encoding**: display math → `$$...$$` inside `class="formula"` div; inline math → `$...$` inside parent `class="text"` div
- **Image detection**: each chart, diagram, or figure gets its own `class="image"` div; adjacent visuals must be split into separate elements
- **Scan order**: top to bottom, exhaustive — no elements skipped

Full prompt is defined as `_DEFAULT_PROMPT` in `qwen_client.py` and passed as the text content of the user message alongside the image.

---

## Retry Behaviour

If the model returns a response with fewer than **20 non-tag characters** (i.e., only `<div class="image">` elements with no text content), the client retries automatically up to **3 times** with the same prompt.

| Constant | Value | Meaning |
|----------|-------|---------|
| `_MIN_TEXT_CHARS` | `20` | Minimum non-tag chars to accept a response |
| `_MAX_RETRIES` | `3` | Max retry attempts before returning last result |

After 3 failed attempts, the last (sparse) response is returned as-is.

---

## Message Structure

```
POST /v1/chat/completions
{
  "model": "Qwen/Qwen3.5-122B-A10B",
  "max_tokens": 8192,
  "temperature": 0.0,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "min_pixels": 524288,
          "max_pixels": 2097152,
          "image_url": { "url": "data:<mime>;base64,<b64>" }
        },
        {
          "type": "text",
          "text": "<_DEFAULT_PROMPT>"
        }
      ]
    }
  ],
  "extra_body": {
    "mm_processor_kwargs": {
      "min_pixels": 524288,
      "max_pixels": 2097152
    }
  }
}
```

---

## Changing Configuration

All defaults are overridable per-call via `QwenVLClient.ocr()` keyword arguments:

```python
client.ocr(
    image_path="page.png",
    max_pixels=8192 * 32 * 32,   # increase for dense pages
    max_tokens=16384,             # increase for very long pages
    temperature=0.0,              # keep 0.0 for reproducibility
    prompt=custom_prompt,         # override default prompt
)
```
