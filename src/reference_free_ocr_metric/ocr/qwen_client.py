"""Qwen VL OCR client using OpenAI-compatible API."""

import base64
import io
import math
import re
from pathlib import Path

import openai

from reference_free_ocr_metric.ocr.base import BaseOCRClient
from reference_free_ocr_metric.reconstruction.html_parser import ParsedDocument, QwenVLHTMLParser

# Default pixel budget — matches the original Qwen3-VL cookbook.
_DEFAULT_MIN_PIXELS = 512 * 32 * 32  # 524,288
_DEFAULT_MAX_PIXELS = 2048 * 32 * 32  # 2,097,152

# Extended prompt that produces per-element bounding boxes.
# Based on the prompt from agentic_document_parser's html_text_extractor.
_DEFAULT_PROMPT = r"""qwenvl html

You are an expert document parsing assistant. Extract ALL content from the image with precise bounding box coordinates.

**CRITICAL VISUAL DETECTION RULES:**

Step 1: Parse the current page and identify EVERY distinct region.

Step 2: For each region, classify it as TEXT or VISUAL:
- TEXT: Pure text content (headings, paragraphs, bullet lists, captions, footnotes)
- VISUAL: Any non-textual content including:
  * Bar charts, line charts, pie charts, area charts, histograms
  * Graphs with axes, gridlines, data points
  * Flowcharts, process diagrams, organizational charts, timelines
  * Infographics with icons and graphics
  * Photos, illustrations, logos, icons
  * Tables with grid lines or row/column structure
  * Complex layouts with multiple visual sections

Step 3: Output format - wrap each element in a <div> with:
- `class`: "text", "formula", "image", or "table"
- `data-bbox`: coordinates as "x1 y1 x2 y2" (normalized 0-1000)

**IMPORTANT VISUAL DETECTION GUIDELINES:**

1. **Split visuals into separate elements**: When multiple visuals are stacked vertically or separated by whitespace (e.g., bar graph + timeline + infographic), emit SEPARATE annotations for EACH visual block. Never merge adjacent visuals into one bounding box.

2. **Detect ALL charts and graphs**: Bar charts, line charts, timelines, process diagrams, organizational charts - each gets its own annotation with class="image".

3. **Complete bounding boxes**: For graphs/charts, coordinates must cover the FULL extent including:
   - Axis labels and titles
   - Legends
   - Data region
   - Any annotations or callouts that are part of the chart

4. **Complex diagrams**: If a diagram has multiple connected parts (timeline with multiple nodes, flowchart with multiple steps), treat it as ONE image with a bbox covering the entire diagram.

5. **Do NOT tag as images**:
   - Pure text paragraphs or headings (even with colored backgrounds)
   - Single-line text callouts
   - Text banners without pictorial elements

6. **Separate each distinct visual**: If there are 2 charts on a page, output 2 image elements. If there's a chart AND a timeline, output 2 image elements.

**FORMULA / EQUATION RULES:**

- Any standalone display math (block equations, numbered equations) → `class="formula"`, include the full LaTeX inside the div wrapped in `$$...$$`
- Inline math embedded within a sentence stays inside its parent `class="text"` div, wrapped in `$...$`
- Do NOT use `class="equation"` — always use `class="formula"` for display math

**OUTPUT FORMAT:**

```html
<div class="text" data-bbox="50 30 950 100">Document Title</div>
<div class="text" data-bbox="50 110 950 200">Let $f$ be holomorphic in $\Omega$. Then</div>
<div class="formula" data-bbox="50 210 950 260">$$\int_{\gamma} f(z)\,dz = 0$$</div>
<div class="image" data-bbox="50 270 500 650">
  <img data-bbox="50 270 500 650" alt="Bar chart showing yearly revenue growth"/>
</div>
<div class="table" data-bbox="50 660 950 950">
  <table>...</table>
</div>
```

**FOR THIS PAGE:**
- Scan from top to bottom
- Identify EVERY visual element (charts, diagrams, timelines, infographics)
- Each visual gets its OWN <div class="image"> with accurate bbox
- Text content goes in <div class="text"> — inline math stays inside as $...$
- Display/block equations go in <div class="formula"> with $$...$$ LaTeX
- Tables go in <div class="table">

Extract ALL content with accurate bounding boxes. Do not skip any visual elements."""

# Minimum non-tag character count to consider an OCR response as having real
# text content (vs. image-only divs).
_MIN_TEXT_CHARS = 20

# Max retries when model returns image-only output.
_MAX_RETRIES = 3


def smart_resize(
    height: int,
    width: int,
    min_pixels: int = _DEFAULT_MIN_PIXELS,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    factor: int = 32,
) -> tuple[int, int]:
    """Compute target (h, w) that fits within pixel budget.

    Matches the Qwen3-VL cookbook logic: rounds to nearest multiple of
    *factor*, then scales down if above max_pixels or up if below min_pixels.
    This is for **display purposes only** — the vLLM server handles actual
    resizing via mm_processor_kwargs.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = int(math.floor(height / beta / factor) * factor)
        w_bar = int(math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = int(math.ceil(height * beta / factor) * factor)
        w_bar = int(math.ceil(width * beta / factor) * factor)
    return h_bar, w_bar


def _strip_tags(html: str) -> str:
    """Return plain text from HTML by removing tags, markdown fences, and whitespace."""
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"```\w*", "", text)  # strip markdown code fences
    text = re.sub(r"\s+", "", text)  # collapse all whitespace
    return text




class QwenVLClient(BaseOCRClient):
    """Client for Qwen3-VL OCR via OpenAI-compatible API.

    Mirrors the original Qwen3-VL document parsing cookbook. Uses
    content-block min/max_pixels (for DashScope compatibility) and also
    passes them via extra_body.mm_processor_kwargs (for vLLM).
    """

    def __init__(self, api_base: str, api_key: str, model_name: str) -> None:
        """Initialize the Qwen OCR client with API base URL, key, and model name."""
        self.model_name = model_name
        self._client = openai.OpenAI(base_url=api_base, api_key=api_key)

    def ocr(  # type: ignore[override]
        self,
        image_path: str,
        prompt: str = _DEFAULT_PROMPT,
        min_pixels: int = _DEFAULT_MIN_PIXELS,
        max_pixels: int = _DEFAULT_MAX_PIXELS,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> str:
        """Run OCR on an image and return the HTML string response.

        Image is resized client-side to fit within min/max_pixels before encoding,
        reducing upload size. min_pixels/max_pixels are also sent in the content block
        (matching the original cookbook for DashScope) and extra_body (for vLLM).

        temperature=0.0 (default) gives deterministic, reproducible output.
        If the response contains only image divs, retries up to _MAX_RETRIES
        times with the same prompt (model is non-deterministic for some images).
        """
        base64_image, mime_type = self._encode_image(image_path, min_pixels, max_pixels)

        content = ""
        for _ in range(_MAX_RETRIES):
            content = self._call_vlm(
                base64_image, mime_type, prompt, min_pixels, max_pixels,
                max_tokens, temperature,
            )
            if len(_strip_tags(content)) >= _MIN_TEXT_CHARS:
                return content

        # Return last attempt even if text is sparse.
        return content

    def ocr_with_logprobs(
        self,
        image_path: str,
        prompt: str = _DEFAULT_PROMPT,
        min_pixels: int = _DEFAULT_MIN_PIXELS,
        max_pixels: int = _DEFAULT_MAX_PIXELS,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_logprobs: int = 5,
    ) -> tuple[str, list[dict]]:
        """Run OCR and also return per-output-token logprobs.

        Returns (html_content, logprobs_list) where logprobs_list is a list of
        {"token": str, "logprob": float, "top_logprobs": [{"token", "logprob"}, ...]}.

        Same retry logic as ocr() — keeps the attempt whose stripped text is
        long enough; returns last attempt's logprobs if all are sparse.
        """
        base64_image, mime_type = self._encode_image(image_path, min_pixels, max_pixels)
        content, logprobs = "", []
        for _ in range(_MAX_RETRIES):
            content, logprobs = self._call_vlm_with_logprobs(
                base64_image, mime_type, prompt, min_pixels, max_pixels,
                max_tokens, temperature, top_logprobs,
            )
            if len(_strip_tags(content)) >= _MIN_TEXT_CHARS:
                return content, logprobs
        return content, logprobs

    def parse(self, raw_output: str) -> ParsedDocument:
        """Parse Qwen HTML output into a ParsedDocument."""
        return QwenVLHTMLParser().parse(raw_output)

    def _call_vlm_with_logprobs(
        self,
        base64_image: str,
        mime_type: str,
        prompt: str,
        min_pixels: int,
        max_pixels: int,
        max_tokens: int,
        temperature: float,
        top_logprobs: int,
    ) -> tuple[str, list[dict]]:
        """Send a single VLM request and return (html_content, logprobs_list)."""
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            extra_body={"mm_processor_kwargs": {"min_pixels": min_pixels, "max_pixels": max_pixels}},
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        lp_list: list[dict] = []
        if choice.logprobs and choice.logprobs.content:
            for tok in choice.logprobs.content:
                lp_list.append({
                    "token": tok.token,
                    "logprob": tok.logprob,
                    "top_logprobs": [
                        {"token": t.token, "logprob": t.logprob}
                        for t in (tok.top_logprobs or [])
                    ],
                })
        return content, lp_list

    def _call_vlm(
        self,
        base64_image: str,
        mime_type: str,
        prompt: str,
        min_pixels: int,
        max_pixels: int,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """Send a single VLM request and return the response text."""
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={
                "mm_processor_kwargs": {
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            },
        )
        result = response.choices[0].message.content
        assert result is not None, "VLM returned empty response"
        return result

    def _encode_image(
        self,
        image_path: str,
        min_pixels: int = _DEFAULT_MIN_PIXELS,
        max_pixels: int = _DEFAULT_MAX_PIXELS,
    ) -> tuple[str, str]:
        """Base64-encode an image, resizing only when far over the pixel budget.

        For images within 4× max_pixels, sends original bytes unchanged so the
        vLLM server applies its own (high-quality) resize via mm_processor_kwargs.
        This preserves fine text and math detail for document pages.

        For very large images (e.g. 27 MP magazine scans at 12× budget), resizes
        client-side to max_pixels as JPEG to avoid server timeout on the upload.
        """
        from PIL import Image as _Image

        img = _Image.open(image_path).convert("RGB")
        h, w = img.height, img.width
        suffix = Path(image_path).suffix.lower()
        is_jpeg = suffix in (".jpg", ".jpeg")

        if h * w <= 4 * max_pixels:
            # Within 4× budget — send original bytes; server handles resize.
            raw = Path(image_path).read_bytes()
            mime = "image/jpeg" if is_jpeg else "image/png"
            return base64.b64encode(raw).decode("utf-8"), mime

        # Very large image: resize client-side to avoid server timeout.
        th, tw = smart_resize(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
        img = img.resize((tw, th), _Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
