"""VLM-based comparison metric using Qwen3-VL for visual similarity scoring."""

import base64
import re
from io import BytesIO
from typing import Any

import httpx
from openai import OpenAI
from PIL import Image

from reference_free_ocr_metric.metrics.base import BaseMetric


class VLMSimilarityMetric(BaseMetric):
    """Metric that uses a VLM to judge visual similarity between
    an original document image and a reconstructed text-only image."""

    def __init__(self, api_base: str, api_key: str, model_name: str) -> None:
        """Initialize with an OpenAI-compatible API base URL, key, and model name."""
        self._client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False),
        )
        self._model_name = model_name

    @property
    def name(self) -> str:
        """Return the metric identifier string."""
        return "vlm_similarity"

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode a PIL image to a base64 JPEG string."""
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_score(self, response: str) -> float:
        """Extract a float score from a VLM response.

        Tries several patterns:
        1. N/10 fraction -> divide by 10
        2. Float between 0 and 1
        3. Any float, normalized if > 1
        4. Default 0.5 if nothing found
        """
        # Pattern 1: N/10
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", response)
        if match:
            return float(match.group(1)) / 10.0

        # Pattern 2: Float between 0 and 1
        for match in re.finditer(r"\b(0\.\d+|1\.0|0|1)\b", response):
            value = float(match.group(1))
            if 0.0 <= value <= 1.0:
                return value

        # Pattern 3: Any float, normalize if > 1
        match = re.search(r"(\d+\.\d+)", response)
        if match:
            value = float(match.group(1))
            if value > 1.0:
                return min(value / 10.0, 1.0)
            return value

        # Pattern 4: Fallback
        return 0.5

    def compare_images(self, original: Image.Image, reconstructed: Image.Image) -> dict:
        """Compare original and reconstructed document images using VLM.

        Args:
            original: The original document image (with images/tables masked).
            reconstructed: The reconstructed text-only image from OCR output.

        Returns:
            Dict with 'vlm_score' (float) and 'raw_response' (str).
        """
        original_b64 = self._encode_image_to_base64(original)
        reconstructed_b64 = self._encode_image_to_base64(reconstructed)

        prompt = (
            "Compare these two document images. The first is the original document "
            "(with images and tables removed). The second is a reconstruction from OCR "
            "output showing only the extracted text at detected positions. Rate how well "
            "the text in the reconstruction matches the original on a scale of 0.0 to 1.0, "
            "where 1.0 means perfectly matching text content and positioning. Respond with "
            "ONLY a single decimal number."
        )

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{original_b64}",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{reconstructed_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        raw_response = response.choices[0].message.content
        score = self._parse_score(raw_response)

        return {"vlm_score": score, "raw_response": raw_response}

    def compute(self, ocr_output: str, **kwargs: Any) -> float:
        """Compute VLM similarity score for OCR output.

        Requires 'original_image' and 'reconstructed_image' as keyword arguments,
        both as PIL Image objects.

        Args:
            ocr_output: The OCR text output (unused directly; images are pre-rendered).
            **kwargs: Must include 'original_image' and 'reconstructed_image'.

        Returns:
            Float score between 0.0 and 1.0.
        """
        original = kwargs["original_image"]
        reconstructed = kwargs["reconstructed_image"]
        result = self.compare_images(original, reconstructed)
        return result["vlm_score"]
