"""OpenCLIP-based image similarity metric for OCR quality evaluation."""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from reference_free_ocr_metric.metrics.base import BaseMetric


class CLIPSimilarityMetric(BaseMetric):
    """Computes cosine similarity between CLIP embeddings of original and reconstructed images.

    Both images are preprocessed to grayscale with sharpening and contrast
    optimization before comparison.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        """Initialize with model name and pretrained weights identifier."""
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess_val = None

    @property
    def name(self) -> str:
        """Return the metric identifier string."""
        return "clip_similarity"

    def _load_model(self) -> None:
        """Lazy-load the OpenCLIP model and preprocessing transforms on first use."""
        import open_clip

        model, _, preprocess_val = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model = model.eval().to(self.device)
        self.model = model
        self.preprocess_val = preprocess_val

    def _preprocess_for_comparison(self, image: Image.Image) -> Image.Image:
        """Convert image to grayscale, sharpen, and auto-contrast for stable CLIP comparison."""
        image = image.convert("L")
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageOps.autocontrast(image)
        image = image.convert("RGB")
        return image

    def compute_from_images(
        self, original: Image.Image, reconstructed: Image.Image
    ) -> Dict[str, float]:
        """Compute CLIP cosine similarity between two PIL images."""
        if self.model is None:
            self._load_model()

        original = self._preprocess_for_comparison(original)
        reconstructed = self._preprocess_for_comparison(reconstructed)

        batch = torch.stack([
            self.preprocess_val(original),
            self.preprocess_val(reconstructed),
        ]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = F.normalize(features, dim=-1)

        cosine_sim = (features[0] @ features[1]).item()
        return {"clip_cosine": float(cosine_sim)}

    def compute(self, ocr_output: str, **kwargs: Any) -> float:
        """Compute CLIP similarity; requires original_image and reconstructed_image in kwargs."""
        original = kwargs.get("original_image")
        reconstructed = kwargs.get("reconstructed_image")
        if original is None or reconstructed is None:
            raise ValueError(
                "Both 'original_image' and 'reconstructed_image' must be provided"
            )
        result = self.compute_from_images(original, reconstructed)
        return result["clip_cosine"]
