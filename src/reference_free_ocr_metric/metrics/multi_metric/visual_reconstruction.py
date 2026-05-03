"""Multi-metric visual reconstruction quality metric (SSIM + MSE + LPIPS)."""

from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from reference_free_ocr_metric.metrics.base import BaseMetric


class VisualReconstructionMetric(BaseMetric):
    """Evaluates OCR quality by comparing a reconstructed text-only image
    against the masked original using SSIM, MSE, and LPIPS sub-metrics
    combined into a composite score.
    """

    def __init__(
        self,
        ssim_weight: float = 0.4,
        mse_weight: float = 0.3,
        lpips_weight: float = 0.3,
    ) -> None:
        """Initialize with per-sub-metric weights for SSIM, MSE, and LPIPS (should sum to 1.0)."""
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self._lpips_model = None

    @property
    def name(self) -> str:
        """Return the metric identifier string."""
        return "visual_reconstruction"

    @property
    def lpips_model(self):
        """Lazy-load the LPIPS model on first use."""
        if self._lpips_model is None:
            import lpips

            self._lpips_model = lpips.LPIPS(net="alex")
        return self._lpips_model

    def compute_from_images(
        self, original: Image.Image, reconstructed: Image.Image
    ) -> Dict[str, float]:
        """Compare original and reconstructed images using SSIM, MSE, and LPIPS.

        Args:
            original: The original (or masked original) PIL Image.
            reconstructed: The reconstructed text-only PIL Image.

        Returns:
            Dict with keys: ssim, mse, lpips, composite.
        """
        from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

        preprocessor = ImagePreprocessor()

        # Binarize both images before SSIM/MSE to remove font-rendering differences.
        # After binarization, comparison measures text placement, not font style.
        orig_bin = preprocessor.adaptive_binarize(original)
        recon_bin = preprocessor.adaptive_binarize(reconstructed)

        # Resize reconstructed to match original if dimensions differ
        if recon_bin.size != orig_bin.size:
            recon_bin = recon_bin.resize(orig_bin.size, Image.Resampling.BILINEAR)

        orig_gray = np.array(orig_bin, dtype=np.float64) / 255.0
        recon_gray = np.array(recon_bin, dtype=np.float64) / 255.0

        # SSIM
        ssim_val = float(
            structural_similarity(orig_gray, recon_gray, data_range=1.0)
        )

        # MSE
        mse_val = float(np.mean((orig_gray - recon_gray) ** 2))

        # LPIPS uses continuous grayscale (not binarized) — learned perceptual metric
        orig_gray_cont = np.array(original.convert("L"), dtype=np.float64) / 255.0
        recon_gray_cont = np.array(reconstructed.convert("L"), dtype=np.float64) / 255.0
        if recon_gray_cont.shape != orig_gray_cont.shape:
            reconstructed_resized = reconstructed.convert("L").resize(
                (orig_gray_cont.shape[1], orig_gray_cont.shape[0]), Image.Resampling.BILINEAR
            )
            recon_gray_cont = np.array(reconstructed_resized, dtype=np.float64) / 255.0

        # LPIPS — use continuous grayscale, not binarized
        orig_tensor = (
            torch.from_numpy(orig_gray_cont).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        )
        recon_tensor = (
            torch.from_numpy(recon_gray_cont).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        )
        # LPIPS expects inputs in [-1, 1]
        orig_tensor = orig_tensor * 2.0 - 1.0
        recon_tensor = recon_tensor * 2.0 - 1.0

        with torch.no_grad():
            lpips_val = float(self.lpips_model(orig_tensor, recon_tensor).item())
        lpips_val = max(0.0, min(1.0, lpips_val))

        # Composite score
        composite = (
            self.ssim_weight * ssim_val
            + self.mse_weight * (1.0 - mse_val)
            + self.lpips_weight * (1.0 - lpips_val)
        )

        return {
            "ssim": ssim_val,
            "mse": mse_val,
            "lpips": lpips_val,
            "composite": composite,
        }

    def compute(self, ocr_output: str, **kwargs: Any) -> float:
        """Compute the visual reconstruction metric from OCR HTML output.

        Args:
            ocr_output: HTML string from OCR (parsed with QwenVLHTMLParser).
            **kwargs: Must include original_image (PIL Image),
                      image_width (int), image_height (int).

        Returns:
            The composite score as a float.
        """
        from reference_free_ocr_metric.reconstruction.html_parser import QwenVLHTMLParser
        from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor
        from reference_free_ocr_metric.reconstruction.image_renderer import ImageRenderer

        original_image: Image.Image = kwargs["original_image"]
        image_width: int = kwargs["image_width"]
        image_height: int = kwargs["image_height"]

        # Parse HTML output
        parsed = QwenVLHTMLParser.parse(ocr_output)
        pixel_doc = parsed.to_pixel_coords(image_width, image_height)

        # Mask the original image (white-fill image/table regions)
        all_regions = pixel_doc.image_regions + pixel_doc.table_regions
        masked_original = ImagePreprocessor.mask_regions(original_image, all_regions)

        # Render text-only image from text elements
        reconstructed = ImageRenderer.render_text_image(
            pixel_doc.text_elements, image_width, image_height
        )

        result = self.compute_from_images(masked_original, reconstructed)
        return result["composite"]

    def compute_from_parsed(
        self,
        parsed_doc: "ParsedDocument",
        original_image: Image.Image,
        image_width: int,
        image_height: int,
    ) -> float:
        """Compute visual reconstruction score from a pre-parsed document.

        Accepts a ParsedDocument directly (from any OCR parser), avoiding
        the need to re-parse HTML. Used by the unified OCR pipeline.
        """
        from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor
        from reference_free_ocr_metric.reconstruction.image_renderer import ImageRenderer

        pixel_doc = parsed_doc.to_pixel_coords(image_width, image_height)
        all_regions = pixel_doc.image_regions + pixel_doc.table_regions
        masked_original = ImagePreprocessor().mask_regions(original_image, all_regions)
        reconstructed = ImageRenderer().render_text_image(
            pixel_doc.text_elements, image_width, image_height
        )
        result = self.compute_from_images(masked_original, reconstructed)
        return result["composite"]
