"""Image preprocessing for OCR metric comparison."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.ndimage import binary_closing
from skimage.filters import threshold_local

from reference_free_ocr_metric.reconstruction.html_parser import Region


class ImagePreprocessor:
    """Preprocesses images for comparison."""

    def mask_regions(
        self,
        image: Image.Image,
        regions: list[Region],
        fill_color: tuple = (255, 255, 255),
    ) -> Image.Image:
        """Mask regions by filling their bounding boxes with fill_color."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for region in regions:
            draw.rectangle(region.bbox, fill=fill_color)
        return img

    def to_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert image to grayscale."""
        return image.convert("L")

    def optimize_for_comparison(self, image: Image.Image) -> Image.Image:
        """Convert to grayscale, sharpen, and apply autocontrast."""
        gray = self.to_grayscale(image)
        sharpened = gray.filter(ImageFilter.SHARPEN)
        return ImageOps.autocontrast(sharpened)

    def adaptive_binarize(self, image: Image.Image) -> Image.Image:
        """Binarize using adaptive (local) thresholding + morphological closing.

        Mirrors r1-p1's prepare_*_for_comparison() pipeline:
          1. Grayscale
          2. Adaptive threshold (Gaussian, 11x11 block, offset=2)
          3. Morphological closing (2x2 kernel) to fill small gaps

        This removes font rendering differences so SSIM/MSE measure text
        placement accuracy rather than font style similarity.
        """
        gray = np.array(image.convert("L"))
        threshold = threshold_local(gray, block_size=11, method="gaussian", offset=2)
        binary = gray > threshold
        closed = binary_closing(binary, structure=np.ones((2, 2)))
        return Image.fromarray((closed * 255).astype(np.uint8))
