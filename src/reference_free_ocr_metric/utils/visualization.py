"""Visualize OCR bounding boxes drawn over the original document image."""

from __future__ import annotations

import os

from bs4 import BeautifulSoup, Tag
from PIL import Image, ImageDraw, ImageFont

# Color per element class — outlines only so text stays readable.
_CLASS_COLOR: dict[str, str] = {
    "text": "blue",
    "formula": "green",
    "equation": "green",
    "table": "orange",
    "image": "purple",
    "figure": "purple",
    "title": "darkblue",
    "caption": "teal",
}
_DEFAULT_COLOR = "red"

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _load_font(size: int = 11) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load or return cached PIL font at the given size."""
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


def draw_bbox_visualization(image: Image.Image, html: str) -> Image.Image:
    """Draw OCR bounding boxes over the original image.

    Bounding box coordinates in Qwen3-VL HTML are in 0–1000 relative space;
    this function converts them to pixel coordinates using the image dimensions.

    Args:
        image: Original document page as a PIL Image.
        html: Raw HTML string from the OCR model (with data-bbox attributes).

    Returns:
        A copy of the image with colored bounding boxes and truncated labels.
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _load_font(11)

    w, h = img.size
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.find_all(attrs={"data-bbox": True})

    # Mirror Qwen cookbook filter: skip <ol> containers, keep <li> children.
    filtered: list[Tag] = []
    for el in elements:
        if not isinstance(el, Tag):
            continue
        if el.name == "ol":
            continue
        filtered.append(el)

    for el in filtered:
        bbox_str = el.get("data-bbox", "")
        if not isinstance(bbox_str, str):
            continue
        parts = bbox_str.split()
        if len(parts) != 4:
            continue
        try:
            rx1, ry1, rx2, ry2 = map(int, parts)
        except ValueError:
            continue

        # Convert from 0-1000 relative space to pixels.
        x1 = int(rx1 / 1000 * w)
        y1 = int(ry1 / 1000 * h)
        x2 = int(rx2 / 1000 * w)
        y2 = int(ry2 / 1000 * h)

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        el_class = (el.get("class") or [""])[0] if el.get("class") else ""
        color = _CLASS_COLOR.get(el_class, _DEFAULT_COLOR)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Label: class name + truncated text, drawn just below the box.
        label_text = el.get_text(strip=True)[:40]
        label = f"[{el_class}] {label_text}" if el_class else label_text
        draw.text((x1, y2 + 1), label, fill=color, font=font)

    return img
