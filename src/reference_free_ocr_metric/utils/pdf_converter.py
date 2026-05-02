"""Convert PDF pages to PIL Images."""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

from pdf2image import convert_from_path
from PIL import Image


def convert_pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Image.Image]:
    """Convert PDF pages to a list of PIL Images."""
    return convert_from_path(
        pdf_path, dpi=dpi, first_page=first_page, last_page=last_page
    )


@contextmanager
def pdf_page_paths(
    pdf_path: str,
    dpi: int = 300,
    first_page: int | None = None,
    last_page: int | None = None,
) -> Generator[list[str]]:
    """Convert PDF pages to temp PNG files and yield their paths.

    Saves each page as a PNG in a temporary directory, yields the list of
    file paths (suitable for ``BaseOCRClient.ocr(image_path)``), then
    cleans up the temp files on exit.
    """
    images = convert_pdf_to_images(
        pdf_path, dpi=dpi, first_page=first_page, last_page=last_page
    )
    tmpdir = tempfile.mkdtemp()
    paths: list[str] = []
    try:
        for i, img in enumerate(images):
            path = os.path.join(tmpdir, f"page_{i}.png")
            img.save(path, "PNG")
            paths.append(path)
        yield paths
    finally:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(tmpdir):
            os.rmdir(tmpdir)
