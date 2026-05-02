#!/usr/bin/env python3
"""Regenerate reconstructed.png per variant using the (fixed) renderer.

Content is variant-independent at the OCR level; render flags pick which
component classes are actually drawn. For ocr_all_no_mask the original
image's pixels for each detected image region are pasted onto the
reconstruction so we don't penalise the metric for blank image regions.

The reconstruction uses a plain white canvas. Using the page background
as the canvas was tried and reverted — see
research/background_isolation_issues.md for why it breaks the metric's
ability to penalise OCR coverage gaps.
"""
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from reference_free_ocr_metric.reconstruction.document_analyzer import DocumentAnalyzer
from reference_free_ocr_metric.reconstruction.html_parser import QwenVLHTMLParser
from reference_free_ocr_metric.reconstruction.image_renderer import ImageRenderer

VARIANTS = ("ocr_all", "ocr_text", "ocr_formula", "ocr_table", "ocr_all_no_mask")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def find_source_image(images_root: Path, page_stem: str) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        p = images_root / f"{page_stem}{suf}"
        if p.exists():
            return p
    return None


RENDER_FLAGS = {
    "ocr_all":         {"render_text": True,  "render_formulas": True,  "render_tables": True},
    "ocr_text":        {"render_text": True,  "render_formulas": False, "render_tables": False},
    "ocr_formula":     {"render_text": False, "render_formulas": True,  "render_tables": False},
    "ocr_table":       {"render_text": False, "render_formulas": False, "render_tables": True},
    "ocr_all_no_mask": {"render_text": True,  "render_formulas": True,  "render_tables": True},
}


def render_one(
    html_file: Path,
    src_image_path: Path,
    parser: QwenVLHTMLParser,
    analyzer: DocumentAnalyzer,
    renderer: ImageRenderer,
    variant: str,
) -> Image.Image:
    image = Image.open(src_image_path).convert("RGB")
    parsed = parser.parse(html_file.read_text()).to_pixel_coords(*image.size)
    flags = RENDER_FLAGS[variant]
    if parsed.text_elements:
        analyzed = analyzer.analyze(image, parsed)
        out = renderer.render_analyzed_document(
            analyzed, *image.size, **flags, original_image=image
        )
    elif variant in ("ocr_all", "ocr_text", "ocr_all_no_mask"):
        plain = parsed.plain_text or ""
        out = renderer.render_plain_text(plain, *image.size)
    else:
        out = Image.new("RGB", image.size, "white")
    if variant == "ocr_all_no_mask":
        for reg in parsed.image_regions:
            x1, y1, x2, y2 = reg.bbox
            if x2 > x1 and y2 > y1:
                out.paste(image.crop((x1, y1, x2, y2)), (x1, y1))
    return out


def process_variant(
    variant_root: Path,
    images_root: Path,
    parser: QwenVLHTMLParser,
    analyzer: DocumentAnalyzer,
    renderer: ImageRenderer,
) -> tuple[int, int, int]:
    ok = missing_html = missing_image = 0
    page_dirs = sorted(p for p in variant_root.iterdir() if p.is_dir())
    total = len(page_dirs)
    variant = variant_root.name
    for i, page_dir in enumerate(page_dirs, 1):
        html_file = page_dir / "ocr_html.html"
        if not html_file.exists():
            missing_html += 1
            continue
        src = find_source_image(images_root, page_dir.name)
        if src is None:
            missing_image += 1
            continue
        try:
            out = render_one(html_file, src, parser, analyzer, renderer, variant)
            out.save(page_dir / "reconstructed.png")
            ok += 1
        except Exception as exc:
            print(f"  ! {page_dir.name}: {exc}", flush=True)
        if i % 50 == 0 or i == total:
            print(f"  [{variant}] {i}/{total} ok={ok} missing_html={missing_html} missing_image={missing_image}", flush=True)
    return ok, missing_html, missing_image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/omnidocbench"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    images_root = args.data_root / "images"
    if not images_root.is_dir():
        print(f"images dir not found: {images_root}", file=sys.stderr)
        return 1

    parser = QwenVLHTMLParser()
    analyzer = DocumentAnalyzer()
    renderer = ImageRenderer()

    for variant in args.variants:
        root = args.data_root / variant
        if not root.is_dir():
            print(f"skipping {variant}: {root} not found")
            continue
        print(f"=== {variant} ===", flush=True)
        ok, mh, mi = process_variant(root, images_root, parser, analyzer, renderer)
        print(f"{variant}: ok={ok} missing_html={mh} missing_image={mi}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
