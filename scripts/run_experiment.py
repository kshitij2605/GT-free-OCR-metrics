#!/usr/bin/env python3
"""Run all reference-free OCR metrics on document images or PDFs.

Usage:
    python scripts/run_experiment.py --input data/sample_pages           # folder of images
    python scripts/run_experiment.py --input document.pdf                # single PDF
    python scripts/run_experiment.py --input page.png                    # single image
    python scripts/run_experiment.py --input data/pdfs/                  # folder of PDFs (or mixed)
"""
# ruff: noqa: E402


import argparse
import json
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Allow large document images

from reference_free_ocr_metric.metrics.clip_compare.clip_similarity import CLIPSimilarityMetric
from reference_free_ocr_metric.metrics.lm_perplexity.perplexity_scorer import LMPerplexityMetric
from reference_free_ocr_metric.metrics.multi_metric.visual_reconstruction import VisualReconstructionMetric
from reference_free_ocr_metric.ocr.qwen_client import QwenVLClient
from reference_free_ocr_metric.reconstruction.document_analyzer import DocumentAnalyzer
from reference_free_ocr_metric.reconstruction.html_parser import QwenVLHTMLParser
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor
from reference_free_ocr_metric.reconstruction.image_renderer import ImageRenderer
from reference_free_ocr_metric.utils.pdf_converter import pdf_page_paths
from reference_free_ocr_metric.utils.visualization import draw_bbox_visualization

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_PDF_SUFFIXES = {".pdf"}


@contextmanager
def collect_images(input_path: Path) -> Generator[list[Path]]:
    """Resolve any input (single file or directory) into image paths.

    Accepts: single image, single PDF, directory of images, directory of PDFs,
    or a mixed directory. PDF pages are converted to temp PNGs and cleaned up
    on exit.
    """
    with ExitStack() as stack:
        images: list[Path] = []

        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix in _PDF_SUFFIXES:
                paths = stack.enter_context(pdf_page_paths(str(input_path)))
                images.extend(Path(p) for p in paths)
            elif suffix in _IMAGE_SUFFIXES:
                images.append(input_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        elif input_path.is_dir():
            for f in sorted(input_path.iterdir()):
                suffix = f.suffix.lower()
                if suffix in _IMAGE_SUFFIXES:
                    images.append(f)
                elif suffix in _PDF_SUFFIXES:
                    paths = stack.enter_context(pdf_page_paths(str(f)))
                    images.extend(Path(p) for p in paths)
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")

        yield images

VLM_API_BASE = "http://difgpu01.tdc.otsuka-shokai.co.jp:9000/v1"
VLM_API_KEY = "tensorflow"
VLM_MODEL = "Qwen/Qwen3.5-122B-A10B"


def process_image(
    image_path: Path,
    ocr_client: QwenVLClient,
    results_dir: Path,
    html_output: str | None = None,
    multi_metric: VisualReconstructionMetric | None = None,
    clip_metric: CLIPSimilarityMetric | None = None,
    lm_metric: LMPerplexityMetric | None = None,
) -> dict:
    """Run OCR and compute all metrics for a single image."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Step 1: OCR with Qwen3-VL (skipped if html_output pre-fetched in parallel)
    if html_output is None:
        print(f"  Running OCR on {image_path.name}...")
        html_output = ocr_client.ocr(str(image_path))

    # Step 2: Parse HTML output
    parser = QwenVLHTMLParser()
    parsed = parser.parse(html_output)
    pixel_doc = parsed.to_pixel_coords(width, height)

    # Extract plain text for perplexity (prefer plain_text from parser)
    plain_text = parsed.plain_text or "\n".join(te.text for te in pixel_doc.text_elements)

    # Step 3: Preprocess images
    preprocessor = ImagePreprocessor()
    all_regions = pixel_doc.image_regions + pixel_doc.table_regions
    masked_original = preprocessor.mask_regions(image, all_regions)

    renderer = ImageRenderer()
    if pixel_doc.text_elements:
        # Analyze original image for precise styling, then render.
        analyzer = DocumentAnalyzer()
        analyzed = analyzer.analyze(image, pixel_doc)
        reconstructed = renderer.render_analyzed_document(analyzed, width, height)
    else:
        # Fallback: render plain text top-to-bottom when no bbox elements
        reconstructed = renderer.render_plain_text(plain_text, width, height)

    # Save intermediate artifacts
    out_dir = results_dir / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ocr_html.html").write_text(html_output)
    (out_dir / "ocr_text.txt").write_text(plain_text)
    ocr_elements = [{"text": e.text, "bbox": list(e.bbox)} for e in pixel_doc.text_elements]
    (out_dir / "ocr_elements.json").write_text(json.dumps(ocr_elements, ensure_ascii=False))
    formula_elements = [{"text": e.text, "bbox": list(e.bbox)} for e in pixel_doc.text_elements if e.tag == "formula"]
    (out_dir / "ocr_formula_elements.json").write_text(json.dumps(formula_elements, ensure_ascii=False))
    table_els = [{"html": te.html_content, "bbox": list(te.bbox)} for te in pixel_doc.table_elements]
    (out_dir / "ocr_table_elements.json").write_text(json.dumps(table_els, ensure_ascii=False))
    masked_original.save(out_dir / "masked_original.png")
    reconstructed.save(out_dir / "reconstructed.png")
    bbox_vis = draw_bbox_visualization(image, html_output)
    bbox_vis.save(out_dir / "bbox_visualization.png")

    results = {
        "image": image_path.name,
        "text_elements": len(pixel_doc.text_elements),
        "image_regions": len(pixel_doc.image_regions),
        "table_regions": len(pixel_doc.table_regions),
        "text_length": len(plain_text),
        "plain_text_length": len(parsed.plain_text),
    }

    # Method 1a: Multi-metric (SSIM + MSE + LPIPS)
    print("  Computing multi-metric (SSIM+MSE+LPIPS)...")
    if multi_metric is None:
        multi_metric = VisualReconstructionMetric()
    multi_result = multi_metric.compute_from_images(masked_original, reconstructed)
    results["multi_metric"] = multi_result

    # Method 1c: OpenCLIP comparison
    print("  Computing CLIP similarity...")
    if clip_metric is None:
        clip_metric = CLIPSimilarityMetric()
    clip_result = clip_metric.compute_from_images(masked_original, reconstructed)
    results["clip_compare"] = clip_result

    # Method 2: LM Perplexity (skipped when lm_metric is None via --no-lm)
    if lm_metric is not None:
        print("  Computing LM perplexity...")
        lm_result = lm_metric.compute_detailed(plain_text)
    else:
        lm_result = {"ngram_score": 0.0, "transformer_score": 0.0, "perplexity": 0.0, "composite": 0.0}
    results["lm_perplexity"] = lm_result

    return results


def main():
    argparser = argparse.ArgumentParser(description="Run reference-free OCR metrics")
    argparser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sample_pages"),
        help="Input: single image, single PDF, or directory of images/PDFs",
    )
    argparser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Reuse cached ocr_html.html from a previous run instead of calling the VLM",
    )
    argparser.add_argument(
        "--no-lm",
        action="store_true",
        help="Skip GPT-2 LM perplexity (fastest per-image step to drop)",
    )
    argparser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts and results.json (default: results/<input_stem>)",
    )
    args = argparser.parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return

    # Initialize OCR client
    ocr_client = QwenVLClient(
        api_base=VLM_API_BASE,
        api_key=VLM_API_KEY,
        model_name=VLM_MODEL,
    )

    # Pre-load metrics once — models stay in memory across all images
    print("Loading metrics...")
    multi_metric = VisualReconstructionMetric()
    clip_metric = CLIPSimilarityMetric()
    lm_metric = None if args.no_lm else LMPerplexityMetric()
    print(f"  LPIPS, CLIP ready. LM perplexity: {'disabled' if args.no_lm else 'will load on first image'}.")

    # Output directory: explicit arg or auto-derived from input name
    results_dir = args.results_dir if args.results_dir else Path("results") / input_path.stem

    all_results = []
    with collect_images(input_path) as image_files:
        if not image_files:
            print(f"No images found from {input_path}")
            return

        print(f"Found {len(image_files)} images from {input_path}")
        print(f"Results directory: {results_dir.resolve()}")


        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Processing {image_path.name}")
            try:
                cached_html: str | None = None
                if args.skip_ocr:
                    cached_path = results_dir / image_path.stem / "ocr_html.html"
                    if cached_path.exists():
                        cached_html = cached_path.read_text()
                        print(f"  Using cached OCR from {cached_path}")
                    else:
                        print("  No cached OCR found, running VLM...")
                result = process_image(
                    image_path, ocr_client, results_dir,
                    html_output=cached_html,
                    multi_metric=multi_metric,
                    clip_metric=clip_metric,
                    lm_metric=lm_metric,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")
                all_results.append({"image": image_path.name, "error": str(e)})

        # Log aggregate metrics
        successful = [r for r in all_results if "error" not in r]
        if successful:
            avg_metrics = {
                "avg_ssim": sum(r["multi_metric"]["ssim"] for r in successful) / len(successful),
                "avg_mse": sum(r["multi_metric"]["mse"] for r in successful) / len(successful),
                "avg_lpips": sum(r["multi_metric"]["lpips"] for r in successful) / len(successful),
                "avg_composite": sum(r["multi_metric"]["composite"] for r in successful) / len(successful),
                "avg_clip_cosine": sum(r["clip_compare"]["clip_cosine"] for r in successful) / len(successful),
                "avg_lm_composite": sum(r["lm_perplexity"]["composite"] for r in successful) / len(successful),
            }
            print(f"\nAggregate metrics: {json.dumps(avg_metrics, indent=2)}")

    output_file = results_dir / "results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()