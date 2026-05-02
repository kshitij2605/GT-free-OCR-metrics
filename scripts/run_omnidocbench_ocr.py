#!/usr/bin/env python3
"""Batch OCR for the full OmniDocBench dataset (1355 images).

Processes all images in data/omnidocbench/images/, saves per-image artifacts
to data/omnidocbench/ocr/<stem>/ and writes results.json incrementally.

Resume-safe: if results.json already exists, images with a successful entry
are skipped. Re-run the same command to continue after interruption.

Usage:
    uv run python scripts/run_omnidocbench_ocr.py
    uv run python scripts/run_omnidocbench_ocr.py --input data/omnidocbench/images --output-dir data/omnidocbench/ocr
    uv run python scripts/run_omnidocbench_ocr.py --skip-ocr   # reuse cached ocr_html.html
"""
# ruff: noqa: E402


import argparse
import json
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from reference_free_ocr_metric.metrics.clip_compare.clip_similarity import CLIPSimilarityMetric  # noqa: E402
from reference_free_ocr_metric.metrics.multi_metric.visual_reconstruction import VisualReconstructionMetric  # noqa: E402
from reference_free_ocr_metric.ocr.qwen_client import QwenVLClient  # noqa: E402
from reference_free_ocr_metric.reconstruction.document_analyzer import DocumentAnalyzer  # noqa: E402
from reference_free_ocr_metric.reconstruction.html_parser import QwenVLHTMLParser  # noqa: E402
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor  # noqa: E402
from reference_free_ocr_metric.reconstruction.image_renderer import ImageRenderer  # noqa: E402
from reference_free_ocr_metric.utils.visualization import draw_bbox_visualization  # noqa: E402

VLM_API_BASE = "http://difgpu01.tdc.otsuka-shokai.co.jp:9000/v1"
VLM_API_KEY = "tensorflow"
VLM_MODEL = "Qwen/Qwen3.5-122B-A10B"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def process_image(
    image_path: Path,
    ocr_client: QwenVLClient,
    out_dir: Path,
    multi_metric: VisualReconstructionMetric,
    clip_metric: CLIPSimilarityMetric,
    skip_ocr: bool = False,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if skip_ocr and (out_dir / "ocr_html.html").exists():
        html_output = (out_dir / "ocr_html.html").read_text()
    else:
        html_output = ocr_client.ocr(str(image_path))

    parser = QwenVLHTMLParser()
    parsed = parser.parse(html_output)
    pixel_doc = parsed.to_pixel_coords(width, height)
    plain_text = parsed.plain_text or "\n".join(te.text for te in pixel_doc.text_elements)

    preprocessor = ImagePreprocessor()
    all_regions = pixel_doc.image_regions + pixel_doc.table_regions
    masked_original = preprocessor.mask_regions(image, all_regions)

    renderer = ImageRenderer()
    if pixel_doc.text_elements:
        analyzer = DocumentAnalyzer()
        analyzed = analyzer.analyze(image, pixel_doc)
        reconstructed = renderer.render_analyzed_document(analyzed, width, height)
    else:
        reconstructed = renderer.render_plain_text(plain_text, width, height)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ocr_html.html").write_text(html_output)
    (out_dir / "ocr_text.txt").write_text(plain_text)
    ocr_elements = [{"text": e.text, "bbox": list(e.bbox)} for e in pixel_doc.text_elements]
    (out_dir / "ocr_elements.json").write_text(json.dumps(ocr_elements, ensure_ascii=False))
    formula_elements = [
        {"text": e.text, "bbox": list(e.bbox)}
        for e in pixel_doc.text_elements
        if e.tag == "formula"
    ]
    (out_dir / "ocr_formula_elements.json").write_text(
        json.dumps(formula_elements, ensure_ascii=False)
    )
    table_els = [
        {"html": te.html_content, "bbox": list(te.bbox)} for te in pixel_doc.table_elements
    ]
    (out_dir / "ocr_table_elements.json").write_text(json.dumps(table_els, ensure_ascii=False))
    masked_original.save(out_dir / "masked_original.png")
    reconstructed.save(out_dir / "reconstructed.png")
    bbox_vis = draw_bbox_visualization(image, html_output)
    bbox_vis.save(out_dir / "bbox_visualization.png")

    multi_result = multi_metric.compute_from_images(masked_original, reconstructed)
    clip_result = clip_metric.compute_from_images(masked_original, reconstructed)

    return {
        "image": image_path.name,
        "text_elements": len(pixel_doc.text_elements),
        "image_regions": len(pixel_doc.image_regions),
        "table_regions": len(pixel_doc.table_regions),
        "text_length": len(plain_text),
        "plain_text_length": len(parsed.plain_text),
        "multi_metric": multi_result,
        "clip_compare": clip_result,
        # LM perplexity excluded for bulk run (poor correlation signal, ~30s per image)
        "lm_perplexity": {"ngram_score": 0.0, "transformer_score": 0.0, "perplexity": 0.0, "composite": 0.0},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch OCR for the full OmniDocBench dataset")
    ap.add_argument("--input", type=Path, default=Path("data/omnidocbench/images"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/omnidocbench/ocr"))
    ap.add_argument("--skip-ocr", action="store_true", help="Reuse cached ocr_html.html")
    args = ap.parse_args()

    images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        print(f"No images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    results_path = args.output_dir / "results.json"
    all_results: list[dict] = []
    done_stems: set[str] = set()
    if results_path.exists():
        all_results = json.loads(results_path.read_text())
        done_stems = {Path(r["image"]).stem for r in all_results if "error" not in r}
        print(f"Resuming: {len(done_stems)} done, {len(images) - len(done_stems)} remaining.")
    else:
        print(f"Starting fresh: {len(images)} images to process.")

    print("Loading metrics (LPIPS, CLIP)...")
    multi_metric = VisualReconstructionMetric()
    clip_metric = CLIPSimilarityMetric()
    ocr_client = QwenVLClient(api_base=VLM_API_BASE, api_key=VLM_API_KEY, model_name=VLM_MODEL)
    print("Ready.\n")

    for i, image_path in enumerate(images):
        if image_path.stem in done_stems:
            continue

        print(f"[{i+1}/{len(images)}] {image_path.name}", flush=True)
        out_dir = args.output_dir / image_path.stem
        try:
            result = process_image(
                image_path, ocr_client, out_dir,
                multi_metric, clip_metric,
                skip_ocr=args.skip_ocr,
            )
            all_results.append(result)
            print(
                f"  OK  elems={result['text_elements']}"
                f"  clip={result['clip_compare']['clip_cosine']:.3f}"
                f"  composite={result['multi_metric']['composite']:.3f}",
                flush=True,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            all_results.append({"image": image_path.name, "error": str(e)})

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(all_results, indent=2))

    successful = sum(1 for r in all_results if "error" not in r)
    print(f"\nDone. {successful}/{len(images)} successful. Results: {results_path}")


if __name__ == "__main__":
    main()
