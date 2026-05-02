#!/usr/bin/env python3
"""Compare reference-free metrics with OmniDocBench reference-based metrics.

Computes Pearson/Spearman correlation between our reference-free scores
and OmniDocBench's Edit_dist metric on matching document samples.

Usage:
    uv run python scripts/run_comparison.py [--experiment-results results/experiment_results.json]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from reference_free_ocr_metric.comparison.omnidocbench_bridge import (
    OmniDocBenchBridge,
    compute_correlation,
)
from reference_free_ocr_metric.experiment.tracker import ExperimentTracker
from reference_free_ocr_metric.reconstruction.html_parser import QwenVLHTMLParser


def _setup_logging(name: str, error_log: Path) -> logging.Logger:
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)-5s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    error_log.parent.mkdir(parents=True, exist_ok=True)
    error_log.write_text("")  # truncate on new run
    fh = logging.FileHandler(error_log)
    fh.setLevel(logging.ERROR)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def main():
    argparser = argparse.ArgumentParser(
        description="Compare reference-free metrics with OmniDocBench"
    )
    argparser.add_argument(
        "--experiment-results",
        type=Path,
        default=Path("results/sample_pages/results.json"),
        help="Path to experiment results JSON from run_experiment.py",
    )
    argparser.add_argument(
        "--omnidocbench-path",
        type=Path,
        default=None,
        help="Path to OmniDocBench repository (optional)",
    )
    argparser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("results/sample_pages"),
        help="Directory containing per-image artifacts (ocr_text.txt files)",
    )
    argparser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparison_results.json"),
        help="Output JSON file path",
    )
    argparser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip CDM (formula) and TEDS (table) — compute text edit distance only",
    )
    argparser.add_argument(
        "--annotations-json",
        type=Path,
        default=None,
        help="Direct path to OmniDocBench.json (overrides --omnidocbench-path)",
    )
    argparser.add_argument(
        "--no-cdm",
        action="store_true",
        help="Skip CDM (expensive pdflatex) but still compute formula edit distance",
    )
    argparser.add_argument(
        "--documentation-output",
        type=Path,
        default=None,
        help="Save correlation summary to this path",
    )
    argparser.add_argument(
        "--focus",
        choices=["text", "formula", "table"],
        default=None,
        help="Focus on one GT dimension: only compute and report that dimension's correlations",
    )
    args = argparser.parse_args()

    # Logger name = variant derived from output path (e.g. "formula" from comparison_formula.json)
    log_name = args.output.stem.removeprefix("comparison_") or "comparison"
    error_log = Path(__file__).parent.parent / "logs" / f"comparison_{log_name}.error.log"
    log = _setup_logging(log_name, error_log)

    t_total = time.time()

    if not args.experiment_results.exists():
        log.error("Experiment results not found at %s", args.experiment_results)
        log.error("Run scripts/run_experiment.py first.")
        return

    # Load experiment results
    t0 = time.time()
    with open(args.experiment_results) as f:
        experiment_results = json.load(f)
    successful_results = [r for r in experiment_results if "error" not in r]
    if not successful_results:
        log.error("No successful results found in experiment data.")
        return
    log.info("Loaded %d experiment results in %.1fs", len(successful_results), time.time() - t0)

    # Load OmniDocBench ground truth
    t0 = time.time()
    bridge = OmniDocBenchBridge(
        annotations_path=str(args.annotations_json) if args.annotations_json else None,
        omnidocbench_path=str(args.omnidocbench_path) if args.omnidocbench_path and not args.annotations_json else None,
    )
    gt_data = bridge.load_ground_truth()
    log.info("Loaded %d OmniDocBench pages in %.1fs", len(gt_data), time.time() - t0)

    # Build lookup from image name to OmniDocBench page
    gt_lookup = {}
    for page in gt_data:
        page_name = page.get("page_info", {}).get("image_path", "")
        if page_name:
            gt_lookup[Path(page_name).stem] = page
            gt_lookup[Path(page_name).name] = page

    # Determine which dimensions to compute based on --focus
    compute_text    = args.focus in (None, "text")
    compute_formula = not args.text_only and args.focus in (None, "formula")
    compute_table   = not args.text_only and args.focus in (None, "table")
    log.info("focus=%s  compute: text=%s formula=%s table=%s",
             args.focus or "all", compute_text, compute_formula, compute_table)

    # Match experiment results to OmniDocBench pages and compute reference metrics
    t0 = time.time()
    matched_pairs = []
    skipped_no_gt = 0
    skipped_no_ocr = 0

    for result in successful_results:
        img_name = result["image"]
        stem = Path(img_name).stem

        gt_page = gt_lookup.get(stem) or gt_lookup.get(img_name)
        if gt_page is None:
            continue

        # Derive the OCR artifacts directory name from the annotation image_path.
        # Using Path(img_name).stem is wrong for names with dots and no extension
        # (e.g. 'PPT_11.2013...page_010' → stem='PPT_11', not the full page name).
        # The annotation image_path always has .jpg, so its stem is always correct.
        ann_page_stem = Path(gt_page.get('page_info', {}).get('image_path', img_name)).stem

        # When focused on formula/table, skip pages with no relevant GT content.
        if args.focus == "formula":
            has_formula_gt = any(
                e.get("category_type") == "equation_isolated"
                and not e.get("ignore", False)
                and (e.get("latex") or e.get("text"))
                for e in gt_page.get("layout_dets", [])
            )
            if not has_formula_gt:
                skipped_no_gt += 1
                continue
        elif args.focus == "table":
            has_table_gt = any(
                e.get("category_type") == "table" and not e.get("ignore", False)
                for e in gt_page.get("layout_dets", [])
            )
            if not has_table_gt:
                skipped_no_gt += 1
                continue

        ocr_elements_path = args.artifacts_dir / ann_page_stem / "ocr_elements.json"
        ocr_formula_path  = args.artifacts_dir / ann_page_stem / "ocr_formula_elements.json"
        ocr_table_path    = args.artifacts_dir / ann_page_stem / "ocr_table_elements.json"
        ocr_html_path     = args.artifacts_dir / ann_page_stem / "ocr_html.html"

        if ocr_elements_path.exists():
            ocr_elements = json.loads(ocr_elements_path.read_text())
            if compute_formula:
                if ocr_formula_path.exists():
                    ocr_formula_elements = json.loads(ocr_formula_path.read_text())
                elif ocr_html_path.exists():
                    page_w = gt_page["page_info"].get("width", 1000)
                    page_h = gt_page["page_info"].get("height", 1000)
                    pixel_doc = QwenVLHTMLParser().parse(ocr_html_path.read_text()).to_pixel_coords(page_w, page_h)
                    ocr_formula_elements = [{"text": e.text, "bbox": list(e.bbox)} for e in pixel_doc.text_elements if e.tag == "formula"]
                else:
                    ocr_formula_elements = []
            else:
                ocr_formula_elements = []
            if compute_table:
                if ocr_table_path.exists():
                    ocr_table_elements = json.loads(ocr_table_path.read_text())
                elif ocr_html_path.exists():
                    page_w = gt_page["page_info"].get("width", 1000)
                    page_h = gt_page["page_info"].get("height", 1000)
                    pixel_doc = QwenVLHTMLParser().parse(ocr_html_path.read_text()).to_pixel_coords(page_w, page_h)
                    ocr_table_elements = [{"html": te.html_content, "bbox": list(te.bbox)} for te in pixel_doc.table_elements]
                else:
                    ocr_table_elements = []
            else:
                ocr_table_elements = []
        elif ocr_html_path.exists():
            page_w = gt_page["page_info"].get("width", 1000)
            page_h = gt_page["page_info"].get("height", 1000)
            parsed = QwenVLHTMLParser().parse(ocr_html_path.read_text())
            pixel_doc = parsed.to_pixel_coords(page_w, page_h)
            ocr_elements = [{"text": e.text, "bbox": list(e.bbox)} for e in pixel_doc.text_elements]
            ocr_formula_elements = [{"text": e.text, "bbox": list(e.bbox)} for e in pixel_doc.text_elements if e.tag == "formula"] if compute_formula else []
            ocr_table_elements = [{"html": te.html_content, "bbox": list(te.bbox)} for te in pixel_doc.table_elements] if compute_table else []
        else:
            log.error("no OCR elements or HTML found: %s", img_name)
            skipped_no_ocr += 1
            continue

        try:
            text_edit_dist    = bridge.compute_page_edit_distance(gt_page, ocr_elements) if compute_text else None
            formula_edit_dist = bridge.compute_page_formula_edit_distance(gt_page, ocr_formula_elements) if compute_formula else None
            table_edit_dist   = bridge.compute_page_table_edit_distance(gt_page, ocr_table_elements) if compute_table else None
            cdm  = (None if args.no_cdm else bridge.compute_page_cdm(gt_page, ocr_formula_elements)) if compute_formula else None
            teds = bridge.compute_page_teds(gt_page, ocr_table_elements) if compute_table else None
        except Exception as e:
            log.error("reference metric failed for %s: %s", img_name, e, exc_info=True)
            continue

        end_to_end_dims = []
        if text_edit_dist is not None:
            end_to_end_dims.append(1.0 - text_edit_dist)
        if cdm is not None:
            end_to_end_dims.append(cdm)
        if teds is not None:
            end_to_end_dims.append(teds)
        end_to_end = sum(end_to_end_dims) / len(end_to_end_dims) if end_to_end_dims else None

        page_attr = gt_page.get("page_info", {}).get("page_attribute", {}) or {}
        matched_pairs.append({
            "image": img_name,
            "category":   page_attr.get("data_source", "unknown"),
            "language":   page_attr.get("language", "unknown"),
            "layout":     page_attr.get("layout", "unknown"),
            "text_edit_distance": text_edit_dist,
            "formula_edit_distance": formula_edit_dist,
            "table_edit_distance": table_edit_dist,
            "cdm": cdm,
            "teds": teds,
            "end_to_end": end_to_end,
            "edit_distance": text_edit_dist,
            "multi_composite": result["multi_metric"]["composite"],
            "clip_cosine": result["clip_compare"]["clip_cosine"],
            "lm_composite": result["lm_perplexity"]["composite"],
            "perplexity": result["lm_perplexity"]["perplexity"],
            "text_accuracy":    None if text_edit_dist    is None else 1.0 - text_edit_dist,
            "formula_accuracy": None if formula_edit_dist is None else 1.0 - formula_edit_dist,
            "table_accuracy":   None if table_edit_dist   is None else 1.0 - table_edit_dist,
        })

    ref_elapsed = time.time() - t0
    log.info(
        "Matched %d pages in %.1fs  (skipped: %d no GT, %d no OCR artifacts)",
        len(matched_pairs), ref_elapsed, skipped_no_gt, skipped_no_ocr,
    )

    if len(matched_pairs) < 3:
        log.warning("Only %d matched samples — need ≥3 for correlation, skipping", len(matched_pairs))
        comparison = {
            "n_matched": len(matched_pairs),
            "message": "Too few matched samples for correlation",
            "matched_pairs": matched_pairs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        log.info("Results saved to %s", args.output)
        return

    # Select dims relevant to --focus (None = all)
    all_dims = {
        "text_accuracy": [p["text_accuracy"]    for p in matched_pairs],
        "formula_cdm": [p["cdm"]              for p in matched_pairs],
        "formula_edit":[p["formula_accuracy"] for p in matched_pairs],
        "table":       [p["teds"]             for p in matched_pairs],
        "table_edit":  [p["table_accuracy"]   for p in matched_pairs],
        "end_to_end":  [p["end_to_end"]       for p in matched_pairs],
    }
    if args.focus == "text":
        active_dims = {"text_accuracy": all_dims["text_accuracy"], "end_to_end": all_dims["end_to_end"]}
    elif args.focus == "formula":
        active_dims = {"formula_edit": all_dims["formula_edit"], "formula_cdm": all_dims["formula_cdm"]}
    elif args.focus == "table":
        active_dims = {"table": all_dims["table"], "table_edit": all_dims["table_edit"]}
    else:
        active_dims = all_dims

    t0 = time.time()
    correlations = {}
    for ref_metric_name, ref_key in [
        ("multi_composite", "multi_composite"),
        ("clip_cosine",     "clip_cosine"),
        ("lm_composite",    "lm_composite"),
    ]:
        ref_scores = [p[ref_key] for p in matched_pairs]
        correlations[ref_metric_name] = {}
        for dim, acc_scores_all in active_dims.items():
            valid = [(a, r) for a, r in zip(acc_scores_all, ref_scores) if a is not None]
            if not valid:
                correlations[ref_metric_name][dim] = {"pearson": None, "spearman": None, "n_samples": 0}
                continue
            acc_valid, ref_valid = zip(*valid)
            corr = compute_correlation(list(acc_valid), list(ref_valid))
            correlations[ref_metric_name][dim] = {
                "pearson": corr.pearson,
                "spearman": corr.spearman,
                "n_samples": corr.n_samples,
            }

        log.info("--- %s ---", ref_metric_name)
        for dim in active_dims:
            c = correlations[ref_metric_name][dim]
            if c["n_samples"] == 0:
                continue
            log.info(
                "  vs %-14s  Pearson %+.4f  Spearman %+.4f  N=%d",
                dim, c["pearson"], c["spearman"], c["n_samples"],
            )

    log.info("Correlations computed in %.1fs", time.time() - t0)

    # ── Per-category breakdown ──────────────────────────────────────────────
    # Group matched_pairs by data_source category and recompute correlations.
    # Categories with <3 pages skip silently. This is diagnostic only — does not
    # change the optimisation target (overall spearman_mean stays primary).
    by_category = {}
    cats = sorted({p.get("category", "unknown") for p in matched_pairs})
    for cat in cats:
        cat_pairs = [p for p in matched_pairs if p.get("category") == cat]
        if len(cat_pairs) < 3:
            by_category[cat] = {"n_pages": len(cat_pairs), "correlations": {}, "note": "fewer than 3 pages, skipped"}
            continue
        cat_dims = {dim: [p[ {
            "text_accuracy": "text_accuracy",
            "formula_cdm": "cdm",
            "formula_edit": "formula_accuracy",
            "table": "teds",
            "table_edit": "table_accuracy",
            "end_to_end": "end_to_end",
        }[dim]] for p in cat_pairs] for dim in active_dims}
        cat_corrs = {}
        for ref_metric_name, ref_key in [
            ("multi_composite", "multi_composite"),
            ("clip_cosine",     "clip_cosine"),
            ("lm_composite",    "lm_composite"),
        ]:
            ref_scores = [p[ref_key] for p in cat_pairs]
            cat_corrs[ref_metric_name] = {}
            for dim, acc_scores_all in cat_dims.items():
                valid = [(a, r) for a, r in zip(acc_scores_all, ref_scores) if a is not None]
                if len(valid) < 3:
                    cat_corrs[ref_metric_name][dim] = {"pearson": None, "spearman": None, "n_samples": len(valid)}
                    continue
                acc_valid, ref_valid = zip(*valid)
                corr = compute_correlation(list(acc_valid), list(ref_valid))
                cat_corrs[ref_metric_name][dim] = {
                    "pearson": corr.pearson,
                    "spearman": corr.spearman,
                    "n_samples": corr.n_samples,
                }
        by_category[cat] = {"n_pages": len(cat_pairs), "correlations": cat_corrs}
    log.info("Per-category breakdown computed for %d categories", len(by_category))

    # Log to MLflow
    tracker = ExperimentTracker(experiment_name="reference-free-ocr-comparison")
    with tracker.start_run(run_name="comparison"):
        tracker.log_params({"n_matched": len(matched_pairs)})
        for ref_metric_name, dim_corrs in correlations.items():
            for dim, corr_data in dim_corrs.items():
                if corr_data.get("pearson") is None:
                    continue
                tracker.log_metrics({
                    f"{ref_metric_name}_pearson_vs_{dim}":  corr_data["pearson"],
                    f"{ref_metric_name}_spearman_vs_{dim}": corr_data["spearman"],
                })

    comparison = {
        "n_matched": len(matched_pairs),
        "focus": args.focus,
        "correlations": correlations,
        "by_category": by_category,
        "matched_pairs": matched_pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info("Results saved to %s", args.output)

    if args.documentation_output:
        n_formula = sum(1 for p in matched_pairs if p.get("formula_accuracy") is not None)
        n_table   = sum(1 for p in matched_pairs if p.get("teds") is not None)
        doc_summary = {
            "n_pages_matched": len(matched_pairs),
            "n_pages_text":    sum(1 for p in matched_pairs if p.get("text_accuracy") is not None),
            "n_pages_formula": n_formula,
            "n_pages_table":   n_table,
            "focus": args.focus,
            "correlations": correlations,
            "by_category": by_category,
        }
        args.documentation_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.documentation_output, "w") as f:
            json.dump(doc_summary, f, indent=2)
        log.info("Documentation summary saved to %s", args.documentation_output)

    log.info("Total elapsed: %.1fs", time.time() - t_total)
    if error_log.stat().st_size > 0:
        log.warning("Errors written to %s", error_log)


if __name__ == "__main__":
    main()
