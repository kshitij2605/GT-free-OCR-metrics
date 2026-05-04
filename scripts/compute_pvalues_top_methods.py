#!/usr/bin/env python3
"""Compute p-value enriched correlation summaries for the top-N methods.

For each (method, variant) pair:
  - Reads results from r1-p2/results/method_runs/ocr_<v>/<method_id>/results.json
  - Runs run_comparison.py with --documentation-output
  - Saves to results/key_methods_pvalues/<method_id>/ocr_<v>.json

Usage:
    cd /home/mac/test/GT-free-ocr-metrics
    uv run python scripts/compute_pvalues_top_methods.py [--top N] [--method <id>]
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
R1P2 = Path(__file__).parent.parent
ANNOTATIONS_JSON = PROJECT / "data" / "omnidocbench" / "OmniDocBench.json"
VARIANTS = ["text", "formula", "table", "all", "all_no_mask"]
DIM = {"text": "text_accuracy", "formula": "formula_edit", "table": "table",
       "all": "end_to_end", "all_no_mask": "end_to_end"}
METRICS = ["multi_composite", "clip_cosine"]


def compute_spearman_mean(method_id: str, base: Path) -> float:
    scores = []
    for v in VARIANTS:
        cfile = base / ("ocr_" + v) / method_id / "correlations.json"
        if not cfile.exists():
            scores.append(None)
            continue
        try:
            raw = cfile.read_text().replace("NaN", "null")
            doc = json.loads(raw)
            corr = doc.get("correlations", {})
            dim = DIM[v]
            best_s = None
            for m in METRICS:
                entry = corr.get(m, {}).get(dim, {})
                s = entry.get("spearman")
                if s is not None and (best_s is None or s > best_s):
                    best_s = s
            scores.append(best_s)
        except Exception:
            scores.append(None)
    vals = [x for x in scores if x is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def get_top_methods(n: int) -> list[str]:
    base = R1P2 / "results" / "method_runs"
    methods: set[str] = set()
    for v in VARIANTS:
        vdir = base / ("ocr_" + v)
        if vdir.exists():
            for d in vdir.iterdir():
                if (d / "correlations.json").exists():
                    methods.add(d.name)

    scored = [(m, compute_spearman_mean(m, base)) for m in methods]
    scored.sort(key=lambda x: -x[1] if not math.isnan(x[1]) else float("-inf"))
    return [m for m, _ in scored[:n]]


def run_one(method_id: str, variant: str) -> bool:
    """Compute p-values for one (method, variant) pair. Returns True on success."""
    results_json = R1P2 / "results" / "method_runs" / f"ocr_{variant}" / method_id / "results.json"
    if not results_json.exists():
        print(f"  SKIP {method_id}/ocr_{variant} — no results.json")
        return True

    out_dir = PROJECT / "results" / "key_methods_pvalues" / method_id
    out_file = out_dir / f"ocr_{variant}.json"
    if out_file.exists():
        print(f"  SKIP {method_id}/ocr_{variant} — already computed")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_comparison = PROJECT / "results" / f"_tmp_comparison_{method_id}_{variant}.json"

    # focus flag: text/formula/table variants use focused computation to speed up CDM/TEDS
    focus_map = {"text": "text", "formula": "formula", "table": "table"}
    focus = focus_map.get(variant)

    cmd = [
        str(Path.home() / ".local/bin/uv"), "run", "python", "scripts/run_comparison.py",
        "--experiment-results", str(results_json),
        "--annotations-json", str(ANNOTATIONS_JSON),
        "--artifacts-dir", str(PROJECT / "data" / "omnidocbench" / f"ocr_{variant}"),
        "--output", str(tmp_comparison),
        "--documentation-output", str(out_file),
        "--no-cdm",  # skip expensive pdflatex; formula_edit still computed
    ]
    if focus:
        cmd += ["--focus", focus]

    print(f"  Computing {method_id}/ocr_{variant}...")
    result = subprocess.run(cmd, cwd=PROJECT, capture_output=False)
    tmp_comparison.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  ERROR: {method_id}/ocr_{variant} failed (exit {result.returncode})")
        out_file.unlink(missing_ok=True)
        return False

    print(f"  -> {out_file}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=30, help="Number of top methods")
    parser.add_argument("--method", type=str, default=None, help="Single method ID to run")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["text", "formula", "table", "all", "all_no_mask"],
                        help="Single variant to run")
    args = parser.parse_args()

    if args.method:
        methods = [args.method]
    else:
        print(f"Computing top-{args.top} methods by spearman_mean...")
        methods = get_top_methods(args.top)
        print(f"Top-{len(methods)} methods:", methods[:5], "...")

    variants = [args.variant] if args.variant else VARIANTS

    # Also always include baseline
    if "baseline" not in methods and not args.method:
        methods.insert(0, "baseline")

    for method in methods:
        print(f"\n=== {method} ===")
        for v in variants:
            run_one(method, v)

    print("\n=== Done. Results in results/key_methods_pvalues/ ===")


if __name__ == "__main__":
    main()
