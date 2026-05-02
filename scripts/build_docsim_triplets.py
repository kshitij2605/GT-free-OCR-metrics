#!/usr/bin/env python3
"""build_docsim_triplets.py — assemble a DreamSim-style triplet manifest for D60 (DocSim).

Creates `data/docsim_triplets/manifest.jsonl` where each line is:
    {"anchor_path": "...masked_original.png",
     "positive_path": "...reconstructed.png",
     "negative_path": "...reconstructed.png",
     "anchor_ed":   0.0,
     "positive_ed": 0.05,
     "negative_ed": 0.42,
     "anchor_page": "PPT_1001115_eng_page_003",
     "positive_page": "PPT_1001115_eng_page_003",
     "negative_page": "book_en_5_page_572",
     "kind": "same_page"|"cross_page"}

Triplet semantics for D60:
    Anchor   = ground-truth render (masked_original.png) — the "clean" target.
    Positive = a reconstruction that is FAITHFUL to the anchor (low text_edit_distance).
    Negative = a reconstruction that is CORRUPTED relative to the anchor (high text_edit_distance).

The trained DocSim metric should learn: dist(anchor, positive) << dist(anchor, negative).

Triplet sources (in priority order):
    (1) SAME-PAGE: for one page X, pick its best-OCR-quality variant as positive
        and its worst-OCR-quality variant as negative. Same content, different
        OCR fidelity. Strongest signal because content is held fixed.
    (2) CROSS-PAGE: for a page X with low edit_distance (= a clean recon exists),
        anchor=GT_X, positive=recon_X, negative=recon_Y where Y has high
        edit_distance. Augments triplet count when same-page contrast is weak.

Usage:
    python scripts/build_docsim_triplets.py \\
        --data-root data/omnidocbench \\
        --comparison-glob "results/method_runs/ocr_*/baseline/comparison.json" \\
        --output data/docsim_triplets/manifest.jsonl \\
        --min-edit-gap 0.15 \\
        --cross-page-multiplier 20

Then train DocSim with that manifest (D60).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path


VARIANTS = ("text", "formula", "table", "all", "all_no_mask")


def load_per_page_edit_distance(comparison_paths: list[Path]) -> dict[tuple[str, str], float]:
    """Return mapping (variant, page_basename_no_ext) -> text_edit_distance.

    `comparison_paths` is a list of `comparison.json` files, one per (variant, method).
    For DocSim we only need the BASELINE method's edit distances per variant — the
    reconstruction `reconstructed.png` is the same across all methods in a (variant, page).
    """
    out: dict[tuple[str, str], float] = {}
    for cp in comparison_paths:
        # comparison.json paths look like: results/method_runs/ocr_<variant>/<method_id>/comparison.json
        try:
            variant = cp.parents[1].name.removeprefix("ocr_")
        except IndexError:
            logging.warning("skip path with unexpected layout: %s", cp)
            continue
        if variant not in VARIANTS:
            continue
        try:
            with cp.open("r", encoding="utf-8", errors="replace") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logging.warning("skip unreadable %s: %s", cp, e)
            continue
        for pair in doc.get("matched_pairs", []):
            ed = pair.get("text_edit_distance")
            img = pair.get("image")
            if ed is None or not isinstance(ed, (int, float)) or img is None:
                continue
            page = Path(img).stem  # strip ".png"
            key = (variant, page)
            # First-seen wins (baseline preferred); other methods would be near-identical anyway
            out.setdefault(key, float(ed))
    return out


def page_image_paths(data_root: Path, variant: str, page: str) -> tuple[Path, Path] | None:
    """Return (anchor_path, recon_path) for a page-variant, or None if either missing."""
    page_dir = data_root / f"ocr_{variant}" / page
    anchor = page_dir / "masked_original.png"
    recon = page_dir / "reconstructed.png"
    if not anchor.exists() or not recon.exists():
        return None
    return anchor, recon


def build_same_page_triplets(
    ed_map: dict[tuple[str, str], float],
    data_root: Path,
    min_gap: float,
) -> list[dict]:
    """For each page, anchor=GT_X, positive=variant_low, negative=variant_high.

    Variants compete on the same page X. Picks the variant with lowest text_edit_distance
    as positive, highest as negative. Filters out triplets with gap < min_gap.
    """
    by_page: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (variant, page), ed in ed_map.items():
        by_page[page].append((variant, ed))

    triplets = []
    for page, variant_eds in by_page.items():
        if len(variant_eds) < 2:
            continue
        variant_eds.sort(key=lambda x: x[1])
        best_v, best_ed = variant_eds[0]
        worst_v, worst_ed = variant_eds[-1]
        if worst_ed - best_ed < min_gap:
            continue
        # Anchor = GT of the WORST-variant (it's the same GT image regardless of variant,
        # but we use the file under that variant directory by convention; both exist).
        # Use best variant's anchor for cleanest layout.
        ap = page_image_paths(data_root, best_v, page)
        pp = page_image_paths(data_root, best_v, page)
        np_ = page_image_paths(data_root, worst_v, page)
        if ap is None or pp is None or np_ is None:
            continue
        triplets.append({
            "anchor_path": str(ap[0]),    # GT of best variant
            "positive_path": str(pp[1]),  # recon of best variant (low ED)
            "negative_path": str(np_[1]), # recon of worst variant (high ED)
            "anchor_ed": 0.0,
            "positive_ed": best_ed,
            "negative_ed": worst_ed,
            "anchor_page": page,
            "positive_page": page,
            "negative_page": page,
            "anchor_variant": best_v,
            "positive_variant": best_v,
            "negative_variant": worst_v,
            "kind": "same_page",
        })
    return triplets


def build_cross_page_triplets(
    ed_map: dict[tuple[str, str], float],
    data_root: Path,
    min_gap: float,
    multiplier: int,
    rng: random.Random,
) -> list[dict]:
    """For each page-variant, anchor=GT, positive=its own recon (if low ED),
    negative=recon of a DIFFERENT page-variant with high ED.

    Generates `multiplier` such triplets per qualifying anchor.
    """
    # Bucket items by ED quartile for fast sampling.
    items = [(v, p, ed) for (v, p), ed in ed_map.items() if page_image_paths(data_root, v, p) is not None]
    if not items:
        return []
    items.sort(key=lambda x: x[2])
    n = len(items)
    low_pool = items[: n // 4]      # lowest 25% ED — clean recons
    high_pool = items[3 * n // 4:]  # highest 25% ED — corrupted recons

    triplets = []
    for v, p, ed in low_pool:
        ap = page_image_paths(data_root, v, p)
        if ap is None:
            continue
        for _ in range(multiplier):
            nv, np_p, n_ed = rng.choice(high_pool)
            if nv == v and np_p == p:
                continue
            if n_ed - ed < min_gap:
                continue
            np_paths = page_image_paths(data_root, nv, np_p)
            if np_paths is None:
                continue
            triplets.append({
                "anchor_path": str(ap[0]),
                "positive_path": str(ap[1]),
                "negative_path": str(np_paths[1]),
                "anchor_ed": 0.0,
                "positive_ed": ed,
                "negative_ed": n_ed,
                "anchor_page": p,
                "positive_page": p,
                "negative_page": np_p,
                "anchor_variant": v,
                "positive_variant": v,
                "negative_variant": nv,
                "kind": "cross_page",
            })
    return triplets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/omnidocbench"),
                   help="Root containing ocr_<variant>/<page>/{masked_original,reconstructed}.png")
    p.add_argument("--comparison-glob", type=str,
                   default="results/method_runs/ocr_*/baseline/comparison.json",
                   help="Glob for comparison.json files to pull text_edit_distance from")
    p.add_argument("--output", type=Path, default=Path("data/docsim_triplets/manifest.jsonl"))
    p.add_argument("--min-edit-gap", type=float, default=0.15,
                   help="Minimum (negative_ed - positive_ed) for a triplet to count")
    p.add_argument("--cross-page-multiplier", type=int, default=20,
                   help="Per qualifying anchor, how many cross-page negatives to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, cap total triplets after sampling (for debugging)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("build_docsim_triplets")

    rng = random.Random(args.seed)

    # 1) Discover comparison.json files
    project_root = Path.cwd()
    comparison_paths = sorted(project_root.glob(args.comparison_glob))
    if not comparison_paths:
        log.error("No comparison.json files matched glob: %s", args.comparison_glob)
        return 1
    log.info("Found %d comparison.json files", len(comparison_paths))

    # 2) Build per-page edit-distance map
    ed_map = load_per_page_edit_distance(comparison_paths)
    log.info("Loaded edit_distance for %d (variant,page) pairs", len(ed_map))
    if not ed_map:
        log.error("No (variant,page) edit distances loaded; aborting")
        return 1
    log.info("ED stats: min=%.3f median=%.3f max=%.3f",
             min(ed_map.values()),
             sorted(ed_map.values())[len(ed_map) // 2],
             max(ed_map.values()))

    # 3) Same-page triplets (high quality)
    same_page = build_same_page_triplets(ed_map, args.data_root, args.min_edit_gap)
    log.info("Same-page triplets: %d (gap >= %.2f)", len(same_page), args.min_edit_gap)

    # 4) Cross-page triplets (augmentation)
    cross_page = build_cross_page_triplets(
        ed_map, args.data_root, args.min_edit_gap, args.cross_page_multiplier, rng
    )
    log.info("Cross-page triplets: %d (multiplier=%d)", len(cross_page), args.cross_page_multiplier)

    triplets = same_page + cross_page
    rng.shuffle(triplets)
    if args.limit > 0:
        triplets = triplets[: args.limit]
        log.info("Capped to --limit=%d", args.limit)

    log.info("Total triplets: %d  (%d same-page + %d cross-page)",
             len(triplets), len(same_page), len(cross_page))

    # 5) Write manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")
    log.info("Wrote %s (%d lines, %.1f MB)",
             args.output, len(triplets), args.output.stat().st_size / 1e6)

    # 6) Sanity sample
    log.info("Sample triplet:\n%s", json.dumps(triplets[0], indent=2) if triplets else "(empty)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
