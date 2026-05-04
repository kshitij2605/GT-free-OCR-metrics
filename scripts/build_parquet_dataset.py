#!/usr/bin/env python3
"""Repackage data/omnidocbench/ocr_<variant>/ directories as parquet shards.

Parquet layout (one row per page):
  page_id              str
  variant              str    # ocr_all / ocr_text / ocr_formula / ocr_table / ocr_all_no_mask
  masked_original      bytes  # PNG
  reconstructed        bytes  # PNG
  ocr_html             str    # HTML text
  ocr_elements         str    # JSON text
  ocr_formula_elements str    # JSON text
  ocr_table_elements   str    # JSON text

Sharded at 200 rows/shard to keep individual files well under HF's 5-GB
recommended max and to stay comfortably under the per-file rate-limit weight.
Result: ~25 parquet files for the whole dataset versus ~23,600 small files.

Usage:
  uv run python scripts/build_parquet_dataset.py [--out parquet_out/]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

VARIANTS = ["ocr_all", "ocr_text", "ocr_formula", "ocr_table", "ocr_all_no_mask"]
ROWS_PER_SHARD = 75

PER_PAGE_FILES = {
    "masked_original": ("masked_original.png", "bytes"),
    "reconstructed":   ("reconstructed.png",   "bytes"),
    "ocr_html":        ("ocr_html.html",       "text"),
    "ocr_elements":    ("ocr_elements.json",          "text"),
    "ocr_formula_elements": ("ocr_formula_elements.json", "text"),
    "ocr_table_elements":   ("ocr_table_elements.json",   "text"),
}


def page_to_row(variant: str, page_dir: Path) -> dict:
    row = {"page_id": page_dir.name, "variant": variant}
    for col, (fname, kind) in PER_PAGE_FILES.items():
        f = page_dir / fname
        if not f.exists():
            row[col] = b"" if kind == "bytes" else ""
            continue
        if kind == "bytes":
            row[col] = f.read_bytes()
        else:
            row[col] = f.read_text(encoding="utf-8")
    return row


def build_variant(src_root: Path, variant: str, out_dir: Path) -> int:
    var_dir = src_root / variant
    if not var_dir.exists():
        print(f"  SKIP {variant}: {var_dir} not found")
        return 0

    page_dirs = sorted(p for p in var_dir.iterdir() if p.is_dir())
    print(f"  {variant}: {len(page_dirs)} pages")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = 0
    for shard_idx in range((len(page_dirs) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD):
        chunk = page_dirs[shard_idx * ROWS_PER_SHARD : (shard_idx + 1) * ROWS_PER_SHARD]
        rows = [page_to_row(variant, p) for p in chunk]
        df = pd.DataFrame(rows)
        shard_path = out_dir / f"{variant}-shard{shard_idx:04d}.parquet"
        df.to_parquet(shard_path, compression="zstd", index=False)
        size_mb = shard_path.stat().st_size / 1e6
        print(f"    {shard_path.name}: {len(rows)} rows, {size_mb:.1f} MB")
        n_shards += 1
    return n_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/omnidocbench"),
                    help="Root with ocr_<variant>/ subdirs")
    ap.add_argument("--out", type=Path, default=Path("parquet_out"),
                    help="Where to write per-variant parquet shards")
    args = ap.parse_args()

    print(f"Source: {args.src.resolve()}")
    print(f"Output: {args.out.resolve()}")

    total = 0
    for v in VARIANTS:
        total += build_variant(args.src, v, args.out)
    print(f"\nTotal shards written: {total}")


if __name__ == "__main__":
    main()
