#!/usr/bin/env python3
"""Inverse of build_parquet_dataset.py — materialize parquet shards back to
on-disk per-page directories so existing method scripts keep working.

After download_data.sh fetches parquet shards, this script writes:
  data/omnidocbench/ocr_<variant>/<page_id>/{masked_original.png, ...}

Usage:
  uv run python scripts/extract_parquet_to_disk.py [--in parquet_in/] [--out data/omnidocbench/]
"""
import argparse
from pathlib import Path

import pandas as pd

PER_PAGE_FILES = {
    "masked_original": ("masked_original.png", "bytes"),
    "reconstructed":   ("reconstructed.png",   "bytes"),
    "ocr_html":        ("ocr_html.html",       "text"),
    "ocr_elements":    ("ocr_elements.json",          "text"),
    "ocr_formula_elements": ("ocr_formula_elements.json", "text"),
    "ocr_table_elements":   ("ocr_table_elements.json",   "text"),
}


def materialize_shard(shard_path: Path, out_root: Path) -> int:
    df = pd.read_parquet(shard_path)
    n = 0
    for _, row in df.iterrows():
        page_dir = out_root / row["variant"] / row["page_id"]
        page_dir.mkdir(parents=True, exist_ok=True)
        for col, (fname, kind) in PER_PAGE_FILES.items():
            data = row[col]
            target = page_dir / fname
            if target.exists():
                continue  # idempotent
            if kind == "bytes":
                if data:
                    target.write_bytes(data)
            else:
                if data:
                    target.write_text(data, encoding="utf-8")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=Path("parquet_in"),
                    help="Directory containing *.parquet shards")
    ap.add_argument("--out", type=Path, default=Path("data/omnidocbench"),
                    help="Destination root for ocr_<variant>/<page_id>/ tree")
    args = ap.parse_args()

    shards = sorted(args.in_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No .parquet shards found in {args.in_dir}")

    print(f"Found {len(shards)} parquet shards in {args.in_dir}")
    total = 0
    for shard in shards:
        n = materialize_shard(shard, args.out)
        print(f"  {shard.name}: {n} pages extracted")
        total += n
    print(f"\nTotal pages extracted: {total}")


if __name__ == "__main__":
    main()
