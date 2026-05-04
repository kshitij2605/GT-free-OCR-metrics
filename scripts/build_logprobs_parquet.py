#!/usr/bin/env python3
"""Repackage data/ocr_logprobs/ and data/ocr_logprobs_per_bbox/ as parquet shards.

Schema (one row per page):
  page_id            str
  ocr_html           str   # rendered HTML from logprobs run
  ocr_logprobs       str   # full token-stream logprobs (large JSON)
  per_bbox_logprobs  str   # per-bbox aggregated logprobs

Usage:
  uv run python scripts/build_logprobs_parquet.py [--out parquet_logprobs/]
"""
import argparse
from pathlib import Path

import pandas as pd

ROWS_PER_SHARD = 200


def page_to_row(lp_dir: Path, pb_dir: Path | None, page_id: str) -> dict:
    row = {"page_id": page_id}
    html_path = lp_dir / "ocr_html.html"
    lp_path = lp_dir / "ocr_logprobs.json"
    row["ocr_html"]     = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
    row["ocr_logprobs"] = lp_path.read_text(encoding="utf-8") if lp_path.exists() else ""
    pb_path = (pb_dir / "per_bbox_logprobs.json") if pb_dir else None
    row["per_bbox_logprobs"] = pb_path.read_text(encoding="utf-8") if pb_path and pb_path.exists() else ""
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp-src",  type=Path, default=Path("data/ocr_logprobs"))
    ap.add_argument("--pb-src",  type=Path, default=Path("data/ocr_logprobs_per_bbox"))
    ap.add_argument("--out",     type=Path, default=Path("parquet_logprobs"))
    args = ap.parse_args()

    lp_pages = sorted(p for p in args.lp_src.iterdir() if p.is_dir()) if args.lp_src.exists() else []
    print(f"Source ocr_logprobs: {len(lp_pages)} pages at {args.lp_src.resolve()}")
    print(f"Source per_bbox    : {args.pb_src.resolve()}")
    args.out.mkdir(parents=True, exist_ok=True)

    n_shards = 0
    for shard_idx in range((len(lp_pages) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD):
        chunk = lp_pages[shard_idx * ROWS_PER_SHARD : (shard_idx + 1) * ROWS_PER_SHARD]
        rows = [page_to_row(p, args.pb_src / p.name, p.name) for p in chunk]
        df = pd.DataFrame(rows)
        out = args.out / f"logprobs-shard{shard_idx:04d}.parquet"
        df.to_parquet(out, compression="zstd", index=False)
        size_mb = out.stat().st_size / 1e6
        print(f"  {out.name}: {len(rows)} rows, {size_mb:.1f} MB")
        n_shards += 1
    print(f"Total shards: {n_shards}")


if __name__ == "__main__":
    main()
