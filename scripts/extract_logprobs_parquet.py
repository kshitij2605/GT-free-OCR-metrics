#!/usr/bin/env python3
"""Inverse of build_logprobs_parquet.py — extract parquet shards back into:
  data/ocr_logprobs/<page_id>/{ocr_html.html, ocr_logprobs.json}
  data/ocr_logprobs_per_bbox/<page_id>/per_bbox_logprobs.json

Usage:
  uv run python scripts/extract_logprobs_parquet.py [--in parquet_logprobs/] [--lp-out data/ocr_logprobs/] [--pb-out data/ocr_logprobs_per_bbox/]
"""
import argparse
from pathlib import Path

import pandas as pd


def materialize_shard(shard: Path, lp_root: Path, pb_root: Path) -> int:
    df = pd.read_parquet(shard)
    n = 0
    for _, row in df.iterrows():
        page_id = row["page_id"]
        lp_dir = lp_root / page_id
        lp_dir.mkdir(parents=True, exist_ok=True)
        if row["ocr_html"]:
            (lp_dir / "ocr_html.html").write_text(row["ocr_html"], encoding="utf-8")
        if row["ocr_logprobs"]:
            (lp_dir / "ocr_logprobs.json").write_text(row["ocr_logprobs"], encoding="utf-8")
        if row["per_bbox_logprobs"]:
            pb_dir = pb_root / page_id
            pb_dir.mkdir(parents=True, exist_ok=True)
            (pb_dir / "per_bbox_logprobs.json").write_text(row["per_bbox_logprobs"], encoding="utf-8")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=Path("parquet_logprobs"))
    ap.add_argument("--lp-out", type=Path, default=Path("data/ocr_logprobs"))
    ap.add_argument("--pb-out", type=Path, default=Path("data/ocr_logprobs_per_bbox"))
    args = ap.parse_args()

    shards = sorted(args.in_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No .parquet shards in {args.in_dir}")

    print(f"Found {len(shards)} parquet shards in {args.in_dir}")
    total = 0
    for s in shards:
        n = materialize_shard(s, args.lp_out, args.pb_out)
        print(f"  {s.name}: {n} pages extracted")
        total += n
    print(f"Total pages: {total}")


if __name__ == "__main__":
    main()
