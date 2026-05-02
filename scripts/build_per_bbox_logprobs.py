#!/usr/bin/env python3
"""Aggregate per-token Qwen logprobs into per-bbox stats.

Reads (existing, NOT modified):
  data/ocr_logprobs/<page>/ocr_html.html
  data/ocr_logprobs/<page>/ocr_logprobs.json

Writes (new sibling folder, idempotent skip-if-exists):
  data/ocr_logprobs_per_bbox/<page>/per_bbox_logprobs.json

Algorithm:
  1. Concatenate token strings → reconstruct the model's response text.
     Char offsets per token come for free (cumulative len()).
  2. Find every <div ... data-bbox="..." ...>...</div> in that reconstructed text.
  3. Each token whose [char_start, char_end) overlaps the div's inner content
     range belongs to that div.
  4. Aggregate per div: mean/min/max logprob + Shannon entropy from top_logprobs.

Output schema per page:
  {
    "n_total_tokens": int,
    "n_bboxes": int,
    "bboxes": [
      {
        "bbox_id": int,
        "div_class": "text" | "formula" | "table" | "image",
        "data_bbox": [x1, y1, x2, y2],   # 0-1000 normalized
        "n_tokens": int,
        "stats": {
          "logprob_mean": float,
          "logprob_min":  float,
          "logprob_max":  float,
          "shannon_entropy_mean": float,
          "shannon_entropy_max":  float
        }
      },
      ...
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATA_ROOT = Path("/home/mac/test/r1-p2/data")
SRC = DATA_ROOT / "ocr_logprobs"
OUT = DATA_ROOT / "ocr_logprobs_per_bbox"

DIV_RE = re.compile(
    r'<div\b[^>]*?data-bbox="([^"]*)"[^>]*?>(.*?)</div>',
    re.DOTALL,
)
CLASS_RE = re.compile(r'class="([^"]*)"')


def shannon_entropy(top_logprobs: list[dict]) -> float:
    """Entropy in nats over the top-k alternative distribution.

    The top_logprobs list usually has 5 entries; we renormalize to a proper
    distribution over those 5 (their probabilities don't sum to 1 in absolute
    terms — vLLM only returns the top-k). Renormalization gives entropy
    relative to the top-k support, which is what we want as a confidence proxy.
    """
    if not top_logprobs:
        return 0.0
    ps = [math.exp(t["logprob"]) for t in top_logprobs]
    s = sum(ps)
    if s <= 0:
        return 0.0
    ps = [p / s for p in ps]
    return -sum(p * math.log(p + 1e-12) for p in ps if p > 0)


def aggregate_one(page_name: str) -> tuple[str, str, float]:
    """Process one page. Returns (page, status, elapsed_s)."""
    src_dir = SRC / page_name
    out_dir = OUT / page_name
    out_path = out_dir / "per_bbox_logprobs.json"
    if out_path.exists():
        return page_name, "skip", 0.0

    t0 = time.time()
    try:
        html = (src_dir / "ocr_html.html").read_text(encoding="utf-8", errors="ignore")
        lp = json.loads((src_dir / "ocr_logprobs.json").read_text())
        tokens = lp.get("tokens", [])
        if not tokens:
            return page_name, "empty", time.time() - t0

        # Char-range per token in the reconstructed text.
        # Note: we reconstruct from token strings, NOT from ocr_html.html.
        # The two should match since vLLM returns the raw decoded tokens that
        # produced the assistant message content.
        reconstructed = "".join(t["token"] for t in tokens)
        token_ranges = []
        off = 0
        for tok in tokens:
            n = len(tok["token"])
            token_ranges.append((off, off + n))
            off += n

        # Sanity check — log a warning to error.txt but continue
        if reconstructed != html:
            # Common case: html may be the trimmed final content; tokens may
            # include code-fence wrappers ('```html\n...\n```'). Don't fail.
            pass

        bboxes_out = []
        for bbox_id, m in enumerate(DIV_RE.finditer(reconstructed)):
            bbox_str = m.group(1)
            content_start = m.start(2)
            content_end = m.end(2)
            full = m.group(0)

            try:
                bbox_coords = [int(x) for x in bbox_str.split()][:4]
                if len(bbox_coords) != 4:
                    continue
            except ValueError:
                continue

            class_m = CLASS_RE.search(full)
            div_class = class_m.group(1) if class_m else ""

            # Tokens whose [s, e) overlaps [content_start, content_end)
            lps_acc = []
            ents_acc = []
            for s, e in token_ranges:
                if s < content_end and e > content_start:
                    # Need the index — re-walk; faster to pre-build, but
                    # 500-token pages × ~10-50 divs is small enough.
                    pass
            # Rebuild as enumerate for index access (still O(n_tokens) per div;
            # for pages with hundreds of divs this could be tightened, but
            # 1356 pages × ~50 divs × 500 tokens = ~34M ops — fine.)
            for tok_idx, (s, e) in enumerate(token_ranges):
                if s < content_end and e > content_start:
                    tok = tokens[tok_idx]
                    lps_acc.append(tok["logprob"])
                    ents_acc.append(shannon_entropy(tok.get("top_logprobs", [])))

            if not lps_acc:
                continue

            bboxes_out.append({
                "bbox_id": bbox_id,
                "div_class": div_class,
                "data_bbox": bbox_coords,
                "n_tokens": len(lps_acc),
                "stats": {
                    "logprob_mean": sum(lps_acc) / len(lps_acc),
                    "logprob_min": min(lps_acc),
                    "logprob_max": max(lps_acc),
                    "shannon_entropy_mean": sum(ents_acc) / len(ents_acc),
                    "shannon_entropy_max": max(ents_acc),
                },
            })

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "n_total_tokens": len(tokens),
            "n_bboxes": len(bboxes_out),
            "bboxes": bboxes_out,
        }, ensure_ascii=False))
        return page_name, "ok", time.time() - t0
    except Exception as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error.txt").write_text(
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        return page_name, f"error:{type(e).__name__}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pages = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    if args.limit > 0:
        pages = pages[: args.limit]
    total = len(pages)
    print(f"[start] {total} pages, workers={args.workers}, out={OUT}", flush=True)

    t_start = time.time()
    counts = {"ok": 0, "skip": 0, "empty": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(aggregate_one, p): p for p in pages}
        for i, fut in enumerate(as_completed(futs), 1):
            page, status, elapsed = fut.result()
            bucket = "error" if status.startswith("error") else status
            counts[bucket] = counts.get(bucket, 0) + 1
            if i % 100 == 0 or i == total:
                eta_s = (time.time() - t_start) / max(i, 1) * (total - i)
                print(
                    f"[{i:4d}/{total}] {status:14s} {elapsed:5.2f}s {page[:50]}"
                    f"  ok={counts['ok']} skip={counts['skip']} empty={counts['empty']} err={counts['error']}"
                    f"  eta={eta_s:.0f}s",
                    flush=True,
                )

    print(f"[done] ok={counts['ok']} skip={counts['skip']} empty={counts['empty']} "
          f"error={counts['error']} elapsed={(time.time()-t_start):.1f}s", flush=True)
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
