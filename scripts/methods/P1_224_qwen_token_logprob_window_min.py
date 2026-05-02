#!/usr/bin/env python3
"""P1_224 — D224: window-MIN of per-token Qwen confidence (entropy axis at finer granularity).

Cheap diagnostic — no image encoder, no SSIM/LPIPS/DISTS. Sets
multi_composite + lm_composite to constants (Spearman becomes NaN), so the
eval picks clip_cosine = window-MIN confidence for every variant. The
per-variant cc Spearman IS the entropy-axis signal in isolation.

Per token in the OCR logprobs sequence, compute confidence = exp(logprob).
Apply rolling window K=10, take mean per window. Take MIN across all
windows = worst-sustained-region confidence. This is the token-level
analogue of D60.p's per-cell-MIN aggregation that broke the table ceiling.
"""

import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

METHOD_ID = "P1_224_qwen_token_logprob_window_min"
WINDOW_K = 10
LOGPROBS_ROOT = Path("/home/mac/test/r1-p2/data/ocr_logprobs")

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <variant>", file=sys.stderr)
    sys.exit(1)

variant = sys.argv[1]
if variant not in {"all", "text", "formula", "table", "all_no_mask"}:
    print(f"Unknown variant: {variant}", file=sys.stderr)
    sys.exit(1)

BASE = Path("/home/mac/test/r1-p2/data/omnidocbench")
var_root = BASE / f"ocr_{variant}"
OUT_DIR = Path("/home/mac/test/r1-p2/results/method_runs") / f"ocr_{variant}" / METHOD_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(f"{variant}_{METHOD_ID}")
log.info("variant=%s method=%s window_k=%d (no image encoder)", variant, METHOD_ID, WINDOW_K)


def _window_min_confidence(page_name: str) -> float:
    """MIN over rolling K-token window means of per-token confidence.

    Returns 0.0 if logprobs missing or token count < window (treated as
    minimal-confidence fallback so the page still ranks).
    """
    lp_path = LOGPROBS_ROOT / page_name / "ocr_logprobs.json"
    if not lp_path.exists():
        return 0.0
    with open(lp_path) as f:
        data = json.load(f)
    tokens = data.get("tokens", [])
    confs = []
    for t in tokens:
        lp = t.get("logprob")
        if lp is None:
            continue
        confs.append(math.exp(lp))
    if not confs:
        return 0.0
    if len(confs) < WINDOW_K:
        # Page too short for windowing — fall back to mean (rare; most pages have ~500 tokens).
        return float(np.mean(confs))
    arr = np.array(confs, dtype=np.float64)
    # Rolling window mean of size WINDOW_K via cumulative sum.
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    window_means = (csum[WINDOW_K:] - csum[:-WINDOW_K]) / WINDOW_K
    return float(window_means.min())


page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
missing_lp = 0
t0 = time.time()
for i, page_dir in enumerate(page_dirs):
    orig_path = page_dir / "masked_original.png"
    if not orig_path.exists():
        continue

    cc = _window_min_confidence(page_dir.name)
    if cc == 0.0 and not (LOGPROBS_ROOT / page_dir.name / "ocr_logprobs.json").exists():
        missing_lp += 1

    meta = {
        "image": page_dir.name,
        "text_elements": 0, "image_regions": 0, "table_regions": 0,
        "text_length": 0, "plain_text_length": 0,
        "multi_metric": {"ssim": 0.0, "mse": 0.0, "lpips": 0.0, "composite": 0.0},
        "clip_compare": {"clip_cosine": cc},
        "lm_perplexity": {
            "ngram_score": 0.0, "transformer_score": 0.0,
            "perplexity": 0.0, "composite": 0.0,
        },
    }
    results.append(meta)

    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0
        log.info("[%d/%d] %.1fs elapsed (%.3fs/page) missing_lp=%d",
                 i + 1, len(page_dirs), elapsed, elapsed / (i + 1), missing_lp)

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s (missing_lp=%d, wall=%.1fs)",
         len(results), out_path, missing_lp, time.time() - t0)
