#!/usr/bin/env python3
"""P1_067: PDQ 256-bit perceptual hash (D67).

Uses imagehash.phash with hash_size=16 (256 bits) as a proxy for PDQ
(Facebook's 256-bit perceptual hash). Unlike D68 (4x4 grid MIN supplement),
this is a full-page standalone metric: hash similarity of the whole page.

The 256-bit resolution should capture finer spatial details than the 64-bit
pHash in D66's composite. Key difference from D66: standalone signal
(not in multi_composite slot) and larger hash = more discriminative bits.

clip_cosine = 1 - hamming_distance / 256

Standalone metric — no D107b blend. Tests whether 256-bit PDQ-proxy hash
correlates with edit distance better than D66's 0.2772.
KEEP if spearman_mean > 0.4820 (D107b best).
"""

import json
import logging
import sys
import time
from pathlib import Path

import imagehash
from PIL import Image

METHOD_ID = "P1_067_pdq_256bit_hash"
HASH_SIZE = 16  # 16*16 = 256 bits
HASH_BITS = HASH_SIZE * HASH_SIZE

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
log.info("variant=%s method=%s hash_bits=%d", variant, METHOD_ID, HASH_BITS)


def _pdq_sim(orig_pil: Image.Image, recon_pil: Image.Image) -> float:
    orig_g = orig_pil.convert("L")
    recon_g = recon_pil.convert("L")
    h1 = imagehash.phash(orig_g, hash_size=HASH_SIZE)
    h2 = imagehash.phash(recon_g, hash_size=HASH_SIZE)
    hamming = h1 - h2
    return float(max(0.0, 1.0 - hamming / HASH_BITS))


t0 = time.time()
page_dirs = sorted(p for p in var_root.iterdir() if p.is_dir())
log.info("Processing %d page dirs for variant=%s", len(page_dirs), variant)

results = []
for i, page_dir in enumerate(page_dirs):
    orig_path = page_dir / "masked_original.png"
    recon_path = page_dir / "reconstructed.png"
    if not orig_path.exists() or not recon_path.exists():
        continue

    orig_pil = Image.open(orig_path).convert("RGB")
    recon_pil = Image.open(recon_path).convert("RGB")

    pdq_sim = _pdq_sim(orig_pil, recon_pil)

    meta = {
        "image": page_dir.name,
        "text_elements": 0,
        "image_regions": 0,
        "table_regions": 0,
        "text_length": 0,
        "plain_text_length": 0,
        "multi_metric": {
            "pdq_sim": pdq_sim,
            "composite": pdq_sim,
        },
        "lm_perplexity": {
            "ngram_score": 0.0,
            "transformer_score": 0.0,
            "perplexity": 0.0,
            "composite": 0.0,
        },
        "clip_compare": {"clip_cosine": pdq_sim},
    }
    results.append(meta)

    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0
        log.info("  %d/%d pages done — %.1fs elapsed", i + 1, len(page_dirs), elapsed)

out_path = OUT_DIR / "results.json"
with out_path.open("w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t0
log.info("Done: %d pages, output=%s, time=%.1fs", len(results), out_path, elapsed)
