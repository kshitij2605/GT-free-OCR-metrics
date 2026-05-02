#!/usr/bin/env python3
"""P2_059: PieAPP as PIM/MILO_L proxy (D59).

PieAPP (Perceptual Image-Error Assessment through Pairwise Preference)
is a learned full-reference perceptual metric trained on human judgments.
It measures perceptual error between original and reconstructed pages.

Used as a proxy for PIM (Unsupervised Information-Theoretic Perceptual
Quality Metric) and MILO_L (Stable Diffusion VAE latent comparison) since
those require TF or diffusers which are not installed.

PieAPP returns lower values for more similar images. We convert to
higher=better: clip_cosine = max(0, 1 - pieapp_score / 2).

Standalone metric — no D107b blend. Tests whether learned perceptual
distance in human-preference feature space correlates with edit distance.
KEEP if spearman_mean > 0.4820 (D107b best).
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import piq
import torch
import torchvision.transforms.functional as TF
from PIL import Image

METHOD_ID = "P2_059_pieapp_perceptual_metric"
IMG_SIZE = 256  # PieAPP works best at 256x256

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

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("device=%s variant=%s method=%s", device, variant, METHOD_ID)

# Load PieAPP metric
log.info("Loading PieAPP...")
_pieapp = piq.PieAPP(reduction="none", stride=27).to(device)


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    img = pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    t = TF.to_tensor(img)  # [C, H, W], float32 in [0,1]
    return t.unsqueeze(0).to(device)  # [1, C, H, W]


def _pieapp_score(orig_pil: Image.Image, recon_pil: Image.Image) -> float:
    with torch.no_grad():
        orig_t = _pil_to_tensor(orig_pil)
        recon_t = _pil_to_tensor(recon_pil)
        score = _pieapp(orig_t, recon_t)
    return float(score.item())


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

    pieapp_val = _pieapp_score(orig_pil, recon_pil)
    # Convert lower=better to higher=better; PieAPP ranges 0..~2
    clip_cosine = max(0.0, 1.0 - pieapp_val / 2.0)

    meta = {
        "image": page_dir.name,
        "text_elements": 0,
        "image_regions": 0,
        "table_regions": 0,
        "text_length": 0,
        "plain_text_length": 0,
        "multi_metric": {
            "pieapp": pieapp_val,
            "composite": clip_cosine,
        },
        "lm_perplexity": {
            "ngram_score": 0.0,
            "transformer_score": 0.0,
            "perplexity": 0.0,
            "composite": 0.0,
        },
        "clip_compare": {"clip_cosine": clip_cosine},
    }
    results.append(meta)

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        log.info("  %d/%d pages done — %.1fs elapsed", i + 1, len(page_dirs), elapsed)

out_path = OUT_DIR / "results.json"
with out_path.open("w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t0
log.info("Done: %d pages, output=%s, time=%.1fs", len(results), out_path, elapsed)
