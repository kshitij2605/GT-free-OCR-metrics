#!/usr/bin/env python3
"""P1_106 — D106 Frontiers-DIQA hand-crafted IQ features.

Per page: extract 6 classical IQ features (focus/Laplacian variance,
gradient magnitude, contrast, edge density, brightness, color variance)
from both orig and recon. Similarity = exp(-mean per-feature relative
distance). Pure-OpenCV; truly orthogonal to current learned-feature
stack.

multi_composite (SSIM/MSE/LPIPS) preserved from baseline so the eval
pipeline has the production fallback. clip_compare.clip_cosine holds
the D106 standalone score.
"""

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import lpips as lpips_lib
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_106_frontiers_diqa_hand_crafted"
SSIM_SIZE = 512
EPS = 1e-6

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <variant>", file=sys.stderr)
    sys.exit(1)

variant = sys.argv[1]
if variant not in {"all", "text", "formula", "table", "all_no_mask"}:
    print(f"Unknown variant: {variant}", file=sys.stderr)
    sys.exit(1)

BASE = Path(__file__).parent.parent.parent / "data" / "omnidocbench"
var_root = BASE / f"ocr_{variant}"
OUT_DIR = Path(__file__).parent.parent.parent / "results" / "method_runs" / f"ocr_{variant}" / METHOD_ID
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

_preprocessor = ImagePreprocessor()

log.info("Loading LPIPS (alex)...")
_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)


def _to_gray_small(pil):
    return pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR)


def _ssim_mse(orig_pil, recon_pil):
    orig_small = _to_gray_small(orig_pil)
    recon_small = _to_gray_small(recon_pil)
    orig_b = _preprocessor.adaptive_binarize(orig_small.convert("RGB"))
    recon_b = _preprocessor.adaptive_binarize(recon_small.convert("RGB"))
    if recon_b.size != orig_b.size:
        recon_b = recon_b.resize(orig_b.size, Image.BILINEAR)
    og = np.array(orig_b, dtype=np.float64) / 255.0
    rg = np.array(recon_b, dtype=np.float64) / 255.0
    return float(structural_similarity(og, rg, data_range=1.0)), float(np.mean((og - rg) ** 2))


def _lpips_score(orig_pil, recon_pil):
    def _to_t(p):
        g = p.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR)
        arr = np.array(g, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        return (t * 2.0 - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(np.clip(_lpips_fn(_to_t(orig_pil), _to_t(recon_pil)).item(), 0.0, 1.0))


def _diqa_features(pil):
    """Extract 6 classical IQ features as a length-6 numpy vector."""
    arr = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    # 1. Focus / sharpness via Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    focus = float(lap.var())
    # 2. Mean Sobel gradient magnitude
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(sx * sx + sy * sy)))
    # 3. Grayscale contrast = std
    contrast = float(gray.std())
    # 4. Canny edge density = mean of binary edge map
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    edge_density = float(edges.mean()) / 255.0  # in [0, 1]
    # 5. Brightness = mean grayscale
    brightness = float(gray.mean())
    # 6. Color variance = mean over 3 channel stds
    color_var = float(np.mean([arr[..., c].std() for c in range(3)]))
    return np.array([focus, grad_mag, contrast, edge_density, brightness, color_var], dtype=np.float64)


def _diqa_similarity(orig_pil, recon_pil):
    """exp(-mean per-feature relative absolute distance) in (0, 1]."""
    f_o = _diqa_features(orig_pil)
    f_r = _diqa_features(recon_pil)
    denom = np.maximum.reduce([np.abs(f_o), np.abs(f_r), np.full_like(f_o, EPS)])
    rel_diff = np.abs(f_o - f_r) / denom
    mean_dist = float(rel_diff.mean())
    return float(np.exp(-mean_dist))


# ── Main ──────────────────────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
t0 = time.time()
for i, page_dir in enumerate(page_dirs):
    orig_path = page_dir / "masked_original.png"
    recon_path = page_dir / "reconstructed.png"
    if not orig_path.exists() or not recon_path.exists():
        continue

    orig_pil = Image.open(orig_path).convert("RGB")
    recon_pil = Image.open(recon_path).convert("RGB")

    ssim_val, mse_val = _ssim_mse(orig_pil, recon_pil)
    lpips_val = _lpips_score(orig_pil, recon_pil)
    composite = 0.4 * ssim_val + 0.3 * max(0.0, 1.0 - mse_val) + 0.3 * max(0.0, 1.0 - lpips_val)
    multi_metric = {"ssim": ssim_val, "mse": mse_val, "lpips": lpips_val, "composite": composite}

    diqa_sim = _diqa_similarity(orig_pil, recon_pil)

    meta = {
        "image": page_dir.name,
        "text_elements": 0, "image_regions": 0, "table_regions": 0,
        "text_length": 0, "plain_text_length": 0,
        "multi_metric": multi_metric,
        "lm_perplexity": {
            "ngram_score": 0.0, "transformer_score": 0.0,
            "perplexity": 0.0, "composite": 0.0,
        },
        "clip_compare": {"clip_cosine": float(diqa_sim)},
    }
    results.append(meta)

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        log.info("[%d/%d] %.1fs elapsed (%.2fs/page)",
                 i + 1, len(page_dirs), elapsed, elapsed / (i + 1))

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s (D106 hand-crafted IQ features)", len(results), out_path)
