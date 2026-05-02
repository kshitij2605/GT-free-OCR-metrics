#!/usr/bin/env python3
"""P1_215 — D215: per-page Qwen confidence_mean as cc replacement.

Diagnostic of token-level self-uncertainty signal. For each page:
  1) Compute production multi_composite (H5.b: SSIM+MSE+LPIPS [+DISTS for
     variant=all]) — preserves baseline cc-alternative for the eval's
     best-of-fields pick.
  2) Compute clip_compare.clip_cosine = confidence_mean = mean(exp(token_lp))
     over Qwen OCR token logprobs at data/ocr_logprobs/<page>/ocr_logprobs.json.
     Universal per-page scalar (variant-independent).

If confidence_mean's per-variant Spearman beats the corresponding D60.p
production per-variant cc (text 0.4139, formula 0.5488, table 0.4767, all
0.3934, all_no_mask 0.3937), that variant ceiling is liftable via a fusion
follow-up. Otherwise the entropy axis is REFUTED at the page-level
granularity (D224 per-bbox would still be testable).
"""

import json
import logging
import math
import sys
import time
from pathlib import Path

import lpips as lpips_lib
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_215_qwen_logprob_confidence_mean"
SSIM_SIZE = 512

LOGPROBS_ROOT = Path(__file__).parent.parent.parent / "data" / "ocr_logprobs"

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <variant>", file=sys.stderr)
    sys.exit(1)

variant = sys.argv[1]
if variant not in {"all", "text", "formula", "table", "all_no_mask"}:
    print(f"Unknown variant: {variant}", file=sys.stderr)
    sys.exit(1)

USE_DISTS = variant == "all"

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
log.info(
    "device=%s variant=%s method=%s use_dists=%s",
    device, variant, METHOD_ID, USE_DISTS,
)

_preprocessor = ImagePreprocessor()

# ── DISTS setup (variant=all only, mirrors P1_120c3b) ─────────────────────────
if USE_DISTS:
    import pyiqa
    log.info("Loading pyiqa DISTS metric (variant=all only)...")
    _dists_metric = pyiqa.create_metric("dists", as_loss=False).to(device)
    _dists_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
else:
    _dists_metric = None
    _dists_transform = None


def _dists_score(orig_pil: Image.Image, recon_pil: Image.Image) -> float:
    orig_t = _dists_transform(orig_pil.convert("RGB")).unsqueeze(0).to(device)
    recon_t = _dists_transform(recon_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = _dists_metric(orig_t, recon_t)
    return float(np.clip(dist.item(), 0.0, 1.0))


# ── LPIPS setup ───────────────────────────────────────────────────────────────
log.info("Loading LPIPS (alex)...")
_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)


def _to_gray_small(pil: Image.Image) -> Image.Image:
    return pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR)


def _ssim_mse(orig_pil: Image.Image, recon_pil: Image.Image) -> tuple:
    orig_small = _to_gray_small(orig_pil)
    recon_small = _to_gray_small(recon_pil)
    orig_b = _preprocessor.adaptive_binarize(orig_small.convert("RGB"))
    recon_b = _preprocessor.adaptive_binarize(recon_small.convert("RGB"))
    if recon_b.size != orig_b.size:
        recon_b = recon_b.resize(orig_b.size, Image.BILINEAR)
    og = np.array(orig_b, dtype=np.float64) / 255.0
    rg = np.array(recon_b, dtype=np.float64) / 255.0
    ssim_val = float(structural_similarity(og, rg, data_range=1.0))
    mse_val = float(np.mean((og - rg) ** 2))
    return ssim_val, mse_val


def _lpips_score(orig_pil: Image.Image, recon_pil: Image.Image) -> float:
    def _to_t(p):
        g = p.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR)
        arr = np.array(g, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        return (t * 2.0 - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = _lpips_fn(_to_t(orig_pil), _to_t(recon_pil))
    return float(np.clip(dist.item(), 0.0, 1.0))


# ── confidence_mean from logprobs ─────────────────────────────────────────────
def _confidence_mean(page_name: str) -> float:
    """mean(exp(token_logprob)) over Qwen OCR tokens for this page.

    Returns 0.0 if logprobs file is missing or empty (treated as
    minimal-confidence fallback so eval still ranks the page).
    """
    lp_path = LOGPROBS_ROOT / page_name / "ocr_logprobs.json"
    if not lp_path.exists():
        return 0.0
    with open(lp_path) as f:
        data = json.load(f)
    tokens = data.get("tokens", [])
    if not tokens:
        return 0.0
    probs = []
    for t in tokens:
        lp = t.get("logprob")
        if lp is None:
            continue
        probs.append(math.exp(lp))
    if not probs:
        return 0.0
    return float(np.mean(probs))


# ── Main processing loop ──────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
missing_lp = 0

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

    if USE_DISTS:
        dists_val = _dists_score(orig_pil, recon_pil)
        composite = (
            0.3 * ssim_val
            + 0.2 * max(0.0, 1.0 - mse_val)
            + 0.2 * max(0.0, 1.0 - lpips_val)
            + 0.3 * max(0.0, 1.0 - dists_val)
        )
        multi_metric = {
            "ssim": ssim_val, "mse": mse_val, "lpips": lpips_val,
            "dists": dists_val, "composite": composite,
        }
    else:
        composite = 0.4 * ssim_val + 0.3 * max(0.0, 1.0 - mse_val) + 0.3 * max(0.0, 1.0 - lpips_val)
        multi_metric = {
            "ssim": ssim_val, "mse": mse_val, "lpips": lpips_val,
            "composite": composite,
        }

    conf = _confidence_mean(page_dir.name)
    if conf == 0.0 and not (LOGPROBS_ROOT / page_dir.name / "ocr_logprobs.json").exists():
        missing_lp += 1

    meta = {
        "image": page_dir.name,
        "text_elements": 0, "image_regions": 0, "table_regions": 0,
        "text_length": 0, "plain_text_length": 0,
        "multi_metric": multi_metric,
        "clip_compare": {"clip_cosine": conf},
        "lm_perplexity": {
            "ngram_score": 0.0, "transformer_score": 0.0,
            "perplexity": 0.0, "composite": 0.0,
        },
    }
    results.append(meta)

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        log.info("[%d/%d] %.1fs elapsed (%.2fs/page) missing_lp=%d",
                 i + 1, len(page_dirs), elapsed, elapsed / (i + 1), missing_lp)

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s (missing_lp=%d)", len(results), out_path, missing_lp)
