#!/usr/bin/env python3
"""P1_082: ST-LPIPS (shift-tolerant LPIPS) drop-in for the LPIPS slot.

Replaces standard LPIPS with ST-LPIPS (Ghildyal & Liu, ECCV 2022) in the
multi_composite formula: 0.4*SSIM + 0.3*(1-MSE) + 0.3*(1-ST_LPIPS).

SSIM is computed via skimage on binarized 512x512 grayscale images (same
convention as the baseline). MSE is unchanged. CLIP cosine batch is unchanged.
ST-LPIPS result is stored in the 'lpips' key so compute_variant_metrics.py
consumes the output without modification.

Reference: priorities.md #82; Ghildyal & Liu, ECCV 2022
  (https://github.com/abhijay9/ShiftTolerant-LPIPS).
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_082_st_lpips"
SSIM_SIZE = 512
LPIPS_SIZE = 256
BATCH_SIZE = 16

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


def _binarize(pil: Image.Image, size: int) -> Image.Image:
    resized = pil.convert("RGB").resize((size, size), Image.LANCZOS)
    binarized = _preprocessor.adaptive_binarize(resized)
    return binarized.convert("RGB")


def _ssim_score(orig_pil: Image.Image, recon_pil: Image.Image, size: int = SSIM_SIZE) -> float:
    orig_b = np.array(_binarize(orig_pil, size).convert("L"))
    recon_b = np.array(_binarize(recon_pil, size).convert("L"))
    score = ssim_fn(orig_b, recon_b, data_range=255)
    return float(score)


def _mse(orig_pil: Image.Image, recon_pil: Image.Image, size: int = SSIM_SIZE) -> float:
    orig_b = np.array(_binarize(orig_pil, size).convert("L")).astype(np.float32)
    recon_b = np.array(_binarize(recon_pil, size).convert("L")).astype(np.float32)
    return float(np.mean((orig_b - recon_b) ** 2) / 255.0 ** 2)


log.info("Loading ST-LPIPS (stlpips_pytorch)...")
from stlpips_pytorch.stlpips import LPIPS as ST_LPIPS
_st_lpips_fn = ST_LPIPS(net="alex", variant="shift_tolerant").to(device).eval()
log.info("ST-LPIPS loaded (shift_tolerant, alex backbone)")


def _st_lpips_score(orig_pil: Image.Image, recon_pil: Image.Image) -> float:
    def _to_t(p):
        arr = (
            np.array(p.convert("RGB").resize((LPIPS_SIZE, LPIPS_SIZE), Image.LANCZOS))
            .astype(np.float32) / 127.5 - 1.0
        )
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = _st_lpips_fn(_to_t(orig_pil), _to_t(recon_pil))
    return float(dist.item())


log.info("Loading CLIP...")
_clip_model, _, _clip_prep = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)


def _clip_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    def _enc(pils):
        batch = torch.stack([_clip_prep(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_image(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


def _process_batch_clip():
    global orig_pils_buf, recon_pils_buf, meta_buf
    if not orig_pils_buf:
        return
    cos_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
    for meta, cos in zip(meta_buf, cos_sims):
        meta["clip_compare"] = {"clip_cosine": cos}
        results.append(meta)
    orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


t0 = time.time()
for i, page_dir in enumerate(page_dirs):
    orig_path = page_dir / "masked_original.png"
    recon_path = page_dir / "reconstructed.png"
    if not orig_path.exists() or not recon_path.exists():
        continue

    orig_pil = Image.open(orig_path).convert("RGB")
    recon_pil = Image.open(recon_path).convert("RGB")

    ssim_score = _ssim_score(orig_pil, recon_pil)
    mse_score = _mse(orig_pil, recon_pil)
    st_lpips_score = _st_lpips_score(orig_pil, recon_pil)
    composite = (
        0.4 * ssim_score
        + 0.3 * max(0.0, 1.0 - mse_score)
        + 0.3 * max(0.0, 1.0 - st_lpips_score)
    )

    meta = {
        "image": page_dir.name,
        "text_elements": 0,
        "image_regions": 0,
        "table_regions": 0,
        "text_length": 0,
        "plain_text_length": 0,
        "multi_metric": {
            "ssim": ssim_score,
            "mse": mse_score,
            "lpips": st_lpips_score,   # ST-LPIPS stored in 'lpips' slot for compatibility
            "composite": composite,
        },
        "lm_perplexity": {"ngram_score": 0.0, "transformer_score": 0.0, "perplexity": 0.0, "composite": 0.0},
    }

    orig_pils_buf.append(orig_pil)
    recon_pils_buf.append(recon_pil)
    meta_buf.append(meta)

    if len(orig_pils_buf) >= BATCH_SIZE:
        _process_batch_clip()

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        log.info("[%d/%d] %.1fs elapsed (%.2fs/page)", i + 1, len(page_dirs), elapsed, elapsed / (i + 1))

_process_batch_clip()

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s", len(results), out_path)
