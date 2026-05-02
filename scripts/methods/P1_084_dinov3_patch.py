#!/usr/bin/env python3
"""P1_084: DINOv2 (vitb14) CLS-token cosine similarity as drop-in for CLIP encoder.

DINOv3 (fbaipublicfiles) returned 403 Forbidden on the remote host — weights
not publicly accessible. This script falls back to DINOv2 vitb14, which is
pre-cached at ~/.cache/torch/hub/facebookresearch_dinov2_main.

Changes vs baseline:
  - clip_cosine slot is now DINOv2 vitb14 CLS-token cosine similarity
    (L2-norm → dot product), not OpenCLIP ViT-B/32.
  - Image preprocessing: resize to 224x224, ImageNet normalisation
    (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
  - multi_composite (SSIM/MSE/LPIPS) is UNCHANGED vs baseline.
  - Output JSON shape identical: multi_metric={ssim,mse,lpips,composite},
    clip_compare={"clip_cosine": <DINOv2 cosine>}, lm_perplexity={zeros}.

This is an ADDITIVE experiment — multi_composite weights are unchanged;
only the clip_cosine top-level metric is replaced.

Reference: hypothesis H4, Branch 4; DINOv2 Oquab et al. 2023.
"""

import json
import logging
import sys
import time
from pathlib import Path

import lpips as lpips_lib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_084_dinov3_patch"
SSIM_SIZE = 512
BATCH_SIZE = 16
DINO_SIZE = 224

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

# ── DINOv2 setup ──────────────────────────────────────────────────────────────
log.info("Loading DINOv2 vitb14 (fallback: DINOv3 weights 403 Forbidden)...")
_dino_model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14",
    trust_repo=True,
)
_dino_model = _dino_model.eval().to(device)

# ImageNet normalisation (standard for DINOv2)
_dino_transform = T.Compose([
    T.Resize((DINO_SIZE, DINO_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── LPIPS setup ───────────────────────────────────────────────────────────────
log.info("Loading LPIPS (alex)...")
_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _binarize(pil: Image.Image, size: int) -> Image.Image:
    resized = pil.convert("RGB").resize((size, size), Image.LANCZOS)
    binarized = _preprocessor.adaptive_binarize(resized)
    return binarized.convert("RGB")


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


# ── DINOv2 cosine similarity (batched) ───────────────────────────────────────

def _dinov2_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    """Compute CLS-token cosine similarity via DINOv2 vitb14."""
    def _enc(pils):
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)  # returns CLS token (B, D)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


# ── Main processing loop ──────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf
    if not orig_pils_buf:
        return
    cos_sims = _dinov2_cosine_batch(orig_pils_buf, recon_pils_buf)
    for meta, cos in zip(meta_buf, cos_sims):
        meta["clip_compare"] = {"clip_cosine": float(cos)}
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

    ssim_val, mse_val = _ssim_mse(orig_pil, recon_pil)
    lpips_val = _lpips_score(orig_pil, recon_pil)
    composite = 0.4 * ssim_val + 0.3 * max(0.0, 1.0 - mse_val) + 0.3 * max(0.0, 1.0 - lpips_val)

    meta = {
        "image": page_dir.name,
        "text_elements": 0,
        "image_regions": 0,
        "table_regions": 0,
        "text_length": 0,
        "plain_text_length": 0,
        "multi_metric": {
            "ssim": ssim_val,
            "mse": mse_val,
            "lpips": lpips_val,
            "composite": composite,
        },
        "lm_perplexity": {
            "ngram_score": 0.0,
            "transformer_score": 0.0,
            "perplexity": 0.0,
            "composite": 0.0,
        },
    }

    orig_pils_buf.append(orig_pil)
    recon_pils_buf.append(recon_pil)
    meta_buf.append(meta)

    if len(orig_pils_buf) >= BATCH_SIZE:
        _flush_batch()

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        log.info("[%d/%d] %.1fs elapsed (%.2fs/page)", i + 1, len(page_dirs), elapsed, elapsed / (i + 1))

_flush_batch()

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s", len(results), out_path)
