#!/usr/bin/env python3
"""P1_002b_haarpsi_all_no_mask_4th_term: H13.1 — HaarPSI as 4th term in mc for variant=all_no_mask only.

H13.1 hypothesis: HaarPSI (Reisenhofer et al. 2018, Signal Processing) is a Haar-wavelet
perceptual similarity index. It uses 6 discrete 2D Haar wavelet filters and weights local
similarity by low-frequency coefficients — strongly responsive to horizontal/vertical edges
where text glyphs live. Direction D2 in program.md (priority #2, Low effort, High impact).

HaarPSI was previously dead-ended ONLY as SSIM-slot replacement (regressed text/all by 0.05).
Additive use as a 4th term in multi_composite has not been tested. The mechanism that worked
for H5.b (DISTS-as-4th-term lifted variant=all mc by +0.065) applies here with a
fundamentally different signal class (Haar wavelets vs DISTS deep-feature stats), targeting
the smallest gap-to-overtake (variant=all_no_mask mc=0.264 vs cc=0.308, gap=0.044).

Diff vs P1_120c3b (H11.2 baseline = current best 0.3600):
- Add piq.haarpsi computation in addition to SSIM/MSE/LPIPS for variant=all_no_mask only
- Composite for variant=all_no_mask:
    composite = 0.30*SSIM + 0.20*(1-MSE) + 0.20*(1-LPIPS) + 0.30*HaarPSI
  (HaarPSI is a similarity measure in [0,1], higher=better, so direct add — no 1-x flip)
- All other variants UNCHANGED:
    variant=all (USE_DISTS=True):    0.3*SSIM + 0.2*(1-MSE) + 0.2*(1-LPIPS) + 0.3*(1-DISTS)
    variant=text/formula/table:      0.4*SSIM + 0.3*(1-MSE) + 0.3*(1-LPIPS) (original)

clip_cosine logic UNCHANGED from H11.2:
- variant=table:        CLIP cosine ONLY, with H6.1 baseline preprocessing
- variant=all_no_mask:  0.5 * DINOv2 + 0.5 * OpenCLIP, both encoders see preprocessed image
- all other variants:   0.5 * DINOv2 + 0.5 * OpenCLIP, raw RGB
"""

import json
import logging
import sys
import time
from pathlib import Path

import lpips as lpips_lib
import numpy as np
import open_clip
import piq
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_002b_haarpsi_all_no_mask_4th_term"
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

# Variant-conditional flags
USE_DISTS = variant == "all"
USE_CLIP_ONLY = variant == "table"
USE_PREPROC = variant in ("table", "all_no_mask")
USE_HAARPSI = variant == "all_no_mask"  # H13.1: HaarPSI 4th term ONLY for all_no_mask

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
    "device=%s variant=%s method=%s use_dists=%s use_clip_only=%s use_preproc=%s use_haarpsi=%s",
    device, variant, METHOD_ID, USE_DISTS, USE_CLIP_ONLY, USE_PREPROC, USE_HAARPSI,
)

_preprocessor = ImagePreprocessor()

# ── DISTS setup via pyiqa (only for variant=all) ──────────────────────────────
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


# ── HaarPSI setup (only for variant=all_no_mask) ──────────────────────────────
if USE_HAARPSI:
    log.info("HaarPSI active for variant=all_no_mask (piq.haarpsi).")
    _haarpsi_transform = T.Compose([
        T.Resize((SSIM_SIZE, SSIM_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),  # -> [0,1] CHW
    ])
else:
    _haarpsi_transform = None


def _haarpsi_score(orig_pil: Image.Image, recon_pil: Image.Image) -> float:
    """piq.haarpsi returns similarity in [0,1] (higher=better). RGB tensors in [0,1]."""
    orig_t = _haarpsi_transform(orig_pil.convert("RGB")).unsqueeze(0).to(device)
    recon_t = _haarpsi_transform(recon_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        s = piq.haarpsi(orig_t, recon_t, data_range=1.0, reduction="none")
    return float(np.clip(s.item(), 0.0, 1.0))


# ── DINOv2 setup ──────────────────────────────────────────────────────────────
if not USE_CLIP_ONLY:
    log.info("Loading DINOv2 vitb14...")
    _dino_model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
        trust_repo=True,
    )
    _dino_model = _dino_model.eval().to(device)

    _dino_transform = T.Compose([
        T.Resize((DINO_SIZE, DINO_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    _dino_model = None
    _dino_transform = None

# ── OpenCLIP setup ────────────────────────────────────────────────────────────
log.info("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)

# ── LPIPS setup ────────────────────────────────────────────────────────────────
log.info("Loading LPIPS (alex)...")
_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

# ── Preprocessing helpers ──────────────────────────────────────────────────────

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


# ── H6.1: Baseline CLIP preprocessing ────────────────────────────────────────

def _baseline_clip_preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")
    return img


# ── DINOv2 cosine variants ───────────────────────────────────────────────────

def _dinov2_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    def _enc(pils):
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


def _dinov2_cosine_batch_preproc(orig_pils: list, recon_pils: list) -> list:
    def _enc(pils):
        preprocessed = [_baseline_clip_preprocess(p) for p in pils]
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in preprocessed]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


# ── OpenCLIP cosine ──────────────────────────────────────────────────────────

def _clip_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    def _enc(pils):
        if USE_PREPROC:
            processed = [_baseline_clip_preprocess(p) for p in pils]
        else:
            processed = [p.convert("RGB") for p in pils]
        batch = torch.stack([_clip_preprocess(p) for p in processed]).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_image(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


# ── Main loop ────────────────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf
    if not orig_pils_buf:
        return
    clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
    if USE_CLIP_ONLY:
        for meta, clip_cos in zip(meta_buf, clip_sims):
            meta["clip_compare"] = {"clip_cosine": float(clip_cos)}
            results.append(meta)
    else:
        if USE_PREPROC:
            dino_sims = _dinov2_cosine_batch_preproc(orig_pils_buf, recon_pils_buf)
        else:
            dino_sims = _dinov2_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, dino_cos, clip_cos in zip(meta_buf, dino_sims, clip_sims):
            avg_cos = 0.5 * dino_cos + 0.5 * clip_cos
            meta["clip_compare"] = {"clip_cosine": float(avg_cos)}
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

    if USE_DISTS:
        dists_val = _dists_score(orig_pil, recon_pil)
        composite = (
            0.3 * ssim_val
            + 0.2 * max(0.0, 1.0 - mse_val)
            + 0.2 * max(0.0, 1.0 - lpips_val)
            + 0.3 * max(0.0, 1.0 - dists_val)
        )
        multi_metric = {
            "ssim": ssim_val,
            "mse": mse_val,
            "lpips": lpips_val,
            "dists": dists_val,
            "composite": composite,
        }
    elif USE_HAARPSI:
        # H13.1: variant=all_no_mask gets HaarPSI 4th term
        haarpsi_val = _haarpsi_score(orig_pil, recon_pil)
        composite = (
            0.30 * ssim_val
            + 0.20 * max(0.0, 1.0 - mse_val)
            + 0.20 * max(0.0, 1.0 - lpips_val)
            + 0.30 * haarpsi_val  # HaarPSI is similarity, NOT distance — direct add
        )
        multi_metric = {
            "ssim": ssim_val,
            "mse": mse_val,
            "lpips": lpips_val,
            "haarpsi": haarpsi_val,
            "composite": composite,
        }
    else:
        composite = 0.4 * ssim_val + 0.3 * max(0.0, 1.0 - mse_val) + 0.3 * max(0.0, 1.0 - lpips_val)
        multi_metric = {
            "ssim": ssim_val,
            "mse": mse_val,
            "lpips": lpips_val,
            "composite": composite,
        }

    meta = {
        "image": page_dir.name,
        "text_elements": 0,
        "image_regions": 0,
        "table_regions": 0,
        "text_length": 0,
        "plain_text_length": 0,
        "multi_metric": multi_metric,
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
