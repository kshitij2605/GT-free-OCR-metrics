#!/usr/bin/env python3
"""P1_117_deepssim_all_4th_term: H14.X / D117 — DeepDC as 4th term REPLACING DISTS for variant=all.

Hypothesis: Replacing DISTS with DeepDC (pyiqa "deepdc", lower_better=True, range [0,1])
as the 4th term in variant=all multi_composite will change spearman_mean vs H11.2 (0.3600).
DeepSSIM was specified in D117 but is not in pyiqa; deepdc is the closest mechanistic
substitute (deep-feature-based structural quality metric, same polarity, same range).

multi_composite formulas:
  variant=all (USE_DEEPDC=True):
    composite = 0.3*SSIM + 0.2*(1-MSE) + 0.2*(1-LPIPS) + 0.3*(1-DeepDC)
  all other variants (USE_DEEPDC=False):
    composite = 0.4*SSIM + 0.3*(1-MSE) + 0.3*(1-LPIPS)  (original, unchanged)

clip_cosine logic (UNCHANGED from H4.e + H6.1 + H11.2 baseline = P1_120c3b):
- variant=table:       CLIP cosine ONLY, with grayscale+SHARPEN+autocontrast->RGB (H6.1)
- variant=all_no_mask: 0.5 * DINOv2_vitb14_CLS + 0.5 * OpenCLIP_ViT-B/32,
                       both encoders receive preprocessed image (H11.2)
- all other variants:  0.5 * DINOv2_vitb14_CLS + 0.5 * OpenCLIP_ViT-B/32, raw RGB

Control: H11.2 (89aee86, spearman_mean=0.3600).
"""

import json
import logging
import sys
import time
from pathlib import Path

import lpips as lpips_lib
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_117_deepssim_all_4th_term"
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
USE_DEEPDC = variant == "all"         # DeepDC-augmented composite only for 'all' variant
USE_CLIP_ONLY = variant == "table"    # CLIP-only cosine for 'table' variant (H4.e strategy)
# H11.2 preprocessing gate: table + all_no_mask get grayscale+SHARPEN+autocontrast->RGB
USE_PREPROC = variant in ("table", "all_no_mask")

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
    "device=%s variant=%s method=%s use_deepdc=%s use_clip_only=%s use_preproc=%s",
    device, variant, METHOD_ID, USE_DEEPDC, USE_CLIP_ONLY, USE_PREPROC,
)

_preprocessor = ImagePreprocessor()

# ── DeepDC setup via pyiqa (only for variant=all) ────────────────────────────
if USE_DEEPDC:
    import pyiqa
    log.info("Loading pyiqa deepdc metric (variant=all only)...")
    _deepdc_metric = pyiqa.create_metric("deepdc", as_loss=False).to(device)
    # deepdc accepts standard [0,1] RGB tensors at any resolution; resize to 256x256 for consistency
    _deepdc_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),  # [0,1]
    ])
else:
    log.info("Skipping deepdc load (variant=%s: original composite formula)", variant)
    _deepdc_metric = None
    _deepdc_transform = None


def _deepdc_score(orig_pil: Image.Image, recon_pil: Image.Image) -> float:
    """Compute DeepDC distance (lower=more similar). Returns clipped to [0,1]."""
    orig_t = _deepdc_transform(orig_pil.convert("RGB")).unsqueeze(0).to(device)
    recon_t = _deepdc_transform(recon_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = _deepdc_metric(orig_t, recon_t)
    return float(np.clip(dist.item(), 0.0, 1.0))


# ── DINOv2 setup (only needed when not CLIP-only) ─────────────────────────────
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
    log.info("Skipping DINOv2 load (table variant: CLIP-only strategy)")
    _dino_model = None
    _dino_transform = None

# ── OpenCLIP setup (always loaded) ────────────────────────────────────────────
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


# ── H6.1 / H11.2: Baseline CLIP preprocessing ────────────────────────────────

def _baseline_clip_preprocess(img: Image.Image) -> Image.Image:
    """Mirror of src/reference_free_ocr_metric/metrics/clip_compare/clip_similarity.py:44-49"""
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")
    return img


# ── DINOv2 cosine (batched, CLS token) ───────────────────────────────────────

def _dinov2_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    def _enc(pils):
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)  # CLS token (B, D)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


def _dinov2_cosine_batch_preproc(orig_pils: list, recon_pils: list) -> list:
    """DINOv2 cosine with grayscale+SHARPEN+autocontrast->RGB preprocessing (for all_no_mask)."""
    def _enc(pils):
        preprocessed = [_baseline_clip_preprocess(p) for p in pils]
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in preprocessed]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)  # CLS token (B, D)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


# ── OpenCLIP cosine (batched) ──────────────────────────────────────────────────

def _clip_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    def _enc(pils):
        if USE_PREPROC:
            # H6.1 / H11.2: apply baseline preprocessing before CLIP preprocess
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


# ── Main processing loop ───────────────────────────────────────────────────────
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
        # table variant: use CLIP cosine alone (H4.e strategy), with H6.1 preprocessing
        for meta, clip_cos in zip(meta_buf, clip_sims):
            meta["clip_compare"] = {"clip_cosine": float(clip_cos)}
            results.append(meta)
    else:
        # all other variants: 50/50 CLIP + DINOv2 average
        if USE_PREPROC:
            # all_no_mask: both encoders see preprocessed image (H11.2)
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

    if USE_DEEPDC:
        # variant=all: DeepDC-augmented rebalanced composite (replaces DISTS slot)
        deepdc_val = _deepdc_score(orig_pil, recon_pil)
        composite = (
            0.3 * ssim_val
            + 0.2 * max(0.0, 1.0 - mse_val)
            + 0.2 * max(0.0, 1.0 - lpips_val)
            + 0.3 * max(0.0, 1.0 - deepdc_val)
        )
        multi_metric = {
            "ssim": ssim_val,
            "mse": mse_val,
            "lpips": lpips_val,
            "deepdc": deepdc_val,
            "composite": composite,
        }
    else:
        # all other variants: original formula (0.4/0.3/0.3)
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
