#!/usr/bin/env python3
"""P1_011_phase_cong_table_cc: H_phasecong / D11 — phase congruency replaces CLIP cosine for variant=table cc only.

Hypothesis: Phase congruency (non-neural, frequency-domain, contrast-invariant) similarity
captures table cell-grid edge structure orthogonally to neural encoders (CLIP, DINOv2,
ViT-L/14, ViT-B/32@336, MaskCLIP, SigLIP — all collapsed 0.22-0.25 on table variant).

Diff vs P1_120c3b_all_no_mask_baseline_preproc (H11.2 baseline at 0.3600):
TWO changes only:
  1. METHOD_ID constant updated
  2. _flush_batch() table branch calls _phase_cong_cosine_batch() instead of _clip_cosine_batch()

All other code paths (text, formula, all, all_no_mask) are byte-identical to H11.2.

multi_composite formulas (UNCHANGED from H5.b / H6.1 / H11.2):
  variant=all (USE_DISTS=True):
    composite = 0.3*SSIM + 0.2*(1-MSE) + 0.2*(1-LPIPS) + 0.3*(1-DISTS)
  all other variants (USE_DISTS=False):
    composite = 0.4*SSIM + 0.3*(1-MSE) + 0.3*(1-LPIPS)  (original, unchanged)

clip_cosine logic:
- variant=table:        phase congruency cosine (H_phasecong CHANGE — non-neural frequency-domain)
- variant=all_no_mask:  0.5 * DINOv2_vitb14_CLS + 0.5 * OpenCLIP_ViT-B/32,
                        both encoders receive preprocessed image (H11.2, UNCHANGED)
- all other variants:   0.5 * DINOv2_vitb14_CLS + 0.5 * OpenCLIP_ViT-B/32, raw RGB
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

sys.path.insert(0, "/home/mac/test/r1-p2/src")
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_011_phase_cong_table_cc"
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
USE_DISTS = variant == "all"          # DISTS-augmented composite only for 'all' variant
USE_CLIP_ONLY = variant == "table"    # CLIP-only cosine for 'table' variant (H4.e strategy)
# H11.2: extend preprocessing gate to also cover all_no_mask (was: variant == "table" only)
USE_PREPROC = variant in ("table", "all_no_mask")

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
log.info(
    "device=%s variant=%s method=%s use_dists=%s use_clip_only=%s use_preproc=%s",
    device, variant, METHOD_ID, USE_DISTS, USE_CLIP_ONLY, USE_PREPROC,
)

_preprocessor = ImagePreprocessor()

# ── DISTS setup via pyiqa (only for variant=all) ──────────────────────────────
if USE_DISTS:
    import pyiqa
    log.info("Loading pyiqa DISTS metric (variant=all only)...")
    _dists_metric = pyiqa.create_metric("dists", as_loss=False).to(device)
    _dists_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),  # [0,1]
    ])
else:
    log.info("Skipping DISTS load (variant=%s: original composite formula)", variant)
    _dists_metric = None
    _dists_transform = None


def _dists_score(orig_pil: Image.Image, recon_pil: Image.Image) -> float:
    """Compute DISTS distance (lower=more similar). Returns clipped to [0,1]."""
    orig_t = _dists_transform(orig_pil.convert("RGB")).unsqueeze(0).to(device)
    recon_t = _dists_transform(recon_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = _dists_metric(orig_t, recon_t)
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
    log.info("Skipping DINOv2 load (table variant: phase-congruency-only strategy)")
    _dino_model = None
    _dino_transform = None

# ── OpenCLIP setup (loaded for non-table variants; skipped if USE_CLIP_ONLY) ──
if not USE_CLIP_ONLY:
    log.info("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _clip_model = _clip_model.eval().to(device)
else:
    log.info("Skipping OpenCLIP load (table variant: phase-congruency-only strategy)")
    _clip_model = None
    _clip_preprocess = None

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


# ── H6.1: Baseline CLIP preprocessing (mirrors clip_similarity.py:44-49) ─────

def _baseline_clip_preprocess(img: Image.Image) -> Image.Image:
    """Mirror of src/reference_free_ocr_metric/metrics/clip_compare/clip_similarity.py:44-49"""
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")
    return img


# ── Phase congruency similarity (variant=table only) ──────────────────────────

def _phase_cong_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    """Compute phase-congruency-based similarity for variant=table only.
    PC map captures contrast-invariant edge significance — distinct from neural encoders.
    Similarity = cosine of L2-normalized flattened PC maps (range [0,1]).
    Uses H6.1-style baseline preprocessing then grayscale at 512x512.
    """
    from phasepack import phasecongmono

    sims = []
    for orig_pil, recon_pil in zip(orig_pils, recon_pils):
        # Use H6.1-style baseline preprocessing then grayscale at 512x512
        og = _baseline_clip_preprocess(orig_pil).convert("L").resize((512, 512), Image.BILINEAR)
        rg = _baseline_clip_preprocess(recon_pil).convert("L").resize((512, 512), Image.BILINEAR)
        og_arr = np.array(og, dtype=np.float32) / 255.0
        rg_arr = np.array(rg, dtype=np.float32) / 255.0
        m_og, _, _, _ = phasecongmono(og_arr)
        m_rg, _, _, _ = phasecongmono(rg_arr)
        v_og = m_og.flatten().astype(np.float64)
        v_rg = m_rg.flatten().astype(np.float64)
        # cosine similarity, clipped to [0,1] (PC magnitudes are non-negative; cos in [0,1])
        denom = float(np.linalg.norm(v_og) * np.linalg.norm(v_rg))
        if denom < 1e-9:
            sims.append(0.0)
            continue
        sim = float(np.clip(np.dot(v_og, v_rg) / denom, 0.0, 1.0))
        sims.append(sim)
    return sims


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


# ── OpenCLIP cosine (batched) ──────────────────────────────────────────────────

def _clip_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    def _enc(pils):
        if USE_PREPROC:
            # H6.1 / H11.2: apply baseline grayscale+SHARPEN+autocontrast->RGB before CLIP preprocess
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
    if USE_CLIP_ONLY:
        # H_phasecong CHANGE: table variant uses phase congruency cosine instead of CLIP cosine
        pc_sims = _phase_cong_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, pc_sim in zip(meta_buf, pc_sims):
            meta["clip_compare"] = {"clip_cosine": float(pc_sim)}
            results.append(meta)
    else:
        # all other variants (including all_no_mask): 50/50 CLIP + DINOv2 average
        # For all_no_mask: DINOv2 receives preprocessed image via _dinov2_cosine_batch_preproc
        # For other non-table variants: DINOv2 receives raw RGB (unchanged from H6.1)
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        if USE_PREPROC:
            # all_no_mask: both encoders see preprocessed image
            dino_sims = _dinov2_cosine_batch_preproc(orig_pils_buf, recon_pils_buf)
        else:
            dino_sims = _dinov2_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, dino_cos, clip_cos in zip(meta_buf, dino_sims, clip_sims):
            avg_cos = 0.5 * dino_cos + 0.5 * clip_cos
            meta["clip_compare"] = {"clip_cosine": float(avg_cos)}
            results.append(meta)
    orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


# ── DINOv2 cosine with preprocessing (for all_no_mask only) ──────────────────

def _dinov2_cosine_batch_preproc(orig_pils: list, recon_pils: list) -> list:
    """DINOv2 cosine where input images are preprocessed (grayscale+SHARPEN+autocontrast->RGB)."""
    def _enc(pils):
        preprocessed = [_baseline_clip_preprocess(p) for p in pils]
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in preprocessed]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)  # CLS token (B, D)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


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
        # variant=all: DISTS-augmented rebalanced composite
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
