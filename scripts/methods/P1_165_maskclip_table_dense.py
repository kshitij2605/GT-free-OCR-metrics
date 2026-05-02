#!/usr/bin/env python3
"""P1_165_maskclip_table_dense: H9.1 — MaskCLIP dense per-patch cosine for variant=table only.

H9.1 hypothesis: CLS pooling in OpenCLIP's encode_image collapses spatial table-grid information.
MaskCLIP-style per-patch feature extraction (per-patch value projection, mean over 49 patches)
preserves cell-level spatial signal and should lift table cc above the 0.2766 CLS-pool ceiling.

Reference: Zhou et al. "Extract Free Dense Labels from CLIP" (ECCV 2022).

For variant=table ONLY:
  - Extract per-patch features [B, 49, D_proj] from OpenCLIP ViT-B/32 last layer
    (conv1 -> embed patches -> add cls+pos emb -> transformer -> ln_post -> proj -> drop CLS)
  - L2-normalize each patch feature
  - Per-patch cosine = (recon_patches * orig_patches).sum(-1) -> [B, 49]
  - Return mean over patches -> [B] scalar

All other variants: UNCHANGED from H6.1 (50/50 CLIP+DINOv2 CLS fusion).
multi_composite: UNCHANGED for all variants (original 0.4/0.3/0.3 or DISTS-augmented for all).
Baseline preprocessing (grayscale+SHARPEN+autocontrast->RGB) preserved before patch extraction.
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

METHOD_ID = "P1_165_maskclip_table_dense"
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
USE_TABLE_PREPROC = variant == "table"  # H6.1: restore baseline preprocessing for table CLIP path
USE_DENSE_PATCHES = variant == "table"  # H9.1: MaskCLIP per-patch cosine for table only

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
    "device=%s variant=%s method=%s use_dists=%s use_clip_only=%s use_table_preproc=%s use_dense_patches=%s",
    device, variant, METHOD_ID, USE_DISTS, USE_CLIP_ONLY, USE_TABLE_PREPROC, USE_DENSE_PATCHES,
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
    log.info("Skipping DINOv2 load (table variant: CLIP-only strategy)")
    _dino_model = None
    _dino_transform = None

# ── OpenCLIP setup (always loaded) ────────────────────────────────────────────
log.info("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)

# Log available visual attributes for debugging open_clip API compatibility
_visual = _clip_model.visual
log.info(
    "OpenCLIP visual attributes: %s",
    [a for a in dir(_visual) if not a.startswith("__")],
)

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


# ── H9.1: MaskCLIP dense per-patch cosine (for variant=table) ─────────────────

def _clip_dense_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    """MaskCLIP-style: per-patch cosine averaged over patches.

    OpenCLIP ViT-B/32: 224x224 input, 32x32 patches -> 7x7 = 49 patches.
    Standard encode_image returns CLS pooled vector; we instead extract the per-patch
    last-layer features (after final LN, projected to text-embedding space), L2-normalize
    per patch, compute per-patch cosine, then average over patches.

    Access pattern: visual.conv1 / visual.class_embedding / visual.positional_embedding /
    visual.ln_pre / visual.transformer / visual.ln_post / visual.proj
    (confirmed for open_clip >= 2.20 VisionTransformer).
    """
    visual = _clip_model.visual

    def _enc(pils):
        if USE_TABLE_PREPROC:
            processed = [_baseline_clip_preprocess(p) for p in pils]
        else:
            processed = [p.convert("RGB") for p in pils]
        batch = torch.stack([_clip_preprocess(p) for p in processed]).to(device)
        with torch.no_grad():
            # Mirror OpenCLIP VisionTransformer.forward but skip the CLS-only pooling
            x = visual.conv1(batch)                          # (B, D, 7, 7)
            B, D, H, W = x.shape
            x = x.reshape(B, D, H * W).permute(0, 2, 1)    # (B, 49, D)
            cls_tok = visual.class_embedding.to(x.dtype).expand(B, 1, -1)
            x = torch.cat([cls_tok, x], dim=1)              # (B, 50, D)
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)                          # (N, B, D) — transformer expects seq-first
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)                          # (B, 50, D)
            x = visual.ln_post(x)
            if visual.proj is not None:
                x = x @ visual.proj                         # (B, 50, D_proj)
            patches = x[:, 1:, :]                           # drop CLS token -> (B, 49, D_proj)
        return F.normalize(patches.float(), dim=-1)         # (B, 49, D_proj) L2-normalized per patch

    orig_p = _enc(orig_pils)   # (B, 49, D_proj)
    recon_p = _enc(recon_pils) # (B, 49, D_proj)
    per_patch_cos = (orig_p * recon_p).sum(dim=-1)         # (B, 49) per-patch cosine
    return per_patch_cos.mean(dim=-1).cpu().tolist()        # (B,) mean over patches


# ── OpenCLIP CLS cosine (batched) — used for non-table variants ───────────────

def _clip_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    def _enc(pils):
        if USE_TABLE_PREPROC:
            # H6.1: apply baseline grayscale+SHARPEN+autocontrast->RGB before CLIP preprocess
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
    if USE_DENSE_PATCHES:
        # H9.1: table variant — MaskCLIP dense per-patch cosine
        clip_sims = _clip_dense_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, clip_cos in zip(meta_buf, clip_sims):
            meta["clip_compare"] = {"clip_cosine": float(clip_cos)}
            results.append(meta)
    elif USE_CLIP_ONLY:
        # Fallback CLS CLIP-only (should not be reached for table, but kept for safety)
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, clip_cos in zip(meta_buf, clip_sims):
            meta["clip_compare"] = {"clip_cosine": float(clip_cos)}
            results.append(meta)
    else:
        # all other variants: 50/50 CLIP + DINOv2 average (H4.e strategy, raw RGB)
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
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
