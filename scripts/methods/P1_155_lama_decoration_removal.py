#!/usr/bin/env python3
"""P1_155_lama_decoration_removal: H14.0 — LaMa inpainting as preprocessing.

H14.0 hypothesis: Decoration regions (banners, headers, figure boxes) present in
ocr_all_no_mask/masked_original.png but absent (white-filled) in ocr_all/masked_original.png
contaminate SSIM/LPIPS/CLIP signals. Replacing those decoration pixels with LaMa FFC-globality
background fill BEFORE the metric pipeline runs should reduce original-vs-reconstruction
texture mismatch, lifting all_no_mask cc and possibly all mc.

Mask source: pixel-level diff between ocr_all/masked_original.png (decorations white-filled)
and ocr_all_no_mask/masked_original.png (decorations present). Where diff > 5, those pixels
are decorations in all_no_mask. We inpaint them away.

Variant-conditional strategy:
  - variant=all_no_mask: LaMa applied. Decorations are present; mask from diff with ocr_all.
  - variant=all: LaMa applied. Decorations are white-filled rectangles in masked_original;
    the matching reconstructed.png also has no decorations. Inpainting the small residual
    decoration pixels may help.
  - variant=text, formula, table: PASS-THROUGH (no LaMa). These variants are saturated;
    risk avoidance dominates.

Pipeline (metric side): identical to P1_120c3b_all_no_mask_baseline_preproc (H11.2 control):
  - multi_composite: variant=all uses DISTS-augmented (0.3/0.2/0.2/0.3); others use 0.4/0.3/0.3
  - clip_cosine: variant=table = CLIP-only+H6.1 preproc; all_no_mask = 50/50 CLIP+DINOv2+H6.1 preproc;
    others = 50/50 CLIP+DINOv2 raw RGB

LaMa model: big-lama (simple_lama_inpainting 0.1.2), Places2 prior.
Sanity probe result (2026-04-27): inpainted region mean [254,254,254], std [6,7,6],
time 1.0s/page — clean white-fill, no hallucinations. File hash differs from input.
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

METHOD_ID = "P1_155_lama_decoration_removal"
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
USE_PREPROC = variant in ("table", "all_no_mask")  # H6.1 / H11.2 preprocessing
USE_LAMA = variant in ("all_no_mask", "all")        # H14.0: LaMa preprocessing

BASE = Path(__file__).parent.parent.parent / "data" / "omnidocbench"
var_root = BASE / f"ocr_{variant}"
# For mask computation: all variant has white-filled decorations (reference for mask)
ALL_ROOT = BASE / "ocr_all"
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
    "device=%s variant=%s method=%s use_dists=%s use_clip_only=%s use_preproc=%s use_lama=%s",
    device, variant, METHOD_ID, USE_DISTS, USE_CLIP_ONLY, USE_PREPROC, USE_LAMA,
)

_preprocessor = ImagePreprocessor()

# ── LaMa setup (only for variant=all_no_mask and variant=all) ─────────────────
if USE_LAMA:
    from simple_lama_inpainting import SimpleLama
    log.info("Loading LaMa model (big-lama, Places2 prior)...")
    _lama = SimpleLama()
    log.info("LaMa loaded")
else:
    _lama = None


def _lama_inpaint(orig_pil: Image.Image, ref_all_path: Path) -> Image.Image:
    """Apply LaMa inpainting to remove decoration pixels.

    For variant=all_no_mask: mask is diff between all_no_mask orig and all orig.
    For variant=all: decorations are already white-filled; mask is near-white
    rectangles that differ significantly from the surrounding background.

    If mask is empty (no decoration pixels), returns orig_pil unchanged.
    """
    orig_arr = np.array(orig_pil.convert("RGB"))

    if ref_all_path.exists():
        ref_arr = np.array(Image.open(ref_all_path).convert("RGB"))
        # Resize ref to match orig if sizes differ slightly
        if ref_arr.shape != orig_arr.shape:
            ref_pil = Image.fromarray(ref_arr).resize(
                (orig_arr.shape[1], orig_arr.shape[0]), Image.BILINEAR
            )
            ref_arr = np.array(ref_pil)
        diff = np.abs(orig_arr.astype(int) - ref_arr.astype(int)).max(axis=2)
        mask_arr = (diff > 5).astype(np.uint8) * 255
    else:
        # Fallback: no reference available (shouldn't happen for ocr_all_no_mask)
        log.warning("No ocr_all reference for %s — skipping LaMa", ref_all_path.parent.name)
        return orig_pil

    n_mask_px = int((mask_arr > 0).sum())
    if n_mask_px == 0:
        return orig_pil  # No decoration pixels on this page

    mask_pil = Image.fromarray(mask_arr, mode="L")
    orig_size = orig_pil.size  # (W, H)

    # LaMa uses TorchScript with 32-bit index math — fails on images where
    # padded tensor elements exceed 2^31. Downsample large images to 2048px
    # max side, inpaint at reduced size, then upsample back.
    LAMA_MAX_SIDE = 2048
    w, h = orig_size
    if max(w, h) > LAMA_MAX_SIDE:
        scale = LAMA_MAX_SIDE / max(w, h)
        small_w, small_h = int(w * scale), int(h * scale)
        # Round to nearest multiple of 8 (LaMa requirement)
        small_w = max(8, (small_w // 8) * 8)
        small_h = max(8, (small_h // 8) * 8)
        orig_small = orig_pil.resize((small_w, small_h), Image.BILINEAR)
        mask_small = mask_pil.resize((small_w, small_h), Image.NEAREST)
        try:
            result_small = _lama(orig_small, mask_small)
        except (RuntimeError, Exception) as e:
            log.warning("LaMa failed at %dx%d (downsampled from %dx%d): %s — skipping page",
                        small_w, small_h, w, h, e)
            return orig_pil
        if result_small.size != (small_w, small_h):
            result_small = result_small.resize((small_w, small_h), Image.BILINEAR)
        result = result_small.resize(orig_size, Image.BILINEAR)
    else:
        try:
            result = _lama(orig_pil, mask_pil)
        except (RuntimeError, Exception) as e:
            log.warning("LaMa failed at %dx%d: %s — skipping page", w, h, e)
            return orig_pil
        # LaMa may resize slightly (pads to multiples of 8); resize back
        if result.size != orig_size:
            result = result.resize(orig_size, Image.BILINEAR)
    return result


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


# ── H6.1: Baseline CLIP preprocessing ─────────────────────────────────────────

def _baseline_clip_preprocess(img: Image.Image) -> Image.Image:
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
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


def _dinov2_cosine_batch_preproc(orig_pils: list, recon_pils: list) -> list:
    """DINOv2 cosine with H6.1 preprocessing (for all_no_mask only)."""
    def _enc(pils):
        preprocessed = [_baseline_clip_preprocess(p) for p in pils]
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in preprocessed]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


# ── OpenCLIP cosine (batched) ──────────────────────────────────────────────────

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


# ── Main processing loop ───────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages (variant=%s, use_lama=%s)", len(page_dirs), variant, USE_LAMA)

results = []
orig_pils_buf, recon_pils_buf, meta_buf = [], [], []
lama_applied_count = 0


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

    # H14.0: Apply LaMa inpainting to original before metric computation
    if USE_LAMA:
        # Reference for mask: matching page in ocr_all (decorations white-filled)
        ref_all_path = ALL_ROOT / page_dir.name / "masked_original.png"
        orig_before = orig_pil
        orig_pil = _lama_inpaint(orig_pil, ref_all_path)
        if orig_pil is not orig_before:
            lama_applied_count += 1

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
        log.info(
            "[%d/%d] %.1fs elapsed (%.2fs/page) lama_applied=%d",
            i + 1, len(page_dirs), elapsed, elapsed / (i + 1), lama_applied_count,
        )

_flush_batch()

log.info("LaMa applied to %d/%d pages", lama_applied_count, len(results))

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s", len(results), out_path)
