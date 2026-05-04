#!/usr/bin/env python3
"""P1_158_gtpoly_content_mask_all_no_mask: H14.1.diag — GT-polygon content masks as preprocessing for variant=all_no_mask only.

CEILING DIAGNOSTIC for mask-based preprocessing axis.

If even GT-perfect content masks (from OmniDocBench.json layout_dets polygons) fail to
preserve correlation on variant=all_no_mask, ALL mask-based preprocessing approaches
(Hi-SAM, CRAFT, etc.) are structurally dead and can be abandoned.

Mechanism:
- Load OmniDocBench.json once at startup; build per-page binary content masks from
  layout_dets[*].poly polygons (excluding category_type=="abandon").
- For variant=all_no_mask ONLY: apply mask to both GT and recon images before metric
  computation: content pixels unchanged, non-content pixels zeroed (black).
- All other variants: byte-identical passthrough to H11.2 baseline.

Diff vs P1_120c3b_all_no_mask_baseline_preproc (H11.2 baseline):
- METHOD_ID updated
- Added USE_CONTENT_MASK flag (True only for all_no_mask)
- Added _load_page_polys(), _make_content_mask(), _apply_content_mask() functions
- Added GT-polygon masking step in main loop (after image load, before SSIM/LPIPS/batch-append)
- All non-all_no_mask code paths: bit-for-bit identical to H11.2

multi_composite formulas (UNCHANGED from H5.b / H6.1 / H11.2):
  variant=all (USE_DISTS=True):
    composite = 0.3*SSIM + 0.2*(1-MSE) + 0.2*(1-LPIPS) + 0.3*(1-DISTS)
  all other variants (USE_DISTS=False):
    composite = 0.4*SSIM + 0.3*(1-MSE) + 0.3*(1-LPIPS)  (original, unchanged)

clip_cosine logic (UNCHANGED from H11.2):
- variant=table:        CLIP cosine ONLY, with preprocessing (H6.1, UNCHANGED)
- variant=all_no_mask:  0.5 * DINOv2_vitb14_CLS + 0.5 * OpenCLIP_ViT-B/32,
                        both encoders receive preprocessed image (H11.2 change)
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
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_158_gtpoly_content_mask_all_no_mask"
SSIM_SIZE = 512
BATCH_SIZE = 16
DINO_SIZE = 224
GT_JSON_PATH = str(Path(__file__).parent.parent.parent / "data/omnidocbench/OmniDocBench.json")

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
# H14.1.diag: GT-polygon content mask — only for all_no_mask (CEILING DIAGNOSTIC)
USE_CONTENT_MASK = variant == "all_no_mask"

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
    "device=%s variant=%s method=%s use_dists=%s use_clip_only=%s use_preproc=%s use_content_mask=%s",
    device, variant, METHOD_ID, USE_DISTS, USE_CLIP_ONLY, USE_PREPROC, USE_CONTENT_MASK,
)

_preprocessor = ImagePreprocessor()


# ── GT-polygon content mask helpers (only used when USE_CONTENT_MASK=True) ────

def _load_page_polys() -> dict:
    """Load OmniDocBench.json and return {page_basename_no_ext: list_of_poly_tuples}.

    Each poly is a list of (x, y) tuples (4 points = 8 coords from flat [x1,y1,...,x4,y4]).
    Entries with category_type == "abandon" are excluded (explicitly ignored regions).
    Returns empty dict on any load failure (caller falls back to passthrough).
    """
    try:
        with open(GT_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning("Failed to load GT JSON %s: %s — falling back to no-mask passthrough", GT_JSON_PATH, e)
        return {}

    page_polys: dict = {}
    # OmniDocBench format: list of page dicts with "page_info" and "layout_dets"
    for page in data:
        try:
            page_info = page.get("page_info", {})
            # image_path may be like "path/to/page_dir/filename.png" or just "filename"
            img_path_str = page_info.get("image_path", "")
            if not img_path_str:
                continue
            # Extract stem (no extension) of the basename — this matches page_dir.name
            basename_no_ext = Path(img_path_str).stem
            layout_dets = page.get("layout_dets", [])
            polys = []
            for det in layout_dets:
                if det.get("category_type") == "abandon":
                    continue
                poly_flat = det.get("poly", [])
                if len(poly_flat) < 8:
                    continue
                # Convert flat [x1,y1,x2,y2,x3,y3,x4,y4] to [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                pts = [(poly_flat[i], poly_flat[i + 1]) for i in range(0, 8, 2)]
                polys.append(pts)
            if polys:
                page_polys[basename_no_ext] = polys
        except Exception as e:
            log.debug("Skipping malformed page entry: %s", e)
            continue

    log.info("Loaded GT polygons for %d pages from %s", len(page_polys), GT_JSON_PATH)
    return page_polys


def _make_content_mask(page_key: str, image_size: tuple) -> "Image.Image | None":
    """Build a binary PIL mask (mode 'L') for the given page.

    image_size: (width, height) matching the loaded image.
    Returns None if page_key not found or no valid polys (caller falls back to passthrough).
    Mask=255 at content polygons, 0 elsewhere.
    """
    polys = _PAGE_POLYS.get(page_key)
    if not polys:
        return None
    try:
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        for pts in polys:
            # pts is list of (x, y) floats — convert to int for ImageDraw
            pts_int = [(int(round(x)), int(round(y))) for x, y in pts]
            draw.polygon(pts_int, fill=255)
        return mask
    except Exception as e:
        log.warning("Failed to build mask for page %s: %s — passthrough", page_key, e)
        return None


def _apply_content_mask(pil_image: Image.Image, mask: Image.Image) -> Image.Image:
    """Zero out non-content regions: Image.composite(image, black, mask).

    Where mask=255: original pixel kept. Where mask=0: black (zero) pixel.
    """
    black = Image.new(pil_image.mode, pil_image.size, 0)
    return Image.composite(pil_image, black, mask)


# Load GT polygons at module startup (only when needed)
_PAGE_POLYS: dict = _load_page_polys() if USE_CONTENT_MASK else {}
_mask_log_count = 0  # throttle coverage logging to first 3 pages


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
    clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
    if USE_CLIP_ONLY:
        # table variant: use CLIP cosine alone (H4.e strategy), with H6.1 preprocessing
        for meta, clip_cos in zip(meta_buf, clip_sims):
            meta["clip_compare"] = {"clip_cosine": float(clip_cos)}
            results.append(meta)
    else:
        # all other variants (including all_no_mask): 50/50 CLIP + DINOv2 average
        # For all_no_mask: DINOv2 receives preprocessed image via _dinov2_cosine_batch_preproc
        # For other non-table variants: DINOv2 receives raw RGB (unchanged from H6.1)
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

    # H14.1.diag: Apply GT-polygon content mask for variant=all_no_mask only
    if USE_CONTENT_MASK:
        page_key = page_dir.name
        content_mask = _make_content_mask(page_key, orig_pil.size)
        if content_mask is not None:
            # Resize mask to match recon if sizes differ
            if recon_pil.size != orig_pil.size:
                recon_mask = content_mask.resize(recon_pil.size, Image.NEAREST)
            else:
                recon_mask = content_mask
            orig_pil = _apply_content_mask(orig_pil, content_mask)
            recon_pil = _apply_content_mask(recon_pil, recon_mask)
            # Log mask coverage for first 3 pages
            if _mask_log_count < 3:
                mask_arr = np.array(content_mask)
                coverage = 100.0 * np.sum(mask_arr > 0) / mask_arr.size
                log.info(
                    "Page %s: content mask applied, coverage=%.1f%% (%d polys)",
                    page_key, coverage, len(_PAGE_POLYS.get(page_key, [])),
                )
                _mask_log_count += 1
        # If no polys found, fall through to passthrough (orig_pil/recon_pil unchanged)

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
