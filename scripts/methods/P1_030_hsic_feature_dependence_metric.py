#!/usr/bin/env python3
"""P1_030_hsic_feature_dependence_metric: D30 — replace cosine with normalized HSIC (CKA)
for variant=all_no_mask cc only. All other variants pass through P1_120c3b (H11.2) stack
unchanged.

HSIC formulation: per-page, treat each D-dim feature vector as D 1D samples; build RBF
kernel matrices with median-heuristic bandwidth; compute normalized HSIC (Centered
Kernel Alignment, CKA) which is bounded in [0, 1] for non-negative kernels. The result
substitutes for the cosine inside the 50/50 fusion of DINOv2 + OpenCLIP for all_no_mask.

Diff vs P1_120c3b:
- New helper _hsic_cka_score(x, y) computes per-page CKA
- For variant=all_no_mask only, swap (orig_f * recon_f).sum(...) -> per-pair CKA
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

METHOD_ID = "P1_030_hsic_feature_dependence_metric"
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

USE_DISTS = variant == "all"
USE_CLIP_ONLY = variant == "table"
USE_PREPROC = variant in ("table", "all_no_mask")
USE_HSIC = variant == "all_no_mask"  # D30: HSIC only for all_no_mask cc

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
    "device=%s variant=%s method=%s use_dists=%s use_clip_only=%s use_preproc=%s use_hsic=%s",
    device, variant, METHOD_ID, USE_DISTS, USE_CLIP_ONLY, USE_PREPROC, USE_HSIC,
)

_preprocessor = ImagePreprocessor()

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

log.info("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)

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


def _baseline_clip_preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")
    return img


# ── HSIC / CKA helper (per-page, on CPU, runs on raw un-normalized features) ──
def _hsic_cka_score(x: torch.Tensor, y: torch.Tensor) -> float:
    """Normalized HSIC (Centered Kernel Alignment) between two feature vectors x, y in R^D.

    Treats each as D 1D samples; uses RBF kernel with median-heuristic bandwidth.
    Returns CKA in [0, 1] for non-negative kernels.
    """
    xn = x.detach().cpu().float().numpy()
    yn = y.detach().cpu().float().numpy()
    D = xn.shape[0]
    if D < 4:
        return 0.0

    # Squared pairwise distances
    dx2 = (xn[:, None] - xn[None, :]) ** 2
    dy2 = (yn[:, None] - yn[None, :]) ** 2

    # Median heuristic on positive entries
    triu_x = dx2[np.triu_indices(D, k=1)]
    triu_y = dy2[np.triu_indices(D, k=1)]
    sx = float(np.median(triu_x[triu_x > 0])) if (triu_x > 0).any() else 1.0
    sy = float(np.median(triu_y[triu_y > 0])) if (triu_y > 0).any() else 1.0
    sx = max(sx, 1e-8)
    sy = max(sy, 1e-8)

    Kx = np.exp(-dx2 / (2.0 * sx))
    Ky = np.exp(-dy2 / (2.0 * sy))

    # Centering: H = I - 1/D * 11^T
    Kx_c = Kx - Kx.mean(axis=0, keepdims=True) - Kx.mean(axis=1, keepdims=True) + Kx.mean()
    Ky_c = Ky - Ky.mean(axis=0, keepdims=True) - Ky.mean(axis=1, keepdims=True) + Ky.mean()

    hsic_xy = float((Kx_c * Ky_c).sum())
    hsic_xx = float((Kx_c * Kx_c).sum())
    hsic_yy = float((Ky_c * Ky_c).sum())

    denom = (hsic_xx * hsic_yy) ** 0.5
    if denom < 1e-12:
        return 0.0
    cka = hsic_xy / denom
    return float(np.clip(cka, 0.0, 1.0))


# ── Encoder cosine batches (un-normalized features returned for HSIC path) ────

def _dinov2_features_batch(orig_pils: list, recon_pils: list, preproc: bool) -> tuple:
    def _enc(pils):
        if preproc:
            pils = [_baseline_clip_preprocess(p) for p in pils]
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)  # CLS (B, D)
        return feats.float()
    return _enc(orig_pils), _enc(recon_pils)


def _clip_features_batch(orig_pils: list, recon_pils: list, preproc: bool) -> tuple:
    def _enc(pils):
        if preproc:
            pils = [_baseline_clip_preprocess(p) for p in pils]
        else:
            pils = [p.convert("RGB") for p in pils]
        batch = torch.stack([_clip_preprocess(p) for p in pils]).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_image(batch)
        return feats.float()
    return _enc(orig_pils), _enc(recon_pils)


def _cosine_from_features(orig_f: torch.Tensor, recon_f: torch.Tensor) -> list:
    o = F.normalize(orig_f, dim=-1)
    r = F.normalize(recon_f, dim=-1)
    return (o * r).sum(dim=-1).cpu().tolist()


def _hsic_from_features(orig_f: torch.Tensor, recon_f: torch.Tensor) -> list:
    return [_hsic_cka_score(o, r) for o, r in zip(orig_f, recon_f)]


page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf
    if not orig_pils_buf:
        return

    clip_orig_f, clip_recon_f = _clip_features_batch(orig_pils_buf, recon_pils_buf, preproc=USE_PREPROC)

    if USE_CLIP_ONLY:
        clip_sims = _cosine_from_features(clip_orig_f, clip_recon_f)
        for meta, clip_cos in zip(meta_buf, clip_sims):
            meta["clip_compare"] = {"clip_cosine": float(clip_cos)}
            results.append(meta)
    else:
        dino_orig_f, dino_recon_f = _dinov2_features_batch(orig_pils_buf, recon_pils_buf, preproc=USE_PREPROC)
        if USE_HSIC:
            clip_sims = _hsic_from_features(clip_orig_f, clip_recon_f)
            dino_sims = _hsic_from_features(dino_orig_f, dino_recon_f)
        else:
            clip_sims = _cosine_from_features(clip_orig_f, clip_recon_f)
            dino_sims = _cosine_from_features(dino_orig_f, dino_recon_f)
        for meta, dino_s, clip_s in zip(meta_buf, dino_sims, clip_sims):
            avg = 0.5 * dino_s + 0.5 * clip_s
            meta["clip_compare"] = {"clip_cosine": float(avg)}
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
