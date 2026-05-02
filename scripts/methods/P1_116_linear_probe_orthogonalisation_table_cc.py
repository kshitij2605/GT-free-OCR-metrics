#!/usr/bin/env python3
"""P1_116_linear_probe_orthogonalisation_table_cc: H15 / D116 — PCA orthogonalisation on CLIP features for variant=table cc.

For variant=table:
  Two-pass approach:
    Pass 1: Encode all GT and reconstruction images with OpenCLIP ViT-B/32 + H6.1 preprocessing.
            Build feature matrix F of shape (2*N_pages, 512).
    PCA:    Compute SVD of centered F. Take top-k (default k=3) principal components.
            Build projection P = I - V_k^T @ V_k (orthogonal complement of nuisance subspace).
    Pass 2: Apply projection P to both GT and reconstruction features, then compute cosine.

All other 4 variants (text, formula, all, all_no_mask) are bit-for-bit identical to
H11.2 (P1_120c3b_all_no_mask_baseline_preproc, spearman_mean=0.3600, control).

multi_composite formulas (UNCHANGED from H5.b / H6.1):
  variant=all (USE_DISTS=True):
    composite = 0.3*SSIM + 0.2*(1-MSE) + 0.2*(1-LPIPS) + 0.3*(1-DISTS)
  all other variants (USE_DISTS=False):
    composite = 0.4*SSIM + 0.3*(1-MSE) + 0.3*(1-LPIPS)  (original, unchanged)

clip_cosine logic:
- variant=table:        CLIP-only (H6.1 preprocessing), PCA-projected cosine (D116 change)
- variant=all_no_mask:  0.5 * DINOv2_vitb14_CLS + 0.5 * OpenCLIP_ViT-B/32,
                        both preprocessed (H11.2 unchanged)
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P1_116_linear_probe_orthogonalisation_table_cc"
SSIM_SIZE = 512
BATCH_SIZE = 16
DINO_SIZE = 224
PCA_K = 3  # number of nuisance principal components to remove for variant=table

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
USE_PREPROC = variant in ("table", "all_no_mask")  # H6.1 / H11.2 preprocessing gate
USE_PCA_ORTHO = variant == "table"    # D116: PCA orthogonalisation ONLY for table variant

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
    "device=%s variant=%s method=%s use_dists=%s use_clip_only=%s use_preproc=%s use_pca_ortho=%s pca_k=%d",
    device, variant, METHOD_ID, USE_DISTS, USE_CLIP_ONLY, USE_PREPROC, USE_PCA_ORTHO, PCA_K,
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


# ── OpenCLIP encode (batched, returns raw normalized features) ─────────────────

def _clip_encode_batch(pils: list, use_preproc: bool) -> torch.Tensor:
    """Return L2-normalized CLIP features (B, D) for a list of PIL images."""
    if use_preproc:
        processed = [_baseline_clip_preprocess(p) for p in pils]
    else:
        processed = [p.convert("RGB") for p in pils]
    batch = torch.stack([_clip_preprocess(p) for p in processed]).to(device)
    with torch.no_grad():
        feats = _clip_model.encode_image(batch)
    return F.normalize(feats.float(), dim=-1)


# ── D116: Two-pass PCA orthogonalisation for variant=table ────────────────────

def _run_table_pca_cosine(page_dirs: list, pca_k: int) -> dict:
    """Two-pass encode + PCA project + cosine for variant=table.

    Returns dict mapping page_name -> clip_cosine (projected).
    """
    # Collect page pairs
    page_names = []
    orig_pil_list = []
    recon_pil_list = []
    for page_dir in page_dirs:
        orig_path = page_dir / "masked_original.png"
        recon_path = page_dir / "reconstructed.png"
        if not orig_path.exists() or not recon_path.exists():
            continue
        page_names.append(page_dir.name)
        orig_pil_list.append(Image.open(orig_path).convert("RGB"))
        recon_pil_list.append(Image.open(recon_path).convert("RGB"))

    N = len(page_names)
    log.info("PCA pass 1: encoding %d GT + %d recon images (total %d)", N, N, 2 * N)

    # Encode all GT features
    orig_feats = []
    for i in range(0, N, BATCH_SIZE):
        batch = orig_pil_list[i : i + BATCH_SIZE]
        orig_feats.append(_clip_encode_batch(batch, use_preproc=True).cpu())
    orig_feats = torch.cat(orig_feats, dim=0).numpy()  # (N, 512)

    # Encode all recon features
    recon_feats = []
    for i in range(0, N, BATCH_SIZE):
        batch = recon_pil_list[i : i + BATCH_SIZE]
        recon_feats.append(_clip_encode_batch(batch, use_preproc=True).cpu())
    recon_feats = torch.cat(recon_feats, dim=0).numpy()  # (N, 512)

    # Build combined matrix F (2N, 512) for PCA
    F = np.concatenate([orig_feats, recon_feats], axis=0).astype(np.float64)
    log.info("PCA: F shape=%s, computing top-%d principal components...", F.shape, pca_k)

    # Center F
    mu = F.mean(axis=0, keepdims=True)  # (1, 512)
    F_c = F - mu  # centered

    # SVD — we only need right singular vectors (rows of V^T = principal components)
    # Use numpy.linalg.svd with full_matrices=False for efficiency
    _, _, Vt = np.linalg.svd(F_c, full_matrices=False)  # Vt: (512, 512) but only min(2N,512) rows
    V_k = Vt[:pca_k, :]  # (k, 512) — top-k principal components (nuisance directions)

    log.info("PCA: captured top-%d PCs; building projection matrix P = I - V_k^T @ V_k", pca_k)

    # Project features: P * f = f - V_k^T @ (V_k @ f)
    # For a row vector f (1, 512): projected = f - (f @ V_k^T) @ V_k
    def _project(feats_np: np.ndarray) -> np.ndarray:
        """Project (N, D) feature matrix onto orthogonal complement of V_k."""
        coords = feats_np @ V_k.T  # (N, k)
        nuisance = coords @ V_k    # (N, D) — component in nuisance subspace
        projected = feats_np - nuisance  # (N, D)
        # Re-normalize after projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1e-8, norms)  # avoid division by zero
        return projected / norms

    orig_proj = _project(orig_feats)   # (N, 512) normalized
    recon_proj = _project(recon_feats)  # (N, 512) normalized

    # Cosine similarity in projected space (dot product of normalized vectors)
    cosines = (orig_proj * recon_proj).sum(axis=1)  # (N,)

    log.info(
        "PCA cosine stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        cosines.mean(), cosines.std(), cosines.min(), cosines.max(),
    )

    # Return page_name -> cosine mapping
    return {name: float(c) for name, c in zip(page_names, cosines)}


# ── Main processing loop ───────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
t0 = time.time()

if USE_PCA_ORTHO:
    # D116: Two-pass PCA + cosine for variant=table
    log.info("D116 mode: running two-pass PCA orthogonalisation for variant=table...")
    pca_cosines = _run_table_pca_cosine(page_dirs, PCA_K)
    log.info("D116 PCA cosine computed for %d pages", len(pca_cosines))

    # Now do the per-page multi_composite pass (same as baseline)
    for i, page_dir in enumerate(page_dirs):
        orig_path = page_dir / "masked_original.png"
        recon_path = page_dir / "reconstructed.png"
        if not orig_path.exists() or not recon_path.exists():
            continue
        if page_dir.name not in pca_cosines:
            continue

        orig_pil = Image.open(orig_path).convert("RGB")
        recon_pil = Image.open(recon_path).convert("RGB")

        ssim_val, mse_val = _ssim_mse(orig_pil, recon_pil)
        lpips_val = _lpips_score(orig_pil, recon_pil)

        # variant=table: original composite formula (no DISTS)
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
            "clip_compare": {"clip_cosine": pca_cosines[page_dir.name]},
        }
        results.append(meta)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            log.info("[%d/%d] %.1fs elapsed", i + 1, len(page_dirs), elapsed)

else:
    # All other variants: original streaming batch loop (bit-identical to P1_120c3b)
    orig_pils_buf, recon_pils_buf, meta_buf = [], [], []

    def _clip_cosine_batch(orig_pils: list, recon_pils: list) -> list:
        orig_f = _clip_encode_batch(orig_pils, use_preproc=USE_PREPROC)
        recon_f = _clip_encode_batch(recon_pils, use_preproc=USE_PREPROC)
        return (orig_f * recon_f).sum(dim=-1).cpu().tolist()

    def _flush_batch():
        global orig_pils_buf, recon_pils_buf, meta_buf
        if not orig_pils_buf:
            return
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        if USE_PREPROC:
            # all_no_mask: both encoders see preprocessed image (H11.2)
            dino_sims = _dinov2_cosine_batch_preproc(orig_pils_buf, recon_pils_buf)
        else:
            dino_sims = _dinov2_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, dino_cos, clip_cos in zip(meta_buf, dino_sims, clip_sims):
            avg_cos = 0.5 * dino_cos + 0.5 * clip_cos
            meta["clip_compare"] = {"clip_cosine": float(avg_cos)}
            results.append(meta)
        orig_pils_buf.clear()
        recon_pils_buf.clear()
        meta_buf.clear()

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
