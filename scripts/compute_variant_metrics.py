"""Compute visual metrics for one render variant. Batched CLIP+LPIPS, parallel SSIM.

Usage: CUDA_VISIBLE_DEVICES=N uv run python compute_variant_metrics.py <variant>
"""
import json
import os
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity

BATCH_SIZE = 32
SSIM_SIZE = 512   # resize before binarize+SSIM — 12x faster than full res
SSIM_WORKERS = 8  # parallel SSIM threads (skimage releases GIL)
MIN_GPU_FREE_MB = 1500  # CLIP ViT-B-32 + LPIPS + batch buffers

variant = sys.argv[1]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor  # noqa: E402

BASE = Path(__file__).parent.parent / "data" / "omnidocbench"
var_root = BASE / f"ocr_{variant}"

import lpips as lpips_lib  # noqa: E402
import open_clip  # noqa: E402


def _setup_logging(name: str) -> logging.Logger:
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)-5s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # INFO+ → stdout (captured by nohup redirect to /tmp/metrics_{variant}.log)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # ERROR+ → dedicated error log for quick grep
    error_path = Path(__file__).parent.parent / "logs" / f"metrics_{name}.error.log"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text("")  # truncate on new run
    fh = logging.FileHandler(error_path)
    fh.setLevel(logging.ERROR)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _setup_logging(variant)


def _select_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free_bytes, _ = torch.cuda.mem_get_info()
    free_mb = free_bytes // (1024 * 1024)
    if free_mb < MIN_GPU_FREE_MB:
        log.warning("GPU has only %dMB free (<%dMB required), using CPU", free_mb, MIN_GPU_FREE_MB)
        return "cpu"
    log.info("GPU free: %dMB", free_mb)
    return "cuda"


device = _select_device()
log.info("device=%s", device)

t_model = time.time()
lpips_model = lpips_lib.LPIPS(net="alex").to(device).eval()
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.eval().to(device)
log.info("models loaded in %.1fs", time.time() - t_model)

_base_path = BASE / "ocr/results.json"
base_results = json.load(open(_base_path, encoding="utf-8")) if _base_path.exists() else []
base_by_dir = {Path(e["image"]).stem: e for e in base_results}

preprocessor = ImagePreprocessor()

page_dirs = sorted(p for p in var_root.iterdir() if p.is_dir())
log.info("%d page directories found", len(page_dirs))


def _clip_preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("L").filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img).convert("RGB")
    return clip_preprocess(img)


def _to_small_gray(img: Image.Image) -> Image.Image:
    """Resize to SSIM_SIZE and convert to grayscale — shared for LPIPS and SSIM."""
    return img.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.Resampling.BILINEAR)


def _lpips_tensor(gray_small: Image.Image) -> torch.Tensor:
    """Build LPIPS tensor from already-resized grayscale image."""
    arr = np.array(gray_small, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
    return t * 2.0 - 1.0  # [-1, 1]


def _ssim_mse(orig_gray: Image.Image, recon_gray: Image.Image):
    """Binarize and compute SSIM+MSE on small grayscale images."""
    ob = preprocessor.adaptive_binarize(orig_gray.convert("RGB"))
    rb = preprocessor.adaptive_binarize(recon_gray.convert("RGB"))
    if rb.size != ob.size:
        rb = rb.resize(ob.size, Image.Resampling.BILINEAR)
    og = np.array(ob, dtype=np.float64) / 255.0
    rg = np.array(rb, dtype=np.float64) / 255.0
    ssim = float(structural_similarity(og, rg, data_range=1.0))
    mse  = float(np.mean((og - rg) ** 2))
    return ssim, mse


def _move_to_cpu():
    """Fall back to CPU after mid-run OOM — moves models and updates global device."""
    global device, lpips_model, clip_model
    torch.cuda.empty_cache()
    device = "cpu"
    lpips_model = lpips_model.to("cpu")
    clip_model = clip_model.to("cpu")
    log.warning("OOM — moved models to CPU, continuing on remaining batches")


def _batch_gpu_scores(origs_full, recons_full, origs_small, recons_small):
    """Run batched CLIP + LPIPS. Returns (clip_scores, lpips_scores) lists."""
    clip_orig_batch  = torch.stack([_clip_preprocess(img) for img in origs_full]).to(device)
    clip_recon_batch = torch.stack([_clip_preprocess(img) for img in recons_full]).to(device)
    with torch.no_grad():
        f_orig  = F.normalize(clip_model.encode_image(clip_orig_batch),  dim=-1)
        f_recon = F.normalize(clip_model.encode_image(clip_recon_batch), dim=-1)
    clip_scores = (f_orig * f_recon).sum(dim=-1).cpu().tolist()

    lp_orig  = torch.stack([_lpips_tensor(g) for g in origs_small]).to(device)
    lp_recon = torch.stack([_lpips_tensor(g) for g in recons_small]).to(device)
    with torch.no_grad():
        lp_raw = lpips_model(lp_orig, lp_recon).squeeze(-1).squeeze(-1).squeeze(-1)
    lpips_scores = lp_raw.cpu().tolist()
    if not isinstance(lpips_scores, list):
        lpips_scores = [float(lpips_scores)]
    return clip_scores, lpips_scores


# Collect valid pages
valid, skip_errors = [], []
for page_dir in page_dirs:
    if (page_dir / "reconstructed.png").exists() and (page_dir / "masked_original.png").exists():
        valid.append(page_dir)
    else:
        skip_errors.append(page_dir.name)

if skip_errors:
    log.warning("%d dirs skipped (missing reconstructed.png or masked_original.png)", len(skip_errors))
    for name in skip_errors:
        log.error("SKIP missing images: %s", name)

log.info("%d valid pages to process", len(valid))

results = []
errors = list(skip_errors)
t0 = time.time()

with ThreadPoolExecutor(max_workers=SSIM_WORKERS) as pool:
    for batch_start in range(0, len(valid), BATCH_SIZE):
        batch_dirs = valid[batch_start:batch_start + BATCH_SIZE]
        t_batch = time.time()

        origs_full   = [Image.open(d / "masked_original.png").convert("RGB") for d in batch_dirs]
        recons_full  = [Image.open(d / "reconstructed.png").convert("RGB")   for d in batch_dirs]
        origs_small  = [_to_small_gray(img) for img in origs_full]
        recons_small = [_to_small_gray(img) for img in recons_full]

        # Batched CLIP + LPIPS — recover from mid-run GPU OOM by moving to CPU
        try:
            clip_scores, lpips_scores = _batch_gpu_scores(
                origs_full, recons_full, origs_small, recons_small
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                _move_to_cpu()
                clip_scores, lpips_scores = _batch_gpu_scores(
                    origs_full, recons_full, origs_small, recons_small
                )
            else:
                raise

        # Parallel SSIM/MSE (ThreadPool, skimage releases GIL)
        ssim_futures = [
            pool.submit(_ssim_mse, origs_small[i], recons_small[i])
            for i in range(len(batch_dirs))
        ]

        for i, page_dir in enumerate(batch_dirs):
            base = base_by_dir.get(page_dir.name, {})
            try:
                ssim_val, mse_val = ssim_futures[i].result()
                lpips_val = float(np.clip(lpips_scores[i], 0.0, 1.0))
                clip_s    = float(clip_scores[i])
                composite = 0.4 * ssim_val + 0.3 * (1.0 - mse_val) + 0.3 * (1.0 - lpips_val)

                results.append({
                    "image":             base.get("image", page_dir.name + ".png"),
                    "text_elements":     base.get("text_elements", 0),
                    "image_regions":     base.get("image_regions", 0),
                    "table_regions":     base.get("table_regions", 0),
                    "text_length":       base.get("text_length", 0),
                    "plain_text_length": base.get("plain_text_length", 0),
                    "multi_metric":  {"ssim": ssim_val, "mse": mse_val, "lpips": lpips_val, "composite": composite},
                    "clip_compare":  {"clip_cosine": clip_s},
                    "lm_perplexity": base.get("lm_perplexity", {"ngram_score": 0, "transformer_score": 0, "perplexity": 0, "composite": 0}),
                })
            except Exception as e:
                errors.append(f"{page_dir.name}: {e}")
                log.error("page %s: %s", page_dir.name, e, exc_info=True)

        done = batch_start + len(batch_dirs)
        elapsed = time.time() - t0
        batch_s = time.time() - t_batch
        if done % 128 < BATCH_SIZE or done >= len(valid):
            rate = done / elapsed if elapsed > 0 else 1
            rem = (len(valid) - done) / rate
            log.info(
                "[%d/%d] elapsed=%.0fs  batch=%.1fs  eta=%.0fs",
                done, len(valid), elapsed, batch_s, rem,
            )

# Output path consolidated to results/method_runs/ocr_<v>/baseline/ — same scheme as all other methods.
out_dir = Path(__file__).parent.parent / "results" / "method_runs" / f"ocr_{variant}" / "baseline"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "results.json"
json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
total = time.time() - t0
log.info(
    "Done. %d ok, %d errors. total=%.0fs (%.2f pages/s) → %s",
    len(results), len(errors), total, len(results) / total if total > 0 else 0, out,
)
if errors:
    log.warning("Error log: logs/metrics_%s.error.log", variant)
