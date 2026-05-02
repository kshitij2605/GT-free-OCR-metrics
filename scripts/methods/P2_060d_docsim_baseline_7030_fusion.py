#!/usr/bin/env python3
"""P2_060c — D60.c DocSim + baseline cc 50/50 fusion for {all, all_no_mask}.

Identical to P2_060b EXCEPT: for variants {all, all_no_mask}, the clip_cosine
slot returns 0.5 * DocSim_cos + 0.5 * baseline_50_50_cos, where baseline_cos
is the H4.e CLIP+DINOv2 fusion on raw RGB (the same fusion used for {text,
formula} in the production stack). Other variants {text, formula, table}
unchanged from P2_060b/P1_120c3b.

Tests whether the trained DocSim cosine carries unique signal vs whether
ensembling with the unsupervised baseline preserves or lifts the gain.

Reuses trained adapter at models/docsim_lora/lora_adapter_best/ unchanged.
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P2_060d_docsim_baseline_7030_fusion"
SSIM_SIZE = 512
BATCH_SIZE = 16
DINO_SIZE = 224
W_DOCSIM = 0.7
W_BASELINE = 0.3

ADAPTER_DIR = Path(__file__).parent.parent.parent / "models" / "docsim_lora" / "lora_adapter_best"
HEAD_STATE_PATH = Path(__file__).parent.parent.parent / "models" / "docsim_lora" / "head_state_best.pt"
HEAD_CONFIG_PATH = Path(__file__).parent.parent.parent / "models" / "docsim_lora" / "config.json"

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
USE_DOCSIM = variant in ("all", "all_no_mask")

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
    "device=%s variant=%s method=%s use_dists=%s use_clip_only=%s use_preproc=%s use_docsim=%s w_docsim=%s w_baseline=%s",
    device, variant, METHOD_ID, USE_DISTS, USE_CLIP_ONLY, USE_PREPROC, USE_DOCSIM, W_DOCSIM, W_BASELINE,
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


def _dists_score(orig_pil, recon_pil):
    orig_t = _dists_transform(orig_pil.convert("RGB")).unsqueeze(0).to(device)
    recon_t = _dists_transform(recon_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = _dists_metric(orig_t, recon_t)
    return float(np.clip(dist.item(), 0.0, 1.0))


# DINOv2 always loaded (needed for both baseline cc fusion and DocSim path)
log.info("Loading DINOv2 vitb14...")
_dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
_dino_model = _dino_model.eval().to(device)
for p in _dino_model.parameters():
    p.requires_grad = False
_dino_transform = T.Compose([
    T.Resize((DINO_SIZE, DINO_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

log.info("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)
for p in _clip_model.parameters():
    p.requires_grad = False

log.info("Loading LPIPS (alex)...")
_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)


class DocSimHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


_docsim_head = None
if USE_DOCSIM:
    from peft import LoraConfig, get_peft_model

    log.info("Loading DocSim head config from %s", HEAD_CONFIG_PATH)
    with HEAD_CONFIG_PATH.open("r") as f:
        head_cfg = json.load(f)

    clip_dim = _clip_model.visual.output_dim
    dino_dim = 768

    _docsim_head = DocSimHead(
        in_dim=clip_dim + dino_dim,
        hidden_dim=head_cfg["hidden_dim"],
        embed_dim=head_cfg["embed_dim"],
    )
    lora_cfg = LoraConfig(
        r=head_cfg["lora_r"],
        lora_alpha=head_cfg["lora_alpha"],
        lora_dropout=head_cfg["lora_dropout"],
        target_modules=["0", "2"],
        bias="none",
    )
    _docsim_head.proj = get_peft_model(_docsim_head.proj, lora_cfg)

    log.info("Loading DocSim head state from %s", HEAD_STATE_PATH)
    state = torch.load(HEAD_STATE_PATH, map_location=device, weights_only=True)
    _docsim_head.load_state_dict(state)
    _docsim_head = _docsim_head.eval().to(device)
    for p in _docsim_head.parameters():
        p.requires_grad = False


def _to_gray_small(pil):
    return pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR)


def _ssim_mse(orig_pil, recon_pil):
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


def _lpips_score(orig_pil, recon_pil):
    def _to_t(p):
        g = p.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR)
        arr = np.array(g, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        return (t * 2.0 - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = _lpips_fn(_to_t(orig_pil), _to_t(recon_pil))
    return float(np.clip(dist.item(), 0.0, 1.0))


def _baseline_clip_preprocess(img):
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")
    return img


def _dinov2_cosine_batch_raw(orig_pils, recon_pils):
    def _enc(pils):
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


def _dinov2_cosine_batch_preproc(orig_pils, recon_pils):
    def _enc(pils):
        preprocessed = [_baseline_clip_preprocess(p) for p in pils]
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in preprocessed]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


def _clip_cosine_batch(orig_pils, recon_pils):
    """CLIP cosine; uses H6.1 preproc when USE_PREPROC, raw RGB otherwise.

    For DOCSIM variants {all, all_no_mask}, USE_PREPROC=True for all_no_mask
    (matches H11.2) and False for all (matches H4.e). Baseline-side fusion
    uses the H4.e-shape recipe: 50/50 CLIP+DINOv2, raw RGB for `all` and
    H6.1-preprocessed for `all_no_mask`.
    """
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


def _docsim_cosine_batch(orig_pils, recon_pils):
    def _encode(pils):
        clip_batch = torch.stack([_clip_preprocess(p.convert("RGB")) for p in pils]).to(device)
        dino_batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            f_clip = _clip_model.encode_image(clip_batch).float()
            f_dino = _dino_model(dino_batch).float()
            f_clip = F.normalize(f_clip, dim=-1)
            f_dino = F.normalize(f_dino, dim=-1)
            x = torch.cat([f_clip, f_dino], dim=-1)
            return _docsim_head(x)
    a = _encode(orig_pils)
    r = _encode(recon_pils)
    return (a * r).sum(dim=-1).cpu().tolist()


def _baseline_5050_fusion_cosine(orig_pils, recon_pils):
    """H4.e-style 50/50 CLIP+DINOv2 fusion. Matches the production stack
    used for variants {text, formula} in P2_060b. Pixel routing follows
    USE_PREPROC: H6.1 preprocess for all_no_mask, raw RGB for all.
    """
    clip_sims = _clip_cosine_batch(orig_pils, recon_pils)
    if USE_PREPROC:
        dino_sims = _dinov2_cosine_batch_preproc(orig_pils, recon_pils)
    else:
        dino_sims = _dinov2_cosine_batch_raw(orig_pils, recon_pils)
    return [0.5 * d + 0.5 * c for d, c in zip(dino_sims, clip_sims)]


page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf
    if not orig_pils_buf:
        return

    if USE_DOCSIM:
        docsim_sims = _docsim_cosine_batch(orig_pils_buf, recon_pils_buf)
        baseline_sims = _baseline_5050_fusion_cosine(orig_pils_buf, recon_pils_buf)
        for meta, ds, bs in zip(meta_buf, docsim_sims, baseline_sims):
            cos = W_DOCSIM * ds + W_BASELINE * bs
            meta["clip_compare"] = {"clip_cosine": float(cos)}
            results.append(meta)
    elif USE_CLIP_ONLY:
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, clip_cos in zip(meta_buf, clip_sims):
            meta["clip_compare"] = {"clip_cosine": float(clip_cos)}
            results.append(meta)
    else:
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        dino_sims = _dinov2_cosine_batch_raw(orig_pils_buf, recon_pils_buf)
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
            "ssim": ssim_val, "mse": mse_val, "lpips": lpips_val,
            "dists": dists_val, "composite": composite,
        }
    else:
        composite = 0.4 * ssim_val + 0.3 * max(0.0, 1.0 - mse_val) + 0.3 * max(0.0, 1.0 - lpips_val)
        multi_metric = {
            "ssim": ssim_val, "mse": mse_val, "lpips": lpips_val,
            "composite": composite,
        }

    meta = {
        "image": page_dir.name,
        "text_elements": 0, "image_regions": 0, "table_regions": 0,
        "text_length": 0, "plain_text_length": 0,
        "multi_metric": multi_metric,
        "lm_perplexity": {
            "ngram_score": 0.0, "transformer_score": 0.0,
            "perplexity": 0.0, "composite": 0.0,
        },
    }

    orig_pils_buf.append(orig_pil)
    recon_pils_buf.append(recon_pil)
    meta_buf.append(meta)

    if len(orig_pils_buf) >= BATCH_SIZE:
        _flush_batch()

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        log.info("[%d/%d] %.1fs elapsed (%.2fs/page)",
                 i + 1, len(page_dirs), elapsed, elapsed / (i + 1))

_flush_batch()

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s", len(results), out_path)
