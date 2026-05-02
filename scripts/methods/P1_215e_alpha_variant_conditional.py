#!/usr/bin/env python3
"""P1_215b — D215.b: D60.p production cc fused with window-MIN confidence at alpha=0.2.

Tests whether the entropy axis adds COMPLEMENTARY signal beyond the D60.p
production stack. For each variant, takes the production-stack clip_cosine
(per-cell-MIN DocSim for table; page-level DocSim for text/all/all_no_mask;
50/50 CLIP+DINOv2 for formula) and FUSES with window-MIN confidence:

    clip_cosine_new = 0.8 * production_cc + 0.2 * confidence_min_window

confidence_min_window = MIN over rolling K=10 token-window means of
exp(token_logprob) — the same signal as D224 (window-MIN of per-token Qwen
confidence). D224 showed window-MIN lifts the entropy axis +0.08 to +0.16
on 4/5 variants vs page-level mean, but standalone is still 0.16-0.36
below per-variant ceilings. Fusion at alpha=0.2 tests whether residual
complementary signal exists. If KEEP > 0.450, sweep alpha=0.1, 0.3, 0.4.
If REVERT, the entropy axis is fully closed (page-mean dead, window-MIN
standalone dead, fusion dead) -> emit COMPLETE.

multi_composite preserved from D60.p (H5.b composite [+ DISTS for
variant=all]) so eval still has the production fallback path.
"""

import json
import logging
import math
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

METHOD_ID = "P1_215e_alpha_variant_conditional"
SSIM_SIZE = 512
BATCH_SIZE = 16
DINO_SIZE = 224
WINDOW_K = 10
# Per-variant alpha from D215.b/c/d sweep (commit 171daa1 + alpha-sweep batch):
# - table: alpha=0.0 (per-cell-MIN already extracts cell-level signal; entropy DILUTES)
# - text/formula/all_no_mask: alpha=0.4 (still climbing at sweep top)
# - all: alpha=0.3 (peak at 0.4061 vs 0.4057 at alpha=0.4 — within noise, use 0.3)

ADAPTER_DIR = Path(__file__).parent.parent.parent / "models" / "docsim_lora" / "lora_adapter_best"
HEAD_STATE_PATH = Path(__file__).parent.parent.parent / "models" / "docsim_lora" / "head_state_best.pt"
HEAD_CONFIG_PATH = Path(__file__).parent.parent.parent / "models" / "docsim_lora" / "config.json"
LOGPROBS_ROOT = Path(__file__).parent.parent.parent / "data" / "ocr_logprobs"

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <variant>", file=sys.stderr)
    sys.exit(1)

variant = sys.argv[1]
if variant not in {"all", "text", "formula", "table", "all_no_mask"}:
    print(f"Unknown variant: {variant}", file=sys.stderr)
    sys.exit(1)

USE_DISTS = variant == "all"
USE_PREPROC = variant in ("table", "all_no_mask")
USE_DOCSIM_PAGE = variant in ("text", "all", "all_no_mask")
USE_DOCSIM_BBOX = variant == "table"
NEED_DOCSIM = USE_DOCSIM_PAGE or USE_DOCSIM_BBOX
NEED_DINO = variant != "table" or USE_DOCSIM_BBOX

# Per-variant alpha (resolved at runtime from variant arg)
ALPHA_ENTROPY = {
    "table": 0.0,
    "all": 0.3,
    "text": 0.4,
    "formula": 0.4,
    "all_no_mask": 0.4,
}[variant]

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
    "device=%s variant=%s method=%s alpha=%.2f window_k=%d use_dists=%s use_preproc=%s use_docsim_page=%s use_docsim_bbox=%s",
    device, variant, METHOD_ID, ALPHA_ENTROPY, WINDOW_K,
    USE_DISTS, USE_PREPROC, USE_DOCSIM_PAGE, USE_DOCSIM_BBOX,
)

_preprocessor = ImagePreprocessor()

# ── DISTS setup ───────────────────────────────────────────────────────────────
if USE_DISTS:
    import pyiqa
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
        return float(np.clip(_dists_metric(orig_t, recon_t).item(), 0.0, 1.0))


# ── DINOv2 ────────────────────────────────────────────────────────────────────
if NEED_DINO:
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
else:
    _dino_model = None
    _dino_transform = None

# ── OpenCLIP ──────────────────────────────────────────────────────────────────
log.info("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)
for p in _clip_model.parameters():
    p.requires_grad = False

# ── LPIPS ─────────────────────────────────────────────────────────────────────
log.info("Loading LPIPS (alex)...")
_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)


# ── DocSim head ───────────────────────────────────────────────────────────────
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
if NEED_DOCSIM:
    from peft import LoraConfig, get_peft_model

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
    state = torch.load(HEAD_STATE_PATH, map_location=device, weights_only=True)
    _docsim_head.load_state_dict(state)
    _docsim_head = _docsim_head.eval().to(device)
    for p in _docsim_head.parameters():
        p.requires_grad = False


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    return float(structural_similarity(og, rg, data_range=1.0)), float(np.mean((og - rg) ** 2))


def _lpips_score(orig_pil, recon_pil):
    def _to_t(p):
        g = p.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR)
        arr = np.array(g, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        return (t * 2.0 - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(np.clip(_lpips_fn(_to_t(orig_pil), _to_t(recon_pil)).item(), 0.0, 1.0))


def _baseline_clip_preprocess(img):
    img = img.convert("L").filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img).convert("RGB")
    return img


def _dinov2_cosine_batch(orig_pils, recon_pils):
    def _enc(pils):
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    return (_enc(orig_pils) * _enc(recon_pils)).sum(dim=-1).cpu().tolist()


def _clip_cosine_batch(orig_pils, recon_pils):
    def _enc(pils):
        if USE_PREPROC:
            processed = [_baseline_clip_preprocess(p) for p in pils]
        else:
            processed = [p.convert("RGB") for p in pils]
        batch = torch.stack([_clip_preprocess(p) for p in processed]).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_image(batch)
        return F.normalize(feats.float(), dim=-1)
    return (_enc(orig_pils) * _enc(recon_pils)).sum(dim=-1).cpu().tolist()


def _docsim_cosine_batch(orig_pils, recon_pils):
    def _encode(pils):
        clip_batch = torch.stack([_clip_preprocess(p.convert("RGB")) for p in pils]).to(device)
        dino_batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            f_clip = F.normalize(_clip_model.encode_image(clip_batch).float(), dim=-1)
            f_dino = F.normalize(_dino_model(dino_batch).float(), dim=-1)
            x = torch.cat([f_clip, f_dino], dim=-1)
            return _docsim_head(x)
    return (_encode(orig_pils) * _encode(recon_pils)).sum(dim=-1).cpu().tolist()


def _crop_bbox(pil, bbox):
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    W, H = pil.size
    x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return pil.crop((x1, y1, x2, y2))


def _docsim_per_bbox_score(orig_pil, recon_pil, bboxes):
    orig_crops, recon_crops = [], []
    for bb in bboxes:
        oc = _crop_bbox(orig_pil, bb); rc = _crop_bbox(recon_pil, bb)
        if oc is None or rc is None:
            continue
        orig_crops.append(oc); recon_crops.append(rc)
    if not orig_crops:
        return None
    cos_sims = _docsim_cosine_batch(orig_crops, recon_crops)
    return float(np.min(cos_sims))


def _window_min_confidence(page_name):
    """MIN over rolling K-token window means of per-token confidence (D224 signal).

    Returns 0.5 (neutral midpoint) if logprobs missing — neutral so fusion
    stays close to production cc rather than zeroing it.
    """
    lp_path = LOGPROBS_ROOT / page_name / "ocr_logprobs.json"
    if not lp_path.exists():
        return 0.5
    with open(lp_path) as f:
        data = json.load(f)
    confs = []
    for t in data.get("tokens", []):
        lp = t.get("logprob")
        if lp is not None:
            confs.append(math.exp(lp))
    if not confs:
        return 0.5
    if len(confs) < WINDOW_K:
        return float(np.mean(confs))
    arr = np.array(confs, dtype=np.float64)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    window_means = (csum[WINDOW_K:] - csum[:-WINDOW_K]) / WINDOW_K
    return float(window_means.min())


def _fuse(prod_cc, page_name):
    """Convex combo of production cc and window-MIN confidence at ALPHA_ENTROPY."""
    conf = _window_min_confidence(page_name)
    return (1.0 - ALPHA_ENTROPY) * prod_cc + ALPHA_ENTROPY * conf


# ── Main ──────────────────────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf, name_buf = [], [], [], []


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf, name_buf
    if not orig_pils_buf:
        return
    if USE_DOCSIM_PAGE:
        cos_sims = _docsim_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, cs, name in zip(meta_buf, cos_sims, name_buf):
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(cs), name))}
            results.append(meta)
    elif variant == "table":
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, clip_cos, name in zip(meta_buf, clip_sims, name_buf):
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(clip_cos), name))}
            results.append(meta)
    else:
        # variant=formula: 50/50 CLIP+DINOv2 fusion
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        dino_sims = _dinov2_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, dino_cos, clip_cos, name in zip(meta_buf, dino_sims, clip_sims, name_buf):
            avg_cos = 0.5 * dino_cos + 0.5 * clip_cos
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(avg_cos), name))}
            results.append(meta)
    orig_pils_buf, recon_pils_buf, meta_buf, name_buf = [], [], [], []


t0 = time.time()
n_bbox_pages = 0
n_fallback_pages = 0
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
        multi_metric = {"ssim": ssim_val, "mse": mse_val, "lpips": lpips_val, "dists": dists_val, "composite": composite}
    else:
        composite = 0.4 * ssim_val + 0.3 * max(0.0, 1.0 - mse_val) + 0.3 * max(0.0, 1.0 - lpips_val)
        multi_metric = {"ssim": ssim_val, "mse": mse_val, "lpips": lpips_val, "composite": composite}

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

    if USE_DOCSIM_BBOX:
        bbox_path = page_dir / "ocr_table_elements.json"
        bboxes = []
        if bbox_path.exists():
            try:
                with bbox_path.open("r") as f:
                    elems = json.load(f)
                bboxes = [e.get("bbox") for e in elems if e.get("bbox")]
            except Exception as e:
                log.warning("page=%s bbox load failed: %s", page_dir.name, e)
                bboxes = []
        score = _docsim_per_bbox_score(orig_pil, recon_pil, bboxes) if bboxes else None
        if score is not None:
            n_bbox_pages += 1
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(score), page_dir.name))}
            results.append(meta)
        else:
            n_fallback_pages += 1
            orig_pils_buf.append(orig_pil)
            recon_pils_buf.append(recon_pil)
            meta_buf.append(meta)
            name_buf.append(page_dir.name)
    else:
        orig_pils_buf.append(orig_pil)
        recon_pils_buf.append(recon_pil)
        meta_buf.append(meta)
        name_buf.append(page_dir.name)

    if len(orig_pils_buf) >= BATCH_SIZE:
        _flush_batch()

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        log.info("[%d/%d] %.1fs elapsed (%.2fs/page)",
                 i + 1, len(page_dirs), elapsed, elapsed / (i + 1))

_flush_batch()

if USE_DOCSIM_BBOX:
    log.info("variant=table bbox-path pages: %d / fallback pages: %d", n_bbox_pages, n_fallback_pages)

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s (alpha=%.2f, window_k=%d)", len(results), out_path, ALPHA_ENTROPY, WINDOW_K)
