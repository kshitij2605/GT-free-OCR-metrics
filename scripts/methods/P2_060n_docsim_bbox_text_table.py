#!/usr/bin/env python3
"""P2_060n — D60.n: per-bbox DocSim cosine for {text, table}.

Extends D60.m by also applying per-bbox DocSim aggregation to variant=text
using ocr_elements.json text bboxes. Other 3 variants (formula, all,
all_no_mask) byte-equal to D60.m.

For variant=text:
  - bbox source: ocr_elements.json (5-50 text-element bboxes per page)
  - fallback for empty-bbox pages: page-level DocSim (matches D60.m text path)

For variant=table: identical to D60.m (per-table-bbox; fallback H6.1
CLIP-only).

Predicted spearman_mean: 0.443-0.475, most likely 0.450 (+0.007 vs
D60.m 0.4432). KEEP if >= 0.450. Kill switch: text cc <= 0.40.
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

sys.path.insert(0, "/home/mac/test/r1-p2/src")
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P2_060n_docsim_bbox_text_table"
SSIM_SIZE = 512
BATCH_SIZE = 16
DINO_SIZE = 224

ADAPTER_DIR = Path("/home/mac/test/r1-p2/models/docsim_lora/lora_adapter_best")
HEAD_STATE_PATH = Path("/home/mac/test/r1-p2/models/docsim_lora/head_state_best.pt")
HEAD_CONFIG_PATH = Path("/home/mac/test/r1-p2/models/docsim_lora/config.json")

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <variant>", file=sys.stderr)
    sys.exit(1)

variant = sys.argv[1]
if variant not in {"all", "text", "formula", "table", "all_no_mask"}:
    print(f"Unknown variant: {variant}", file=sys.stderr)
    sys.exit(1)

# Strategy flags
USE_DISTS = variant == "all"
USE_PREPROC = variant in ("table", "all_no_mask")  # H6.1 preproc gate
# Per-bbox DocSim for {text, table}; page-level DocSim for {all, all_no_mask}.
USE_DOCSIM_BBOX = variant in ("text", "table")
USE_DOCSIM_PAGE = variant in ("all", "all_no_mask")
NEED_DOCSIM = USE_DOCSIM_PAGE or USE_DOCSIM_BBOX
NEED_DINO = NEED_DOCSIM or variant == "formula"  # DINO needed for DocSim or formula 50/50 fusion
# Per-variant bbox file + fallback policy (text empty -> page-level DocSim; table empty -> H6.1 CLIP-only)
BBOX_FILE = {"text": "ocr_elements.json", "table": "ocr_table_elements.json"}.get(variant)
TEXT_FALLBACK_PAGE_DOCSIM = variant == "text"  # text empty-bbox pages use page-level DocSim, not CLIP-only

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
    "device=%s variant=%s method=%s use_dists=%s use_preproc=%s use_docsim_page=%s use_docsim_bbox=%s",
    device, variant, METHOD_ID, USE_DISTS, USE_PREPROC, USE_DOCSIM_PAGE, USE_DOCSIM_BBOX,
)

_preprocessor = ImagePreprocessor()

# ── DISTS setup (variant=all only) ────────────────────────────────────────────
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


# ── DINOv2 setup ──────────────────────────────────────────────────────────────
if NEED_DINO:
    log.info("Loading DINOv2 vitb14...")
    _dino_model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True,
    )
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

# ── OpenCLIP setup ────────────────────────────────────────────────────────────
log.info("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)
for p in _clip_model.parameters():
    p.requires_grad = False

# ── LPIPS setup ───────────────────────────────────────────────────────────────
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

    log.info("Loading DocSim head config from %s", HEAD_CONFIG_PATH)
    with HEAD_CONFIG_PATH.open("r") as f:
        head_cfg = json.load(f)

    clip_dim = _clip_model.visual.output_dim  # 512
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


# ── Preprocessing helpers ─────────────────────────────────────────────────────
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


# ── Cosine implementations ────────────────────────────────────────────────────
def _dinov2_cosine_batch(orig_pils, recon_pils):
    def _enc(pils):
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


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
    orig_f = _enc(orig_pils)
    recon_f = _enc(recon_pils)
    return (orig_f * recon_f).sum(dim=-1).cpu().tolist()


def _docsim_cosine_batch(orig_pils, recon_pils):
    """DocSim cosine: LoRA-projected fusion of frozen CLIP + DINOv2 CLS embeddings."""
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


def _crop_bbox(pil: Image.Image, bbox) -> Image.Image | None:
    """Crop a bbox [x1, y1, x2, y2] from a PIL image; return None if degenerate."""
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    W, H = pil.size
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return pil.crop((x1, y1, x2, y2))


def _docsim_per_bbox_score(orig_pil: Image.Image, recon_pil: Image.Image, bboxes) -> float | None:
    """Mean DocSim cosine over per-table-bbox crops. None if no usable bboxes."""
    orig_crops = []
    recon_crops = []
    for bb in bboxes:
        oc = _crop_bbox(orig_pil, bb)
        rc = _crop_bbox(recon_pil, bb)
        if oc is None or rc is None:
            continue
        orig_crops.append(oc)
        recon_crops.append(rc)
    if not orig_crops:
        return None
    cos_sims = _docsim_cosine_batch(orig_crops, recon_crops)
    return float(np.mean(cos_sims))


# ── Main processing loop ──────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf = [], [], []  # for full-page batched cosine paths


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf
    if not orig_pils_buf:
        return

    if USE_DOCSIM_PAGE or (variant == "text" and TEXT_FALLBACK_PAGE_DOCSIM):
        # variant in {all, all_no_mask}: page-level DocSim (D60.m).
        # variant=text bbox-empty pages: page-level DocSim fallback.
        cos_sims = _docsim_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, cs in zip(meta_buf, cos_sims):
            meta["clip_compare"] = {"clip_cosine": float(cs)}
            results.append(meta)
    elif variant == "table":
        # variant=table bbox-empty pages: H6.1 CLIP-only fallback (D60.m semantics).
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, clip_cos in zip(meta_buf, clip_sims):
            meta["clip_compare"] = {"clip_cosine": float(clip_cos)}
            results.append(meta)
    else:
        # variant=formula: 50/50 CLIP + DINOv2 fusion (D60.j baseline).
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        dino_sims = _dinov2_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, dino_cos, clip_cos in zip(meta_buf, dino_sims, clip_sims):
            avg_cos = 0.5 * dino_cos + 0.5 * clip_cos
            meta["clip_compare"] = {"clip_cosine": float(avg_cos)}
            results.append(meta)
    orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


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

    if USE_DOCSIM_BBOX:
        # Per-bbox DocSim path. Process each page individually.
        bbox_path = page_dir / BBOX_FILE
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
            meta["clip_compare"] = {"clip_cosine": float(score)}
            results.append(meta)
        else:
            # Fallback to baseline H6.1 CLIP-only — buffer for batched flush.
            n_fallback_pages += 1
            orig_pils_buf.append(orig_pil)
            recon_pils_buf.append(recon_pil)
            meta_buf.append(meta)
    else:
        # All non-table variants: standard batched flush.
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

if USE_DOCSIM_BBOX:
    log.info("variant=table bbox-path pages: %d / fallback pages: %d", n_bbox_pages, n_fallback_pages)

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s", len(results), out_path)
