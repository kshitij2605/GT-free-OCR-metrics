#!/usr/bin/env python3
"""D143 — per-element CLIP P10 extended to formula variant at β=0.05.

D137 applied elem_p10 to all/all_no_mask only. D141 extends to text variant.
D143 extends to formula variant: formula pages have ocr_formula_elements.json
bboxes; per-element CLIP P10 over formula crops may capture per-formula
rendering quality (compound symbols, spacing errors).

  formula: (1-0.3-0.5-0.05-0.05)*DocSim + 0.3*entropy + 0.5*IQ + 0.05*ssim + 0.05*elem_p10
         = 0.10*DocSim + 0.30*entropy + 0.50*IQ + 0.05*ssim + 0.05*elem_p10
  text/all/all_no_mask: same as D141 (unchanged)
  table: unchanged

KEEP if spearman_mean > 0.4932 (D137/D140 best). REVERT → formula elem_p10 does not help.
"""

import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import cv2
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

METHOD_ID = "P1_143_elem_p10_formula_beta05"
SSIM_SIZE = 512
BATCH_SIZE = 16
DINO_SIZE = 224

ADAPTER_DIR = Path("/home/mac/test/r1-p2/models/docsim_lora/lora_adapter_best")
HEAD_STATE_PATH = Path("/home/mac/test/r1-p2/models/docsim_lora/head_state_best.pt")
HEAD_CONFIG_PATH = Path("/home/mac/test/r1-p2/models/docsim_lora/config.json")
LOGPROBS_ROOT = Path("/home/mac/test/r1-p2/data/ocr_logprobs")
PER_BBOX_ROOT = Path("/home/mac/test/r1-p2/data/ocr_logprobs_per_bbox")
TOP_K = 5
_LOG_K_TOP = math.log(TOP_K)

SOURCE_ROUTING = {
    "text":        "class_filtered_per_bbox_max_entropy_min",
    "formula":     "window_min_mean_shannon",
    "table":       "window_min_mean_shannon",
    "all":         "token_class_filtered_window_min",
    "all_no_mask": "token_class_filtered_window_min",
}

TARGET_CLASSES_TEXT = {
    "text", "title", "plain_text", "header", "footer", "caption",
    "page_header", "page_footer", "page_number",
    "普通文字", "标题", "页眉文字", "页脚文字", "页码",
}

TARGET_CLASSES_CONTENT = {
    "text", "title", "plain_text", "caption", "table_caption",
    "table_footnote", "image_caption", "image", "table",
    "formula", "isolate_formula", "inline_formula", "equation",
    "普通文字", "标题", "公式", "行内公式", "独立公式",
    "表格", "表格标题", "图像", "图像标题",
}

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

ALPHA_ENTROPY = {
    "table": 0.0,
    "all": 0.3,
    "text": 0.4,
    "formula": 0.3,
    "all_no_mask": 0.4,
}[variant]
WINDOW_K = {
    "table": 10,
    "all": 3,
    "text": 3,
    "formula": 20,
    "all_no_mask": 3,
}[variant]
GAMMA_DIQA = {
    "table": 0.1,
    "all": 0.0,
    "text": 0.0,
    "formula": 0.5,
    "all_no_mask": 0.0,
}[variant]
DIQA_EPS = 1e-6
BETA_TABLE_CELL = {
    "table": 0.0,
    "all": 0.15,
    "text": 0.0,
    "formula": 0.0,
    "all_no_mask": 0.10,
}[variant]
BETA_SSIM = {
    "table": 0.15,
    "all": 0.20,
    "text": 0.15,
    "formula": 0.05,
    "all_no_mask": 0.10,
}[variant]
BETA_CONTENT_ELEM = {
    "table": 0.0,
    "text": 0.10,       # D141: text variant elem_p10 β=0.10
    "formula": 0.05,    # D143: formula variant elem_p10 β=0.05
    "all": 0.10,
    "all_no_mask": 0.10,
}[variant]

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


def _ssim_page_supplement(orig_pil, recon_pil):
    """D111: SSIM between grayscale pages at SSIM_SIZE (raw, not binarized)."""
    og = np.array(orig_pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float64) / 255.0
    rg = np.array(recon_pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float64) / 255.0
    return float(structural_similarity(og, rg, data_range=1.0))


def _clip_cosine_batch_plain(orig_pils, recon_pils):
    """CLIP cosine batch without USE_PREPROC — for per-element crops."""
    def _enc(pils):
        batch = torch.stack([_clip_preprocess(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_image(batch)
        return F.normalize(feats.float(), dim=-1)
    return (_enc(orig_pils) * _enc(recon_pils)).sum(dim=-1).cpu().tolist()


def _content_element_clip_p10(orig_pil, recon_pil, page_dir):
    """CLIP cosine P10 over all content-element bboxes (text+formula+table).

    Uses P10 (10th percentile) for robustness to heterogeneous element types.
    Returns 0.0 (neutral) if no valid crops found.
    """
    bboxes = []
    for fname in ("ocr_elements.json", "ocr_formula_elements.json", "ocr_table_elements.json"):
        ep = page_dir / fname
        if ep.exists():
            try:
                with ep.open("r") as f:
                    elems = json.load(f)
                bboxes.extend(e.get("bbox") for e in elems if e.get("bbox"))
            except Exception:
                pass
    if not bboxes:
        return 0.0
    orig_crops, recon_crops = [], []
    for bb in bboxes:
        oc = _crop_bbox(orig_pil, bb)
        rc = _crop_bbox(recon_pil, bb)
        if oc is not None and rc is not None:
            orig_crops.append(oc)
            recon_crops.append(rc)
    if not orig_crops:
        return 0.0
    sims = []
    for s in range(0, len(orig_crops), BATCH_SIZE):
        sims.extend(_clip_cosine_batch_plain(orig_crops[s:s + BATCH_SIZE], recon_crops[s:s + BATCH_SIZE]))
    return float(np.percentile(sims, 10))


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
    """D215.l Shannon-entropy variant: MIN over rolling K-token window means."""
    lp_path = LOGPROBS_ROOT / page_name / "ocr_logprobs.json"
    if not lp_path.exists():
        return 0.5
    with open(lp_path) as f:
        data = json.load(f)
    confs = []
    for t in data.get("tokens", []):
        top = t.get("top_logprobs", [])
        if not top:
            continue
        probs = [math.exp(tt.get("logprob", -100.0)) for tt in top]
        s = sum(probs)
        if s <= 0:
            continue
        probs = [p / s for p in probs]
        H = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        log_k = math.log(len(top))
        if log_k <= 0:
            continue
        confs.append(1.0 - H / log_k)
    if not confs:
        return 0.5
    if len(confs) < WINDOW_K:
        return float(np.mean(confs))
    arr = np.array(confs, dtype=np.float64)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    window_means = (csum[WINDOW_K:] - csum[:-WINDOW_K]) / WINDOW_K
    return float(window_means.min())


def _class_filtered_per_bbox_max_entropy_min_text(page_name):
    """D224.c source: class-filtered MIN over per-bbox shannon_entropy_max."""
    pb_path = PER_BBOX_ROOT / page_name / "per_bbox_logprobs.json"
    if not pb_path.exists():
        return 0.5
    with open(pb_path) as f:
        data = json.load(f)
    bboxes = data.get("bboxes", [])
    if not bboxes:
        return 0.5

    def _bb_conf(bb):
        s = bb.get("stats", {})
        h_max = s.get("shannon_entropy_max")
        if h_max is None:
            return None
        c = 1.0 - float(h_max) / _LOG_K_TOP
        return max(0.0, min(1.0, c))

    matched = [bb for bb in bboxes if bb.get("div_class") in TARGET_CLASSES_TEXT]
    confs = [c for c in (_bb_conf(bb) for bb in matched) if c is not None]
    if not confs:
        confs = [c for c in (_bb_conf(bb) for bb in bboxes) if c is not None]
    if not confs:
        return 0.5
    return float(np.min(confs))


_BBOX_MARKER_RE = re.compile(r'data-bbox="')


def _token_class_filtered_window_min_confidence(page_name):
    """D224.e source: class-filtered token stream then rolling-K window-MIN."""
    lp_path = LOGPROBS_ROOT / page_name / "ocr_logprobs.json"
    pb_path = PER_BBOX_ROOT / page_name / "per_bbox_logprobs.json"
    if not lp_path.exists() or not pb_path.exists():
        return _window_min_confidence(page_name)
    with open(lp_path) as f:
        lp_data = json.load(f)
    with open(pb_path) as f:
        pb_data = json.load(f)
    tokens = lp_data.get("tokens", [])
    bbox_classes = [bb.get("div_class") for bb in pb_data.get("bboxes", [])]
    if not tokens or not bbox_classes:
        return _window_min_confidence(page_name)

    confs = []
    for t in tokens:
        top = t.get("top_logprobs", [])
        if not top:
            confs.append(None); continue
        probs = [math.exp(tt.get("logprob", -100.0)) for tt in top]
        s = sum(probs)
        if s <= 0:
            confs.append(None); continue
        probs = [p / s for p in probs]
        H = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        log_k = math.log(len(top)) if len(top) > 1 else 0.0
        if log_k <= 0:
            confs.append(None); continue
        confs.append(1.0 - H / log_k)

    text_chars = []
    pos = 0
    for t in tokens:
        text_chars.append(pos)
        pos += len(t["token"])
    full = "".join(t["token"] for t in tokens)
    marker_positions = [m.start() for m in _BBOX_MARKER_RE.finditer(full)]

    n_pairs = min(len(marker_positions), len(bbox_classes))
    if n_pairs == 0:
        return _window_min_confidence(page_name)
    marker_positions = marker_positions[:n_pairs]
    bbox_classes = bbox_classes[:n_pairs]

    token_starts = np.array(text_chars)
    marker_arr = np.array(marker_positions)
    bbox_idx_per_token = np.searchsorted(marker_arr, token_starts, side="right") - 1

    filtered = []
    for t_idx, b_idx in enumerate(bbox_idx_per_token):
        if b_idx < 0:
            continue
        cls = bbox_classes[int(b_idx)]
        if cls not in TARGET_CLASSES_CONTENT:
            continue
        c = confs[t_idx]
        if c is None:
            continue
        filtered.append(c)

    if len(filtered) < WINDOW_K:
        return _window_min_confidence(page_name)

    arr = np.array(filtered, dtype=np.float64)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    window_means = (csum[WINDOW_K:] - csum[:-WINDOW_K]) / WINDOW_K
    return float(window_means.min())


def _diqa_features(pil):
    """D106 hand-crafted IQ feature vector (length 6)."""
    arr = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    focus = float(lap.var())
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(sx * sx + sy * sy)))
    contrast = float(gray.std())
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    edge_density = float(edges.mean()) / 255.0
    brightness = float(gray.mean())
    color_var = float(np.mean([arr[..., c].std() for c in range(3)]))
    return np.array([focus, grad_mag, contrast, edge_density, brightness, color_var], dtype=np.float64)


def _diqa_similarity(orig_pil, recon_pil):
    """exp(-mean per-feature relative absolute distance) in (0, 1]."""
    f_o = _diqa_features(orig_pil)
    f_r = _diqa_features(recon_pil)
    denom = np.maximum.reduce([np.abs(f_o), np.abs(f_r), np.full_like(f_o, DIQA_EPS)])
    rel_diff = np.abs(f_o - f_r) / denom
    return float(np.exp(-float(rel_diff.mean())))


def _fuse(prod_cc, page_name, diqa_sim=0.0, table_cell_score=0.0, ssim_score=0.0, content_elem_score=0.0):
    """Convex combo: production cc + entropy + optional D106 IQ + optional supplements."""
    src = SOURCE_ROUTING[variant]
    if src == "class_filtered_per_bbox_max_entropy_min":
        conf = _class_filtered_per_bbox_max_entropy_min_text(page_name)
    elif src == "token_class_filtered_window_min":
        conf = _token_class_filtered_window_min_confidence(page_name)
    else:
        conf = _window_min_confidence(page_name)
    if GAMMA_DIQA > 0.0 and BETA_SSIM > 0.0:
        # D143: include BETA_CONTENT_ELEM in the DIQA+SSIM branch (formula variant)
        beta_e = BETA_CONTENT_ELEM if content_elem_score > 0.0 else 0.0
        return (1.0 - ALPHA_ENTROPY - GAMMA_DIQA - BETA_SSIM - beta_e) * prod_cc + ALPHA_ENTROPY * conf + GAMMA_DIQA * diqa_sim + BETA_SSIM * ssim_score + beta_e * content_elem_score
    if GAMMA_DIQA > 0.0:
        return (1.0 - ALPHA_ENTROPY - GAMMA_DIQA) * prod_cc + ALPHA_ENTROPY * conf + GAMMA_DIQA * diqa_sim
    if BETA_TABLE_CELL > 0.0 or BETA_SSIM > 0.0 or BETA_CONTENT_ELEM > 0.0:
        beta_t = BETA_TABLE_CELL if table_cell_score > 0.0 else 0.0
        beta_g = BETA_SSIM
        beta_e = BETA_CONTENT_ELEM if content_elem_score > 0.0 else 0.0
        total_beta = beta_t + beta_g + beta_e
        if total_beta > 0.0:
            return (1.0 - ALPHA_ENTROPY - total_beta) * prod_cc + ALPHA_ENTROPY * conf + beta_t * table_cell_score + beta_g * ssim_score + beta_e * content_elem_score
    return (1.0 - ALPHA_ENTROPY) * prod_cc + ALPHA_ENTROPY * conf


# ── Main ──────────────────────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf, name_buf, table_cell_score_buf, ssim_score_buf, content_elem_score_buf = [], [], [], [], [], [], []


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf, name_buf, table_cell_score_buf, ssim_score_buf, content_elem_score_buf
    if not orig_pils_buf:
        return
    if USE_DOCSIM_PAGE:
        cos_sims = _docsim_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, cs, name, tcs, ghs, ces in zip(meta_buf, cos_sims, name_buf, table_cell_score_buf, ssim_score_buf, content_elem_score_buf):
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(cs), name, table_cell_score=tcs, ssim_score=ghs, content_elem_score=ces))}
            results.append(meta)
    elif variant == "table":
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, clip_cos, name, orig_p, recon_p, ghs in zip(
            meta_buf, clip_sims, name_buf, orig_pils_buf, recon_pils_buf, ssim_score_buf
        ):
            diqa_sim_t = _diqa_similarity(orig_p, recon_p) if GAMMA_DIQA > 0.0 else 0.0
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(clip_cos), name, diqa_sim_t, ssim_score=ghs))}
            results.append(meta)
    else:
        # variant=formula: 50/50 CLIP+DINOv2 fusion + D106 IQ term + optional SSIM + elem_p10
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        dino_sims = _dinov2_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, dino_cos, clip_cos, name, orig_p, recon_p, ghs, ces in zip(
            meta_buf, dino_sims, clip_sims, name_buf, orig_pils_buf, recon_pils_buf, ssim_score_buf, content_elem_score_buf
        ):
            avg_cos = 0.5 * dino_cos + 0.5 * clip_cos
            diqa_sim = _diqa_similarity(orig_p, recon_p) if GAMMA_DIQA > 0.0 else 0.0
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(avg_cos), name, diqa_sim, ssim_score=ghs, content_elem_score=ces))}
            results.append(meta)
    orig_pils_buf, recon_pils_buf, meta_buf, name_buf, table_cell_score_buf, ssim_score_buf, content_elem_score_buf = [], [], [], [], [], [], []


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
            diqa_sim_t = _diqa_similarity(orig_pil, recon_pil) if GAMMA_DIQA > 0.0 else 0.0
            ssim_t = ssim_val if BETA_SSIM > 0.0 else 0.0
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(score), page_dir.name, diqa_sim_t, ssim_score=ssim_t))}
            results.append(meta)
        else:
            n_fallback_pages += 1
            table_cell_score_buf.append(0.0)
            ssim_score_buf.append(ssim_val if BETA_SSIM > 0.0 else 0.0)
            content_elem_score_buf.append(0.0)
            orig_pils_buf.append(orig_pil)
            recon_pils_buf.append(recon_pil)
            meta_buf.append(meta)
            name_buf.append(page_dir.name)
    else:
        tcs = 0.0
        if BETA_TABLE_CELL > 0.0:
            table_bbox_path = page_dir / "ocr_table_elements.json"
            table_bboxes = []
            if table_bbox_path.exists():
                try:
                    with table_bbox_path.open("r") as _tf:
                        _telems = json.load(_tf)
                    table_bboxes = [e.get("bbox") for e in _telems if e.get("bbox")]
                except Exception:
                    table_bboxes = []
            if table_bboxes:
                tcs = _docsim_per_bbox_score(orig_pil, recon_pil, table_bboxes) or 0.0
        table_cell_score_buf.append(tcs)
        ghs = 0.0
        if BETA_SSIM > 0.0:
            ghs = _ssim_page_supplement(orig_pil, recon_pil)
        ssim_score_buf.append(ghs)
        ces = 0.0
        if BETA_CONTENT_ELEM > 0.0:
            ces = _content_element_clip_p10(orig_pil, recon_pil, page_dir)
        content_elem_score_buf.append(ces)
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

log.info("Done. %d pages -> %s (alpha=%.2f, window_k=%d, source=%s)",
         len(results), out_path, ALPHA_ENTROPY, WINDOW_K, SOURCE_ROUTING[variant])
