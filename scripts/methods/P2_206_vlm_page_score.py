#!/usr/bin/env python3
"""D206 — VLM page-level comparison score (Qwen3.5-122B visual judge).

Replaces DocSim page-level similarity (prod_cc) with a VLM-based quality
score for text/all/all_no_mask variants. For each page, Qwen3.5-122B
receives the masked original and the OCR reconstruction and rates the
OCR fidelity 0.0-1.0.

  text:        VLM_score × (1-0.40-0.15) + 0.40*entropy + 0.15*ssim
  all:         VLM_score × (1-0.30-0.15-0.20-0.10) + 0.30*entropy + 0.15*table_cell + 0.20*ssim + 0.10*elem_p10
  all_no_mask: VLM_score × (1-0.40-0.10-0.10-0.10) + 0.40*entropy + 0.10*table_cell + 0.10*ssim + 0.10*elem_p10
  formula:     unchanged from D137 (CLIP+DINOv2 50/50 + DIQA + entropy)
  table:       unchanged from D137 (per-cell-MIN DocSim + IQ + SSIM)

All supplements (entropy, SSIM, elem_p10, table_cell) unchanged from D137.
VLM calls are concurrent (ThreadPoolExecutor with N workers) to minimize
wall time. Scores cached per page to avoid re-runs.

KEEP if spearman_mean > 0.4932 (D137).
"""

import base64
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import cv2
import httpx
import lpips as lpips_lib
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from openai import OpenAI
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P2_206_vlm_page_score"
SSIM_SIZE = 512
BATCH_SIZE = 16
DINO_SIZE = 224
VLM_MAX_WORKERS = 8
VLM_MAX_IMG_DIM = 1024  # resize images before sending to VLM (speed)

VLM_API_BASE = "http://your-vlm-server:9000/v1"  # replace with your endpoint
VLM_API_KEY = "tensorflow"
VLM_MODEL = "Qwen/Qwen3.5-122B-A10B"

TARGET_CLASSES_TEXT = {
    "text", "title", "plain_text", "header", "footer", "caption",
    "page_header", "page_footer", "page_number",
    "普通文字", "标题", "页眉", "页脚", "页码",
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

USE_VLM_SCORE = variant in ("text", "all", "all_no_mask")
USE_DISTS = variant == "all"
USE_PREPROC = variant in ("table", "all_no_mask")
USE_DOCSIM_PAGE = (not USE_VLM_SCORE) and variant in ("text", "all", "all_no_mask")
USE_DOCSIM_BBOX = variant == "table"
NEED_DOCSIM = USE_DOCSIM_PAGE or USE_DOCSIM_BBOX
NEED_DINO = variant in ("formula",)

ALPHA_ENTROPY = {
    "table": 0.0, "all": 0.3, "text": 0.4, "formula": 0.3, "all_no_mask": 0.4,
}[variant]
WINDOW_K = {
    "table": 10, "all": 3, "text": 3, "formula": 20, "all_no_mask": 3,
}[variant]
GAMMA_DIQA = {
    "table": 0.1, "all": 0.0, "text": 0.0, "formula": 0.5, "all_no_mask": 0.0,
}[variant]
DIQA_EPS = 1e-6
BETA_TABLE_CELL = {
    "table": 0.0, "all": 0.15, "text": 0.0, "formula": 0.0, "all_no_mask": 0.10,
}[variant]
BETA_SSIM = {
    "table": 0.15, "all": 0.20, "text": 0.15, "formula": 0.05, "all_no_mask": 0.10,
}[variant]
BETA_CONTENT_ELEM = {
    "table": 0.0, "text": 0.0, "formula": 0.0, "all": 0.10, "all_no_mask": 0.10,
}[variant]

BASE = Path(__file__).parent.parent.parent / "data/omnidocbench"
var_root = BASE / f"ocr_{variant}"
OUT_DIR = Path(__file__).parent.parent.parent / "results/method_runs" / f"ocr_{variant}" / METHOD_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)
log.info(
    "device=cpu variant=%s method=%s alpha=%.2f window_k=%d use_vlm=%s",
    variant, METHOD_ID, ALPHA_ENTROPY, WINDOW_K, USE_VLM_SCORE,
)

_preprocessor = ImagePreprocessor()

# ── DISTS ─────────────────────────────────────────────────────────────────────
if USE_DISTS:
    import pyiqa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _dists_metric = pyiqa.create_metric("dists", as_loss=False).to(device)
    _dists_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
else:
    _dists_metric = None
    _dists_transform = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ── LPIPS (always needed for multi_metric) ────────────────────────────────────
log.info("Loading LPIPS (alex)...")
_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

# ── CLIP (needed for formula, table fallback, and elem_p10) ───────────────────
log.info("Loading OpenCLIP ViT-B-32...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)
for p in _clip_model.parameters():
    p.requires_grad = False

# ── DINOv2 (formula) ──────────────────────────────────────────────────────────
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


# ── DocSim (table variant only) ───────────────────────────────────────────────
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


ADAPTER_DIR = Path(__file__).parent.parent.parent / "models/docsim_lora/lora_adapter_best"
HEAD_STATE_PATH = Path(__file__).parent.parent.parent / "models/docsim_lora/head_state_best.pt"
HEAD_CONFIG_PATH = Path(__file__).parent.parent.parent / "models/docsim_lora/config.json"
LOGPROBS_ROOT = Path(__file__).parent.parent.parent / "data/ocr_logprobs"
PER_BBOX_ROOT = Path(__file__).parent.parent.parent / "data/ocr_logprobs_per_bbox"
TOP_K = 5
_LOG_K_TOP = math.log(TOP_K)

_docsim_head = None
if NEED_DOCSIM:
    from peft import LoraConfig, get_peft_model  # noqa: E402
    with HEAD_CONFIG_PATH.open("r") as f:
        head_cfg = json.load(f)
    clip_dim = _clip_model.visual.output_dim
    dino_dim = 768
    in_dim = clip_dim + dino_dim
    _docsim_head = DocSimHead(in_dim, head_cfg["hidden_dim"], head_cfg["embed_dim"]).to(device)
    _docsim_head.load_state_dict(torch.load(HEAD_STATE_PATH, map_location=device))
    _docsim_head.eval()
    for p in _docsim_head.parameters():
        p.requires_grad = False
    lora_cfg = LoraConfig(r=head_cfg["lora_r"], lora_alpha=head_cfg["lora_alpha"], target_modules=["q", "v"])
    _clip_model = get_peft_model(_clip_model, lora_cfg)
    sd = torch.load(ADAPTER_DIR / "adapter_model.safetensors", map_location=device)
    _clip_model.load_state_dict(sd, strict=False)
    _clip_model.eval()
    if NEED_DINO and _dino_model is not None:
        lora_cfg_dino = LoraConfig(r=head_cfg["lora_r"], lora_alpha=head_cfg["lora_alpha"], target_modules=["q", "v"])
        _dino_model = get_peft_model(_dino_model, lora_cfg_dino)
        sd_dino = torch.load(ADAPTER_DIR / "adapter_model_dino.safetensors", map_location=device)
        _dino_model.load_state_dict(sd_dino, strict=False)
        _dino_model.eval()

# ── SOURCE ROUTING ────────────────────────────────────────────────────────────
SOURCE_ROUTING = {
    "text":        "class_filtered_per_bbox_max_entropy_min",
    "formula":     "window_min_mean_shannon",
    "table":       "window_min_mean_shannon",
    "all":         "token_class_filtered_window_min",
    "all_no_mask": "token_class_filtered_window_min",
}

# ── VLM client (text/all/all_no_mask only) ────────────────────────────────────
_vlm_client = None
if USE_VLM_SCORE:
    _vlm_client = OpenAI(
        base_url=VLM_API_BASE,
        api_key=VLM_API_KEY,
        http_client=httpx.Client(trust_env=False, timeout=60.0),
    )
    log.info("VLM client ready: %s workers, max_img_dim=%d", VLM_MAX_WORKERS, VLM_MAX_IMG_DIM)


def _encode_pil_b64(pil: Image.Image) -> str:
    """Resize and encode PIL image to base64 JPEG for VLM input."""
    w, h = pil.size
    if max(w, h) > VLM_MAX_IMG_DIM:
        scale = VLM_MAX_IMG_DIM / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_VLM_PROMPT = (
    "Image 1 is an original document page. "
    "Image 2 is an OCR text reconstruction of that page. "
    "Rate how faithfully Image 2 captures the text content from Image 1. "
    "Consider: text completeness, character accuracy, reading order. "
    "Respond with ONLY a single decimal number from 0.0 (no match) to 1.0 (perfect)."
)


def _vlm_score_single(page_name: str, orig_b64: str, recon_b64: str) -> float:
    """Call VLM API and return 0-1 quality score. Returns 0.5 on failure."""
    try:
        resp = _vlm_client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{recon_b64}"}},
                    {"type": "text", "text": _VLM_PROMPT},
                ],
            }],
            max_tokens=16,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            v = float(m.group(1))
            return float(np.clip(v / 10.0 if v > 1.5 else v, 0.0, 1.0))
    except Exception as e:
        log.warning("VLM call failed for %s: %s", page_name, e)
    return 0.5


def _compute_vlm_scores(page_dirs_list) -> dict:
    """Concurrently compute VLM scores for all pages. Returns {page_name: score}."""
    cache_path = OUT_DIR / "vlm_score_cache.json"
    cached = {}
    if cache_path.exists():
        try:
            with cache_path.open() as f:
                cached = json.load(f)
            log.info("Loaded %d cached VLM scores", len(cached))
        except Exception:
            pass

    to_compute = [(pd.name, pd / "masked_original.png", pd / "reconstructed.png")
                  for pd in page_dirs_list
                  if pd.name not in cached
                  and (pd / "masked_original.png").exists()
                  and (pd / "reconstructed.png").exists()]

    if not to_compute:
        log.info("All VLM scores cached (%d pages)", len(cached))
        return cached

    log.info("Computing VLM scores for %d pages (workers=%d)...", len(to_compute), VLM_MAX_WORKERS)
    t0 = time.time()

    def _worker(args):
        name, orig_p, recon_p = args
        orig_b64 = _encode_pil_b64(Image.open(orig_p).convert("RGB"))
        recon_b64 = _encode_pil_b64(Image.open(recon_p).convert("RGB"))
        return name, _vlm_score_single(name, orig_b64, recon_b64)

    scores = dict(cached)
    done = 0
    with ThreadPoolExecutor(max_workers=VLM_MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, args): args[0] for args in to_compute}
        for fut in as_completed(futures):
            name, score = fut.result()
            scores[name] = score
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                log.info("[VLM] %d/%d done (%.1fs elapsed)", done, len(to_compute), elapsed)

    # Cache results
    try:
        with cache_path.open("w") as f:
            json.dump(scores, f)
    except Exception as e:
        log.warning("VLM cache save failed: %s", e)

    log.info("VLM scoring done: %d pages in %.1fs", len(to_compute), time.time() - t0)
    return scores


# ── Similarity helpers ────────────────────────────────────────────────────────
_lpips_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _ssim_mse(orig_pil, recon_pil):
    og = np.array(orig_pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float64) / 255.0
    rg = np.array(recon_pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float64) / 255.0
    ssim_val = float(structural_similarity(og, rg, data_range=1.0))
    mse_val = float(np.mean((og - rg) ** 2))
    return ssim_val, mse_val


def _lpips_score(orig_pil, recon_pil):
    o = _lpips_transform(orig_pil.convert("RGB")).unsqueeze(0).to(device)
    r = _lpips_transform(recon_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(_lpips_fn(o, r).item())


def _dists_score(orig_pil, recon_pil):
    o = _dists_transform(orig_pil.convert("RGB")).unsqueeze(0).to(device)
    r = _dists_transform(recon_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(np.clip(_dists_metric(o, r).item(), 0.0, 1.0))


def _clip_cosine_batch(orig_pils, recon_pils):
    def _preproc(pil):
        return _preprocessor.preprocess(pil) if USE_PREPROC else pil
    def _enc(pils):
        batch = torch.stack([_clip_preprocess(_preproc(p).convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_image(batch)
        return F.normalize(feats.float(), dim=-1)
    return (_enc(orig_pils) * _enc(recon_pils)).sum(dim=-1).cpu().tolist()


def _clip_cosine_batch_plain(orig_pils, recon_pils):
    def _enc(pils):
        batch = torch.stack([_clip_preprocess(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_image(batch)
        return F.normalize(feats.float(), dim=-1)
    return (_enc(orig_pils) * _enc(recon_pils)).sum(dim=-1).cpu().tolist()


def _dinov2_cosine_batch(orig_pils, recon_pils):
    def _enc(pils):
        batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            feats = _dino_model(batch)
        return F.normalize(feats.float(), dim=-1)
    return (_enc(orig_pils) * _enc(recon_pils)).sum(dim=-1).cpu().tolist()


def _docsim_cosine_batch(orig_pils, recon_pils):
    def _enc(pils):
        def _preproc(p):
            return _preprocessor.preprocess(p) if USE_PREPROC else p
        clip_batch = torch.stack([_clip_preprocess(_preproc(p).convert("RGB")) for p in pils]).to(device)
        dino_batch = torch.stack([_dino_transform(_preproc(p).convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            clip_feats = _clip_model.encode_image(clip_batch)
            dino_feats = _dino_model(dino_batch)
        combined = torch.cat([clip_feats.float(), dino_feats.float()], dim=-1)
        return _docsim_head(combined)
    return (_enc(orig_pils) * _enc(recon_pils)).sum(dim=-1).cpu().tolist()


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


def _diqa_similarity(orig_pil, recon_pil):
    def _iq(pil):
        g = cv2.cvtColor(np.array(pil.resize((512, 512))), cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        focus = float(lap.var())
        grad = float(cv2.Sobel(g, cv2.CV_32F, 1, 0).var() + cv2.Sobel(g, cv2.CV_32F, 0, 1).var())
        contrast = float(g.std())
        edges = float((cv2.Canny((g * 255).astype(np.uint8), 50, 150).sum()) / (512 * 512 * 255.0))
        brightness = float(g.mean() / 255.0)
        return np.array([focus, grad, contrast, edges, brightness], dtype=np.float32)
    f_o, f_r = _iq(orig_pil), _iq(recon_pil)
    denom = np.maximum(np.abs(f_o) + DIQA_EPS, 1.0)
    rel_diff = np.abs(f_o - f_r) / denom
    return float(np.exp(-float(rel_diff.mean())))


def _ssim_page_supplement(orig_pil, recon_pil):
    og = np.array(orig_pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float64) / 255.0
    rg = np.array(recon_pil.convert("L").resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float64) / 255.0
    return float(structural_similarity(og, rg, data_range=1.0))


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


def _content_element_clip_p10(orig_pil, recon_pil, page_dir):
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


# ── Entropy helpers (copied from D137) ────────────────────────────────────────
def _window_min_confidence(page_name):
    lp_path = LOGPROBS_ROOT / f"{page_name}.json"
    if not lp_path.exists():
        return 0.5
    try:
        with lp_path.open() as f:
            data = json.load(f)
        tokens = data.get("tokens") or data.get("logprobs") or []
        if not tokens:
            return 0.5
        confs = []
        for t in tokens:
            lp = t.get("logprob") if isinstance(t, dict) else t
            if lp is not None:
                top_lps = t.get("top_logprobs", {}) if isinstance(t, dict) else {}
                if top_lps:
                    lps = sorted(top_lps.values(), reverse=True)[:TOP_K]
                    lps = [lp for lp in lps if lp > -1e9]
                    if len(lps) >= 2:
                        max_lp = max(lps)
                        log_sum = max_lp + math.log(sum(math.exp(l - max_lp) for l in lps))
                        probs = [math.exp(l - log_sum) for l in lps]
                        H = -sum(p * math.log(p + 1e-12) for p in probs)
                        conf = 1.0 - H / _LOG_K_TOP
                        confs.append(max(0.0, min(1.0, conf)))
        if not confs:
            return 0.5
        if WINDOW_K <= 1:
            return float(min(confs))
        window_mins = [min(confs[i:i + WINDOW_K]) for i in range(len(confs) - WINDOW_K + 1)]
        return float(min(window_mins)) if window_mins else float(min(confs))
    except Exception:
        return 0.5


def _class_filtered_per_bbox_max_entropy_min_text(page_name):
    per_bbox_path = PER_BBOX_ROOT / f"{page_name}.json"
    if not per_bbox_path.exists():
        return 0.5
    try:
        with per_bbox_path.open() as f:
            data = json.load(f)
        bboxes = data if isinstance(data, list) else data.get("bboxes", [])
        scores = []
        for b in bboxes:
            cls = b.get("class", "")
            if cls not in TARGET_CLASSES_TEXT:
                continue
            tokens = b.get("tokens", [])
            if not tokens:
                continue
            max_ent = 0.0
            for t in tokens:
                top_lps = t.get("top_logprobs", {})
                if not top_lps:
                    continue
                lps = sorted(top_lps.values(), reverse=True)[:TOP_K]
                lps = [lp for lp in lps if lp > -1e9]
                if len(lps) < 2:
                    continue
                max_lp = max(lps)
                log_sum = max_lp + math.log(sum(math.exp(l - max_lp) for l in lps))
                probs = [math.exp(l - log_sum) for l in lps]
                H = -sum(p * math.log(p + 1e-12) for p in probs)
                conf = 1.0 - H / _LOG_K_TOP
                max_ent = max(max_ent, 1.0 - max(0.0, min(1.0, conf)))
            if max_ent > 0:
                scores.append(1.0 - max_ent)
        return float(min(scores)) if scores else 0.5
    except Exception:
        return 0.5


def _token_class_filtered_window_min_confidence(page_name):
    per_bbox_path = PER_BBOX_ROOT / f"{page_name}.json"
    if not per_bbox_path.exists():
        return _window_min_confidence(page_name)
    try:
        with per_bbox_path.open() as f:
            data = json.load(f)
        bboxes = data if isinstance(data, list) else data.get("bboxes", [])
        all_confs = []
        for b in bboxes:
            cls = b.get("class", "")
            if cls not in TARGET_CLASSES_CONTENT:
                continue
            tokens = b.get("tokens", [])
            for t in tokens:
                top_lps = t.get("top_logprobs", {})
                if not top_lps:
                    continue
                lps = sorted(top_lps.values(), reverse=True)[:TOP_K]
                lps = [lp for lp in lps if lp > -1e9]
                if len(lps) < 2:
                    continue
                max_lp = max(lps)
                log_sum = max_lp + math.log(sum(math.exp(l - max_lp) for l in lps))
                probs = [math.exp(l - log_sum) for l in lps]
                H = -sum(p * math.log(p + 1e-12) for p in probs)
                conf = 1.0 - H / _LOG_K_TOP
                all_confs.append(max(0.0, min(1.0, conf)))
        if not all_confs:
            return _window_min_confidence(page_name)
        if WINDOW_K <= 1:
            return float(min(all_confs))
        window_mins = [min(all_confs[i:i + WINDOW_K]) for i in range(len(all_confs) - WINDOW_K + 1)]
        return float(min(window_mins)) if window_mins else float(min(all_confs))
    except Exception:
        return _window_min_confidence(page_name)


def _fuse(prod_cc, page_name, diqa_sim=0.0, table_cell_score=0.0, ssim_score=0.0, content_elem_score=0.0):
    src = SOURCE_ROUTING[variant]
    if src == "class_filtered_per_bbox_max_entropy_min":
        conf = _class_filtered_per_bbox_max_entropy_min_text(page_name)
    elif src == "token_class_filtered_window_min":
        conf = _token_class_filtered_window_min_confidence(page_name)
    else:
        conf = _window_min_confidence(page_name)
    if GAMMA_DIQA > 0.0 and BETA_SSIM > 0.0:
        return (1.0 - ALPHA_ENTROPY - GAMMA_DIQA - BETA_SSIM) * prod_cc + ALPHA_ENTROPY * conf + GAMMA_DIQA * diqa_sim + BETA_SSIM * ssim_score
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

# Pre-compute VLM scores concurrently
vlm_scores = {}
if USE_VLM_SCORE:
    vlm_scores = _compute_vlm_scores(page_dirs)

results = []
orig_pils_buf, recon_pils_buf, meta_buf, name_buf, table_cell_score_buf, ssim_score_buf, content_elem_score_buf = [], [], [], [], [], [], []


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf, name_buf, table_cell_score_buf, ssim_score_buf, content_elem_score_buf
    if not orig_pils_buf:
        return
    if USE_VLM_SCORE:
        for meta, name, tcs, ghs, ces in zip(meta_buf, name_buf, table_cell_score_buf, ssim_score_buf, content_elem_score_buf):
            vlm_s = vlm_scores.get(name, 0.5)
            meta["clip_compare"] = {"clip_cosine": float(_fuse(vlm_s, name, table_cell_score=tcs, ssim_score=ghs, content_elem_score=ces))}
            results.append(meta)
    elif USE_DOCSIM_PAGE:
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
        # formula: 50/50 CLIP+DINOv2 + IQ + SSIM
        clip_sims = _clip_cosine_batch(orig_pils_buf, recon_pils_buf)
        dino_sims = _dinov2_cosine_batch(orig_pils_buf, recon_pils_buf)
        for meta, dino_cos, clip_cos, name, orig_p, recon_p, ghs in zip(
            meta_buf, dino_sims, clip_sims, name_buf, orig_pils_buf, recon_pils_buf, ssim_score_buf
        ):
            avg_cos = 0.5 * dino_cos + 0.5 * clip_cos
            diqa_sim = _diqa_similarity(orig_p, recon_p) if GAMMA_DIQA > 0.0 else 0.0
            meta["clip_compare"] = {"clip_cosine": float(_fuse(float(avg_cos), name, diqa_sim, ssim_score=ghs))}
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
        composite = 0.3 * ssim_val + 0.2 * max(0.0, 1.0 - mse_val) + 0.2 * max(0.0, 1.0 - lpips_val) + 0.3 * max(0.0, 1.0 - dists_val)
        multi_metric = {"ssim": ssim_val, "mse": mse_val, "lpips": lpips_val, "dists": dists_val, "composite": composite}
    else:
        composite = 0.4 * ssim_val + 0.3 * max(0.0, 1.0 - mse_val) + 0.3 * max(0.0, 1.0 - lpips_val)
        multi_metric = {"ssim": ssim_val, "mse": mse_val, "lpips": lpips_val, "composite": composite}

    meta = {
        "image": page_dir.name,
        "text_elements": 0, "image_regions": 0, "table_regions": 0,
        "text_length": 0, "plain_text_length": 0,
        "multi_metric": multi_metric,
        "lm_perplexity": {"ngram_score": 0.0, "transformer_score": 0.0, "perplexity": 0.0, "composite": 0.0},
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
        ghs = _ssim_page_supplement(orig_pil, recon_pil) if BETA_SSIM > 0.0 else 0.0
        ssim_score_buf.append(ghs)
        ces = _content_element_clip_p10(orig_pil, recon_pil, page_dir) if BETA_CONTENT_ELEM > 0.0 else 0.0
        content_elem_score_buf.append(ces)
        orig_pils_buf.append(orig_pil)
        recon_pils_buf.append(recon_pil)
        meta_buf.append(meta)
        name_buf.append(page_dir.name)

    if len(orig_pils_buf) >= BATCH_SIZE:
        _flush_batch()

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        log.info("[%d/%d] %.1fs elapsed (%.2fs/page)", i + 1, len(page_dirs), elapsed, elapsed / (i + 1))

_flush_batch()

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info("Done. %d pages -> %s (alpha=%.2f, source=%s, use_vlm=%s)",
         len(results), out_path, ALPHA_ENTROPY, SOURCE_ROUTING[variant], USE_VLM_SCORE)
