#!/usr/bin/env python3
"""P2_060 DocSim — DreamSim recipe applied to documents.

clip_cosine = cosine_similarity(
    DocSimHead(concat(OpenCLIP(GT), DINOv2(GT))),
    DocSimHead(concat(OpenCLIP(recon), DINOv2(recon)))
)

DocSimHead = LoRA-fine-tuned MLP trained on 20,280 edit-distance triplets.
Trained adapter at models/docsim_lora/lora_adapter_best/.

multi_composite (SSIM/MSE/LPIPS) is UNCHANGED vs baseline — only clip_cosine
is replaced. Output JSON shape unchanged: clip_compare={"clip_cosine": <docsim>}.
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
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, "/home/mac/test/r1-p2/src")
from reference_free_ocr_metric.reconstruction.image_preprocessor import ImagePreprocessor

METHOD_ID = "P2_060_docsim_dreamsim_recipe_with_document_embeddings_208"
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
log.info("device=%s variant=%s method=%s", device, variant, METHOD_ID)

_preprocessor = ImagePreprocessor()

# ── DINOv2 setup ──────────────────────────────────────────────────────────────
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

# ── OpenCLIP setup ────────────────────────────────────────────────────────────
log.info("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model = _clip_model.eval().to(device)
for p in _clip_model.parameters():
    p.requires_grad = False

# ── LPIPS setup (for multi_composite) ─────────────────────────────────────────
log.info("Loading LPIPS (alex)...")
_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

# ── DocSim head with LoRA adapter ─────────────────────────────────────────────


class DocSimHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


log.info("Loading DocSim head config from %s", HEAD_CONFIG_PATH)
with HEAD_CONFIG_PATH.open("r") as f:
    head_cfg = json.load(f)

clip_dim = _clip_model.visual.output_dim  # 512
dino_dim = 768

_head = DocSimHead(
    in_dim=clip_dim + dino_dim,
    hidden_dim=head_cfg["hidden_dim"],
    embed_dim=head_cfg["embed_dim"],
)

# Wrap with PEFT LoRA — must match training config
from peft import LoraConfig, get_peft_model, PeftModel

lora_cfg = LoraConfig(
    r=head_cfg["lora_r"],
    lora_alpha=head_cfg["lora_alpha"],
    lora_dropout=head_cfg["lora_dropout"],
    target_modules=["0", "2"],
    bias="none",
)
_head.proj = get_peft_model(_head.proj, lora_cfg)

log.info("Loading head state from %s", HEAD_STATE_PATH)
state = torch.load(HEAD_STATE_PATH, map_location=device, weights_only=True)
_head.load_state_dict(state)

# Load LoRA adapter weights ON TOP — load_state_dict already covers them since
# the head_state file includes the LoRA params, but be defensive.
log.info("Loading LoRA adapter from %s", ADAPTER_DIR)
# PEFT's adapter saves weights separately; load_state_dict above already
# restored everything (LoRA weights live in head.proj.base_model.* keys).
_head = _head.eval().to(device)
for p in _head.parameters():
    p.requires_grad = False

n_total = sum(p.numel() for p in _head.parameters())
log.info("DocSim head loaded. params=%d", n_total)


# ── Preprocessing helpers (same as P1_084c baseline) ──────────────────────────


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
    def _to_t(pil):
        arr = np.array(pil.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        d = _lpips_fn(_to_t(orig_pil), _to_t(recon_pil))
    return float(d.item())


# ── DocSim cosine (batched) ───────────────────────────────────────────────────


def _docsim_cosine_batch(orig_pils: list, recon_pils: list) -> list:
    def _encode(pils: list) -> torch.Tensor:
        # CLIP branch
        clip_batch = torch.stack([_clip_preprocess(p.convert("RGB")) for p in pils]).to(device)
        # DINO branch
        dino_batch = torch.stack([_dino_transform(p.convert("RGB")) for p in pils]).to(device)
        with torch.no_grad():
            f_clip = _clip_model.encode_image(clip_batch).float()
            f_dino = _dino_model(dino_batch).float()
            f_clip = F.normalize(f_clip, dim=-1)
            f_dino = F.normalize(f_dino, dim=-1)
            x = torch.cat([f_clip, f_dino], dim=-1)  # (B, 1280)
            return _head(x)  # (B, 256), L2-normed

    a = _encode(orig_pils)
    r = _encode(recon_pils)
    return (a * r).sum(dim=-1).cpu().tolist()


# ── Main processing loop ──────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages", len(page_dirs))

results = []
orig_pils_buf, recon_pils_buf, meta_buf = [], [], []


def _flush_batch():
    global orig_pils_buf, recon_pils_buf, meta_buf
    if not orig_pils_buf:
        return
    docsim_sims = _docsim_cosine_batch(orig_pils_buf, recon_pils_buf)
    for meta, dsim in zip(meta_buf, docsim_sims):
        meta["clip_compare"] = {"clip_cosine": float(dsim)}
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
    composite = 0.4 * ssim_val + 0.3 * max(0.0, 1.0 - mse_val) + 0.3 * max(0.0, 1.0 - lpips_val)

    meta = {
        "image": page_dir.name,
        "text_elements": 0,
        "image_regions": 0,
        "table_regions": 0,
        "text_length": 0,
        "plain_text_length": 0,
        "multi_metric": {
            "ssim": ssim_val,
            "mse": mse_val,
            "lpips": lpips_val,
            "composite": composite,
        },
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
            log.info("Processed %d/%d pages (%.1fs elapsed, %.2f pages/s)",
                     i + 1, len(page_dirs), elapsed, (i + 1) / elapsed)

_flush_batch()

elapsed = time.time() - t0
log.info("Completed %d pages in %.1fs (%.2f pages/s)",
         len(results), elapsed, len(results) / elapsed)

out_path = OUT_DIR / "results.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log.info("Wrote %s (%d entries)", out_path, len(results))
