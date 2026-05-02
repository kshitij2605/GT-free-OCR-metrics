#!/usr/bin/env python3
"""P2_104_recognition_feature_distance: D104 — TrOCR recognition-feature distance.

Hypothesis: Distance in TrOCR encoder feature space measures text-character fidelity
better than CLIP/DINOv2 cosine, because TrOCR was trained to recognise characters
and its internal representations are font-invariant and character-discriminative.
A low feature-distance between GT-crop and recon-crop means the recogniser "sees"
the same characters — independent of font, colour, or layout.

Mechanism (variant=text, variant=all only):
  1. Read per-page text-element bboxes from ocr_elements.json.
  2. Crop each element from masked_original.png and reconstructed.png.
  3. Run TrOCR encoder forward pass on both sets of crops (batch=32).
     Use mean of last_hidden_state as the per-crop feature vector (NOT decoded text).
  4. Per-element cosine similarity between GT-crop and recon-crop features.
  5. Per-page score = mean(per-element cosines), or 0.5 if no text elements.
  6. Write score to BOTH multi_composite.composite AND clip_compare.clip_cosine.

Passthrough for formula, table, all_no_mask:
  Reads H11.2 baseline (P1_120c3b_all_no_mask_baseline_preproc) results.json
  and writes verbatim — these variants' ceilings are protected.

Model: microsoft/trocr-small-printed (encoder ViT-S/16, 384-dim hidden, ~239 MB).
Encoder-only forward pass — NO text decoding — 10-100x faster than full TrOCR generation.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <variant>", file=sys.stderr)
    sys.exit(1)

variant = sys.argv[1]
if variant not in {"all", "text", "formula", "table", "all_no_mask"}:
    print(f"Unknown variant: {variant}", file=sys.stderr)
    sys.exit(1)

METHOD_ID = "P2_104_recognition_feature_distance"
BASELINE_METHOD_ID = "P1_120c3b_all_no_mask_baseline_preproc"

CROP_BATCH_SIZE = 32
MIN_CROP_PX = 4          # skip degenerate bboxes
NEUTRAL_SCORE = 0.5      # for pages with no text elements

BASE = Path(__file__).parent.parent.parent
DATA_BASE = BASE / "data" / "omnidocbench"
var_root = DATA_BASE / f"ocr_{variant}"
OUT_DIR = BASE / "results" / "method_runs" / f"ocr_{variant}" / METHOD_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(f"{variant}_{METHOD_ID}")

log.info("variant=%s method=%s", variant, METHOD_ID)

# ── Non-active variants: passthrough to H11.2 baseline ───────────────────────
ACTIVE_VARIANTS = {"text", "all"}

if variant not in ACTIVE_VARIANTS:
    baseline_results_path = (
        BASE / "results" / "method_runs" / f"ocr_{variant}" / BASELINE_METHOD_ID / "results.json"
    )
    if not baseline_results_path.exists():
        log.error(
            "Baseline results not found for variant=%s at %s — aborting.",
            variant, baseline_results_path,
        )
        sys.exit(1)

    log.info("Passthrough: reading H11.2 baseline from %s", baseline_results_path)
    with open(baseline_results_path) as f:
        results = json.load(f)

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Passthrough done. %d pages -> %s", len(results), out_path)
    sys.exit(0)

# ── Active variant path (text, all): TrOCR recognition-feature distance ───────

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("device=%s", device)

log.info("Loading TrOCR small printed (encoder-only)...")
_proc = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-small-printed", use_fast=False
)
_full_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
_encoder = _full_model.encoder.eval().to(device)
del _full_model  # free decoder weights
log.info("TrOCR encoder loaded, hidden_size=%d", _encoder.config.hidden_size)


def _extract_features(crops: list) -> torch.Tensor:
    """Run TrOCR encoder on a list of PIL crops, return (N, D) normalised features."""
    results_f = []
    for i in range(0, len(crops), CROP_BATCH_SIZE):
        batch_crops = crops[i : i + CROP_BATCH_SIZE]
        inputs = _proc(images=batch_crops, return_tensors="pt").to(device)
        with torch.no_grad():
            out = _encoder(**inputs)
        # mean of last_hidden_state: (B, seq_len, D) -> (B, D)
        feats = out.last_hidden_state.mean(dim=1)
        feats = F.normalize(feats.float(), dim=-1)
        results_f.append(feats.cpu())
    return torch.cat(results_f, dim=0)  # (N, D)


def _recognition_score(page_dir: Path, orig_img: Image.Image, recon_img: Image.Image) -> float:
    """Compute per-page TrOCR recognition-feature cosine score."""
    elements_path = page_dir / "ocr_elements.json"
    if not elements_path.exists():
        log.warning("No ocr_elements.json for %s — returning neutral", page_dir.name)
        return NEUTRAL_SCORE

    with open(elements_path) as f:
        elements = json.load(f)

    if not elements:
        return NEUTRAL_SCORE

    w, h = orig_img.size

    orig_crops, recon_crops = [], []
    for el in elements:
        bbox = el.get("bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            continue
        orig_crops.append(orig_img.crop((x1, y1, x2, y2)))
        recon_crops.append(recon_img.crop((x1, y1, x2, y2)))

    if not orig_crops:
        return NEUTRAL_SCORE

    orig_f = _extract_features(orig_crops)   # (N, D)
    recon_f = _extract_features(recon_crops)  # (N, D)

    cosines = (orig_f * recon_f).sum(dim=-1).numpy()  # (N,)
    return float(np.mean(cosines))


# ── Main processing loop ───────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages for variant=%s", len(page_dirs), variant)

results = []
t0 = time.time()

for i, page_dir in enumerate(page_dirs):
    orig_path = page_dir / "masked_original.png"
    recon_path = page_dir / "reconstructed.png"
    if not orig_path.exists() or not recon_path.exists():
        log.warning("Missing images for %s — skipping", page_dir.name)
        continue

    orig_img = Image.open(orig_path).convert("RGB")
    recon_img = Image.open(recon_path).convert("RGB")

    score = _recognition_score(page_dir, orig_img, recon_img)

    # Write score to BOTH multi_composite and clip_cosine channels so
    # best_over_metrics(mc, cc) in the eval pipeline evaluates both.
    result = {
        "image": page_dir.name,
        "text_elements": 0,
        "image_regions": 0,
        "table_regions": 0,
        "text_length": 0,
        "plain_text_length": 0,
        "multi_metric": {
            "ssim": score,
            "mse": 1.0 - score,
            "lpips": 1.0 - score,
            "composite": score,
        },
        "lm_perplexity": {
            "ngram_score": 0.0,
            "transformer_score": 0.0,
            "perplexity": 0.0,
            "composite": 0.0,
        },
        "clip_compare": {
            "clip_cosine": score,
        },
    }
    results.append(result)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        log.info(
            "[%d/%d] %.1fs elapsed (%.2fs/page), last score=%.4f",
            i + 1, len(page_dirs), elapsed, elapsed / (i + 1), score,
        )

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t0
log.info("Done. %d pages -> %s (%.1fs total)", len(results), out_path, elapsed)
