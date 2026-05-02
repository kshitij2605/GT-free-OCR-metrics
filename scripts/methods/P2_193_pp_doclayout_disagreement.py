#!/usr/bin/env python3
"""P2_193_pp_doclayout_disagreement: D193 — DocLayout-YOLO layout-profile disagreement signal.

Hypothesis: Comparing DocLayout-YOLO layout class distributions between GT and
reconstruction images produces a reference-free quality signal for variant=all.
Pages where reconstruction preserves non-text layout structure (similar fraction of
formula/table/figure areas) should correlate with higher OCR quality (lower edit
distance). Signal: 1 - |nontext_area_fraction_GT - nontext_area_fraction_recon|.

Scope: variant=all only uses the new mechanism.
All other variants (text, formula, table, all_no_mask) delegate to the H11.2 baseline
by reading from P1_120c3b_all_no_mask_baseline_preproc/results.json directly.
This way only variant=all is probed; other variants are unchanged.

DocLayout-YOLO class labels:
  0: title, 1: plain text, 2: abandon, 3: figure, 4: figure_caption,
  5: table, 6: table_caption, 7: table_footnote, 8: isolate_formula, 9: formula_caption

NON_TEXT_CLASSES = {figure, table, isolate_formula, figure_caption, table_caption,
                    table_footnote, formula_caption}

The score (multi_composite) is: 1 - |f_gt - f_recon|
where f_x = sum(bbox_area of NON_TEXT_CLASSES) / image_area for image x.
clip_cosine is set equal to multi_composite (same value duplicated so both channels
are populated and the eval pipeline's best_over_metrics picks whichever ranks higher).
"""

import json
import logging
import sys
import time
from pathlib import Path

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <variant>", file=sys.stderr)
    sys.exit(1)

variant = sys.argv[1]
if variant not in {"all", "text", "formula", "table", "all_no_mask"}:
    print(f"Unknown variant: {variant}", file=sys.stderr)
    sys.exit(1)

METHOD_ID = "P2_193_pp_doclayout_disagreement"
BASELINE_METHOD_ID = "P1_120c3b_all_no_mask_baseline_preproc"

BASE = Path("/home/mac/test/r1-p2")
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

# ── Non-all variants: passthrough to H11.2 baseline ───────────────────────────
if variant != "all":
    baseline_results_path = (
        BASE / "results" / "method_runs" / f"ocr_{variant}" / BASELINE_METHOD_ID / "results.json"
    )
    if not baseline_results_path.exists():
        log.error(
            "Baseline results not found for variant=%s at %s — aborting passthrough.",
            variant, baseline_results_path,
        )
        sys.exit(1)

    log.info("Passthrough: reading baseline results from %s", baseline_results_path)
    with open(baseline_results_path) as f:
        results = json.load(f)

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Done (passthrough). %d pages -> %s", len(results), out_path)
    sys.exit(0)

# ── variant=all: DocLayout-YOLO layout-profile disagreement ───────────────────
import torch
from doclayout_yolo import YOLOv10

MODEL_PATH = (
    "/home/mac/.cache/huggingface/hub/models--juliozhao--DocLayout-YOLO-DocStructBench"
    "/snapshots/8c3299a30b8ff29a1503c4431b035b93220f7b11"
    "/doclayout_yolo_docstructbench_imgsz1024.pt"
)

# Non-text structure classes — disagreement between Qwen text and these classes is the signal
NON_TEXT_CLASSES = {
    "figure",
    "table",
    "isolate_formula",
    "figure_caption",
    "table_caption",
    "table_footnote",
    "formula_caption",
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
log.info("Loading DocLayout-YOLO from %s ...", MODEL_PATH)
model = YOLOv10(MODEL_PATH)
log.info("Model loaded. Classes: %s", model.names)


def compute_nontext_area_fraction(img_path: Path) -> float:
    """Run DocLayout-YOLO on img and return fraction of image area classified as non-text."""
    results = model.predict(
        source=str(img_path),
        imgsz=1024,
        conf=0.25,
        device=device,
        verbose=False,
    )
    r = results[0]
    h, w = r.orig_shape
    total_area = float(h * w)
    if total_area == 0 or r.boxes is None:
        return 0.0

    nontext_area = 0.0
    for box, cls_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        cls_name = model.names[int(cls_id)]
        if cls_name in NON_TEXT_CLASSES:
            bw = max(0.0, float(box[2] - box[0]))
            bh = max(0.0, float(box[3] - box[1]))
            nontext_area += bw * bh

    return min(1.0, nontext_area / total_area)


def layout_disagreement_score(f_gt: float, f_recon: float) -> float:
    """Score = 1 - |nontext_fraction_GT - nontext_fraction_recon|, clipped to [0,1]."""
    return float(max(0.0, 1.0 - abs(f_gt - f_recon)))


# ── Main processing loop ───────────────────────────────────────────────────────
page_dirs = sorted(d for d in var_root.iterdir() if d.is_dir())
log.info("Found %d pages in %s", len(page_dirs), var_root)

results = []
t0 = time.time()

for i, page_dir in enumerate(page_dirs):
    orig_path = page_dir / "masked_original.png"
    recon_path = page_dir / "reconstructed.png"
    if not orig_path.exists() or not recon_path.exists():
        log.warning("Missing images in %s — skipping", page_dir.name)
        continue

    try:
        f_gt = compute_nontext_area_fraction(orig_path)
        f_recon = compute_nontext_area_fraction(recon_path)
    except Exception as exc:
        log.warning("Error on page %s: %s — skipping", page_dir.name, exc)
        continue

    score = layout_disagreement_score(f_gt, f_recon)

    # Both channels populated with the same value so best_over_metrics can pick either
    result = {
        "image": page_dir.name,
        "text_elements": 0,
        "image_regions": 0,
        "table_regions": 0,
        "text_length": 0,
        "plain_text_length": 0,
        "multi_metric": {
            "composite": score,
            "nontext_fraction_gt": f_gt,
            "nontext_fraction_recon": f_recon,
        },
        "lm_perplexity": {
            "ngram_score": 0.0,
            "transformer_score": 0.0,
            "perplexity": 0.0,
            "composite": 0.0,
        },
        "clip_compare": {
            "clip_cosine": score,  # duplicated so both mc and cc channels are populated
        },
    }
    results.append(result)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        log.info(
            "[%d/%d] %.1fs elapsed (%.2fs/page)",
            i + 1, len(page_dirs), elapsed, elapsed / (i + 1),
        )

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t0
log.info("Done. %d pages -> %s (%.1fs total)", len(results), out_path, elapsed)
