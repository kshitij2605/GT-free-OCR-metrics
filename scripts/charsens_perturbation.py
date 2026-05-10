"""
Character-level sensitivity perturbation study (Appendix C of the paper).

For a sample of OmniDocBench pages, hold bounding boxes and rendering pipeline
fixed and corrupt OCR text content at controlled character edit-distance
levels. Re-render each corrupted version and measure the per-element CLIP
cosine (10th-percentile aggregation, matching the leading P1_137 / P2_210
family methods) against the cached masked_original. A monotonically
decreasing score curve as corruption increases confirms that the
reference-free metric responds to character-level fidelity rather than
gross layout alone.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/charsens_perturbation.py

Outputs:
    results/charsens/results.json   (per-level mean / std / per-page scores)

The companion script paper/build_charsens_fig.py renders this JSON into
paper/figures/fig_charsens.pdf.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from reference_free_ocr_metric.reconstruction.html_parser import TextElement
from reference_free_ocr_metric.reconstruction.image_renderer import ImageRenderer

OCR_DIR = REPO / "data" / "omnidocbench" / "ocr_all"  # parquet-extracted layout uses variant-suffixed dirs
OUT_DIR = REPO / "results" / "charsens"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PAGES = 100
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.50]
BATCH_SIZE = 16
MIN_CROP = 8
ALPHANUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def perturb_text(text: str, frac: float, rng: random.Random) -> str:
    """Replace `frac` fraction of characters with random alphanumeric chars."""
    if frac == 0.0 or not text:
        return text
    chars = list(text)
    n = len(chars)
    n_perturb = max(1, int(round(n * frac)))
    indices = rng.sample(range(n), min(n_perturb, n))
    for i in indices:
        chars[i] = rng.choice(ALPHANUM)
    return "".join(chars)


def crop_bbox(pil: Image.Image, bbox) -> Image.Image | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = pil.size
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    if (x2 - x1) < MIN_CROP or (y2 - y1) < MIN_CROP:
        return None
    return pil.crop((x1, y1, x2, y2))


def main() -> None:
    if not OCR_DIR.exists():
        sys.exit(
            f"[charsens] OCR artifact directory not found: {OCR_DIR}\n"
            "  Run download_data.sh first to populate data/omnidocbench/."
        )

    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[charsens] device={device}", file=sys.stderr)
    clip_model, _, clip_pp = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(device).eval()

    @torch.no_grad()
    def clip_cosine_batch(orig_pils, recon_pils):
        def enc(pils):
            ts = torch.stack([clip_pp(p.convert("RGB")) for p in pils]).to(device)
            f = clip_model.encode_image(ts)
            return f / f.norm(dim=-1, keepdim=True)

        a = enc(orig_pils)
        b = enc(recon_pils)
        return (a * b).sum(-1).cpu().numpy().tolist()

    all_pages = sorted(p.name for p in OCR_DIR.iterdir() if p.is_dir())
    rng_pages = random.Random(SEED)
    rng_pages.shuffle(all_pages)
    sample = all_pages[:N_PAGES]
    print(
        f"[charsens] sampled {len(sample)} pages from {len(all_pages)} total",
        file=sys.stderr,
    )

    renderer = ImageRenderer()
    perturb_rng = random.Random(SEED)

    results = {f"{lv:.2f}": [] for lv in LEVELS}
    successful = 0
    t0 = time.time()

    for idx, pname in enumerate(sample, 1):
        page_dir = OCR_DIR / pname
        elem_path = page_dir / "ocr_elements.json"
        masked_path = page_dir / "masked_original.png"
        if not elem_path.exists() or not masked_path.exists():
            continue
        try:
            with elem_path.open("r") as f:
                elems = json.load(f)
            masked = Image.open(masked_path).convert("RGB")
            W, H = masked.size
        except Exception as e:
            print(f"[{idx}] {pname}: load fail: {e}", file=sys.stderr)
            continue

        bboxes, texts = [], []
        for e in elems:
            bb = e.get("bbox")
            t = e.get("text", "")
            if bb and t and t.strip():
                bboxes.append(bb)
                texts.append(t)
        if not bboxes:
            continue

        orig_crops_full = [crop_bbox(masked, bb) for bb in bboxes]

        page_p10 = {}
        for lv in LEVELS:
            perturbed_texts = [perturb_text(t, lv, perturb_rng) for t in texts]
            text_elems = [
                TextElement(text=t, bbox=tuple(bb), tag="text")
                for t, bb in zip(perturbed_texts, bboxes)
            ]
            try:
                recon = renderer.render_text_image(text_elems, W, H, background="white")
            except Exception as e:
                print(
                    f"[{idx}] {pname}: render fail at lv={lv}: {e}", file=sys.stderr
                )
                continue

            recon_crops_full = [crop_bbox(recon, bb) for bb in bboxes]
            pairs_o, pairs_r = [], []
            for oc, rc in zip(orig_crops_full, recon_crops_full):
                if oc is not None and rc is not None:
                    pairs_o.append(oc)
                    pairs_r.append(rc)
            if not pairs_o:
                continue
            sims = []
            for s in range(0, len(pairs_o), BATCH_SIZE):
                sims.extend(
                    clip_cosine_batch(
                        pairs_o[s : s + BATCH_SIZE], pairs_r[s : s + BATCH_SIZE]
                    )
                )
            page_p10[f"{lv:.2f}"] = float(np.percentile(sims, 10))

        if len(page_p10) == len(LEVELS):
            for k, v in page_p10.items():
                results[k].append(v)
            successful += 1
        if idx % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[{idx}/{len(sample)}] successful={successful} elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )

    summary = {
        "config": {
            "n_pages_target": N_PAGES,
            "n_pages_successful": successful,
            "levels": LEVELS,
            "seed": SEED,
            "metric": "per_element_clip_cosine_p10",
            "clip_model": "ViT-B-32 / laion2b_s34b_b79k",
            "perturbation": "random alphanumeric character substitution at fixed fraction",
        },
        "per_level_mean": {k: float(np.mean(v)) for k, v in results.items() if v},
        "per_level_std": {k: float(np.std(v)) for k, v in results.items() if v},
        "per_level_n": {k: len(v) for k, v in results.items()},
        "per_page_scores": results,
    }
    out_path = OUT_DIR / "results.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\n[charsens] done in {time.time() - t0:.1f}s, "
        f"{successful} pages × {len(LEVELS)} levels"
    )
    print(f"[charsens] results: {out_path}")
    for lv in LEVELS:
        k = f"{lv:.2f}"
        m = summary["per_level_mean"].get(k)
        n = summary["per_level_n"][k]
        if m is not None:
            print(f"  level {k}: mean P10 = {m:.4f}  (n={n})")


if __name__ == "__main__":
    main()
