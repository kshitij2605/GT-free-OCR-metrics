#!/usr/bin/env python3
"""Re-run OCR with logprobs enabled, writing to data/ocr_logprobs/.

Idempotent: skips pages whose ocr_logprobs.json already exists.
Writes nothing into existing data/omnidocbench/ocr*/ folders.

Outputs per page:
  data/ocr_logprobs/<page_name>/
    ocr_html.html        — full HTML response (re-OCR'd, may differ slightly)
    ocr_logprobs.json    — list of {token, logprob, top_logprobs[]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load qwen_client directly to avoid pulling the full package __init__ chain
# (which imports skimage and other heavy deps not needed for OCR-only work).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qwen_client_only",
    str(Path(__file__).parent.parent / "src" / "reference_free_ocr_metric" / "ocr" / "qwen_client.py"),
)
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
# Stub the BaseOCRClient import that qwen_client.py expects.
import types as _types
_base = _types.ModuleType("reference_free_ocr_metric.ocr.base")
class _BaseOCRClient: ...
_base.BaseOCRClient = _BaseOCRClient
sys.modules.setdefault("reference_free_ocr_metric", _types.ModuleType("reference_free_ocr_metric"))
sys.modules.setdefault("reference_free_ocr_metric.ocr", _types.ModuleType("reference_free_ocr_metric.ocr"))
sys.modules["reference_free_ocr_metric.ocr.base"] = _base
# html_parser is also imported by qwen_client; stub it too (we don't use parse() here).
_hp = _types.ModuleType("reference_free_ocr_metric.reconstruction.html_parser")
class _ParsedDocument: ...
class _QwenVLHTMLParser:
    def parse(self, *_a, **_k): return _ParsedDocument()
_hp.ParsedDocument = _ParsedDocument
_hp.QwenVLHTMLParser = _QwenVLHTMLParser
sys.modules.setdefault("reference_free_ocr_metric.reconstruction", _types.ModuleType("reference_free_ocr_metric.reconstruction"))
sys.modules["reference_free_ocr_metric.reconstruction.html_parser"] = _hp
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
QwenVLClient = _mod.QwenVLClient

API_BASE = os.environ.get("OCR_ENDPOINT_URL", "http://localhost:9000/v1")
API_KEY = "tensorflow"
MODEL = "Qwen/Qwen3.5-122B-A10B"

DATA_ROOT = Path(__file__).parent.parent / "data" / "omnidocbench"
IMAGES_DIR = DATA_ROOT / "images"
OUT_ROOT = Path(__file__).parent.parent / "data" / "ocr_logprobs"


def process_one(image_path: Path, client: QwenVLClient) -> tuple[str, str, float]:
    """Process a single page. Returns (page_name, status, elapsed_s)."""
    page_name = image_path.stem
    out_dir = OUT_ROOT / page_name
    lp_path = out_dir / "ocr_logprobs.json"
    html_path = out_dir / "ocr_html.html"

    if lp_path.exists() and html_path.exists():
        return page_name, "skip", 0.0

    t0 = time.time()
    try:
        html, logprobs = client.ocr_with_logprobs(str(image_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        lp_path.write_text(json.dumps({"tokens": logprobs}, ensure_ascii=False))
        return page_name, "ok", time.time() - t0
    except Exception as e:
        err_path = out_dir / "error.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        err_path.write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return page_name, f"error:{type(e).__name__}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls")
    ap.add_argument("--limit", type=int, default=0, help="0=all pages, N=first N")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    client = QwenVLClient(api_base=API_BASE, api_key=API_KEY, model_name=MODEL)
    images = sorted(p for p in IMAGES_DIR.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.limit > 0:
        images = images[: args.limit]
    total = len(images)
    print(f"[start] {total} pages, workers={args.workers}, out={OUT_ROOT}", flush=True)

    t_start = time.time()
    counts = {"ok": 0, "skip": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, p, client): p for p in images}
        for i, fut in enumerate(as_completed(futs), 1):
            page, status, elapsed = fut.result()
            bucket = "error" if status.startswith("error") else status
            counts[bucket] = counts.get(bucket, 0) + 1
            if i % 25 == 0 or i == total:
                eta_s = (time.time() - t_start) / max(i, 1) * (total - i)
                print(
                    f"[{i:4d}/{total}] {status:14s} {elapsed:5.1f}s {page[:60]}"
                    f"  ok={counts['ok']} skip={counts['skip']} err={counts['error']}"
                    f"  eta={eta_s/60:.1f}min",
                    flush=True,
                )

    print(f"[done] ok={counts['ok']} skip={counts['skip']} error={counts['error']} "
          f"total_elapsed={(time.time()-t_start)/60:.1f}min", flush=True)
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
