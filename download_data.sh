#!/usr/bin/env bash
# Download OmniDocBench dataset and pre-computed OCR artifacts from HuggingFace.
# Idempotent — safe to run multiple times.
# Usage: bash download_data.sh
#
# Required disk: ~50 GB
# Requires: uv  (pip install uv && uv sync)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

UV=$(command -v uv 2>/dev/null || echo uv)

echo "=== Downloading OmniDocBench (ground-truth annotations + page images) ==="
$UV run huggingface-cli download opendatalab/OmniDocBench \
    --repo-type dataset \
    --local-dir data/omnidocbench \
    --local-dir-use-symlinks False

echo ""
echo "=== Downloading pre-computed render-and-compare OCR artifacts ==="
# Dataset contains ocr_all/, ocr_text/, ocr_formula/, ocr_table/, ocr_all_no_mask/
# subfolders, each with per-page directories holding:
#   masked_original.png, reconstructed.png, ocr_html.html,
#   ocr_elements.json, ocr_formula_elements.json, ocr_table_elements.json
# Download to data/omnidocbench/ so paths become data/omnidocbench/ocr_{variant}/<page_id>/
$UV run huggingface-cli download gt-free-ocr-metrics/omnidocbench-render-compare \
    --repo-type dataset \
    --local-dir data/omnidocbench \
    --local-dir-use-symlinks False

echo ""
echo "=== Downloading OCR log-probabilities ==="
$UV run huggingface-cli download gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs \
    --repo-type dataset \
    --local-dir data/ocr_logprobs \
    --local-dir-use-symlinks False

echo ""
echo "=== Download complete ==="
echo "Data layout:"
echo "  data/omnidocbench/OmniDocBench.json     — GT annotations"
echo "  data/omnidocbench/images/               — original page scans"
echo "  data/omnidocbench/ocr_{all,text,formula,table,all_no_mask}/<page_id>/"
echo "                                          — pre-computed OCR artifacts"
echo "  data/ocr_logprobs/                       — per-token OCR log-probabilities"
