#!/usr/bin/env bash
# Download OmniDocBench dataset and pre-computed OCR artifacts from HuggingFace.
# Idempotent — safe to run multiple times.
# Usage: bash download_data.sh
#
# Required disk: ~50 GB
# Requires: huggingface-cli  (pip install huggingface-hub)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v huggingface-cli >/dev/null 2>&1 || {
    echo "ERROR: huggingface-cli not found. Install via: pip install huggingface-hub"
    exit 1
}

echo "=== Downloading OmniDocBench (ground-truth annotations + page images) ==="
huggingface-cli download opendatalab/OmniDocBench \
    --repo-type dataset \
    --local-dir data/omnidocbench \
    --local-dir-use-symlinks False

echo ""
echo "=== Downloading pre-computed render-and-compare OCR artifacts ==="
# Contains: masked_original.png, reconstructed.png, ocr_html.html,
#           ocr_elements.json, ocr_formula_elements.json, ocr_table_elements.json
# Also includes OmniDocBench.json for convenience.
huggingface-cli download puku128/omnidocbench-render-compare \
    --repo-type dataset \
    --local-dir data/omnidocbench/ocr \
    --local-dir-use-symlinks False

echo ""
echo "=== Downloading OCR log-probabilities ==="
huggingface-cli download puku128/omnidocbench-qwen-ocr-logprobs \
    --repo-type dataset \
    --local-dir data/omnidocbench/ocr_logprobs \
    --local-dir-use-symlinks False

echo ""
echo "=== Download complete ==="
echo "Data layout:"
echo "  data/omnidocbench/OmniDocBench.json     — GT annotations"
echo "  data/omnidocbench/images/               — original page scans"
echo "  data/omnidocbench/ocr/                  — pre-computed OCR artifacts"
echo "  data/omnidocbench/ocr_logprobs/         — per-token OCR log-probabilities"
