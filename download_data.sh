#!/usr/bin/env bash
# Download OmniDocBench dataset and pre-computed OCR artifacts from HuggingFace.
# Idempotent — safe to run multiple times.
#
# Usage:
#   bash download_data.sh                # default: parquet (fast, no rate limit)
#   DATA_FORMAT=raw bash download_data.sh  # legacy per-page-directory layout
#
# Required disk: ~50 GB
# Requires: uv  (pip install uv && uv sync)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

UV=$(command -v uv 2>/dev/null || echo uv)
FORMAT="${DATA_FORMAT:-parquet}"

echo "=== Downloading OmniDocBench (ground-truth annotations + page images) ==="
$UV run huggingface-cli download opendatalab/OmniDocBench \
    --repo-type dataset \
    --local-dir data/omnidocbench \
    --local-dir-use-symlinks False

echo ""
if [ "$FORMAT" = "parquet" ]; then
    # Default fast path: 64 parquet shards (~9.4 GB compressed) → extract locally
    # This avoids HuggingFace's 5,000-resolver-cache-requests/5min rate limit
    # that the per-file layout (23,600 files) trips for unauthenticated users.
    echo "=== Downloading render-and-compare parquet shards ==="
    $UV run huggingface-cli download gt-free-ocr-metrics/omnidocbench-render-compare-parquet \
        --repo-type dataset \
        --local-dir data/parquet_in \
        --local-dir-use-symlinks False

    echo ""
    echo "=== Extracting parquet shards into data/omnidocbench/ocr_<variant>/ ==="
    $UV run python scripts/extract_parquet_to_disk.py \
        --in data/parquet_in \
        --out data/omnidocbench
else
    # Legacy raw layout: per-page directories with individual files. Slow due to
    # HF rate limits — set HF_TOKEN to avoid 429 errors.
    echo "=== Downloading render-and-compare raw per-page files (legacy) ==="
    $UV run huggingface-cli download gt-free-ocr-metrics/omnidocbench-render-compare \
        --repo-type dataset \
        --local-dir data/omnidocbench \
        --local-dir-use-symlinks False
fi

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
