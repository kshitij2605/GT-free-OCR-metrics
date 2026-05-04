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
set -uo pipefail

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
    echo "=== Downloading render-and-compare parquet shards ==="
    # 64 shards × ~75 rows = 4,610 page-rows in 64 HTTP requests.
    # No HF rate-limit issues; no HF_TOKEN required.
    $UV run huggingface-cli download gt-free-ocr-metrics/omnidocbench-render-compare-parquet \
        --repo-type dataset \
        --local-dir data/parquet_in \
        --local-dir-use-symlinks False

    echo ""
    echo "=== Extracting render-compare shards into data/omnidocbench/ocr_<variant>/ ==="
    $UV run python scripts/extract_parquet_to_disk.py \
        --in data/parquet_in \
        --out data/omnidocbench

    echo ""
    echo "=== Downloading OCR log-probabilities (parquet) ==="
    # 7 shards × ~200 rows = 1,355 logprobs rows in 7 HTTP requests.
    $UV run huggingface-cli download gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs-parquet \
        --repo-type dataset \
        --local-dir data/parquet_logprobs_in \
        --local-dir-use-symlinks False

    echo ""
    echo "=== Extracting logprobs shards into data/ocr_logprobs/ + data/ocr_logprobs_per_bbox/ ==="
    $UV run python scripts/extract_logprobs_parquet.py \
        --in data/parquet_logprobs_in \
        --lp-out data/ocr_logprobs \
        --pb-out data/ocr_logprobs_per_bbox
else
    # Legacy raw layout: per-page directories. Hits HF rate limits — set HF_TOKEN to avoid 429.
    echo "=== Downloading render-and-compare raw per-page files (legacy) ==="
    $UV run huggingface-cli download gt-free-ocr-metrics/omnidocbench-render-compare \
        --repo-type dataset \
        --local-dir data/omnidocbench \
        --local-dir-use-symlinks False

    echo ""
    echo "=== Downloading OCR log-probabilities (raw) ==="
    $UV run huggingface-cli download gt-free-ocr-metrics/omnidocbench-qwen-ocr-logprobs \
        --repo-type dataset \
        --local-dir data/ocr_logprobs \
        --local-dir-use-symlinks False
fi

echo ""
echo "=== Download complete ==="
echo "Data layout:"
echo "  data/omnidocbench/OmniDocBench.json     — GT annotations"
echo "  data/omnidocbench/images/               — original page scans"
echo "  data/omnidocbench/ocr_{all,text,formula,table,all_no_mask}/<page_id>/"
echo "                                          — pre-computed OCR artifacts"
echo "  data/ocr_logprobs/<page_id>/            — per-token OCR log-probabilities"
echo "  data/ocr_logprobs_per_bbox/<page_id>/   — per-bbox aggregated logprobs"
