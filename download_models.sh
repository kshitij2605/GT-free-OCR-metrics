#!/usr/bin/env bash
# Download DocSim LoRA weights from HuggingFace.
# DocSim is a LoRA-adapted CLIP+DINOv2 similarity head trained on 20,280 triplets
# derived from the OmniDocBench render-and-compare dataset.
#
# Usage: bash download_models.sh
# Requires: uv  (pip install uv && uv sync)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

UV=$(command -v uv 2>/dev/null || echo uv)
HF_MODEL_REPO="gt-free-ocr-metrics/docsim-lora"

echo "=== Downloading DocSim LoRA weights ==="
$UV run huggingface-cli download "$HF_MODEL_REPO" \
    --repo-type model \
    --local-dir models/docsim_lora \
    --local-dir-use-symlinks False

echo ""
echo "Weights downloaded to models/docsim_lora/"
