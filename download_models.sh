#!/usr/bin/env bash
# Download DocSim LoRA weights from HuggingFace.
# DocSim is a LoRA-adapted CLIP+DINOv2 similarity head trained on 20,280 triplets
# derived from the OmniDocBench render-and-compare dataset.
#
# Usage: bash download_models.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v huggingface-cli >/dev/null 2>&1 || {
    echo "ERROR: huggingface-cli not found. Install via: pip install huggingface-hub"
    exit 1
}

HF_MODEL_REPO="puku128/docsim-lora"

echo "=== Downloading DocSim LoRA weights ==="
huggingface-cli download "$HF_MODEL_REPO" \
    --repo-type model \
    --local-dir models/docsim_lora \
    --local-dir-use-symlinks False

echo ""
echo "Weights downloaded to models/docsim_lora/"
