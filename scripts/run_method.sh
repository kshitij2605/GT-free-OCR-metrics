#!/bin/bash
# Run a single method against all 4 variants, compute correlations.
# Does NOT update the leaderboard — main session does that after verifying output.
#
# Usage: scripts/run_method.sh <method_id> [gpu_id]
#   method_id: must match a YAML in methods/<method_id>.yaml
#   gpu_id: CUDA device (default 0)
#
# Idempotent: skips existing results.json and correlations.json.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

METHOD_ID="${1:?Usage: run_method.sh <method_id> [gpu_id]}"
GPU="${2:-0}"
YAML_PATH="methods/${METHOD_ID}.yaml"
ANNOT="data/omnidocbench/OmniDocBench.json"
OCR_DIR="data/omnidocbench/ocr"
VARIANTS="all text formula table"
UV=$(command -v uv 2>/dev/null || echo uv)

[ ! -f "$YAML_PATH" ] && { echo "ERROR: $YAML_PATH not found"; exit 1; }

SCRIPT_PATH=$(python3 -c "
import yaml
d = yaml.safe_load(open('$YAML_PATH'))
print(d.get('script', ''))
")
[ -z "$SCRIPT_PATH" ] && { echo "ERROR: no 'script' field in $YAML_PATH"; exit 1; }

echo "=== run_method: $METHOD_ID  gpu=$GPU  script=$SCRIPT_PATH ==="
mkdir -p logs

for variant in $VARIANTS; do
    out_dir="results/method_runs/ocr_${variant}/${METHOD_ID}"
    results="${out_dir}/results.json"
    corr_out="${out_dir}/correlations.json"
    mkdir -p "$out_dir"

    # --- Step 1: compute per-page metrics ---
    if [ -f "$results" ]; then
        echo "  SKIP metrics (exists): $variant/$METHOD_ID"
    else
        echo "  Computing metrics: $variant/$METHOD_ID"
        CUDA_VISIBLE_DEVICES=$GPU $UV run python "$SCRIPT_PATH" "$variant" \
            > "logs/method_${variant}_${METHOD_ID}.log" 2>&1 \
            || { echo "  ERROR computing metrics for $variant/$METHOD_ID — see logs/method_${variant}_${METHOD_ID}.log"; continue; }
    fi

    # --- Step 2: compute correlations ---
    if [ -f "$corr_out" ]; then
        echo "  SKIP correlations (exists): $variant/$METHOD_ID"
        continue
    fi

    focus_arg=""
    [ "$variant" = "formula" ] && focus_arg="--focus formula"
    [ "$variant" = "text" ]    && focus_arg="--focus text"
    [ "$variant" = "table" ]   && focus_arg="--focus table"

    echo "  Computing correlations: $variant/$METHOD_ID"
    $UV run python scripts/run_comparison.py \
        --experiment-results "$results" \
        --annotations-json "$ANNOT" \
        --artifacts-dir "$OCR_DIR" \
        --output "${out_dir}/comparison.json" \
        --no-cdm \
        --documentation-output "$corr_out" \
        $focus_arg \
        > "logs/corr_${variant}_${METHOD_ID}.log" 2>&1 \
        || echo "  ERROR in correlations for $variant/$METHOD_ID — see logs/corr_${variant}_${METHOD_ID}.log"
done

echo "=== $METHOD_ID complete ==="
echo "    Next: $UV run python scripts/update_leaderboard.py --technique-yaml $YAML_PATH"
