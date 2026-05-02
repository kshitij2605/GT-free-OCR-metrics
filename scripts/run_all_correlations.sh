#!/bin/bash
# Run correlations and leaderboard for all background test methods
# Safe to run even if some results.json are missing (skips those)
set -uo pipefail
cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/..

ANNOT="data/omnidocbench/OmniDocBench.json"
OCR_DIR="data/omnidocbench/ocr"
METHODS="A1 A2 A3 A4 B1_ns B1_telea C2_dino C3_dreamsim C6_stlpips"
VARIANTS="all text formula table"

YAML_MAP="A1:A1_gaussian_divide_otsu A2:A2_blackhat_otsu A3:A3_sauvola A4:A4_hsv_whitewash B1_ns:B1_ns B1_telea:B1_telea C2_dino:C2_dino C3_dreamsim:C3_dreamsim C6_stlpips:C6_stlpips"

echo "=== Step 1: Run correlations ==="
for variant in $VARIANTS; do
    for method in $METHODS; do
        results="results/method_runs/ocr_${variant}/${method}/results.json"
        corr_out="results/method_runs/ocr_${variant}/${method}/correlations.json"
        if [ ! -f "$results" ]; then
            echo "  SKIP (no results): $variant/$method"
            continue
        fi
        if [ -f "$corr_out" ]; then
            echo "  SKIP (already done): $variant/$method"
            continue
        fi
        focus_arg=""
        [ "$variant" = "formula" ] && focus_arg="--focus formula"
        [ "$variant" = "text" ]    && focus_arg="--focus text"
        [ "$variant" = "table" ]   && focus_arg="--focus table"
        echo "  Running correlation: $variant/$method"
        /home/mac/.local/bin/uv run python scripts/run_comparison.py \
            --experiment-results "$results" \
            --annotations-json "$ANNOT" \
            --artifacts-dir "$OCR_DIR" \
            --output "results/method_runs/ocr_${variant}/${method}/comparison.json" \
            --no-cdm \
            --documentation-output "$corr_out" \
            $focus_arg \
            > "logs/bg_corr_${variant}_${method}.log" 2>&1 &
    done
done
wait
echo "=== Correlations done ==="

echo "=== Step 2: Update leaderboard ==="
for pair in $YAML_MAP; do
    method="${pair%%:*}"
    yaml_id="${pair##*:}"
    yaml_path="methods/${yaml_id}.yaml"
    if [ -f "$yaml_path" ]; then
        echo "  Updating leaderboard: $yaml_id"
        /home/mac/.local/bin/uv run python scripts/update_leaderboard.py --technique-yaml "$yaml_path"
    else
        echo "  SKIP: $yaml_path not found"
    fi
done
echo "=== Leaderboard update done ==="
