#!/bin/bash
# Run all methods and update the leaderboard.
# Iterates over every YAML in methods/, deriving method_id from filename.
# Safe to resume: skips variants where correlations.json already exists.
set -uo pipefail
cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/..

ANNOT="data/omnidocbench/OmniDocBench.json"
OCR_DIR="data/omnidocbench/ocr_all"
VARIANTS="all text formula table all_no_mask"

UV=$(command -v uv 2>/dev/null || echo uv)
mkdir -p logs

echo "=== Running all $(ls methods/*.yaml | wc -l) methods ==="

for yaml_path in methods/*.yaml; do
    method_id="${yaml_path##methods/}"
    method_id="${method_id%.yaml}"
    echo "--- Method: $method_id ---"

    for variant in $VARIANTS; do
        out_dir="results/method_runs/ocr_${variant}/${method_id}"
        results_json="${out_dir}/results.json"
        corr_json="${out_dir}/correlations.json"

        if [ -f "$corr_json" ]; then
            echo "  SKIP (done): $variant/$method_id"
            continue
        fi

        if [ ! -f "$results_json" ]; then
            # Need to run the method first
            GPU_ID=${1:-0}
            echo "  Running method: $variant/$method_id (GPU $GPU_ID)"
            bash scripts/run_method.sh "$method_id" "$GPU_ID"
        fi

        if [ ! -f "$results_json" ]; then
            echo "  ERROR: results.json missing after run: $variant/$method_id"
            continue
        fi

        focus_arg=""
        [ "$variant" = "formula" ] && focus_arg="--focus formula"
        [ "$variant" = "text" ]    && focus_arg="--focus text"
        [ "$variant" = "table" ]   && focus_arg="--focus table"
        echo "  Computing correlation: $variant/$method_id"
        $UV run python scripts/run_comparison.py             --experiment-results "$results_json"             --annotations-json "$ANNOT"             --artifacts-dir "$OCR_DIR"             --output "${out_dir}/comparison.json"             --no-cdm             --documentation-output "$corr_json"             $focus_arg             > "logs/corr_${variant}_${method_id}.log" 2>&1
    done

    echo "  Updating leaderboard: $method_id"
    $UV run python scripts/update_leaderboard.py --technique-yaml "$yaml_path"
done

echo "=== All methods done. Final leaderboard: ==="
$UV run python scripts/show_leaderboard.py --top 30
