#!/bin/bash
# Run batched visual metric computation for all 4 target variants in parallel.
# Dynamically assigns GPUs by free memory; falls back to CPU if a GPU is too occupied.
#
# Usage: bash scripts/run_variant_metrics.sh
# Logs:  logs/metrics_{variant}.log  logs/metrics_{variant}.error.log

cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/..
mkdir -p logs

SCRIPT="scripts/compute_variant_metrics.py"
VARIANTS=(all text formula table)
MIN_FREE_MB=1500  # minimum free VRAM required to use a GPU

# Returns GPU indices (one per line) that have >= MIN_FREE_MB free, sorted most-free first.
available_gpus() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v min="$MIN_FREE_MB" '$2 >= min {print $1, $2}' \
        | sort -k2 -rn \
        | awk '{print $1}'
}

# Read available GPUs into an array
mapfile -t GPUS < <(available_gpus)

echo "Available GPUs (free >= ${MIN_FREE_MB}MB): ${GPUS[*]:-none}"

for i in "${!VARIANTS[@]}"; do
    v="${VARIANTS[$i]}"
    if [ $i -lt ${#GPUS[@]} ]; then
        g="${GPUS[$i]}"
        echo "Launching $v on GPU $g"
        CUDA_VISIBLE_DEVICES=$g nohup /home/mac/.local/bin/uv run python "$SCRIPT" $v \
            > logs/metrics_${v}.log 2>&1 &
    else
        # No GPU available for this variant — Python script will use CPU
        echo "Launching $v on CPU (no free GPU)"
        CUDA_VISIBLE_DEVICES="" nohup /home/mac/.local/bin/uv run python "$SCRIPT" $v \
            > logs/metrics_${v}.log 2>&1 &
    fi
done

echo "All ${#VARIANTS[@]} variants launched. Follow logs with:"
echo "  tail -f logs/metrics_{all,text,formula,table}.log"
