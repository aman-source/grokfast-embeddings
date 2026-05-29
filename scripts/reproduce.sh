#!/usr/bin/env bash
# Reproduce Stage 2 results.
# Works with any number of GPUs -- assign seeds to GPUs however you like.
# Examples below show 1-GPU and 4-GPU setups; adapt as needed.

set -e
cd "$(dirname "$0")/.."

CONFIG="configs/stage2.yaml"
SEEDS=(0 1)   # edit to add more seeds

NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "GPUs available: $NGPU"

if [ "$NGPU" -le 1 ]; then
    # Single GPU: run all seeds sequentially
    echo "Running all seeds on GPU 0 ..."
    python src/train.py --config "$CONFIG" --seeds "${SEEDS[@]}"
else
    # Multiple GPUs: one seed per GPU, all in parallel
    PIDS=()
    for i in "${!SEEDS[@]}"; do
        GPU=$(( i % NGPU ))
        SEED="${SEEDS[$i]}"
        echo "GPU $GPU: seed $SEED"
        CUDA_VISIBLE_DEVICES=$GPU python src/train.py \
            --config "$CONFIG" --seeds "$SEED" --tag "gpu${GPU}" &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do
        wait "$PID"
    done
fi

echo ""
echo "=== Final verdict ==="
python src/train.py --config "$CONFIG" --report-only

echo ""
echo "=== Regenerating figures ==="
python scripts/make_figures.py
