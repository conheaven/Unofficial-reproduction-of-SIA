#!/bin/bash
# SIA Test Run - 5 samples for quick testing

# Use GPU 6 (avoid GPU 3 - faulty)
export CUDA_VISIBLE_DEVICES=6

cd /home/gwj/gwj_sdd/baseline/sia

echo "Running SIA test evaluation (5 samples)..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

uv run python eval_vlguard.py \
    --model-path /home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct \
    --model-type qwen2.5-vl \
    --data-file /home/gwj/gwj_sdd/dataset/VLGuard/vlguard_dataset.json \
    --output-file results/vlguard_sia_test.json \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --limit 5

echo ""
echo "Test complete! Check results/vlguard_sia_test.json"
