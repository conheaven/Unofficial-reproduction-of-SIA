#!/bin/bash
# SIA Evaluation on BeaverTails-V dataset - Test run (5 samples)

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

cd /home/gwj/gwj_sdd/baseline/sia

echo "Running SIA test evaluation on BeaverTails-V dataset (5 samples)..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

uv run python eval_vlguard.py \
    --model-path /home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct \
    --model-type qwen2.5-vl \
    --data-file /home/gwj/gwj_sdd/dataset/BeaverTails-V/image_index.json \
    --output-file results/beavertails_sia_test.json \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --limit 5

echo ""
echo "Test complete! Check results/beavertails_sia_test.json"
