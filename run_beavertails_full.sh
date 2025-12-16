#!/bin/bash
# SIA Full Evaluation on BeaverTails-V dataset (590 samples)

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

cd /home/gwj/gwj_sdd/baseline/sia

echo "Running SIA full evaluation on BeaverTails-V dataset (590 samples)..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "This should take about 3-4 hours. Progress will be shown."
echo ""

uv run python eval_vlguard.py \
    --model-path /home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct \
    --model-type qwen2.5-vl \
    --data-file /home/gwj/gwj_sdd/dataset/BeaverTails-V/image_index.json \
    --output-file results/beavertails_sia_qwen25vl_full.json \
    --temperature 0.2 \
    --max-new-tokens 1024

echo ""
echo "Full evaluation complete! Check results/beavertails_sia_qwen25vl_full.json"
