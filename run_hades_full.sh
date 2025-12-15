#!/bin/bash
# SIA Full Evaluation on HADES dataset (750 samples)

# Use GPU 2
export CUDA_VISIBLE_DEVICES=2

cd /home/gwj/gwj_sdd/baseline/sia

echo "Running SIA full evaluation on HADES dataset (750 samples)..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "This should take about 4-5 hours. Progress will be shown."
echo ""

uv run python eval_vlguard.py \
    --model-path /home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct \
    --model-type qwen2.5-vl \
    --data-file /home/gwj/gwj_sdd/dataset/HADES/hades_dataset.json \
    --output-file results/hades_sia_qwen25vl_full.json \
    --temperature 0.2 \
    --max-new-tokens 1024

echo ""
echo "Full evaluation complete! Check results/hades_sia_qwen25vl_full.json"
