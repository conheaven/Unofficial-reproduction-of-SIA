#!/bin/bash
# SIA Full Evaluation on SIUO dataset (167 samples)

# Use GPU 7
export CUDA_VISIBLE_DEVICES=7

cd /home/gwj/gwj_sdd/baseline/sia

echo "Running SIA full evaluation on SIUO dataset (167 samples)..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "This should take about 1 hour. Progress will be shown."
echo ""

uv run python eval_vlguard.py \
    --model-path /home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct \
    --model-type qwen2.5-vl \
    --data-file /home/gwj/gwj_sdd/dataset/SIUO/siuo_dataset.json \
    --output-file results/siuo_sia_qwen25vl_full.json \
    --temperature 0.2 \
    --max-new-tokens 1024

echo ""
echo "Full evaluation complete! Check results/siuo_sia_qwen25vl_full.json"
