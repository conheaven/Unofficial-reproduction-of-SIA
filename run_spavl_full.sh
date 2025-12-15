#!/bin/bash
# SIA Full Evaluation on SPA-VL dataset (265 samples)

# Use GPU 6 (avoid GPU 3 - faulty)
export CUDA_VISIBLE_DEVICES=5

cd /home/gwj/gwj_sdd/baseline/sia

echo "Running SIA full evaluation on SPA-VL dataset (265 samples)..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "This should take about 1-2 hours. Progress will be shown."
echo ""

uv run python eval_vlguard.py \
    --model-path /home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct \
    --model-type qwen2.5-vl \
    --data-file /home/gwj/gwj_sdd/dataset/SPA-VL/spavl_dataset.json \
    --output-file results/spavl_sia_qwen25vl_full.json \
    --temperature 0.2 \
    --max-new-tokens 1024

echo ""
echo "Full evaluation complete! Check results/spavl_sia_qwen25vl_full.json"
