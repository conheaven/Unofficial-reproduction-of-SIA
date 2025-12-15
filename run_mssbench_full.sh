#!/bin/bash
# SIA Full Evaluation on MM-SafetyBench (mssbench) dataset (376 samples)

# Use GPU 1
export CUDA_VISIBLE_DEVICES=1

cd /home/gwj/gwj_sdd/baseline/sia

echo "Running SIA full evaluation on MM-SafetyBench dataset (376 samples)..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "This should take about 2-3 hours. Progress will be shown."
echo ""

uv run python eval_vlguard.py \
    --model-path /home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct \
    --model-type qwen2.5-vl \
    --data-file /home/gwj/gwj_sdd/dataset/mssbench/mssbench_dataset.json \
    --output-file results/mssbench_sia_qwen25vl_full.json \
    --temperature 0.2 \
    --max-new-tokens 1024

echo ""
echo "Full evaluation complete! Check results/mssbench_sia_qwen25vl_full.json"
