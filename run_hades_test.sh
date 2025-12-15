#!/bin/bash
# SIA Evaluation on HADES dataset - Test run (5 samples)

# Use GPU 2
export CUDA_VISIBLE_DEVICES=2

cd /home/gwj/gwj_sdd/baseline/sia

echo "Running SIA test evaluation on HADES dataset (5 samples)..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

uv run python eval_vlguard.py \
    --model-path /home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct \
    --model-type qwen2.5-vl \
    --data-file /home/gwj/gwj_sdd/dataset/HADES/hades_dataset.json \
    --output-file results/hades_sia_test.json \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --limit 5

echo ""
echo "Test complete! Check results/hades_sia_test.json"
