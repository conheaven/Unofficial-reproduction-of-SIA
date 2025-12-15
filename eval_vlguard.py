#!/usr/bin/env python3
"""
SIA evaluation on VLGuard dataset using Qwen2.5-VL.

This script implements the 3-stage SIA pipeline and evaluates on VLGuard:
1. Visual Abstraction (caption generation)
2. Intent Inference (text-only CoT)
3. Intent-Conditioned Response
"""

import argparse
import os
import json
import sys
from tqdm import tqdm
from PIL import Image

# Add ECSO path for VLM adapter
sys.path.insert(0, '/home/gwj/gwj_sdd/baseline/ECSO-main')
from llava.model.vlm_adapter import create_adapter

# Add current directory for SIA modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sia_pipeline import SIAPipeline
from utils import detect_unsafe_intent, detect_refusal, calculate_metrics


def process_vlguard_item(item, sia_pipeline):
    """
    Process a single VLGuard item through SIA pipeline.

    Args:
        item: VLGuard data item with format:
              {
                  "problem_id": int,
                  "problem": str (instruction),
                  "path": str (image path),
                  "problem_type": str,
                  "data_type": str,
                  "solution": str (ground truth answer)
              }
        sia_pipeline: SIAPipeline instance

    Returns:
        Result dictionary or None if error
    """
    try:
        # Validate required fields
        if not all(k in item for k in ['problem_id', 'problem', 'path']):
            print(f"Warning: Item missing required fields, skipping")
            return None

        # Load image
        image_path = item['path']
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return None

        image = Image.open(image_path).convert('RGB')
        query = item['problem']

        # Run SIA pipeline
        sia_outputs = sia_pipeline.run_full_pipeline(image, query)

        # Detect unsafe intent and refusal
        sia_detected_unsafe = detect_unsafe_intent(
            sia_outputs['stage2_intent'],
            sia_outputs['stage2_reasoning']
        )
        sia_refused = detect_refusal(sia_outputs['stage3_final_response'])

        # Build result
        result = {
            # Original fields (preserved from VLGuard)
            'problem_id': item['problem_id'],
            'problem': query,
            'path': item['path'],
            'problem_type': item.get('problem_type', ''),
            'data_type': item.get('data_type', ''),
            'solution': item.get('solution', ''),  # Ground truth

            # SIA stage outputs
            'stage1_caption': sia_outputs['stage1_caption'],
            'stage2_intent': sia_outputs['stage2_intent'],
            'stage2_reasoning': sia_outputs['stage2_reasoning'],
            'stage2_raw_output': sia_outputs['stage2_raw_output'],
            'stage3_final_response': sia_outputs['stage3_final_response'],

            # Metrics
            'sia_detected_unsafe': sia_detected_unsafe,
            'sia_refused': sia_refused
        }

        return result

    except Exception as e:
        print(f"Error processing item {item.get('problem_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="SIA evaluation on VLGuard dataset using Qwen2.5-VL"
    )

    # Model arguments
    parser.add_argument("--model-path", type=str,
                       default="/home/gwj/gwj_sdd/model/Qwen2.5-VL-3B-Instruct",
                       help="Path to Qwen2.5-VL model")
    parser.add_argument("--model-type", type=str, default="qwen2.5-vl",
                       help="Type of VLM (default: qwen2.5-vl)")

    # Data arguments
    parser.add_argument("--data-file", type=str,
                       default="/home/gwj/gwj_sdd/dataset/VLGuard/vlguard_dataset.json",
                       help="Path to VLGuard dataset JSON")
    parser.add_argument("--output-file", type=str,
                       default="results/vlguard_sia_qwen25vl_results.json",
                       help="Output file path for results")

    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                       help="Maximum tokens per generation")

    # Evaluation arguments
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--offset", type=int, default=0,
                       help="Starting offset in dataset")

    args = parser.parse_args()

    print("="*60)
    print("SIA Evaluation on VLGuard Dataset")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.output_file}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_new_tokens}")
    print("="*60)

    # Load model
    print("\nLoading model...")
    adapter = create_adapter(args.model_type)
    adapter.load_model(args.model_path)
    print("Model loaded successfully!")

    # Initialize SIA pipeline
    print("\nInitializing SIA pipeline...")
    sia_pipeline = SIAPipeline(
        adapter,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    print("SIA pipeline initialized!")

    # Load data
    print(f"\nLoading data from {args.data_file}...")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples in dataset: {len(data)}")

    # Apply offset and limit
    if args.offset > 0:
        data = data[args.offset:]
        print(f"Starting from offset {args.offset}")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to {args.limit} samples")

    print(f"Samples to process: {len(data)}")

    # Process all items
    print("\nProcessing samples through SIA pipeline...")
    results = []
    successful = 0
    failed = 0

    for item in tqdm(data, desc="Processing VLGuard with SIA"):
        result = process_vlguard_item(item, sia_pipeline)
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    output_data = {
        'metadata': {
            'model_path': args.model_path,
            'model_type': args.model_type,
            'data_file': args.data_file,
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'total_samples': len(data),
            'successful': successful,
            'failed': failed
        },
        'metrics': metrics,
        'results': results
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n" + "="*60)
    print("SIA Evaluation Complete!")
    print("="*60)
    print(f"Total processed: {successful}")
    print(f"Failed: {failed}")
    print(f"\nMetrics:")
    print(f"  Detection Rate: {metrics['detection_rate']:.2%} "
          f"({metrics['detected_unsafe_count']}/{metrics['total_samples']})")
    print(f"  Refusal Rate: {metrics['refusal_rate']:.2%} "
          f"({metrics['refused_count']}/{metrics['total_samples']})")
    print(f"  Alignment Rate: {metrics['alignment_rate']:.2%} "
          f"({metrics['aligned_count']}/{metrics['total_samples']})")
    print(f"\nResults saved to: {args.output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
