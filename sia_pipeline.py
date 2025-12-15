"""
SIA Pipeline - 3-stage safety inference framework.

This module implements the core SIA (Safety via Intent Awareness) pipeline:
- Stage 1: Visual Abstraction (caption generation)
- Stage 2: Intent Inference (text-only CoT reasoning)
- Stage 3: Intent-Conditioned Response (final safe answer)
"""

import re
from typing import Dict, Tuple, Optional
from PIL import Image


class SIAPipeline:
    """
    SIA (Safety via Intent Awareness) 3-stage pipeline.

    The pipeline follows the methodology from the paper:
    1. Generate objective caption from image
    2. Infer user intent via few-shot CoT (text-only)
    3. Generate final response conditioned on inferred intent
    """

    def __init__(self, adapter, temperature: float = 0.2, max_new_tokens: int = 1024):
        """
        Initialize SIA pipeline.

        Args:
            adapter: VLM adapter (e.g., Qwen25VLAdapter)
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum tokens per generation
        """
        self.adapter = adapter
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Import prompts
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from prompts import P_CAPTION, P_FEWSHOT, P_RESPONSE
        self.P_CAPTION = P_CAPTION
        self.P_FEWSHOT = P_FEWSHOT
        self.P_RESPONSE = P_RESPONSE

    def stage1_caption(self, image: Image.Image) -> str:
        """
        Stage 1: Generate objective image caption.

        Args:
            image: PIL Image to caption

        Returns:
            Caption text describing visible content
        """
        caption = self.adapter.generate(
            query=self.P_CAPTION,
            image=image,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )
        return caption.strip()

    def stage2_intent_inference(self, caption: str, query: str) -> Tuple[str, str, str]:
        """
        Stage 2: Infer user intent via few-shot CoT (text-only).

        This stage is CRITICAL: it must use text-only inference without
        the image. The model reasons about the caption (not raw image).

        Args:
            caption: Generated caption from Stage 1
            query: User's original question

        Returns:
            Tuple of (intent, reasoning, raw_output)
        """
        # Format the few-shot prompt with caption and query
        prompt = self.P_FEWSHOT.format(caption=caption, query=query)

        # CRITICAL: Text-only inference (no image!)
        raw_output = self.adapter.generate(
            query=prompt,
            image=None,  # No image input!
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )

        # Parse intent and reasoning from output
        intent, reasoning = self._parse_intent_reasoning(raw_output)

        return intent, reasoning, raw_output

    def stage3_response(
        self,
        image: Image.Image,
        query: str,
        caption: str,
        intent: str,
        reasoning: str
    ) -> str:
        """
        Stage 3: Generate final response conditioned on intent.

        Args:
            image: PIL Image (restored for final response)
            query: User's original question
            caption: Caption from Stage 1
            intent: Intent from Stage 2
            reasoning: Reasoning from Stage 2

        Returns:
            Final response text
        """
        # Combine intent and reasoning
        intent_reasoning = f"{intent} {reasoning}".strip()

        # Format response prompt
        prompt = self.P_RESPONSE.format(
            caption=caption,
            query=query,
            intent_reasoning=intent_reasoning
        )

        # Generate final response with image
        response = self.adapter.generate(
            query=prompt,
            image=image,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )

        return response.strip()

    def _parse_intent_reasoning(self, raw_output: str) -> Tuple[str, str]:
        """
        Parse 'Intent:' and 'Reasoning:' from Stage 2 output.

        Expected format:
            Intent: <intent text>
            Reasoning: <reasoning text>

        Uses regex with fallback to line-by-line parsing for robustness.

        Args:
            raw_output: Raw text output from Stage 2

        Returns:
            Tuple of (intent, reasoning)
        """
        intent = ""
        reasoning = ""

        # Try regex first (handles multi-line, flexible spacing)
        intent_match = re.search(
            r'Intent:\s*(.+?)(?=Reasoning:|$)',
            raw_output,
            re.DOTALL | re.IGNORECASE
        )
        reasoning_match = re.search(
            r'Reasoning:\s*(.+?)$',
            raw_output,
            re.DOTALL | re.IGNORECASE
        )

        if intent_match:
            intent = intent_match.group(1).strip()
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Fallback: line-by-line parsing
        if not intent or not reasoning:
            lines = raw_output.split('\n')
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if 'intent:' in line_lower:
                    # Extract intent (may be on same line or next line)
                    if ':' in line:
                        intent_text = line.split(':', 1)[1].strip()
                        if intent_text:
                            intent = intent_text
                    # Check next lines for continuation
                    for j in range(i+1, min(i+5, len(lines))):
                        if 'reasoning:' in lines[j].lower():
                            break
                        if lines[j].strip() and not intent:
                            intent = lines[j].strip()
                            break

                if 'reasoning:' in line_lower:
                    # Extract reasoning (may span multiple lines)
                    if ':' in line:
                        reasoning_text = line.split(':', 1)[1].strip()
                        if reasoning_text:
                            reasoning = reasoning_text
                    # Collect remaining lines as reasoning
                    remaining = '\n'.join(lines[i+1:]).strip()
                    if remaining and not reasoning:
                        reasoning = remaining
                    break

        return intent, reasoning

    def run_full_pipeline(self, image: Image.Image, query: str) -> Dict:
        """
        Run complete SIA pipeline (3 stages).

        Args:
            image: PIL Image
            query: User query

        Returns:
            Dictionary with all stage outputs:
            {
                'stage1_caption': str,
                'stage2_intent': str,
                'stage2_reasoning': str,
                'stage2_raw_output': str,
                'stage3_final_response': str
            }
        """
        # Stage 1: Generate caption
        caption = self.stage1_caption(image)

        # Stage 2: Infer intent (text-only)
        intent, reasoning, raw_stage2 = self.stage2_intent_inference(caption, query)

        # Stage 3: Generate final response
        final_response = self.stage3_response(image, query, caption, intent, reasoning)

        return {
            'stage1_caption': caption,
            'stage2_intent': intent,
            'stage2_reasoning': reasoning,
            'stage2_raw_output': raw_stage2,
            'stage3_final_response': final_response
        }
