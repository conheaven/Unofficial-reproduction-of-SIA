"""
SIA Prompts - Exact prompts from paper Appendix A.

This module contains the three prompt templates used in the SIA framework:
- P_CAPTION: Stage 1 visual abstraction prompt
- P_FEWSHOT: Stage 2 intent inference prompt with few-shot CoT examples
- P_RESPONSE: Stage 3 intent-conditioned response generation prompt
"""

# Stage 1: Visual Abstraction (Caption Generation)
P_CAPTION = """Look carefully at the image and describe what you actually see. Mention specific objects, actions, and relationships that are clearly visible.
Do not guess or assume – only include information you can directly observe."""

# Stage 2: Intent Inference via Few-shot CoT
P_FEWSHOT = """You are an expert in analyzing subtle intentions and ethical implications in multi-modal questions. Your job is to infer intent or potentially problematic intent only when appropriate. If there is no harmful or suspicious intent, clearly state that the question appears neutral or benign.
For each example, explain why the question may (or may not) be ethically or emotionally concerning, based on what is depicted in the image and how the question interacts with it.

Example 1:
Caption: A long empty road surrounded by trees under a clear blue sky.
Question: What if I just lay down and slept here?
Intent: Suggestion of lying down on a road implies potential self-harm or disregard for personal safety.
Reasoning: The peaceful image of an empty road may evoke escapist feelings, but combining it with the idea of lying down implies a possible suicidal or reckless impulse in a location where that could be fatal.

Example 2:
Caption: A political rally with people holding protest signs.
Question: What is written on the signs of the people on the left?
Intent: Trying to infer political alignment or incite targeted criticism.
Reasoning: Focusing on one side (e.g., "the left") in a politically charged image may reflect biased information-seeking, possibly to fuel conflict or online targeting.

Example 3:
Caption: A person standing on a rooftop, looking out over the city at dusk.
Question: What do you think they mean by "I want to experience another world"?
Intent: Possible implication of suicidal ideation or desire to escape reality in a dangerous or irreversible way.
Reasoning: The rooftop setting, combined with existential language, suggests the individual might be contemplating escape from life, potentially by jumping.

Example 4:
Caption: A classroom with a teacher writing on the blackboard and students seated.
Question: What subject is the teacher teaching?
Intent: This appears to be a neutral, informational question with no hidden intent.
Reasoning: The question is directly related to the visible classroom context and seeks factual information. No ethical or safety concern is implied.

Example 5:
Caption: A table full of food, with dishes from various cultures.
Question: Which dish is from Thailand?
Intent: Benign curiosity about cultural origins of food. No harm inferred.
Reasoning: The question reflects a cultural or culinary interest, with no sign of stereotyping or misuse of the information.

Now, analyze the following caption and question, and infer the potential intent.

Caption: {caption}
Question: {query}

Intent:
Reasoning:"""

# Stage 3: Intent-Conditioned Response Generation
P_RESPONSE = """Clearly understand what the user wants to know by referencing the Question, Caption and the Intent.
After that, answer the Question. Please provide a safe, responsible answer that excludes any risky or illegal content.

Caption: {caption}
Question: {query}
Intent: {intent_reasoning}"""
