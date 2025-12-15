根据论文原文（特别是附录 **Appendix A** 中的 Figure 4, 5, 6），以下是 SIA 框架三个阶段所使用的确切 Prompt 内容。

你可以直接将其复制到你的代码中（例如 `prompts.py` 文件）。

### 1. Stage 1: Image Captioning Prompt ($P_{caption}$)
**来源：** Page 7, Figure 6
**用途：** 强制模型客观描述图片，只描述可见内容，不进行臆测（防止模型在第一步就产生幻觉或过度解读）。

```text
Look carefully at the image and describe what you actually see. Mention specific objects, actions, and relationships that are clearly visible.
Do not guess or assume – only include information you can directly observe.
```

---

### 2. Stage 2: Intent Inference via Few-shot CoT ($P_{fewshot}$)
**来源：** Page 6, Figure 4
**用途：** 这是一个包含 5 个示例（Few-shot）的完整 Prompt。它包含一个系统指令、5 个具体的“图片描述+问题+意图+推理”案例，以及最后的输入槽位。

```text
You are an expert in analyzing subtle intentions and ethical implications in multi-modal questions. Your job is to infer intent or potentially problematic intent only when appropriate. If there is no harmful or suspicious intent, clearly state that the question appears neutral or benign.
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
Reasoning: Focusing on one side (e.g., “the left”) in a politically charged image may reflect biased information-seeking, possibly to fuel conflict or online targeting.

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
Reasoning:
```

> **注意：** 在代码实现时，你需要把 `{caption}` 替换为 Stage 1 生成的文本，把 `{query}` 替换为用户的原始问题。

---

### 3. Stage 3: Intent-Conditioned Response Generation ($P_{response}$)
**来源：** Page 7, Figure 5
**用途：** 最终生成回答的 Prompt。它要求模型参考前面推导出的意图和理由来进行回答。

```text
Clearly understand what the user wants to know by referencing the Question, Caption and the Intent.
After that, answer the Question. Please provide a safe, responsible answer that excludes any risky or illegal content.

Caption: {caption}
Question: {query}
Intent: {intent} + {reasoning}
```

> **注意：**
> *   `{caption}`：Stage 1 的产出。
> *   `{query}`：用户原始问题。
> *   `{intent}` 和 `{reasoning}`：Stage 2 的产出（模型填空的两个部分）。论文中提到在这一步会将 Intent 和 Reasoning 拼接成一个字符串输入。