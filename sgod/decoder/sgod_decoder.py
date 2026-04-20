"""
SGODDecoder — Component 5 (Section 3.6).

Main integration class that combines all components:
    1. SGGModule      → extract scene graph (one-time)
    2. VisualOracle   → build oracle from SG + CLIP (one-time)
    3. AnchorDetector → detect anchor positions (per token)
    4. GenerationCtx  → track context, compute adaptive lambda (per token)

Key method:
    generate(image, question, max_new_tokens=256) -> str

Single forward pass per token step (no 2x compute like VCD/ICD).
Applies: logit_final = logit_lm + lambda * oracle_scores
"""
