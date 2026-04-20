"""
GenerationContext — Component 4 (Section 3.5).

Tracks generation state to adaptively adjust oracle lambda:
- Negation detection and depth tracking (decay-based)
- Question type classification (existential, descriptive, comparative, hypothetical)
- Context-aware lambda computation

Key class: GenerationContext
    - update(new_token) -> None
    - get_lambda() -> float
"""
