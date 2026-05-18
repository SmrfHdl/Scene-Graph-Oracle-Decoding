"""sgod.backbones — VLM wrappers conforming to the `Backbone` interface.

Currently implemented:
- LLaVAv15Backbone — LLaVA-1.5-7B (Phase 1 target)

Planned (Phase 3):
- Qwen25VLBackbone   — Qwen 2.5-VL-7B-Instruct
- InternVL25Backbone — InternVL 2.5-8B
"""
from sgod.backbones.llava15 import LLaVAv15Backbone

__all__ = ["LLaVAv15Backbone"]
