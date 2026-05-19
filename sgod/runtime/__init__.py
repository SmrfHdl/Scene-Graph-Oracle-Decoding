"""sgod.runtime — orchestrator that drives Backbone + Oracle + Policy.

The runtime is intentionally thin: it owns the decode loop and the
GenerationState plumbing, while every model-specific behavior lives behind
the three ABCs in `sgod.core`.
"""
from sgod.runtime.builder import build_from_config, load_config
from sgod.runtime.decoder import HallucinationDecoder
from sgod.runtime.trace_collector import StepTrace, TraceCollector

__all__ = [
    "HallucinationDecoder",
    "StepTrace",
    "TraceCollector",
    "build_from_config",
    "load_config",
]
