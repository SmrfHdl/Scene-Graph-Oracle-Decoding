"""Shared types for the SGOD framework.

`SceneGraph` is re-exported from `sgod.sgg.scene_graph` (the canonical location)
so that downstream code can import it from the neutral `sgod.core` namespace
without coupling to the RelTR-specific package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from torch import Tensor

from sgod.sgg.scene_graph import (
    AttributeNode,
    ObjectNode,
    RelationEdge,
    SceneGraph,
)


@dataclass
class OracleEvidence:
    """Aggregated belief produced by an Oracle.

    For now this is just a SceneGraph; reserved as a dataclass so future
    oracles (e.g., dual-vision ensemble) can attach metadata without breaking
    the Policy interface.
    """
    scene_graph: SceneGraph
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationState:
    """Runtime state passed through every decode step.

    Built by the orchestrator that drives Backbone + Policy.

    Fields:
        prompt_ids:       [B, L_prompt] token ids of the user prompt.
        generated_ids:    [B, t]        tokens decoded so far.
        hidden_states:    [B, d]        last hidden state from the Backbone at step t.
        lm_logits:        [B, V]        raw LM logits at step t (before policy adjustment).
        evidence:         OracleEvidence (scene graph + extras), built once per image.
        step:             current decode step index (0-based).
        policy_state:     opaque dict owned by the Policy for slot states, gate counters, etc.
        rule_anchor_type: optional anchor label from rule-based detector (used as ATG warmup signal).
        hidden_buffer:    rolling list of the last K_t backbone hidden states
                          (each [B, d]), oldest first, newest last. Maintained by the
                          orchestrator. Policies that want a prefix-summary read this
                          (DT-SGOD does so to feed the slow module). May be None
                          for orchestrators that do not maintain a buffer.
    """
    prompt_ids: Tensor
    generated_ids: Tensor
    hidden_states: Tensor
    lm_logits: Tensor
    evidence: OracleEvidence
    step: int
    policy_state: dict[str, Any] = field(default_factory=dict)
    rule_anchor_type: Optional[str] = None
    hidden_buffer: Optional[list] = None


__all__ = [
    "AttributeNode",
    "GenerationState",
    "ObjectNode",
    "OracleEvidence",
    "RelationEdge",
    "SceneGraph",
]
