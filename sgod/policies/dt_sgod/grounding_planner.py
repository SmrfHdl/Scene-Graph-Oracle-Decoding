"""Grounding Planner (slow module).

Per-fire computation:

    1. Featurize each object in the scene graph into a [embed_dim + 5] vector
       (label embedding ⊕ normalized bbox ⊕ confidence).
    2. Project to hidden_dim via a learned linear layer → g_v ∈ ℝ^{N × d}.
    3. Slot attention: each of S slots soft-attends over the N nodes.
    4. GRU update per slot: input = [context, prefix_summary] → new h_slow_s.

Label embeddings come from a pluggable `LabelEmbedder` (see
`sgod.policies.dt_sgod.label_embedders`). The default is `HashLabelEmbedder`,
which yields md5-seeded random vectors — cheap, deterministic, dependency-
free, but carries no semantic structure. Pass a `CLIPTextLabelEmbedder` to
get open-vocab CLIP text features; the GP's `node_proj` layer absorbs any
dim mismatch automatically.

Relations / edges are not yet consumed — slot attention runs over node
features only. Adding a GAT layer with edge-conditioned aggregation is a
Week-2 enhancement.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from sgod.core.types import OracleEvidence
from sgod.policies.dt_sgod.config import DTSGODConfig
from sgod.policies.dt_sgod.label_embedders import (
    HashLabelEmbedder,
    LabelEmbedder,
)

# Bbox(4) + confidence(1) appended to the label embedding before projection.
_BBOX_CONF_DIM = 5


class GroundingPlanner(nn.Module):
    """Slow module: produces an evolving grounding state h_slow ∈ ℝ^[B, S, d]."""

    def __init__(
        self,
        hidden_dim: int,
        config: DTSGODConfig,
        node_embed_dim: int = 128,
        label_embedder: LabelEmbedder | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        self.num_slots = config.num_slots
        # `label_embedder.embed_dim` is the authoritative source of the label
        # feature dimensionality. `node_embed_dim` is only used to build the
        # default HashLabelEmbedder when no embedder is supplied.
        self.label_embedder: LabelEmbedder = (
            label_embedder if label_embedder is not None
            else HashLabelEmbedder(embed_dim=node_embed_dim)
        )
        self.node_embed_dim = self.label_embedder.embed_dim

        # Project [label_embed ⊕ bbox(4) ⊕ confidence(1)] → hidden_dim.
        self.node_proj = nn.Linear(self.node_embed_dim + _BBOX_CONF_DIM, hidden_dim, bias=False)

        # Diverse per-slot initialization. Each slot is its own learnable vector.
        # Week 2: optionally init from K-means clustering of SG node embeddings.
        self.slot_init = nn.Parameter(torch.randn(config.num_slots, hidden_dim) * 0.02)

        # Slot attention — slots are queries, nodes are keys/values.
        self.q_slot = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_node = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # GRU update: concatenate [context_s, prefix_summary] → project → GRUCell.
        self.input_proj = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # The embedder owns its own cache; nothing more to set up here.

    # ── public API ────────────────────────────────────────────────────────

    def init_state(
        self,
        evidence: OracleEvidence,
        batch_size: int,
        device: str | torch.device,
    ) -> Tensor:
        """Build the initial h_slow tensor [B, S, d] for a fresh generation."""
        del evidence  # Week-2 may use SG nodes to bias the init; not needed yet.
        init = self.slot_init.to(device)
        return init.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def forward(
        self,
        h_slow_prev: Tensor,           # [B, S, d]
        prefix_summary: Tensor,        # [B, d]
        evidence: OracleEvidence,
    ) -> Tensor:
        """Advance h_slow by one slow step.

        On an empty scene graph (no detected objects) we deliberately return
        h_slow_prev unchanged — there's no evidence to integrate, so any
        update would inject noise. This matches the variational reading
        (posterior = prior when there's nothing to condition on).
        """
        B, S, d = h_slow_prev.shape
        device = h_slow_prev.device

        # Build node feature matrix [N, node_embed_dim + 5] from the scene graph.
        node_raw = self._build_node_features(evidence, device, h_slow_prev.dtype)
        N = node_raw.shape[0]
        if N == 0:
            return h_slow_prev

        # Project to hidden_dim.
        g = self.node_proj(node_raw)                             # [N, d]

        # Slot attention: each slot queries all nodes.
        q = self.q_slot(h_slow_prev)                              # [B, S, d]
        k = self.k_node(g)                                        # [N, d]
        v = self.v_node(g)                                        # [N, d]

        scale = 1.0 / (d ** 0.5)
        scores = torch.einsum("bsd,nd->bsn", q, k) * scale        # [B, S, N]
        attn = torch.softmax(scores, dim=-1)                       # [B, S, N]
        context = torch.einsum("bsn,nd->bsd", attn, v)             # [B, S, d]

        # GRU update per slot, with prefix_summary broadcast across slots.
        prefix_expand = prefix_summary.unsqueeze(1).expand(-1, S, -1)   # [B, S, d]
        gru_input = self.input_proj(torch.cat([context, prefix_expand], dim=-1))  # [B, S, d]

        # Flatten batch × slot, run GRU, then reshape back.
        gru_input_flat = gru_input.reshape(B * S, d)
        h_prev_flat = h_slow_prev.reshape(B * S, d)
        h_new_flat = self.gru(gru_input_flat, h_prev_flat)
        return h_new_flat.reshape(B, S, d)

    # ── featurization ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _featurize_label(self, label: str) -> Tensor:
        """Delegate to the configured `LabelEmbedder`.

        The returned tensor lives on CPU; `_build_node_features` handles the
        move to the active device. Determinism + caching are the embedder's
        responsibility (see `LabelEmbedder.embed`).
        """
        return self.label_embedder.embed(label)

    @torch.no_grad()
    def _build_node_features(
        self,
        evidence: OracleEvidence,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Stack per-object features into [N, node_embed_dim + 5]."""
        sg = evidence.scene_graph
        if not sg.objects:
            return torch.zeros(0, self.node_embed_dim + _BBOX_CONF_DIM, device=device, dtype=dtype)

        img_w, img_h = sg.image_size if sg.image_size is not None else (1.0, 1.0)
        img_w = max(float(img_w), 1.0)
        img_h = max(float(img_h), 1.0)

        rows: list[Tensor] = []
        for obj in sg.objects:
            label_feat = self._featurize_label(obj.label)               # [E]
            bbox_norm = torch.tensor(
                [obj.bbox[0] / img_w, obj.bbox[1] / img_h,
                 obj.bbox[2] / img_w, obj.bbox[3] / img_h],
                dtype=label_feat.dtype,
            )
            conf = torch.tensor([obj.confidence], dtype=label_feat.dtype)
            rows.append(torch.cat([label_feat, bbox_norm, conf], dim=0))

        return torch.stack(rows, dim=0).to(device=device, dtype=dtype)
