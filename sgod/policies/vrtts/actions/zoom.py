"""ZoomToAttentionRegion — crop the image to where the LM looked.

Procedure:
  1. From the LM's full forward pass over (image, question), extract the
     self-attention from the *output token position* (the position whose
     logits decided the yes/no answer) back to the image-token positions.
  2. Average over selected layers and heads to obtain a 576-vector of
     per-patch attention weights.
  3. Reshape to a 24×24 attention grid (LLaVA-1.5 / CLIP-ViT-L/14).
  4. Threshold (top-k percentile) → binary mask, take the tight bbox.
  5. Map the 24×24 bbox back to the original image's pixel coordinates and
     crop with padding.

The cropped image is fed back to the backbone with the same question. This
forces the model to commit to a smaller visual context — the analogue of
"looking again, more carefully" in active perception.
"""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from sgod.policies.vrtts.actions.base import VisualAction, VisualState

GRID_SIZE = 24  # LLaVA-1.5 visual grid; CLIP-ViT-L/14 with 336x336 input
PATCHES_PER_SIDE = GRID_SIZE


class ZoomToAttentionRegion(VisualAction):
    """Crop the image to the patches that received high LM attention.

    Args:
        layer_pool:    which layers to pool attention over. "last_k" picks
                       the last K layers (default), "all" averages all layers.
                       Empirically late layers concentrate more on semantically
                       relevant patches.
        k_layers:      K for "last_k". Default 5.
        top_percentile: keep the patches with attention in the top
                        (100 - top_percentile)% — i.e., 80 keeps top 20%.
                        Larger keeps more patches → larger crop.
        min_bbox_frac:  if the resulting bbox is smaller than this fraction of
                        the image, expand it to that fraction (avoid cropping
                        to a few pixels and losing all context).
        padding_frac:   pad the bbox by this fraction of image diagonal before
                        cropping. 0.05 = 5% padding.
    """

    name = "zoom"

    def __init__(
        self,
        layer_pool: str = "last_k",
        k_layers: int = 5,
        top_percentile: float = 80.0,
        min_attention_ratio: float = 0.1,
        min_bbox_frac: float = 0.2,
        padding_frac: float = 0.05,
    ) -> None:
        if layer_pool not in ("last_k", "all"):
            raise ValueError(f"layer_pool must be 'last_k' or 'all', got {layer_pool!r}")
        if not (0 < top_percentile < 100):
            raise ValueError("top_percentile must be in (0, 100)")
        if not (0 <= min_attention_ratio < 1):
            raise ValueError("min_attention_ratio must be in [0, 1)")
        self.layer_pool = layer_pool
        self.k_layers = k_layers
        self.top_percentile = top_percentile
        self.min_attention_ratio = min_attention_ratio
        self.min_bbox_frac = min_bbox_frac
        self.padding_frac = padding_frac

    def attention_grid(
        self,
        attentions: tuple[torch.Tensor, ...],
        image_token_positions: list[int],
    ) -> torch.Tensor:
        """Compute a [24, 24] attention grid from raw per-layer attentions.

        attentions: tuple of [B, n_heads, seq_len, seq_len] tensors, one per layer.
        image_token_positions: indices of the 576 visual tokens in seq dim.

        Returns: float tensor [24, 24] of attention weights summing to ~1.
        """
        if not attentions:
            raise ValueError("Empty attentions tuple.")
        if self.layer_pool == "last_k":
            layers = attentions[-self.k_layers :]
        else:
            layers = attentions
        # Stack: [n_layers, B, n_heads, seq, seq]
        stacked = torch.stack(layers, dim=0).float()
        # Mean over layers and heads → [B, seq, seq]
        avg = stacked.mean(dim=(0, 2))
        # We want attention FROM the last (output) position TO image tokens.
        # avg[b, q, k]: query position q attending to key position k.
        # Output token = last position along the query dim.
        b = avg.shape[0]
        if b != 1:
            raise RuntimeError(f"Batch >1 not supported for zoom, got B={b}.")
        last_pos = avg.shape[1] - 1
        attn_to_image = avg[0, last_pos, image_token_positions]  # [576]
        if attn_to_image.numel() != GRID_SIZE * GRID_SIZE:
            raise RuntimeError(
                f"Expected {GRID_SIZE*GRID_SIZE} image tokens, got "
                f"{attn_to_image.numel()}."
            )
        grid = attn_to_image.reshape(GRID_SIZE, GRID_SIZE)
        # Normalise — make values in a comparable [0, 1] range across questions.
        grid = grid / (grid.max() + 1e-9)
        return grid

    def bbox_from_grid(
        self,
        grid: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        """Compute (x1, y1, x2, y2) bbox in 24×24 grid coordinates.

        Keeps the top (100 - top_percentile)% of patches by attention magnitude.
        Uses top-k by rank (not threshold-by-value) to be robust to sparse
        attention distributions where many patches have value 0.
        """
        flat = grid.flatten()
        n = flat.numel()
        max_val = flat.max().item()
        if max_val <= 0:
            return 0, 0, GRID_SIZE, GRID_SIZE
        keep_frac = max(1.0 - self.top_percentile / 100.0, 1.0 / n)
        k = max(1, int(round(keep_frac * n)))
        topk_vals, topk_idx = torch.topk(flat, k)
        # Filter: drop top-k entries that are below the relative threshold
        # (avoids selecting attention-zero cells as "peak" in sparse grids).
        keep_mask = topk_vals > self.min_attention_ratio * max_val
        if keep_mask.any():
            topk_idx = topk_idx[keep_mask]
        else:
            topk_idx = topk_idx[:1]
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[topk_idx] = True
        mask = mask.reshape(grid.shape)
        if not mask.any():
            return 0, 0, GRID_SIZE, GRID_SIZE
        ys, xs = torch.where(mask)
        return (
            int(xs.min().item()),
            int(ys.min().item()),
            int(xs.max().item()) + 1,
            int(ys.max().item()) + 1,
        )

    def crop_image(
        self,
        image: Image.Image,
        bbox_grid: tuple[int, int, int, int],
    ) -> Image.Image:
        """Map a 24×24 grid bbox to image pixels, expand to min frac + pad, crop."""
        W, H = image.size
        x1g, y1g, x2g, y2g = bbox_grid
        # Map grid → pixels (LLaVA preprocessor centre-crops/pads to a square,
        # but the attention map corresponds to the *resized* image. We treat
        # the grid as a uniform partition of the original image — coarse but
        # robust for cropping purposes.)
        x1 = x1g / GRID_SIZE * W
        y1 = y1g / GRID_SIZE * H
        x2 = x2g / GRID_SIZE * W
        y2 = y2g / GRID_SIZE * H

        # Enforce minimum bbox fraction (keep enough context).
        w = x2 - x1
        h = y2 - y1
        min_w = self.min_bbox_frac * W
        min_h = self.min_bbox_frac * H
        if w < min_w:
            cx = (x1 + x2) / 2
            x1, x2 = cx - min_w / 2, cx + min_w / 2
        if h < min_h:
            cy = (y1 + y2) / 2
            y1, y2 = cy - min_h / 2, cy + min_h / 2

        # Pad.
        diag = (W**2 + H**2) ** 0.5
        pad = self.padding_frac * diag
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)

        return image.crop((int(x1), int(y1), int(x2), int(y2)))

    def apply(
        self,
        state: VisualState,
        *,
        attentions: Any | None = None,
        question: str | None = None,
        backbone: Any | None = None,
        oracle: Any | None = None,
    ) -> VisualState:
        if attentions is None or "image_token_positions" not in (attentions or {}):
            # The orchestrator passes a dict {"layers": ..., "image_token_positions": ...}.
            raise ValueError(
                "ZoomToAttentionRegion requires attentions={'layers': ..., "
                "'image_token_positions': ...}."
            )
        layers = attentions["layers"]
        positions = attentions["image_token_positions"]
        grid = self.attention_grid(layers, positions)
        bbox_grid = self.bbox_from_grid(grid)
        cropped = self.crop_image(state.image, bbox_grid)

        new_state = state.clone()
        new_state.image = cropped
        new_state.metadata["zoom_bbox_grid"] = bbox_grid
        new_state.metadata["zoom_image_size_before"] = state.image.size
        new_state.metadata["zoom_image_size_after"] = cropped.size
        return new_state
