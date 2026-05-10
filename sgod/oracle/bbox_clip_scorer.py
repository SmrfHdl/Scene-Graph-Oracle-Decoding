"""
BboxClipScorer — bbox-conditioned CLIP scoring (M1.5 of BG-SGOD).

Why this exists
---------------
Image-level CLIP scoring (``CLIPScorer``) assigns a single similarity score
between a token and the entire image. That works for "is concept X in this
photo" but fails for **attribute attribution**:

    Q: "What color is the pillow?"
    GT: yellow

Image-level CLIP cannot say *which* object is yellow — many images contain
several colors, and the global similarity for "yellow" reflects a mixture.
On the MMHal regression set this was the dominant failure mode.

The fix is to crop the image at the question's target bbox (e.g., the
"pillow" object's bbox from Grounding DINO), pad it slightly for context,
and score tokens against the crop instead of the whole image. This is the
standard approach in RegionCLIP / Crop-CLIP / SoftCLIP literature.

Design
------
Construction is cheap: we share the open_clip model + tokenizer + preprocess
with ``CLIPScorer`` via a ``_preloaded`` dict, so a single image pass loads
weights only once.

Crops are encoded **lazily and cached** by rounded bbox key, so callers can
freely call ``score_token_at_bbox`` for many (token, bbox) pairs without
repeated forward passes.

Each crop gets its **own calibration baseline** (mean + std of cosine over
the vocab cache, computed against THAT crop's feature). A "yellow" pillow
crop and a "metal kitchen" crop have very different cosine distributions;
without per-crop calibration, raw scores are not comparable.

Usage
-----
    >>> scorer = BboxClipScorer(image, _preloaded=oracle._clip_preloaded)
    >>> for tok in ["yellow", "blue", "red"]:
    ...     score = scorer.score_token_at_bbox(tok, pillow_bbox)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]


class BboxClipScorer:
    """Bbox-conditioned CLIP scorer with per-crop calibration.

    Encodes image crops on demand and caches their features so that scoring
    many tokens against the same bbox is cheap.

    Args:
        image:           PIL image to crop from.
        pad_ratio:       Fraction of bbox width/height to pad on each side.
                         0.15 = +30% total per dimension (RegionCLIP default-ish).
        model_name:      open_clip model id (must match _preloaded if provided).
        pretrained:      open_clip pretrained weights name.
        device:          Torch device; auto-detected if None.
        _preloaded:      Internal — dict from CLIPScorer with model/tokenizer/
                         preprocess/vocab_embeddings/vocab_words/vocab_index.
                         Strongly recommended to share with the host VisualOracle's
                         CLIPScorer so weights load once.
    """

    def __init__(
        self,
        image: Image.Image,
        pad_ratio: float = 0.15,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        _preloaded: Optional[dict] = None,
    ):
        import open_clip  # deferred so unit tests can mock without loading CLIP

        if not 0.0 <= pad_ratio <= 1.0:
            raise ValueError(f"pad_ratio must be in [0, 1], got {pad_ratio}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._image = image
        self._pad_ratio = pad_ratio

        if _preloaded is not None:
            self._model       = _preloaded["model"]
            self._tokenizer   = _preloaded["tokenizer"]
            self._preprocess  = _preloaded["preprocess"]
            self._vocab_embeddings = _preloaded.get("vocab_embeddings")
            self._vocab_words      = _preloaded.get("vocab_words")
            self._vocab_index      = _preloaded.get("vocab_index")
        else:
            logger.info("Loading CLIP %s/%s on %s (BboxClipScorer)",
                        model_name, pretrained, self.device)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._model = model.to(self.device).eval()
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._preprocess = preprocess
            self._vocab_embeddings = None
            self._vocab_words = None
            self._vocab_index = None

        # Per-bbox cache: bbox_key → (crop_feat [1,D], baseline_mean, baseline_std)
        self._crop_cache: dict[tuple, tuple[torch.Tensor, float, float]] = {}
        # Per-token cache: token → text_feat [1,D]. Reused across bboxes.
        self._token_cache: dict[str, torch.Tensor] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def score_token_at_bbox(self, token: str, bbox: Bbox) -> float:
        """Calibrated similarity in [0, 1] between ``token`` and the bbox crop.

        0.5 = average match for this crop, > 0.5 = better than average,
        < 0.5 = worse than average. See ``_calibrate`` for the mapping.
        """
        return self.score_tokens_at_bbox([token], bbox)[0].item()

    def score_tokens_at_bbox(self, tokens: list[str], bbox: Bbox) -> torch.Tensor:
        """Vectorised version. Returns a tensor of shape [len(tokens)]."""
        if not tokens:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        crop_feat, base_mean, base_std = self._get_crop_feat(bbox)
        text_feats = self._get_text_feats(tokens)              # [N, D]
        sims = (text_feats @ crop_feat.T).squeeze(-1)           # [N]
        return self._calibrate(sims, base_mean, base_std)

    def score_token_max_across_bboxes(
        self, token: str, bboxes: list[Bbox]
    ) -> float:
        """Max calibrated similarity across multiple candidate target bboxes.

        Used when the question parser yields several targets and we don't
        know which one the attribute attaches to ("Is the dog on the table?"
        + adjective "wooden" — wooden is more likely table than dog, but
        without dependency parsing we just take the best match).
        """
        if not bboxes:
            return 0.5  # no target → neutral
        scores = [self.score_token_at_bbox(token, bb) for bb in bboxes]
        return max(scores)

    def score_tokens_max_across_bboxes(
        self, tokens: list[str], bboxes: list[Bbox]
    ) -> torch.Tensor:
        """Batched: per-token max over bboxes. Shape [len(tokens)]."""
        if not tokens:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        if not bboxes:
            return torch.full((len(tokens),), 0.5, device=self.device)
        # Stack scores across bboxes [B, N], then max over B.
        per_bbox = torch.stack([
            self.score_tokens_at_bbox(tokens, bb) for bb in bboxes
        ], dim=0)
        return per_bbox.max(dim=0).values

    # ── Internal ─────────────────────────────────────────────────────────

    def _get_crop_feat(
        self, bbox: Bbox
    ) -> tuple[torch.Tensor, float, float]:
        """Encode (and cache) the padded crop at ``bbox``. Returns (feat, mean, std)."""
        key = self._bbox_key(bbox)
        cached = self._crop_cache.get(key)
        if cached is not None:
            return cached

        crop = self._padded_crop(bbox)
        with torch.no_grad():
            tensor = self._preprocess(crop).unsqueeze(0).to(self.device)
            feat = F.normalize(self._model.encode_image(tensor), dim=-1)  # [1, D]

            mean, std = 0.0, 1.0
            if self._vocab_embeddings is not None:
                sims = (self._vocab_embeddings @ feat.T).squeeze(-1)
                mean = float(sims.mean().item())
                std  = float(sims.std().item())

        self._crop_cache[key] = (feat, mean, std)
        return feat, mean, std

    def _get_text_feats(self, tokens: list[str]) -> torch.Tensor:
        """Encode (and cache) text features for a list of tokens. Shape [N, D]."""
        # Vocab-cache fast path: every token in cache.
        if self._vocab_index is not None and self._vocab_embeddings is not None:
            indices = [self._vocab_index.get(t) for t in tokens]
            if all(i is not None for i in indices):
                return self._vocab_embeddings[indices]

        # Per-token cache + encode missing ones in a single batch.
        missing = [t for t in tokens if t not in self._token_cache]
        if missing:
            with torch.no_grad():
                ids = self._tokenizer(missing).to(self.device)
                feats = F.normalize(self._model.encode_text(ids), dim=-1)
            for tok, feat in zip(missing, feats):
                self._token_cache[tok] = feat.unsqueeze(0)
        return torch.cat([self._token_cache[t] for t in tokens], dim=0)

    def _padded_crop(self, bbox: Bbox) -> Image.Image:
        """Crop ``self._image`` at ``bbox`` with ``pad_ratio`` padding per side."""
        x1, y1, x2, y2 = bbox
        w_img, h_img = self._image.size
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        pad_x = bw * self._pad_ratio
        pad_y = bh * self._pad_ratio
        cx1 = max(0,    int(round(x1 - pad_x)))
        cy1 = max(0,    int(round(y1 - pad_y)))
        cx2 = min(w_img, int(round(x2 + pad_x)))
        cy2 = min(h_img, int(round(y2 + pad_y)))
        # Defensive: if bbox is outside image entirely, fall back to image.
        if cx2 <= cx1 or cy2 <= cy1:
            return self._image
        return self._image.crop((cx1, cy1, cx2, cy2))

    @staticmethod
    def _bbox_key(bbox: Bbox) -> tuple[int, int, int, int]:
        """Round bbox to ints for cache hashing (sub-pixel jitter ignored)."""
        x1, y1, x2, y2 = bbox
        return (round(x1), round(y1), round(x2), round(y2))

    def _calibrate(
        self, sims: torch.Tensor, mean: float, std: float
    ) -> torch.Tensor:
        """Map raw cosines to [0, 1] using the per-crop baseline.

        Mirrors CLIPScorer._calibrate but with a crop-specific baseline:
        0.5 + 0.15 * z, clamped — +1σ ≈ 0.65, +2σ ≈ 0.80.
        Falls back to (sims+1)/2 if no vocab cache is available.
        """
        if self._vocab_embeddings is None:
            return ((sims + 1) / 2).clamp(0.0, 1.0)
        z = (sims - mean) / max(std, 0.01)
        return (0.5 + 0.15 * z).clamp(0.0, 1.0)

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def num_cached_crops(self) -> int:
        return len(self._crop_cache)
