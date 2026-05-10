"""
CLIP-based scoring utilities.

Handles:
- Pre-computing CLIP embeddings for VLM vocabulary (one-time, offline)
- Fast cosine similarity via matrix multiplication at inference (~1ms)
- Vocabulary cache management (clip_vocab_cache.pt)

Addresses Risk 1 (Section 5) and Risk 3 (CLIP vs LLaVA tokenizer mismatch).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPScorer:
    """CLIP visual-text scorer.

    Encodes an image once at construction, then scores word lists by
    cosine similarity against the stored image embedding.

    For full-vocabulary scoring at inference, pre-compute embeddings with
    scripts/precompute_clip_vocab.py and pass vocab_cache_path here
    to enable ~1ms matrix-multiply instead of per-word encoding.
    """

    def __init__(
        self,
        image: Image.Image,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        vocab_cache_path: Optional[Path | str] = None,
        _preloaded: Optional[dict] = None,
    ):
        """
        Args:
            image: PIL image to score words against.
            model_name: open_clip model identifier (ViT-B-32 for speed,
                ViT-L-14 for quality).
            pretrained: open_clip pretrained weights name.
            device: Torch device string; auto-detected if None.
            vocab_cache_path: Path to pre-computed vocab embeddings file
                (produced by scripts/precompute_clip_vocab.py).
            _preloaded: Internal — dict with keys model, preprocess, tokenizer,
                vocab_embeddings, vocab_words to skip re-loading.
        """
        import open_clip  # deferred so unit tests can mock without loading CLIP

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if _preloaded is not None:
            self._model     = _preloaded["model"]
            self._tokenizer = _preloaded["tokenizer"]
            preprocess      = _preloaded["preprocess"]
            self._vocab_embeddings = _preloaded.get("vocab_embeddings")
            self._vocab_words      = _preloaded.get("vocab_words")
            self._vocab_index      = _preloaded.get("vocab_index")
        else:
            logger.info("Loading CLIP %s/%s on %s", model_name, pretrained, self.device)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._model = model.to(self.device).eval()
            self._tokenizer = open_clip.get_tokenizer(model_name)

            self._vocab_embeddings = None
            self._vocab_words      = None
            self._vocab_index      = None

            if vocab_cache_path:
                cache_path = Path(vocab_cache_path)
                if cache_path.exists():
                    cached = torch.load(cache_path, map_location=self.device, weights_only=True)
                    self._vocab_embeddings = F.normalize(
                        cached["embeddings"].to(self.device).float(), dim=-1
                    )  # [V, D]
                    self._vocab_words = cached["words"]
                    self._vocab_index = {w: i for i, w in enumerate(self._vocab_words)}
                    logger.info("Loaded vocab cache: %d words", len(self._vocab_words))
                else:
                    logger.warning("Vocab cache not found at %s — live encoding will be used", cache_path)

        # Stash preprocess so a sibling BboxClipScorer can share the transform.
        self._preprocess = preprocess

        with torch.no_grad():
            img_tensor = preprocess(image).unsqueeze(0).to(self.device)
            feat = self._model.encode_image(img_tensor)
            self._image_feat = F.normalize(feat, dim=-1)  # [1, D]

        # Per-image CLIP baseline: mean and std of cosine over the vocab cache.
        # Used to z-score-center scores so "above-average for THIS image" → > 0.5
        # and "below average" → < 0.5. Without this, raw cosine is always ~0.2-0.3
        # for every word and the (sims+1)/2 rescale yields a uniform positive bias.
        self._image_baseline_mean: float | None = None
        self._image_baseline_std:  float | None = None
        if self._vocab_embeddings is not None:
            with torch.no_grad():
                all_sims = (self._vocab_embeddings @ self._image_feat.T).squeeze(-1)
            self._image_baseline_mean = float(all_sims.mean().item())
            self._image_baseline_std  = float(all_sims.std().item())

    # ── Public API ───────────────────────────────────────────────────────

    def score_words(self, words: list[str]) -> torch.Tensor:
        """Encode words and return calibrated similarity in [0, 1].

        Uses an image-specific empirical baseline (mean+std of cosine over the
        vocab cache) so that "above-average for this image" maps to > 0.5 and
        "below average" maps to < 0.5. Falls back to (sims+1)/2 when no cache.
        """
        if not words:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        # Try fast cache lookup first
        if self._vocab_index is not None and self._vocab_embeddings is not None:
            indices = [self._vocab_index.get(w) for w in words]
            if all(i is not None for i in indices):
                cached_feats = self._vocab_embeddings[indices]  # [N, D]
                sims = (cached_feats @ self._image_feat.T).squeeze(-1)  # [N]
                return self._calibrate(sims)

        return self._encode_and_score(words)

    def score_single(self, word: str) -> float:
        """Score a single word. Returns float in [0, 1]."""
        return self.score_words([word]).item()

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the CLIP embedding space."""
        return self._image_feat.shape[-1]

    def to_preloaded(self) -> dict:
        """Return a ``_preloaded`` dict for sibling scorers (e.g. BboxClipScorer).

        Lets us spin up a bbox-conditioned scorer for the same image without
        reloading CLIP weights or the vocab cache.
        """
        return {
            "model": self._model,
            "tokenizer": self._tokenizer,
            "preprocess": self._preprocess,
            "vocab_embeddings": self._vocab_embeddings,
            "vocab_words": self._vocab_words,
            "vocab_index": self._vocab_index,
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _encode_and_score(self, words: list[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self._tokenizer(words).to(self.device)
            text_feats = F.normalize(self._model.encode_text(tokens), dim=-1)  # [N, D]
        sims = (text_feats @ self._image_feat.T).squeeze(-1)  # [N]
        return self._calibrate(sims)

    def _calibrate(self, sims: torch.Tensor) -> torch.Tensor:
        """Map raw cosine similarities to [0, 1] with the image-specific baseline.

        With baseline:  0.5 + 0.15 * (sims - mean) / max(std, 0.01), clamped.
                        +1σ → ~0.65, +2σ → ~0.80, perfect match (~3σ) → ~0.95.
        Without:        fallback (sims+1)/2 — same as legacy behavior.
        """
        if self._image_baseline_mean is None or self._image_baseline_std is None:
            return ((sims + 1) / 2).clamp(0.0, 1.0)
        std = max(self._image_baseline_std, 0.01)
        z = (sims - self._image_baseline_mean) / std
        return (0.5 + 0.15 * z).clamp(0.0, 1.0)

    def encode_words(self, words: list[str]) -> torch.Tensor:
        """Return normalized text embeddings for a word list.

        Shape: [N, D]. Used by precompute_clip_vocab.py.
        """
        if not words:
            return torch.zeros(0, self.embedding_dim, device=self.device)
        with torch.no_grad():
            tokens = self._tokenizer(words).to(self.device)
            feats = self._model.encode_text(tokens)
        return F.normalize(feats, dim=-1)
