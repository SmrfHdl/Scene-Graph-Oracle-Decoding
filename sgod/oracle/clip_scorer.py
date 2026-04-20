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
        """
        import open_clip  # deferred so unit tests can mock without loading CLIP

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        logger.info("Loading CLIP %s/%s on %s", model_name, pretrained, self.device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            img_tensor = preprocess(image).unsqueeze(0).to(self.device)
            feat = self._model.encode_image(img_tensor)
            self._image_feat = F.normalize(feat, dim=-1)  # [1, D]

        self._vocab_embeddings: Optional[torch.Tensor] = None
        self._vocab_words: Optional[list[str]] = None
        self._vocab_index: Optional[dict[str, int]] = None

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

    # ── Public API ───────────────────────────────────────────────────────

    def score_words(self, words: list[str]) -> torch.Tensor:
        """Encode words and return cosine similarities with the image.

        Returns:
            Float tensor of shape [N] with values in [0, 1].
            Uses vocab cache when all words are present; otherwise encodes live.
        """
        if not words:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        # Try fast cache lookup first
        if self._vocab_index is not None and self._vocab_embeddings is not None:
            indices = [self._vocab_index.get(w) for w in words]
            if all(i is not None for i in indices):
                cached_feats = self._vocab_embeddings[indices]  # [N, D]
                sims = (cached_feats @ self._image_feat.T).squeeze(-1)  # [N]
                return ((sims + 1) / 2).clamp(0.0, 1.0)

        return self._encode_and_score(words)

    def score_single(self, word: str) -> float:
        """Score a single word. Returns float in [0, 1]."""
        return self.score_words([word]).item()

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the CLIP embedding space."""
        return self._image_feat.shape[-1]

    # ── Internal ─────────────────────────────────────────────────────────

    def _encode_and_score(self, words: list[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self._tokenizer(words).to(self.device)
            text_feats = F.normalize(self._model.encode_text(tokens), dim=-1)  # [N, D]
        sims = (text_feats @ self._image_feat.T).squeeze(-1)  # [N]
        return ((sims + 1) / 2).clamp(0.0, 1.0)

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
