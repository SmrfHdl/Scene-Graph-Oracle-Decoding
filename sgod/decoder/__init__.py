"""
SGOD Decoder — Component 5 (Section 3.6).
Main integration module: drop-in replacement for VLM's generate() method.
"""

from sgod.decoder.sgod_decoder import SGODDecoder
from sgod.decoder.token_utils import apply_oracle_scores, decode_top_k

__all__ = [
    "SGODDecoder",
    "decode_top_k",
    "apply_oracle_scores",
]
