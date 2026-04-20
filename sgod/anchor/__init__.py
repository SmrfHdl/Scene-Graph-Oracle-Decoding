"""
Anchor Position Detector — Component 3 (Section 3.4).
Determines whether current token position requires oracle verification.
"""

from sgod.anchor.detector import detect_anchor
from sgod.anchor.vocabularies import (
    COLOR_ATTRS,
    DETERMINERS,
    SIZE_ATTRS,
    SPATIAL_PREPS,
    SPATIAL_PREPS_MULTI,
)

__all__ = [
    "detect_anchor",
    "DETERMINERS",
    "SPATIAL_PREPS",
    "SPATIAL_PREPS_MULTI",
    "COLOR_ATTRS",
    "SIZE_ATTRS",
]
