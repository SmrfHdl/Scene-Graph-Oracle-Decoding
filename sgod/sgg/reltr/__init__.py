"""
Vendored RelTR model package for scene graph generation.

Source: https://github.com/yrcong/RelTR
Paper: RelTR: Relation Transformer for Scene Graph Generation (TPAMI 2023)
"""
from .reltr import build


def build_model(args):
    """Build RelTR model from args. Returns (model,)."""
    return build(args)
