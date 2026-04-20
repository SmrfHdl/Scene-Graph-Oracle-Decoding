"""
SGG Module — Scene Graph Generation (Component 1).

Provides:
- SGGModule: Main wrapper for RelTR-based scene graph extraction
- SceneGraph: Structured scene representation dataclass
- ObjectNode, RelationEdge, AttributeNode: Component data types
"""
from .scene_graph import SceneGraph, ObjectNode, RelationEdge, AttributeNode
from .sgg_module import SGGModule

__all__ = [
    "SGGModule",
    "SceneGraph",
    "ObjectNode",
    "RelationEdge",
    "AttributeNode",
]
