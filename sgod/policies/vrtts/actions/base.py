"""Visual action base class and shared visual-state container."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class VisualState:
    """Mutable container threaded through the VR-TTS iteration.

    `image` is the *current* image fed to the backbone. Actions update it
    (cropped, marked, etc.) without touching the original.

    `prompt_suffix` accumulates action-specific prompt extensions (e.g.,
    "Marked objects: {1: cat, 2: dog}"). `sub_qa_trace` records prior
    sub-question / sub-answer pairs for `AskSubQuestion`.
    """

    image: Image.Image
    original_image: Image.Image
    prompt_suffix: str = ""
    sub_qa_trace: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "VisualState":
        return VisualState(
            image=self.image.copy(),
            original_image=self.original_image,
            prompt_suffix=self.prompt_suffix,
            sub_qa_trace=list(self.sub_qa_trace),
            metadata=dict(self.metadata),
        )


class VisualAction(ABC):
    """Abstract action: given current `VisualState` + diagnostic context, produce next state.

    Implementations must be deterministic given the same inputs to keep the
    iteration reproducible.
    """

    name: str = "abstract"

    @abstractmethod
    def apply(
        self,
        state: VisualState,
        *,
        attentions: Any | None = None,
        question: str | None = None,
        backbone: Any | None = None,
        oracle: Any | None = None,
    ) -> VisualState:
        """Return a new state after applying this action.

        Optional kwargs are passed by the orchestrator so each action can
        pull what it needs without expanding the interface for others:
          - `attentions`: layer-stacked attention from the latest forward
                          (used by zoom).
          - `question`: the main question text (used by sub-question).
          - `backbone`: the VLM backbone (used by sub-question for inner calls).
          - `oracle`: external detector oracle (used by SoM marking).
        """
