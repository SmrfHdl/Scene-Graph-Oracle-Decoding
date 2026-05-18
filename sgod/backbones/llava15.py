"""LLaVA-1.5-7B Backbone wrapper.

Wraps the existing `sgod.utils.model_loader.load_llava` function so that LLaVA
fits the framework's `Backbone` interface. The underlying HF model and processor
are loaded lazily on first use to keep instantiation cheap for tests.

The `forward_step` implementation is a stub for Day 1; Week 1 fills it in
once the cache/KV-handling strategy is finalized.
"""
from __future__ import annotations

from typing import Any

from torch import Tensor

from sgod.core.interfaces import Backbone
from sgod.core.registry import register

# Known static facts about LLaVA-1.5-7B (avoids loading the model just to query dims).
_LLAVA15_7B_HIDDEN_DIM = 4096
_LLAVA15_7B_VOCAB_SIZE = 32064  # 32000 LLaMA vocab + image-token expansion
_LLAVA15_7B_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


@register("backbone", "llava-1.5-7b")
class LLaVAv15Backbone(Backbone):
    """LLaVA-1.5-7B HF wrapper conforming to the framework `Backbone` interface.

    Args:
        model_id:    HuggingFace model id (defaults to `llava-hf/llava-1.5-7b-hf`).
        dtype:       torch dtype for the model weights.
        device_map:  device map passed to `from_pretrained` (e.g. "auto", "cuda:0").
        load_in_4bit: enable bitsandbytes 4-bit quantization for small GPUs.
        lazy:        if True (default), defer model load until first `forward_step` /
                     `tokenizer` access — keeps unit tests fast.
        hidden_dim_override / vocab_size_override:
                     escape hatches for testing on fake/mock models (e.g., tiny
                     model with different dims). Production code should leave these
                     as None to use the known constants.
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        dtype: str = "float16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        lazy: bool = True,
        hidden_dim_override: int | None = None,
        vocab_size_override: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self._lazy = lazy
        self._model: Any | None = None
        self._processor: Any | None = None
        self._hidden_dim = hidden_dim_override or _LLAVA15_7B_HIDDEN_DIM
        self._vocab_size = vocab_size_override or _LLAVA15_7B_VOCAB_SIZE

        if not lazy:
            self._ensure_loaded()

    # ── lazy loader ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from sgod.utils.model_loader import load_llava

        torch_dtype = torch.float16 if self.dtype == "float16" else torch.bfloat16
        self._model, self._processor = load_llava(
            model_id=self.model_id,
            dtype=torch_dtype,
            device_map=self.device_map,
            load_in_4bit=self.load_in_4bit,
        )

    # ── interface ─────────────────────────────────────────────────────────

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def lora_target_modules(self) -> list[str]:
        return list(_LLAVA15_7B_LORA_TARGETS)

    @property
    def device(self) -> str:
        # If the model is loaded, ask it. Otherwise return a conservative default.
        if self._model is None:
            return "cuda" if self.device_map != "cpu" else "cpu"
        # When loaded with device_map="auto", parameters can span devices; pick the
        # device of the first parameter as a representative.
        return str(next(self._model.parameters()).device)

    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._processor.tokenizer

    def prepare_inputs(self, image: Any, prompt: str) -> dict[str, Any]:
        """Build the dict consumed by `forward_step`.

        For LLaVA-1.5 the processor handles both image preprocessing and
        prompt tokenization. The returned dict carries the processor output
        plus the raw image for any policy that needs it.
        """
        self._ensure_loaded()
        proc_out = self._processor(images=image, text=prompt, return_tensors="pt")
        return {
            "proc_out": proc_out,
            "image": image,
            "prompt": prompt,
        }

    def forward_step(
        self,
        inputs: dict[str, Any],
        generated_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single-step forward returning (hidden_state, logits).

        TODO(week-1): wire up KV-cache-aware single-token forward via
        `model.generate` step or `model.forward` with `use_cache=True`.
        This is intentionally a NotImplementedError at the Day-1 skeleton stage —
        the smoke test exercises the Policy interface directly with mock hidden
        states rather than driving a real LLaVA forward pass.
        """
        raise NotImplementedError(
            "LLaVAv15Backbone.forward_step is wired in Week 1, after the "
            "framework-readiness smoke test has validated the surface area."
        )
