"""LLaVA-1.5-7B Backbone wrapper.

Wraps the existing `sgod.utils.model_loader.load_llava` function so that LLaVA
fits the framework's `Backbone` interface. The underlying HF model and processor
are loaded lazily on first use to keep instantiation cheap for tests.

`forward_step` is KV-cache aware:
  - On the first call, runs the full prompt + pixel_values forward and stashes
    `past_key_values` in `inputs`.
  - On subsequent calls, sends only the latest generated token along with the
    cached `past_key_values`, so per-step compute is O(1) in prompt length.

The cache lives in the `inputs` dict that the orchestrator threads through the
generation loop — there is no hidden state on the backbone itself, so a single
backbone instance can be shared across generate() calls (each call uses a fresh
`inputs` dict from `prepare_inputs`).
"""
from __future__ import annotations

from typing import Any

import torch
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
        prompt tokenization. The returned dict also carries an initially-empty
        `past_key_values` slot that `forward_step` mutates in place to maintain
        the KV cache across decode steps.
        """
        self._ensure_loaded()
        proc_out = self._processor(images=image, text=prompt, return_tensors="pt")
        return {
            "proc_out": proc_out,
            "image": image,
            "prompt": prompt,
            "past_key_values": None,
        }

    def forward_step(
        self,
        inputs: dict[str, Any],
        generated_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single-step forward returning (hidden_state, logits).

        Uses HF's `past_key_values` interface for incremental decoding:
          - First call (cache is None): runs full prompt + pixel_values, captures
            the cache from the model output, and returns the LAST token's hidden
            state and logits.
          - Subsequent calls: passes only `generated_ids[:, -1:]` along with the
            cached `past_key_values`. HF computes a single new token's hidden
            state and logits.

        Args:
            inputs:        Dict from `prepare_inputs`. The `past_key_values`
                           slot is mutated in place.
            generated_ids: [B, t] all tokens decoded so far. On the first call
                           may be a zero-length tensor (orchestrator hasn't
                           sampled yet); we ignore it and use the prompt instead.

        Returns:
            hidden_state:  [B, hidden_dim] last hidden state of the final layer.
            lm_logits:     [B, vocab_size] raw LM logits at the new token position.
        """
        self._ensure_loaded()
        model = self._model
        device = next(model.parameters()).device
        cache = inputs.get("past_key_values")

        if cache is None:
            # First step: drive the full prompt + image through the model so
            # the encoder pass over image patches happens exactly once.
            proc = inputs["proc_out"]
            input_ids = proc["input_ids"].to(device)
            pixel_values = proc["pixel_values"].to(device)
            attention_mask = proc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            # Incremental step: only the new token. HF threads the cache for us.
            last_id = generated_ids[:, -1:].to(device)
            with torch.no_grad():
                out = model(
                    input_ids=last_id,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

        inputs["past_key_values"] = out.past_key_values
        # `hidden_states` is a tuple per layer; -1 is the final layer.
        # Slice last position (the newly produced token) along the seq axis.
        hidden = out.hidden_states[-1][:, -1, :].contiguous()  # [B, d]
        logits = out.logits[:, -1, :].contiguous()             # [B, V]
        return hidden, logits
