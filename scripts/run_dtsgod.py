"""CLI entry-point: load a config and run DT-SGOD on an (image, question) pair.

Usage:
    python scripts/run_dtsgod.py \\
        --config configs/dt_sgod_llava15.yaml \\
        --image  /path/to/image.jpg \\
        --question "Is there a dog in the image?"

The script loads the full backbone + oracle + policy stack defined by the
config and prints the model's generated answer. Useful for:
  - Smoke-testing the framework end-to-end on a single example.
  - Quick qualitative inspection during development.
  - Direction-checking after architecture or oracle changes.

This is the highest-level entry point — it exercises every layer of the
framework. If something breaks here, the bisecting target is the individual
layer (Backbone, Oracle, or Policy) that the orchestrator drives.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from sgod.runtime import build_from_config, load_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    p.add_argument("--image", type=Path, required=True, help="Path to input image.")
    p.add_argument("--question", type=str, required=True, help="User question.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Full prompt incl. chat template. Defaults to a LLaVA-1.5 USER/ASSISTANT wrap of --question.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override runtime.max_new_tokens.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override runtime.temperature.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def _default_llava_prompt(question: str) -> str:
    """LLaVA-1.5 chat template wrapper."""
    return f"USER: <image>\n{question} ASSISTANT:"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    log = logging.getLogger("dt_sgod")

    log.info("Loading config: %s", args.config)
    config = load_config(args.config)

    log.info("Building backbone + oracle + policy …")
    # lazy=False so the full stack is loaded up-front — surface any model-load
    # failure before the first generate() call.
    decoder = build_from_config(config, lazy=False)

    log.info("Reading image: %s", args.image)
    image = Image.open(args.image)

    prompt = args.prompt if args.prompt is not None else _default_llava_prompt(args.question)
    log.info("Generating …  prompt=%r", prompt)
    answer = decoder.generate(
        image=image,
        prompt=prompt,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("\n=== Model answer ===")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
