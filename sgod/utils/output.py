"""
Output directory and logging utilities for SGOD.

Every task (infer, eval, train, test) writes to:
    outputs/<task>/<YYYY-MM-DD_HH-MM-SS>/
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def make_run_dir(task: str, tag: str = "") -> Path:
    """Create and return a timestamped output directory for a task run.

    Args:
        task: One of "infer", "eval", "train", "test"
        tag:  Optional label appended to the directory name (e.g. "reltr_coco")

    Returns:
        Path to the newly created run directory, e.g.
        outputs/infer/2026-04-20_14-30-00_reltr_coco/
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}_{tag}" if tag else ts
    run_dir = OUTPUTS_ROOT / task / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: object, path: Path, indent: int = 2) -> None:
    """Write data as JSON, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
