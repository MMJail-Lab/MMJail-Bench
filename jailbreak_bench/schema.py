from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_SAMPLE_FIELDS = ("sample_id", "task_meta", "input_config")
REQUIRED_TASK_FIELDS = ("task_id", "task_label", "category", "tier2", "severity")
REQUIRED_INPUT_FIELDS = ("query_type", "prompt", "carrier_mode", "image_type")
ALLOWED_CARRIER_MODES = {"text", "ocr"}


def validate_sample(sample: dict[str, Any], line_number: int | None = None) -> None:
    prefix = f"line {line_number}: " if line_number is not None else ""
    for field in REQUIRED_SAMPLE_FIELDS:
        if field not in sample:
            raise ValueError(f"{prefix}missing required field '{field}'")
    task_meta = sample["task_meta"]
    input_config = sample["input_config"]
    if not isinstance(task_meta, dict):
        raise ValueError(f"{prefix}'task_meta' must be an object")
    if not isinstance(input_config, dict):
        raise ValueError(f"{prefix}'input_config' must be an object")
    for field in REQUIRED_TASK_FIELDS:
        if field not in task_meta:
            raise ValueError(f"{prefix}missing task_meta field '{field}'")
    for field in REQUIRED_INPUT_FIELDS:
        if field not in input_config:
            raise ValueError(f"{prefix}missing input_config field '{field}'")
    carrier_mode = input_config.get("carrier_mode")
    if carrier_mode not in ALLOWED_CARRIER_MODES:
        raise ValueError(
            f"{prefix}input_config.carrier_mode must be one of {sorted(ALLOWED_CARRIER_MODES)}"
        )


def validate_dataset(
    samples: list[dict[str, Any]],
    data_root: str | Path | None = None,
    stage: str | None = None,
) -> None:
    if stage not in {None, "base", "final"}:
        raise ValueError("stage must be None, 'base', or 'final'")
    seen: set[str] = set()
    root = Path(data_root).resolve() if data_root else None
    for index, sample in enumerate(samples, start=1):
        validate_sample(sample, index)
        sample_id = str(sample["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        image_path = sample["input_config"].get("image_path")
        if not image_path:
            raise ValueError(f"{sample_id} has no image_path")
        prompt = str(sample["input_config"].get("prompt", ""))
        carrier_mode = str(sample["input_config"].get("carrier_mode", ""))
        if stage == "base" and not prompt:
            raise ValueError(f"Base row {sample_id} has an empty prompt")
        if stage == "final":
            if carrier_mode == "text" and not prompt:
                raise ValueError(f"Final text row {sample_id} has an empty prompt")
            if carrier_mode == "ocr" and prompt:
                raise ValueError(f"Final OCR row {sample_id} must have an empty prompt")
        if root:
            path = (root / str(image_path)).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Image path for {sample_id} escapes data root") from exc
            if not path.is_file():
                raise FileNotFoundError(f"image file referenced by {sample_id} not found: {path}")
