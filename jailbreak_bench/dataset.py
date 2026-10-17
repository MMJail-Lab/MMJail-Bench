from __future__ import annotations

import random
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .determinism import PAPER_SEED
from .io import read_jsonl


QUERY_TYPES = ("Academic", "System", "Structure", "Story", "Code", "Paradox")
IMAGE_TYPES = ("auth_doc", "danger", "id_card", "role", "scenario")
PAPER_TASK_COUNT = 272


def _folder_name(task_label: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in task_label)


def _normalize_prompt(query_type: str, prompt: str) -> str:
    """Apply the prompt normalization used for the paper data."""
    if query_type != "Academic":
        return prompt
    normalized = prompt.replace(
        "- 1. discuss general concepts in it.",
        "- 1. discuss general concepts in it. (50 words)",
    ).replace(
        "- 2. outline hypothetical risks.",
        "- 2. outline hypothetical risks. (50 words)",
    )
    if not normalized.endswith("step by step."):
        raise ValueError("Academic prompt does not match the paper normalization rule")
    return normalized + " (400 words)"


def _p2_carrier_map() -> dict[str, str]:
    """Reproduce the p2 assignment against the complete 272-intent universe."""
    universe_ids = [
        f"muse-{task_id:03d}-{query_type}-{image_type}"
        for task_id in range(1, PAPER_TASK_COUNT + 1)
        for query_type in QUERY_TYPES
        for image_type in IMAGE_TYPES
    ]
    shuffled_ids = list(universe_ids)
    random.Random(PAPER_SEED).shuffle(shuffled_ids)
    p1_text_ids = set(shuffled_ids[: len(shuffled_ids) // 2])
    return {
        sample_id: ("ocr" if sample_id in p1_text_ids else "text")
        for sample_id in universe_ids
    }


def build_dataset(
    task_info_path: str | Path,
    task_label_path: str | Path,
    *,
    semantic_image_dir: str = "imgs",
) -> list[dict[str, Any]]:
    """Build the p2 base data from full or subset intent metadata.

    OCR rows retain their original prompt and semantic image path in this base
    data because the deterministic OCR renderer consumes both. The carrier map
    is always calculated over all 272 original intents, so releasing a subset
    cannot reshuffle its text/OCR assignments.
    """
    labels = {item["task_label"]: item for item in read_jsonl(task_label_path)}
    task_info = read_jsonl(task_info_path)
    carrier_map = _p2_carrier_map()
    samples: list[dict[str, Any]] = []
    seen_task_ids: set[int] = set()

    for task in task_info:
        task_label = str(task["task_label"])
        if task_label not in labels:
            raise ValueError(f"Missing task metadata for {task_label!r}")
        meta = labels[task_label]
        queries = task.get("queries")
        if not isinstance(queries, dict):
            raise ValueError(f"Task {task_label!r} has no queries object")
        missing_queries = [name for name in QUERY_TYPES if name not in queries]
        extra_queries = [name for name in queries if name not in QUERY_TYPES]
        if missing_queries or extra_queries:
            raise ValueError(
                f"Task {task_label!r} query keys differ from the paper design: "
                f"missing={missing_queries}, extra={extra_queries}"
            )

        task_id_value = task["id"]
        task_id = int(task_id_value)
        if task_id not in range(1, PAPER_TASK_COUNT + 1):
            raise ValueError(f"Task id {task_id} is outside the paper universe 1-272")
        if task_id in seen_task_ids:
            raise ValueError(f"Duplicate task id: {task_id}")
        seen_task_ids.add(task_id)
        folder = _folder_name(task_label)

        for query_type in QUERY_TYPES:
            for image_type in IMAGE_TYPES:
                sample_id = f"muse-{task_id:03d}-{query_type}-{image_type}"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "task_meta": {
                            "task_id": task_id_value,
                            "task_label": task_label,
                            "category": meta.get("category"),
                            "tier2": meta.get("tier2"),
                            "severity": meta.get("severity"),
                        },
                        "input_config": {
                            "query_type": query_type,
                            "prompt": _normalize_prompt(query_type, str(queries[query_type])),
                            "carrier_mode": carrier_map[sample_id],
                            "image_type": image_type,
                            "image_path": str(
                                PurePosixPath(semantic_image_dir, folder, f"{image_type}.png")
                            ),
                        },
                    }
                )

    samples.sort(key=lambda item: str(item["sample_id"]))
    return samples


def dataset_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def count(section: str, field: str) -> dict[str, int]:
        values = Counter(str(item[section].get(field, "")) for item in samples)
        return dict(sorted(values.items()))

    return {
        "rows": len(samples),
        "unique_sample_ids": len({str(item["sample_id"]) for item in samples}),
        "category": count("task_meta", "category"),
        "tier2": count("task_meta", "tier2"),
        "severity": count("task_meta", "severity"),
        "query_type": count("input_config", "query_type"),
        "carrier_mode": count("input_config", "carrier_mode"),
        "image_type": count("input_config", "image_type"),
    }
