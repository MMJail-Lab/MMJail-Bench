from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")
_GENERATION_KEYS = {
    "max_new_tokens",
    "min_new_tokens",
    "temperature",
    "top_p",
    "do_sample",
    "seed",
}


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        current = os.getenv(name)
        if current is not None:
            return current
        if default is not None:
            return default
        raise EnvironmentError(
            f"Environment variable {name} is required by the config file."
        )

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            config = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML model configuration") from exc
            config = yaml.safe_load(handle) or {}
    return _resolve_env(config)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_model_config(
    config: dict[str, Any], model_name: str, role: str = "victim"
) -> dict[str, Any]:
    models = config.get("models", {})
    if model_name not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")

    default_runtime = config.get("default_runtime", {})
    role_runtime = config.get(f"{role}_models", {})
    merged = merge_dicts(default_runtime, role_runtime)
    merged = merge_dicts(merged, models[model_name])
    generation = copy.deepcopy(merged.get("gen_config", {}))
    for key in _GENERATION_KEYS:
        if key in merged:
            generation[key] = copy.deepcopy(merged[key])
    merged["gen_config"] = generation
    return merged
