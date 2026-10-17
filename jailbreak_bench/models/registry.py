from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import MLLM


MODEL_TYPES = {
    "openai_compatible": "jailbreak_bench.models.openai_compatible:OpenAICompatibleModel",
    "qwen_vl": "jailbreak_bench.models.hf_qwen_vl:QwenVLModel",
    "hf_chat_template": "jailbreak_bench.models.hf_chat_template:HFChatTemplateModel",
    "internvl": "jailbreak_bench.models.hf_internvl:InternVLModel",
    "llava_onevision": "jailbreak_bench.models.hf_llava_onevision:LlavaOneVisionModel",
    "ministral": "jailbreak_bench.models.hf_ministral:MinistralModel",
}


def _resolve_model_class(model_type: str):
    value = MODEL_TYPES[model_type]
    if isinstance(value, str):
        module_name, class_name = value.split(":")
        return getattr(import_module(module_name), class_name)
    return value


def load_model(model_config: dict[str, Any]) -> MLLM:
    model_type = str(model_config.get("type", "")).lower()
    if model_type not in MODEL_TYPES:
        available = ", ".join(sorted(MODEL_TYPES))
        raise KeyError(f"Unknown model type '{model_type}'. Available types: {available}")
    cls = _resolve_model_class(model_type)
    kwargs = {key: value for key, value in model_config.items() if key not in {"type", "path", "device"}}
    return cls(
        model_path=model_config.get("path"),
        device=model_config.get("device", "cpu"),
        **kwargs,
    )
