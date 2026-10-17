from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


def build_user_content(image_block: dict[str, Any] | None, prompt: str) -> list[dict[str, Any]]:
    """Build a multimodal user turn without adding hidden or empty text blocks."""
    content: list[dict[str, Any]] = []
    if image_block is not None:
        content.append(image_block)
    if prompt:
        content.append({"type": "text", "text": prompt})
    if not content:
        raise ValueError("A victim input must contain an image or a non-empty prompt")
    return content


class MLLM(ABC):
    def __init__(self, model_path: str | None = None, device: str = "cpu", **kwargs: Any):
        self.model_path = model_path
        self.device = device
        self.default_gen_config = kwargs.get("gen_config", {})

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        raise NotImplementedError
