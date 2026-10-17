from __future__ import annotations

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from PIL import Image
from openai import BadRequestError, OpenAI

from .base import MLLM, build_user_content


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class OpenAICompatibleModel(MLLM):
    """Wrapper for OpenAI-compatible chat completion APIs."""

    def __init__(self, model_path: str, device: str = "cpu", **kwargs: Any):
        super().__init__(model_path=model_path, device=device, **kwargs)
        api_key = kwargs.get("api_key") or os.getenv(str(kwargs.get("api_key_env", "OPENAI_API_KEY")))
        base_url = kwargs.get("base_url") or os.getenv(str(kwargs.get("base_url_env", "OPENAI_BASE_URL")))
        if not api_key:
            raise EnvironmentError("API key is missing. Set api_key_env in config or OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_workers = int(kwargs.get("max_workers", 8))

    def _single(self, prompt: str, image: Image.Image | None, **kwargs: Any) -> str:
        judge_system_prompt = kwargs.get("judge_system_prompt")
        messages: list[dict[str, Any]] = []
        if judge_system_prompt:
            messages.append({"role": "system", "content": str(judge_system_prompt)})
        content = build_user_content(
            (
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"},
                }
                if image is not None
                else None
            ),
            prompt,
        )
        messages.append({"role": "user", "content": content})
        gen_config = {**self.default_gen_config, **kwargs}
        try:
            response = self.client.chat.completions.create(
                model=str(self.model_path),
                messages=messages,
                max_tokens=int(gen_config.get("max_new_tokens", 512)),
                temperature=float(gen_config.get("temperature", 0.0)),
                timeout=60.0,
            )
            return response.choices[0].message.content or ""
        except BadRequestError as exc:
            message = str(exc).lower()
            if "content_policy" in message or "sensitive" in message:
                return "[[API_SAFETY_REFUSAL]]"
            return f"ERROR_API_400: {exc}"
        except Exception as exc:
            return f"ERROR_API: {type(exc).__name__}"

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        images = images or [None] * len(prompts)
        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._single, prompt, image, **kwargs): index
                for index, (prompt, image) in enumerate(zip(prompts, images))
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results
