from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

from .base import MLLM, build_user_content


class MinistralModel(MLLM):
    """Serial adapter matching the paper's Ministral-3 inference path."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs: Any):
        super().__init__(model_path=model_path, device=device, **kwargs)
        import torch
        from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration, MistralCommonBackend

        self.torch = torch
        self.tokenizer = MistralCommonBackend.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = None
        if bool(kwargs.get("fp8_dequantize", True)):
            quantization = FineGrainedFP8Config(dequantize=True)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map=kwargs.get("device_map") or "auto",
            quantization_config=quantization,
        ).eval()
        self.model.generation_config.max_length = None

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        images = images or [None] * len(prompts)
        gen_config = {**self.default_gen_config, **kwargs}
        device = next(self.model.parameters()).device
        responses = []
        for prompt, image in zip(prompts, images):
            messages = []
            content = build_user_content(
                (
                    {"type": "image_url", "image_url": {"url": self._data_url(image)}}
                    if image is not None
                    else None
                ),
                prompt,
            )
            messages.append({"role": "user", "content": content})
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            tokenized = {
                key: value.to(device) if isinstance(value, self.torch.Tensor) else value
                for key, value in tokenized.items()
            }
            image_sizes = None
            if tokenized.get("pixel_values") is not None:
                height, width = tokenized["pixel_values"].shape[-2:]
                image_sizes = [(height, width)]
            with self.torch.inference_mode():
                output = self.model.generate(
                    **tokenized,
                    image_sizes=image_sizes,
                    max_new_tokens=int(gen_config.get("max_new_tokens", 512)),
                    do_sample=bool(gen_config.get("do_sample", False)),
                )[0]
            responses.append(
                self.tokenizer.decode(
                    output[len(tokenized["input_ids"][0]) :],
                    skip_special_tokens=True,
                )
            )
        return responses
