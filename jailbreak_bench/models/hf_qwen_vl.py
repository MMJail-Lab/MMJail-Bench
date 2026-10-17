from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from .base import MLLM, build_user_content


class QwenVLModel(MLLM):
    """Qwen2.5-VL wrapper used by the benchmark.

    The wrapper is intentionally narrow and conservative. Add additional local
    model wrappers in a separate module if your reproduction needs them.
    """

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs: Any):
        super().__init__(model_path=model_path, device=device, **kwargs)
        device_map = kwargs.get("device_map") or device
        dtype = torch.bfloat16 if kwargs.get("precision", "bf16") == "bf16" else torch.float16
        architecture = str(kwargs.get("architecture", "auto")).lower()
        if architecture == "auto":
            architecture = "qwen3_moe" if "a3b" in model_path.lower() else (
                "qwen3" if "qwen3" in model_path.lower() else "qwen2_5"
            )
        if architecture == "qwen2_5":
            from transformers import Qwen2_5_VLForConditionalGeneration as model_class
        elif architecture == "qwen3":
            from transformers import Qwen3VLForConditionalGeneration as model_class
        elif architecture == "qwen3_moe":
            from transformers import Qwen3VLMoeForConditionalGeneration as model_class
        else:
            raise ValueError(f"Unsupported Qwen architecture: {architecture}")

        load_options: dict[str, Any] = {"trust_remote_code": True}
        if kwargs.get("revision"):
            load_options["revision"] = str(kwargs["revision"])
        if kwargs.get("local_files_only") is not None:
            load_options["local_files_only"] = bool(kwargs["local_files_only"])
        self.processor = AutoProcessor.from_pretrained(model_path, **load_options)
        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            **load_options,
        ).eval()
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        images = images or [None] * len(prompts)
        gen_config = {**self.default_gen_config, **kwargs}
        batch_messages = []
        for prompt, image in zip(prompts, images):
            messages = []
            content = build_user_content(
                {"type": "image", "image": image} if image is not None else None,
                prompt,
            )
            messages.append({"role": "user", "content": content})
            batch_messages.append(messages)

        text_inputs = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        do_sample = bool(gen_config.get("do_sample", False))
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(gen_config.get("max_new_tokens", 512)),
            "min_new_tokens": int(gen_config.get("min_new_tokens", 0)),
            "do_sample": do_sample,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                temperature=float(gen_config.get("temperature", 1.0)),
                top_p=float(gen_config.get("top_p", 1.0)),
            )
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
