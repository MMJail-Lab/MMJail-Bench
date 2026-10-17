from __future__ import annotations

from typing import Any

from PIL import Image

from .base import MLLM, build_user_content


class LlavaOneVisionModel(MLLM):
    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs: Any):
        super().__init__(model_path=model_path, device=device, **kwargs)
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=kwargs.get("device_map") or device,
            trust_remote_code=True,
        ).eval()

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        from qwen_vl_utils import process_vision_info

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

        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=texts,
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
                top_p=float(gen_config.get("top_p", 0.95)),
            )
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        # Retain the paper runner's attention-length trimming for compatibility.
        lengths = inputs.get("attention_mask", self.torch.ones_like(inputs["input_ids"])).sum(dim=-1)
        return [
            self.processor.decode(
                generated[index, int(length) :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for index, length in enumerate(lengths)
        ]
