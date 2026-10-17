from __future__ import annotations

from typing import Any

from PIL import Image

from .base import MLLM, build_user_content


class HFChatTemplateModel(MLLM):
    """Paper-compatible adapter for Gemma3, GLM-4.1V, and Step3-VL."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs: Any):
        super().__init__(model_path=model_path, device=device, **kwargs)
        import torch
        from transformers import AutoProcessor

        architecture = str(kwargs.get("architecture", "auto")).lower()
        self.architecture = architecture
        if architecture == "gemma3":
            from modelscope import AutoProcessor as processor_class
            from modelscope import Gemma3ForConditionalGeneration as model_class
            model_kwargs: dict[str, Any] = {}
        elif architecture == "glm4v":
            processor_class = AutoProcessor
            from transformers import Glm4vForConditionalGeneration as model_class
            model_kwargs = {}
        elif architecture == "step3":
            processor_class = AutoProcessor
            from transformers import AutoModelForCausalLM as model_class
            model_kwargs = {
                "key_mapping": {
                    "^vision_model": "model.vision_model",
                    r"^model(?!\.(language_model|vision_model))": "model.language_model",
                    "vit_large_projector": "model.vit_large_projector",
                }
            }
        else:
            raise ValueError(f"Unsupported chat-template architecture: {architecture}")

        self.torch = torch
        self.processor = processor_class.from_pretrained(model_path, trust_remote_code=True)
        device_map = kwargs.get("device_map") or device
        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs,
        ).eval()
        if hasattr(self.processor, "tokenizer"):
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
        messages_batch = []
        for prompt, image in zip(prompts, images):
            messages = []
            content = build_user_content(
                {"type": "image", "image": image} if image is not None else None,
                prompt,
            )
            messages.append({"role": "user", "content": content})
            messages_batch.append(messages)

        inputs = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **(
                {"processor_kwargs": {"padding": True}}
                if self.architecture == "gemma3"
                else {"padding": True}
            ),
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
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
