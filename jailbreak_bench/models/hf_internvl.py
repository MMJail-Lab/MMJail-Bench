from __future__ import annotations

from typing import Any

from PIL import Image

from .base import MLLM


class InternVLModel(MLLM):
    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs: Any):
        super().__init__(model_path=model_path, device=device, **kwargs)
        import torch
        import torchvision.transforms as transforms
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_size = int(kwargs.get("input_size", 448))
        self.max_num = int(kwargs.get("max_num", 12))
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
                transforms.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @staticmethod
    def _closest_ratio(
        aspect_ratio: float,
        target_ratios: list[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_difference = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            difference = abs(aspect_ratio - ratio[0] / ratio[1])
            if difference < best_difference or (
                difference == best_difference
                and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
            ):
                best_difference = difference
                best_ratio = ratio
        return best_ratio

    def _preprocess_image(self, image: Image.Image):
        width, height = image.size
        ratios = sorted(
            {
                (left, right)
                for size in range(1, self.max_num + 1)
                for left in range(1, size + 1)
                for right in range(1, size + 1)
                if 1 <= left * right <= self.max_num
            },
            key=lambda ratio: ratio[0] * ratio[1],
        )
        cols, rows = self._closest_ratio(width / height, ratios, width, height, self.input_size)
        resized = image.resize((self.input_size * cols, self.input_size * rows))
        patches = []
        for index in range(cols * rows):
            left = (index % cols) * self.input_size
            top = (index // cols) * self.input_size
            patches.append(resized.crop((left, top, left + self.input_size, top + self.input_size)))
        if len(patches) != 1:
            patches.append(image.resize((self.input_size, self.input_size)))
        return self.torch.stack([self.transform(patch) for patch in patches])

    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        images = images or [None] * len(prompts)
        gen_config = {**self.default_gen_config, **kwargs}
        questions = []
        pixel_values_list = []
        patch_counts = []
        for prompt, image in zip(prompts, images):
            question = (
                (f"<image>\n{prompt}" if prompt else "<image>")
                if image is not None
                else prompt
            )
            questions.append(question)
            if image is None:
                patch_counts.append(0)
            else:
                pixels = self._preprocess_image(image).to(self.torch.bfloat16).to(self.device)
                pixel_values_list.append(pixels)
                patch_counts.append(pixels.size(0))
        pixel_values = self.torch.cat(pixel_values_list) if pixel_values_list else None
        do_sample = bool(gen_config.get("do_sample", False))
        generation_config: dict[str, Any] = {
            "max_new_tokens": int(gen_config.get("max_new_tokens", 512)),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_config["temperature"] = float(gen_config.get("temperature", 1.0))
        with self.torch.inference_mode():
            return self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                num_patches_list=patch_counts,
                questions=questions,
                generation_config=generation_config,
            )
