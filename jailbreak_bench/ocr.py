from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .image_factory import ImageFactory
from .io import sha256_file


PAPER_FONT_SHA256 = "674a5e96280c31b866c4c166f9723233d545595d3604b438ff060d0b872e9f98"


def render_ocr_dataset(
    samples: list[dict[str, Any]],
    *,
    data_root: str | Path,
    output_image_dir: str | Path,
    image_path_prefix: str,
    font_path: str | Path,
    expected_font_sha256: str = PAPER_FONT_SHA256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render the paper's [instruction image, semantic image] horizontal stitch."""
    root = Path(data_root).resolve()
    output = Path(output_image_dir).resolve()
    font = Path(font_path).resolve()
    if sha256_file(font).lower() != expected_font_sha256.lower():
        raise RuntimeError(f"Font hash mismatch for {font}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output image directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prefix = Path(image_path_prefix).as_posix().strip("/")
    if not prefix or prefix.startswith(".."):
        raise ValueError("image_path_prefix must be a safe relative directory")

    factory = ImageFactory(font_path=font, font_size=40)
    rendered = 0
    for sample in samples:
        config = sample["input_config"]
        if config.get("carrier_mode") != "ocr":
            if not str(config.get("prompt", "")):
                raise ValueError(f"Text row has an empty prompt: {sample['sample_id']}")
            continue
        prompt = str(config.get("prompt", ""))
        if not prompt:
            raise ValueError(f"OCR base row has an empty prompt: {sample['sample_id']}")
        relative = Path(str(config.get("image_path", "")))
        semantic = (root / relative).resolve()
        try:
            semantic.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Image path escapes data root: {relative}") from exc
        if not semantic.is_file():
            raise FileNotFoundError(semantic)
        target = output / f"{sample['sample_id'].replace('/', '_')}.png"
        if target.exists():
            raise FileExistsError(target)
        with Image.open(semantic) as source:
            background = source.convert("RGB")
        text_image = factory.create_ocr_image(prompt)
        ImageFactory.stitch_images([text_image, background]).save(target, format="PNG")
        config["image_path"] = f"{prefix}/{target.name}"
        config["prompt"] = ""
        rendered += 1

    return samples, {
        "rendered": rendered,
        "font_sha256": sha256_file(font),
        "output_image_dir": output.as_posix(),
        "image_path_prefix": prefix,
    }
