from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


class ImageFactory:
    """Minimal paper renderer; unused perturbation/analysis code is excluded."""

    def __init__(self, font_path: str | Path, font_size: int = 40):
        self.font_path = str(font_path)
        self.max_font_size = font_size
        self.default_font = self._font(font_size)

    def _font(self, size: int):
        return ImageFont.truetype(self.font_path, size)

    @staticmethod
    def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
        lines: list[str] = []
        for paragraph in text.split("\n"):
            # Retained exactly: the paper renderer inserts a blank line before
            # every explicit paragraph, including empty paragraphs.
            lines.append("")
            current: list[str] = []
            for word in paragraph.split():
                candidate = " ".join(current + [word])
                left, _, right, _ = draw.textbbox((0, 0), candidate, font=font)
                if right - left <= max_width:
                    current.append(word)
                else:
                    if current:
                        lines.append(" ".join(current))
                    current = [word]
            if current:
                lines.append(" ".join(current))
        return lines

    def create_ocr_image(
        self,
        text: str,
        width: int = 1000,
        height: int = 1000,
        background: tuple[int, int, int] = (255, 255, 255),
        text_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        image = Image.new("RGB", (width, height), color=background)
        draw = ImageDraw.Draw(image)
        margin = 50
        max_width = width - 2 * margin
        max_height = height - 2 * margin
        best_font = self.default_font
        best_lines: list[str] = []
        best_size = self.max_font_size
        for size in range(self.max_font_size, 11, -2):
            font = self._font(size)
            lines = self._wrap_text(text, font, max_width, draw)
            spacing = int(size * 0.25)
            if len(lines) * (size + spacing) <= max_height or size <= 13:
                best_font, best_lines, best_size = font, lines, size
                break
        y = margin
        spacing = int(best_size * 0.25)
        for line in best_lines:
            draw.text((margin, y), line, font=best_font, fill=text_color)
            y += best_size + spacing
        return image

    @staticmethod
    def stitch_images(images: list[Image.Image]) -> Image.Image:
        if not images:
            raise ValueError("At least one image is required")
        canvas = Image.new(
            "RGB",
            (sum(image.width for image in images), max(image.height for image in images)),
            color="white",
        )
        x = 0
        for image in images:
            canvas.paste(image, (x, 0))
            x += image.width
        return canvas
