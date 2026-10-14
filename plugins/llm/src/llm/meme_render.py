"""Draw captions onto a meme template with Pillow.

The text fitting — shrink-to-fit, the two/three-row wrap choice, stroke width
and the vertical centring — is memegen's (https://github.com/jacebrowning/
memegen, app/utils/images.py and app/models/text.py, MIT, Copyright 2020
Jace Browning), trimmed to plain Pillow. Template boxes use memegen's
config.yml fields, so its hand-made layouts load unchanged.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).parent / "fonts"
FONT_FILES = {
    "thick": "TitilliumWeb-Black.ttf",
    "thin": "TitilliumWeb-SemiBold.ttf",
    "comic": "Kalam-Regular.ttf",
}
DEFAULT_FONT = "thick"
MINIMUM_FONT_SIZE = 7
# A custom image larger than this is shrunk before drawing; nobody reads a
# 4000-px meme in an IRC client, and it bounds the render's memory.
MAX_EDGE = 1200


@dataclass(frozen=True)
class Box:
    """One caption box, as fractions of the image (memegen's ``text:`` entry)."""

    anchor_x: float
    anchor_y: float
    scale_x: float
    scale_y: float
    angle: float = 0.0
    color: str = "white"
    style: str = "upper"
    font: str = DEFAULT_FONT
    align: str = "center"
    # Where a short caption sits in a tall box: top, middle or bottom.
    valign: str = "middle"


# memegen's ``custom`` template: a top and a bottom band.
TOP_BOTTOM = (
    Box(0.0, 0.0, 1.0, 0.2),
    Box(0.0, 0.8, 1.0, 0.2),
)


@lru_cache(maxsize=256)
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / FONT_FILES.get(name, FONT_FILES[DEFAULT_FONT])
    return ImageFont.truetype(str(path), size=size)


def _stroke_width(font: ImageFont.FreeTypeFont) -> int:
    return min(3, max(1, int(font.size / 12)))


_MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def _text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    _, _, width, height = _MEASURE.textbbox((0, 0), text, font)
    stroke = _stroke_width(font)
    return int(width) + stroke, int(height) + stroke


def _fit_font(
    name: str, text: str, max_size: tuple[int, int], max_font_size: int
) -> ImageFont.FreeTypeFont:
    """The largest size at which ``text`` fits the box."""
    max_width = max_size[0] - max_size[0] / 35
    max_height = max_size[1] - max_size[1] / 10
    font = _font(name, MINIMUM_FONT_SIZE)
    for size in range(max(MINIMUM_FONT_SIZE, max_font_size), 6, -1):
        font = _font(name, size)
        width, height = _text_size(text, font)
        x_offset, y_offset, _, _ = font.getbbox(text)
        if width - x_offset <= max_width and height - y_offset <= max_height:
            break
    return font


def _split_2(line: str) -> str:
    midpoint = len(line) // 2 - 1
    for offset in range(0, len(line) // 4):
        for index in (midpoint - offset, midpoint + offset):
            if line[index] == " ":
                return line[:index].strip() + "\n" + line[index:].strip()
    return line


def _split_3(line: str) -> str:
    max_len = len(line) / 3
    rows = ["", "", ""]
    index = 0
    for word in line.split(" "):
        if len(rows[index]) + len(word) * 0.7 > max_len and index < 2:
            index += 1
        rows[index] += word + " "
    return "\n".join(rows).strip()


def _wrap(name: str, line: str, max_size: tuple[int, int], max_font_size: int) -> str:
    """One, two or three rows: whichever lets the text be biggest."""
    one, two, three = line, _split_2(line), _split_3(line)
    font_1 = _fit_font(name, one, max_size, max_font_size)
    font_2 = _fit_font(name, two, max_size, max_font_size)
    font_3 = _fit_font(name, three, max_size, max_font_size)
    if font_1.size == font_2.size and font_2.size <= MINIMUM_FONT_SIZE:
        return two
    if font_1.size >= font_2.size:
        return one
    if _text_size(three, font_3)[0] >= max_size[0] * 0.60:
        return three
    if _text_size(two, font_2)[0] >= max_size[0] * 0.60:
        return two
    return one


def _offset(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_size: tuple[int, int],
    align: str,
) -> tuple[float, float]:
    """memegen's draw origin for text centred in the box (negated)."""
    width, height = _text_size(text, font)
    stroke = _stroke_width(font)
    x_offset, y_offset, _, _ = font.getbbox(text)
    x: float = x_offset - stroke
    y: float = y_offset - stroke
    rows = text.split("\n")
    y_adjust = 1.1 if len(rows) >= 3 else 1 + (3 - len(rows)) * 0.25
    if align == "right":
        x -= max_size[0] - width
    elif align != "left":
        x -= (max_size[0] - width) / 2
    y -= (max_size[1] - height / y_adjust) / 2
    if any(letter in rows[-1] for letter in "gjpqy"):
        y += height // 20
    return x, y


def _stroke(color: str, width: int) -> tuple[int, str]:
    if color == "black":
        # imgflip outlines black text in white at full width; memegen used a
        # faint 1-px halo, which vanishes on photos.
        return width, "white"
    if color.startswith("#"):
        return 1, "#000000"
    return width, "black"


_LONE_I = re.compile(r"\bi\b")


def _stylize(style: str, text: str, lines: list[str]) -> str:
    if style == "upper":
        return text.upper()
    if style == "default":
        # memegen: capitalise a lowercase caption when all of them are
        # lowercase or one reads as a sentence; always "I", never "i".
        filled = [x for x in lines if x.strip()]
        all_lower = all(x.islower() for x in filled)
        sentence = any(x.endswith((".", "?", "!")) for x in filled)
        if text.islower() and (all_lower or sentence):
            text = text.capitalize()
        return _LONE_I.sub("I", text)
    if style == "mock":
        return "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(text))
    return text


def _top_layout(count: int) -> tuple[Box, ...]:
    """memegen's ``layout=top``: black rows in a white band over the picture."""
    height = 0.2 / max(count, 1)
    return tuple(
        Box(0.01, i * height, 0.98, height, color="black", style="none", font="thin", align="left")
        for i in range(count)
    )


def _pad_top(image: Image.Image) -> Image.Image:
    padded = Image.new("RGB", (image.width, image.height + image.height // 4), "white")
    padded.paste(image, (0, image.height // 4))
    return padded


@dataclass(frozen=True)
class Slot:
    """Where a picture goes: a box (fractions) and the mask inside it."""

    box: tuple[float, float, float, float]
    mask: Path
    # Top-left, top-right, bottom-right, bottom-left (fractions of the
    # template) of a blank seen in perspective — the two sheets of paper in
    # "same". The picture is warped onto them instead of cropped square.
    corners: tuple[tuple[float, float], ...] = ()


def slot_size(image: Image.Image, slot: Slot) -> tuple[int, int]:
    return max(1, int(image.width * slot.box[2])), max(1, int(image.height * slot.box[3]))


def insert_pictures(
    image: Image.Image, slots: tuple[Slot, ...], pictures: list[Image.Image | None]
) -> Image.Image:
    """Paste each picture into its slot through the slot's mask.

    The picture is scaled to cover the slot's box and centre-cropped, so it
    fills the blank edge to edge; the mask (the blank's exact outline, made
    at catalog build time) keeps it off the template around it. A None
    picture leaves its slot blank.
    """
    image = image.copy()
    for slot, picture in zip(slots, pictures, strict=False):
        if picture is None:
            continue
        width, height = slot_size(image, slot)
        left, top = int(image.width * slot.box[0]), int(image.height * slot.box[1])
        mask = Image.open(slot.mask).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        if len(slot.corners) == 4:
            quad = [(x * image.width, y * image.height) for x, y in slot.corners]
            image.paste(_warp(picture.convert("RGB"), quad, image.size), (0, 0), mask)
            continue
        fitted = _cover(picture.convert("RGB"), width, height)
        image.paste(fitted, (left, top), mask.crop((left, top, left + width, top + height)))
    return image


def _cover(picture: Image.Image, width: int, height: int) -> Image.Image:
    """``picture`` scaled to cover ``width`` x ``height``, centre-cropped."""
    scale = max(width / picture.width, height / picture.height)
    resized = picture.resize(
        (max(width, round(picture.width * scale)), max(height, round(picture.height * scale))),
        Image.Resampling.LANCZOS,
    )
    cx, cy = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((cx, cy, cx + width, cy + height))


def _solve(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; 8x8 here, no numpy."""
    n = len(values)
    rows = [[*matrix[i], values[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r != col:
                factor = rows[r][col] / rows[col][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col], strict=True)]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def _warp(
    picture: Image.Image, quad: list[tuple[float, float]], size: tuple[int, int]
) -> Image.Image:
    """``picture`` warped so its corners land on ``quad`` (TL, TR, BR, BL)
    on a canvas of ``size``. Pillow's PERSPECTIVE maps output pixels back
    to input ones, so the coefficients solve quad -> picture corners."""
    w, h = picture.size
    source = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    matrix: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(quad, source, strict=True):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    coeffs = _solve(matrix, values)
    return picture.transform(
        size, Image.Transform.PERSPECTIVE, tuple(coeffs), Image.Resampling.BICUBIC
    )


def load_image(source: Path | bytes) -> Image.Image:
    """The first frame as RGB, transparency flattened onto white."""
    image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    image.seek(0)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.split()[-1])
    else:
        image = image.convert("RGB")
    if max(image.size) > MAX_EDGE:
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    return image


def render(
    image: Image.Image,
    boxes: tuple[Box, ...],
    lines: list[str],
    *,
    font: str | None = None,
    layout_top: bool = False,
) -> bytes:
    """Draw ``lines`` into ``boxes`` (in order) and return PNG bytes.

    A blank line leaves its box empty. ``font`` overrides every box's own.
    ``layout_top`` ignores the template's boxes and stacks the lines in a
    white band added above the picture.
    """
    if layout_top:
        image = _pad_top(image)
        boxes = _top_layout(len(lines))
    else:
        image = image.copy()
    width, height = image.size
    for box, line in zip(boxes, lines, strict=False):
        if not line.strip():
            continue
        name = font or box.font
        max_size = (max(1, int(width * box.scale_x)), max(1, int(height * box.scale_y)))
        max_font_size = int(height / (4 if box.angle else 9))
        text = _stylize(box.style, _wrap(name, line, max_size, max_font_size), lines)
        fitted = _fit_font(name, text, max_size, max_font_size)
        x, y = _offset(text, fitted, max_size, box.align)
        stroke_width, stroke_fill = _stroke(box.color, _stroke_width(fitted))
        layer = Image.new("RGBA", max_size)
        rows = text.count("\n") + 1
        # memegen ties row spacing to the centred offset; keep that, then
        # move top/bottom text by its measured extent so no row hangs out
        # of the box (memegen's centring height is a fudge that only
        # balances when the error splits above and below).
        spacing = -y / (rows * 2)
        if box.valign in ("top", "bottom"):
            _, ink_top, _, ink_bottom = _MEASURE.multiline_textbbox(
                (0, 0), text, fitted, spacing=spacing, align=box.align, stroke_width=stroke_width
            )
            pad = max_size[1] / 20
            y = ink_top - pad if box.valign == "top" else ink_bottom - (max_size[1] - pad)
        ImageDraw.Draw(layer).text(
            (-x, -y),
            text,
            box.color,
            fitted,
            spacing=spacing,
            align=box.align,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        left, top = int(width * box.anchor_x), int(height * box.anchor_y)
        if box.angle:
            # Turn about the box's centre, as imgflip's editor does.
            centre = (left + max_size[0] / 2, top + max_size[1] / 2)
            layer = layer.rotate(box.angle, resample=Image.Resampling.BICUBIC, expand=True)
            left = int(centre[0] - layer.width / 2)
            top = int(centre[1] - layer.height / 2)
        image.paste(layer, (left, top), layer)
    out = io.BytesIO()
    image.save(out, "PNG", optimize=True)
    return out.getvalue()
