from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from moviepy import AudioFileClip, CompositeAudioClip, CompositeVideoClip, ImageClip, VideoClip, VideoFileClip
    from moviepy.audio.fx import AudioFadeOut
    from moviepy.video.fx import Loop
except ImportError:  # moviepy < 2
    from moviepy.editor import AudioFileClip, CompositeAudioClip, CompositeVideoClip, ImageClip, VideoClip, VideoFileClip
    from moviepy.audio.fx import audio_fadeout
    from moviepy.video.fx import loop


WIDTH = 1080
HEIGHT = 1920
MORPH_4K_WIDTH = 3840
MORPH_4K_HEIGHT = 2160
MORPH_PREVIEW_MAX_WIDTH = 1280
MORPH_PREVIEW_MAX_HEIGHT = 720
MORPH_MAX_IMAGES = 10
MORPH_MIN_IMAGES = 1
SPLITTER_MAX_GRID_SIZE = 12
SPLITTER_MAX_PREVIEW_IMAGES = 36
SPLITTER_4K_LANDSCAPE = (3840, 2160)
SPLITTER_4K_PORTRAIT = (2160, 3840)
SPLITTER_4K_SQUARE = (3840, 3840)
MAX_DURATION_SECONDS = 60
PREVIEW_DURATION_SECONDS = 8
FPS = 30
BACKGROUND_ZOOM = 0.08
AUDIO_FADE_OUT_SECONDS = 4
TEXT_FADE_SECONDS = 1.4
TEXT_BLUR_TRANSITION_SECONDS = 1.8
TEXT_BLUR_RADIUS = 5
CAPTION_WORDS_PER_CARD = 7
CAPTION_POSITIONS = ["Bottom", "Above Reference", "Below Verse", "Above Verse", "Top"]
BOX_FILL = (245, 240, 228, 58)
BOX_OUTLINE = (255, 255, 255, 115)
BOX_SHADOW = (22, 18, 14, 42)
WORK_DIR = Path("generated")
FONT_CACHE_DIR = WORK_DIR / "fonts"
BACKGROUND_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
TEXT_COLOR = (92, 47, 5)
TEXT_STROKE = (255, 244, 216, 150)
TEXT_SHADOW = (43, 24, 8, 125)
TEXT_RENDER_SCALE = 2
FONT_FAMILIES = [
    "Serif",
    "Playfair Display",
    "Cinzel",
    "Cormorant Garamond",
    "Montserrat",
    "Helvetica Neue",
    "Sans Serif",
    "Georgia",
    "Times New Roman",
    "Arial",
    "DejaVu Serif",
    "DejaVu Sans",
]
DOWNLOADABLE_FONTS = {
    "Playfair Display": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Cinzel": "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf",
    "Cormorant Garamond": "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
    "Montserrat": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
}

VIDEO_DIMENSIONS = {
    "YouTube": {
        "Horizontal": (3840, 2160, "4K landscape video"),
        "Vertical": (2160, 3840, "4K YouTube Shorts style video"),
    },
    "TikTok": {
        "Vertical": (1080, 1920, "TikTok/Reels portrait video"),
        "Horizontal": (1920, 1080, "Landscape upload video"),
    },
    "Facebook": {
        "Vertical": (1080, 1920, "Facebook/Reels portrait video"),
        "Horizontal": (1920, 1080, "Facebook landscape feed video"),
    },
    "Instagram": {
        "Vertical": (1080, 1920, "Instagram Reels/Stories portrait video"),
        "Horizontal": (1920, 1080, "Instagram landscape feed video"),
    },
    "Twitter / X": {
        "Horizontal": (1920, 1080, "X landscape video"),
        "Vertical": (1080, 1920, "X portrait video"),
    },
}


@dataclass(frozen=True)
class VideoDetails:
    date_text: str
    verse_reference: str
    verse_text: str
    duration: int


@dataclass(frozen=True)
class TextStyle:
    font_family: str
    text_color: str
    glow_color: str
    date_size: int
    verse_size: int
    reference_size: int
    glow_strength: int
    shadow_strength: int
    show_date_box: bool
    show_verse_box: bool
    show_reference_box: bool


@dataclass(frozen=True)
class LogoStyle:
    size_percent: int
    position: str
    opacity: int


def save_upload(uploaded_file, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    return destination


def file_digest(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:12]


def fit_image_to_short(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    source_ratio = image.width / image.height
    target_ratio = WIDTH / HEIGHT

    if source_ratio > target_ratio:
        new_height = HEIGHT
        new_width = math.ceil(HEIGHT * source_ratio)
    else:
        new_width = WIDTH
        new_height = math.ceil(WIDTH / source_ratio)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - WIDTH) // 2
    top = (new_height - HEIGHT) // 2
    return image.crop((left, top, left + WIDTH, top + HEIGHT))


def fit_image_to_canvas(image_path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    source_ratio = image.width / image.height
    target_ratio = width / height

    if source_ratio > target_ratio:
        new_height = height
        new_width = math.ceil(height * source_ratio)
    else:
        new_width = width
        new_height = math.ceil(width / source_ratio)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def contain_image_on_canvas(image_path: Path, width: int, height: int, background: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    scale = min(width / image.width, height / image.height)
    fitted_width = max(1, int(image.width * scale))
    fitted_height = max(1, int(image.height * scale))
    fitted = image.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS)

    if background == "Blurred background":
        canvas = fit_image_to_canvas(image_path, width, height).filter(ImageFilter.GaussianBlur(radius=max(12, width // 80)))
        canvas = Image.blend(canvas, Image.new("RGB", (width, height), (0, 0, 0)), 0.35)
    else:
        canvas = Image.new("RGB", (width, height), (0, 0, 0))

    left = (width - fitted_width) // 2
    top = (height - fitted_height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def prepare_canvas_image(image_path: Path, width: int, height: int, fit_mode: str) -> Image.Image:
    if fit_mode == "Fill frame":
        return fit_image_to_canvas(image_path, width, height)
    if fit_mode == "Fit with blurred background":
        return contain_image_on_canvas(image_path, width, height, "Blurred background")
    return contain_image_on_canvas(image_path, width, height, "Black bars")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_serif_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return load_font(size, bold=bold)


def downloadable_font_path(family: str) -> Path | None:
    url = DOWNLOADABLE_FONTS.get(family)
    if not url:
        return None

    FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    font_path = FONT_CACHE_DIR / f"{family.replace(' ', '_')}.ttf"
    if font_path.exists() and font_path.stat().st_size > 0:
        return font_path

    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            font_bytes = response.read()
        if font_bytes:
            font_path.write_bytes(font_bytes)
    except Exception:
        if font_path.exists() and font_path.stat().st_size == 0:
            font_path.unlink()
        return None
    return font_path if font_path.exists() and font_path.stat().st_size > 0 else None


def load_named_font(size: int, family: str, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    serif_candidates = [
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]
    sans_candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    candidates_by_family = {
        "Georgia": ["C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf"],
        "Times New Roman": ["C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"],
        "Arial": ["C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"],
        "Helvetica Neue": [
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/Library/Fonts/HelveticaNeue.ttc",
            "C:/Windows/Fonts/HelveticaNeue.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "DejaVu Serif": ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
        "DejaVu Sans": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        "Serif": serif_candidates,
        "Sans Serif": sans_candidates,
    }

    downloaded = downloadable_font_path(family)
    if downloaded:
        try:
            return ImageFont.truetype(str(downloaded), size)
        except OSError:
            pass

    for candidate in candidates_by_family.get(family, serif_candidates):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return load_serif_font(size, bold=bold) if family != "Sans Serif" else load_font(size, bold=bold)


def hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)
    return lines


def text_block_height(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
    line_gap: int,
) -> int:
    if not lines:
        return 0
    heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines]
    return sum(heights) + line_gap * (len(lines) - 1)


def fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    start_size: int,
    min_size: int,
    family: str,
    bold: bool = False,
    line_gap_ratio: float = 0.16,
) -> tuple[ImageFont.ImageFont, list[str], int]:
    for size in range(start_size, min_size - 1, -2):
        font = load_named_font(size, family, bold=bold)
        line_gap = max(8, int(size * line_gap_ratio))
        lines = wrap_text(draw, text, font, max_width)
        if text_block_height(draw, lines, font, line_gap) <= max_height:
            return font, lines, line_gap

    font = load_named_font(min_size, family, bold=bold)
    line_gap = max(8, int(min_size * line_gap_ratio))
    return font, wrap_text(draw, text, font, max_width), line_gap


def format_reference(reference: str) -> str:
    normalized = " ".join(reference.upper().split())
    return normalized.replace(" :", ":").replace(": ", ":")


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] | None = None,
    canvas_width: int = WIDTH,
) -> int:
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (canvas_width - (bbox[2] - bbox[0])) // 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        y += bbox[3] - bbox[1] + line_gap
    return y


def draw_centered_text_with_contrast(
    text_draw: ImageDraw.ImageDraw,
    glow_draw: ImageDraw.ImageDraw,
    shadow_draw: ImageDraw.ImageDraw,
    lines: list[str],
    y: int,
    font: ImageFont.ImageFont,
    line_gap: int,
    stroke_width: int,
    text_color: tuple[int, int, int, int],
    glow_color: tuple[int, int, int, int],
    shadow_color: tuple[int, int, int, int],
    canvas_width: int,
) -> int:
    for line in lines:
        bbox = text_draw.textbbox((0, 0), line, font=font)
        x = (canvas_width - (bbox[2] - bbox[0])) // 2
        shadow_draw.text((x + 8, y + 10), line, font=font, fill=shadow_color)
        glow_draw.text(
            (x, y),
            line,
            font=font,
            fill=glow_color,
            stroke_width=stroke_width,
            stroke_fill=glow_color,
        )
        text_draw.text(
            (x, y),
            line,
            font=font,
            fill=text_color,
        )
        y += bbox[3] - bbox[1] + line_gap
    return y


def scaled_box(box: tuple[int, int, int, int], scale: int) -> tuple[int, int, int, int]:
    return tuple(value * scale for value in box)


def draw_translucent_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    scale: int,
) -> None:
    scaled = scaled_box(box, scale)
    scaled_radius = radius * scale
    shadow_offset = 7 * scale
    shadow_box = (
        scaled[0] + shadow_offset,
        scaled[1] + shadow_offset,
        scaled[2] + shadow_offset,
        scaled[3] + shadow_offset,
    )
    draw.rounded_rectangle(shadow_box, radius=scaled_radius, fill=BOX_SHADOW)
    draw.rounded_rectangle(scaled, radius=scaled_radius, fill=BOX_FILL, outline=BOX_OUTLINE, width=max(1, 2 * scale))


def create_background_frame(image_path: Path, output_path: Path) -> Path:
    frame = fit_image_to_short(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(output_path, quality=95)
    return output_path


def create_text_overlay(details: VideoDetails, style: TextStyle, output_path: Path) -> Path:
    scale = TEXT_RENDER_SCALE
    canvas_size = (WIDTH * scale, HEIGHT * scale)
    text_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    glow_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    box_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    box_draw = ImageDraw.Draw(box_layer)
    draw = ImageDraw.Draw(text_layer)
    glow_draw = ImageDraw.Draw(glow_layer)
    shadow_draw = ImageDraw.Draw(shadow_layer)

    date_box = (170, 105, 910, 230)
    verse_box = (105, 615, 975, 1200)
    reference_box = (190, 1585, 890, 1700)

    if style.show_date_box:
        draw_translucent_box(box_draw, date_box, radius=0, scale=scale)
    if style.show_verse_box:
        draw_translucent_box(box_draw, verse_box, radius=22, scale=scale)
    if style.show_reference_box:
        draw_translucent_box(box_draw, reference_box, radius=0, scale=scale)

    date_font = load_named_font(style.date_size * scale, style.font_family, bold=True)
    ref_font = load_named_font(style.reference_size * scale, style.font_family, bold=True)
    text_color = hex_to_rgba(style.text_color)
    glow_color = hex_to_rgba(style.glow_color, alpha=185)
    shadow_color = (0, 0, 0, max(0, min(255, style.shadow_strength)))
    verse_text = details.verse_text.strip()
    if not (verse_text.startswith('"') or verse_text.startswith("'")):
        verse_text = f'"{verse_text}"'
    verse_font, verse_lines, verse_gap = fit_wrapped_text(
        draw,
        verse_text,
        max_width=860 * scale,
        max_height=670 * scale,
        start_size=style.verse_size * scale,
        min_size=44 * scale,
        family=style.font_family,
        bold=True,
        line_gap_ratio=0.1,
    )
    verse_height = text_block_height(draw, verse_lines, verse_font, verse_gap)
    verse_y = max((verse_box[1] + 60) * scale, ((verse_box[1] + verse_box[3]) // 2) * scale - (verse_height // 2))

    draw_centered_text_with_contrast(
        draw,
        glow_draw,
        shadow_draw,
        wrap_text(draw, details.date_text.upper(), date_font, 940 * scale),
        132 * scale,
        date_font,
        18 * scale,
        max(1, style.glow_strength * scale),
        text_color,
        glow_color,
        shadow_color,
        WIDTH * scale,
    )

    draw_centered_text_with_contrast(
        draw,
        glow_draw,
        shadow_draw,
        verse_lines,
        verse_y,
        verse_font,
        verse_gap,
        max(1, style.glow_strength * scale),
        text_color,
        glow_color,
        shadow_color,
        WIDTH * scale,
    )
    draw_centered_text_with_contrast(
        draw,
        glow_draw,
        shadow_draw,
        wrap_text(draw, format_reference(details.verse_reference), ref_font, 900 * scale),
        1612 * scale,
        ref_font,
        12 * scale,
        max(1, style.glow_strength * scale),
        text_color,
        glow_color,
        shadow_color,
        WIDTH * scale,
    )

    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=max(1, style.glow_strength * scale)))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=max(1, 2 * scale)))
    overlay = Image.alpha_composite(box_layer, shadow_layer)
    overlay = Image.alpha_composite(overlay, glow_layer)
    overlay = Image.alpha_composite(overlay, text_layer)
    overlay = overlay.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path


def create_blurred_text_overlay(text_overlay_path: Path, output_path: Path) -> Path:
    overlay = Image.open(text_overlay_path).convert("RGBA")
    blurred = overlay.filter(ImageFilter.GaussianBlur(radius=TEXT_BLUR_RADIUS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blurred.save(output_path)
    return output_path


def logo_position_coordinates(logo_size: tuple[int, int], position: str, margin: int = 60) -> tuple[int, int]:
    logo_width, logo_height = logo_size
    horizontal = {
        "Left": margin,
        "Middle": (WIDTH - logo_width) // 2,
        "Right": WIDTH - logo_width - margin,
    }
    vertical = {
        "Top": margin,
        "Middle": (HEIGHT - logo_height) // 2,
        "Bottom": HEIGHT - logo_height - margin,
    }

    parts = position.split()
    if len(parts) == 1 and parts[0] == "Center":
        return horizontal["Middle"], vertical["Middle"]
    if parts[0] == "Middle":
        return horizontal[parts[1]], vertical["Middle"]
    return horizontal[parts[1]], vertical[parts[0]]


def create_logo_overlay(logo_path: Path, style: LogoStyle, output_path: Path) -> Path:
    logo = Image.open(logo_path).convert("RGBA")
    target_width = max(24, int(WIDTH * (style.size_percent / 100)))
    target_height = max(1, int(logo.height * (target_width / logo.width)))
    logo = logo.resize((target_width, target_height), Image.Resampling.LANCZOS)

    if style.opacity < 100:
        alpha = logo.getchannel("A")
        alpha = alpha.point(lambda value: int(value * style.opacity / 100))
        logo.putalpha(alpha)

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    overlay.alpha_composite(logo, logo_position_coordinates(logo.size, style.position))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path


def chunk_caption_text(text: str, words_per_card: int = CAPTION_WORDS_PER_CARD) -> list[str]:
    words = text.replace("\n", " ").split()
    if not words:
        return []
    return [" ".join(words[index : index + words_per_card]) for index in range(0, len(words), words_per_card)]


def caption_y_for_position(position: str, caption_height: int) -> int:
    if position == "Top":
        return 320
    if position == "Above Verse":
        return 500
    if position == "Below Verse":
        return 1245
    if position == "Above Reference":
        return 1430
    return HEIGHT - caption_height - 80


def create_caption_card(text: str, position: str, output_path: Path) -> Path:
    card_width = 900
    card_height = 150
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_named_font(44, "Montserrat", bold=True)
    lines = wrap_text(draw, text, font, 820)
    line_gap = 8
    text_height = text_block_height(draw, lines, font, line_gap)
    x = (WIDTH - card_width) // 2
    y = caption_y_for_position(position, card_height)
    draw.rounded_rectangle((x, y, x + card_width, y + card_height), radius=22, fill=(0, 0, 0, 135))
    draw.rounded_rectangle((x, y, x + card_width, y + card_height), radius=22, outline=(255, 255, 255, 120), width=2)
    draw_centered_text(
        draw,
        lines,
        y + (card_height - text_height) // 2,
        font,
        (255, 255, 255, 255),
        line_gap,
        stroke_width=1,
        stroke_fill=(0, 0, 0, 180),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path


def create_caption_clips(caption_text: str | None, position: str, duration: float, output_dir: Path) -> list:
    if not caption_text:
        return []
    chunks = chunk_caption_text(caption_text)
    if not chunks:
        return []
    start_offset = min(1.0, duration * 0.1)
    available = max(0.5, duration - (start_offset * 2))
    chunk_duration = available / len(chunks)
    clips = []
    for index, chunk in enumerate(chunks):
        card_path = create_caption_card(chunk, position, output_dir / f"caption_{index + 1:02d}.png")
        clip = clip_with_duration(ImageClip(str(card_path)), chunk_duration)
        if hasattr(clip, "with_start"):
            clip = clip.with_start(start_offset + (index * chunk_duration))
        else:
            clip = clip.set_start(start_offset + (index * chunk_duration))
        clip = clip.with_position(("center", "center")) if hasattr(clip, "with_position") else clip.set_position(("center", "center"))
        clips.append(clip)
    return clips


def create_frame(image_path: Path, details: VideoDetails, style: TextStyle, output_path: Path) -> Path:
    frame = fit_image_to_short(image_path)
    overlay = Image.open(create_text_overlay(details, style, output_path.parent / "text_overlay.png")).convert("RGBA")
    composed = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path, quality=95)
    return output_path


def clip_with_duration(clip, duration: float):
    return clip.with_duration(duration) if hasattr(clip, "with_duration") else clip.set_duration(duration)


def clip_with_audio(video, audio):
    return video.with_audio(audio) if hasattr(video, "with_audio") else video.set_audio(audio)


def clip_without_audio(video):
    return video.without_audio() if hasattr(video, "without_audio") else video.set_audio(None)


def clip_with_mask(clip, mask):
    return clip.with_mask(mask) if hasattr(clip, "with_mask") else clip.set_mask(mask)


def apply_audio_fade_out(audio, duration: float):
    fade_duration = min(AUDIO_FADE_OUT_SECONDS, max(1, duration / 3))
    if hasattr(audio, "with_effects"):
        return audio.with_effects([AudioFadeOut(fade_duration)])
    return audio_fadeout(audio, fade_duration)


def is_background_video(path_or_name: str | Path) -> bool:
    return Path(str(path_or_name)).suffix.lower() in BACKGROUND_VIDEO_EXTENSIONS


def loop_clip_to_duration(clip, duration: float):
    if clip.duration and clip.duration >= duration:
        return clip.subclipped(0, duration) if hasattr(clip, "subclipped") else clip.subclip(0, duration)
    if hasattr(clip, "with_effects"):
        return clip.with_effects([Loop(duration=duration)])
    return loop(clip, duration=duration)


def fit_clip_to_short(clip):
    clip_width, clip_height = clip.size
    scale = max(WIDTH / clip_width, HEIGHT / clip_height)
    if hasattr(clip, "resized"):
        clip = clip.resized(scale)
    else:
        clip = clip.resize(scale)
    resized_width, resized_height = clip.size
    if hasattr(clip, "cropped"):
        return clip.cropped(x_center=resized_width / 2, y_center=resized_height / 2, width=WIDTH, height=HEIGHT)
    return clip.crop(x_center=resized_width / 2, y_center=resized_height / 2, width=WIDTH, height=HEIGHT)


def create_background_clip(background_path: Path, render_duration: float, include_video_audio: bool):
    if is_background_video(background_path):
        source = VideoFileClip(str(background_path))
        clip = loop_clip_to_duration(source, render_duration)
        clip = fit_clip_to_short(clip)
        if hasattr(clip, "resized"):
            clip = clip.resized(lambda t: 1 + BACKGROUND_ZOOM * (t / render_duration))
        else:
            clip = clip.resize(lambda t: 1 + BACKGROUND_ZOOM * (t / render_duration))
        clip = clip.with_position(("center", "center")) if hasattr(clip, "with_position") else clip.set_position(("center", "center"))
        if not include_video_audio:
            clip = clip_without_audio(clip)
        return clip, source

    background = clip_with_duration(ImageClip(str(background_path)), render_duration)
    if hasattr(background, "resized"):
        background = background.resized(lambda t: 1 + BACKGROUND_ZOOM * (t / render_duration))
    else:
        background = background.resize(lambda t: 1 + BACKGROUND_ZOOM * (t / render_duration))
    background = background.with_position(("center", "center")) if hasattr(background, "with_position") else background.set_position(("center", "center"))
    return background, None


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def text_opacity_at(time_value: float, duration: float) -> float:
    fade_in = smoothstep(time_value / min(TEXT_FADE_SECONDS, duration / 3))
    fade_out = smoothstep((duration - time_value) / min(TEXT_FADE_SECONDS, duration / 3))
    return min(fade_in, fade_out)


def blur_opacity_at(time_value: float, duration: float) -> float:
    visible_text = text_opacity_at(time_value, duration)
    intro_blur = 1 - smoothstep(time_value / min(TEXT_BLUR_TRANSITION_SECONDS, duration / 2))
    outro_blur = 1 - smoothstep((duration - time_value) / min(TEXT_BLUR_TRANSITION_SECONDS, duration / 2))
    return visible_text * max(intro_blur, outro_blur) * 0.85


def animated_alpha_mask(image_path: Path, duration: float, opacity_function) -> VideoClip:
    alpha = np.asarray(Image.open(image_path).convert("RGBA").getchannel("A"), dtype=float) / 255.0

    def make_frame(time_value: float):
        return alpha * opacity_function(time_value)

    mask = VideoClip(make_frame, is_mask=True, duration=duration)
    return mask


def render_video(
    background_path: Path,
    text_overlay_path: Path,
    logo_overlay_path: Path | None,
    audio_path: Path,
    narration_path: Path | None,
    caption_text: str | None,
    caption_position: str,
    output_path: Path,
    duration: int,
    bitrate: str,
    include_background_video_audio: bool,
    use_gpu: bool,
) -> Path:
    audio_clip = AudioFileClip(str(audio_path))
    render_duration = min(duration, MAX_DURATION_SECONDS, audio_clip.duration or MAX_DURATION_SECONDS)

    music_audio = audio_clip.subclipped(0, render_duration) if hasattr(audio_clip, "subclipped") else audio_clip.subclip(0, render_duration)
    background, background_source = create_background_clip(background_path, render_duration, include_background_video_audio)
    audio_layers = [music_audio]
    if include_background_video_audio and getattr(background, "audio", None):
        audio_layers.append(background.audio)
    narration_clip = None
    if narration_path:
        narration_clip = AudioFileClip(str(narration_path))
        narration_audio = narration_clip.subclipped(0, render_duration) if hasattr(narration_clip, "subclipped") else narration_clip.subclip(0, render_duration)
        audio_layers.append(narration_audio)
    audio = CompositeAudioClip(audio_layers) if len(audio_layers) > 1 else music_audio
    audio = apply_audio_fade_out(audio, render_duration)

    blurred_text_path = create_blurred_text_overlay(text_overlay_path, text_overlay_path.parent / "text_overlay_blur.png")
    sharp_text = clip_with_duration(ImageClip(str(text_overlay_path)), render_duration)
    blurred_text = clip_with_duration(ImageClip(str(blurred_text_path)), render_duration)
    sharp_mask = animated_alpha_mask(text_overlay_path, render_duration, lambda t: text_opacity_at(t, render_duration))
    blurred_mask = animated_alpha_mask(blurred_text_path, render_duration, lambda t: blur_opacity_at(t, render_duration))
    sharp_text = clip_with_mask(sharp_text, sharp_mask)
    blurred_text = clip_with_mask(blurred_text, blurred_mask)
    sharp_text = sharp_text.with_position(("center", "center")) if hasattr(sharp_text, "with_position") else sharp_text.set_position(("center", "center"))
    blurred_text = blurred_text.with_position(("center", "center")) if hasattr(blurred_text, "with_position") else blurred_text.set_position(("center", "center"))
    layers = [background, blurred_text, sharp_text]
    logo_clip = None
    if logo_overlay_path:
        logo_clip = clip_with_duration(ImageClip(str(logo_overlay_path)), render_duration)
        logo_clip = logo_clip.with_position(("center", "center")) if hasattr(logo_clip, "with_position") else logo_clip.set_position(("center", "center"))
        layers.append(logo_clip)
    caption_clips = create_caption_clips(caption_text, caption_position, render_duration, output_path.parent / f"{output_path.stem}_captions")
    layers.extend(caption_clips)
    video = CompositeVideoClip(layers, size=(WIDTH, HEIGHT))
    video = clip_with_audio(video, audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4_with_gpu_fallback(video, output_path, FPS, bitrate, use_gpu=use_gpu, audio_codec="aac")

    audio.close()
    audio_clip.close()
    background.close()
    if background_source:
        background_source.close()
    if narration_clip:
        narration_clip.close()
    if logo_clip:
        logo_clip.close()
    for caption_clip in caption_clips:
        caption_clip.close()
    blurred_text.close()
    sharp_text.close()
    blurred_mask.close()
    sharp_mask.close()
    video.close()
    return output_path


def uploaded_files_digest(uploaded_files, *parts: str) -> str:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(str(uploaded_file.size).encode("utf-8"))
        digest.update(uploaded_file.getbuffer())
    for part in parts:
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:12]


def save_ordered_images(uploaded_files, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        suffix = Path(uploaded_file.name).suffix.lower() or ".png"
        image_path = save_upload(uploaded_file, destination / f"{index:02d}_{Path(uploaded_file.name).stem}{suffix}")
        paths.append(image_path)
    return paths


def detect_grid_layout(image: Image.Image, max_grid_size: int = SPLITTER_MAX_GRID_SIZE) -> tuple[int, int, float]:
    sample = image.convert("L")
    scale = min(900 / sample.width, 900 / sample.height, 1.0)
    if scale < 1.0:
        sample = sample.resize((max(1, int(sample.width * scale)), max(1, int(sample.height * scale))), Image.Resampling.BILINEAR)

    pixels = np.asarray(sample, dtype=np.float32)
    if pixels.shape[0] < 2 or pixels.shape[1] < 2:
        return 1, 1, 0.0

    vertical_changes = np.mean(np.abs(pixels[:, 1:] - pixels[:, :-1]), axis=0)
    horizontal_changes = np.mean(np.abs(pixels[1:, :] - pixels[:-1, :]), axis=1)

    def boundary_score(changes: np.ndarray, size: int, pieces: int) -> float:
        if pieces <= 1:
            return 0.0
        window_radius = max(2, min(8, size // 180))
        boundaries = [round((size * part) / pieces) - 1 for part in range(1, pieces)]
        baseline = float(np.mean(changes))
        spread = float(np.std(changes)) or 1.0
        scores = []
        for boundary in boundaries:
            start = min(max(boundary - window_radius, 0), len(changes) - 1)
            end = min(max(boundary + window_radius + 1, start + 1), len(changes))
            scores.append((float(np.max(changes[start:end])) - baseline) / spread)
        return float(np.mean(scores))

    best_rows, best_cols, best_score = 1, 1, -999.0
    sample_width, sample_height = sample.size
    for rows in range(1, max_grid_size + 1):
        for cols in range(1, max_grid_size + 1):
            if rows * cols == 1:
                continue
            if image.width // cols < 32 or image.height // rows < 32:
                continue

            seam_score = boundary_score(vertical_changes, sample_width, cols) + boundary_score(horizontal_changes, sample_height, rows)
            divisibility_bonus = 0.0
            if image.width % cols == 0:
                divisibility_bonus += 0.25
            if image.height % rows == 0:
                divisibility_bonus += 0.25
            grid_complexity_penalty = 0.025 * (rows + cols)
            score = seam_score + divisibility_bonus - grid_complexity_penalty

            if score > best_score:
                best_rows, best_cols, best_score = rows, cols, score

    confidence = max(0.0, min(1.0, (best_score + 0.5) / 4.0))
    return best_rows, best_cols, confidence


def splitter_4k_size(width: int, height: int) -> tuple[int, int]:
    if width > height:
        return SPLITTER_4K_LANDSCAPE
    if height > width:
        return SPLITTER_4K_PORTRAIT
    return SPLITTER_4K_SQUARE


def resize_image_fill(image: Image.Image, width: int, height: int) -> Image.Image:
    source_ratio = image.width / image.height
    target_ratio = width / height

    if source_ratio > target_ratio:
        new_height = height
        new_width = math.ceil(height * source_ratio)
    else:
        new_width = width
        new_height = math.ceil(width / source_ratio)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def split_grid_image(image: Image.Image, rows: int, cols: int, output_dir: Path, base_name: str, upscale_to_4k: bool = True) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_width = image.width // cols
    cell_height = image.height // rows
    usable_width = cell_width * cols
    usable_height = cell_height * rows
    left_margin = (image.width - usable_width) // 2
    top_margin = (image.height - usable_height) // 2

    saved_paths: list[Path] = []
    for row in range(rows):
        for col in range(cols):
            left = left_margin + (col * cell_width)
            top = top_margin + (row * cell_height)
            crop = image.crop((left, top, left + cell_width, top + cell_height))
            if upscale_to_4k:
                crop = resize_image_fill(crop, *splitter_4k_size(cell_width, cell_height))
            suffix = "_4k" if upscale_to_4k else ""
            output_path = output_dir / f"{base_name}_{row + 1:02d}_{col + 1:02d}{suffix}.png"
            crop.save(output_path)
            saved_paths.append(output_path)
    return saved_paths


def zip_files(file_paths: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in file_paths:
            archive.write(file_path, arcname=file_path.name)
    return buffer.getvalue()


def render_image_splitter_section(uploaded_file) -> None:
    WORK_DIR.mkdir(exist_ok=True)
    st.subheader("Image Splitter")
    st.caption("Upload one collage image, detect the grid, then export every tile as a separate PNG.")

    if not uploaded_file:
        st.info("Upload a collage image to detect and split it.")
        return

    token = uploaded_files_digest([uploaded_file], "image-splitter")
    temp_dir = WORK_DIR / token / "splitter"
    source_suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    source_path = save_upload(uploaded_file, temp_dir / f"source{source_suffix}")
    image = Image.open(source_path).convert("RGB")
    auto_rows, auto_cols, confidence = detect_grid_layout(image)

    st.image(image, caption=f"{uploaded_file.name} - {image.width}x{image.height}px", use_container_width=True)
    st.write(
        {
            "detected_rows": auto_rows,
            "detected_columns": auto_cols,
            "detected_images": auto_rows * auto_cols,
            "confidence": f"{confidence:.0%}",
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        rows = st.number_input("Rows", min_value=1, max_value=SPLITTER_MAX_GRID_SIZE, value=auto_rows, step=1)
    with col2:
        cols = st.number_input("Columns", min_value=1, max_value=SPLITTER_MAX_GRID_SIZE, value=auto_cols, step=1)

    rows = int(rows)
    cols = int(cols)
    cell_width = image.width // cols
    cell_height = image.height // rows
    if cell_width < 1 or cell_height < 1:
        st.error("The selected rows and columns are too high for this image.")
        return

    upscale_to_4k = st.toggle("Upscale output images to 4K", value=True)
    output_width, output_height = splitter_4k_size(cell_width, cell_height) if upscale_to_4k else (cell_width, cell_height)
    image_count = rows * cols
    st.write(
        {
            "output_images": image_count,
            "cropped_size_each": f"{cell_width}x{cell_height}px",
            "final_size_each": f"{output_width}x{output_height}px",
        }
    )

    if confidence < 0.35:
        st.warning("The automatic detection is uncertain. Adjust rows and columns if the preview does not match your collage.")

    if st.button("Split Image", type="primary", key="splitter_create_files"):
        output_dir = temp_dir / f"{rows}x{cols}_{'4k' if upscale_to_4k else 'original'}"
        split_paths = split_grid_image(image, rows, cols, output_dir, Path(uploaded_file.name).stem or "split", upscale_to_4k)
        st.session_state["splitter_paths"] = [str(path) for path in split_paths]
        st.session_state["splitter_token"] = token
        st.session_state["splitter_grid"] = f"{rows}x{cols}_{upscale_to_4k}"

    current_paths = st.session_state.get("splitter_paths", [])
    current_token = st.session_state.get("splitter_token")
    current_grid = st.session_state.get("splitter_grid")
    if current_paths and current_token == token and current_grid == f"{rows}x{cols}_{upscale_to_4k}":
        split_paths = [Path(path) for path in current_paths if Path(path).exists()]
        if split_paths:
            st.success(f"Created {len(split_paths)} image files.")
            preview_paths = split_paths[:SPLITTER_MAX_PREVIEW_IMAGES]
            preview_cols = st.columns(min(3, len(preview_paths)))
            for index, split_path in enumerate(preview_paths):
                with preview_cols[index % len(preview_cols)]:
                    st.image(str(split_path), caption=split_path.name, use_container_width=True)
            if len(split_paths) > SPLITTER_MAX_PREVIEW_IMAGES:
                st.caption(f"Showing the first {SPLITTER_MAX_PREVIEW_IMAGES} images.")

            zip_suffix = "4k_images" if upscale_to_4k else "images"
            zip_name = f"{Path(uploaded_file.name).stem or 'split'}_{rows}x{cols}_{zip_suffix}.zip"
            st.download_button(
                "Download All Images",
                data=zip_files(split_paths),
                file_name=zip_name,
                mime="application/zip",
                key="splitter_download_zip",
            )


def preview_dimensions(width: int, height: int) -> tuple[int, int]:
    scale = min(MORPH_PREVIEW_MAX_WIDTH / width, MORPH_PREVIEW_MAX_HEIGHT / height, 1.0)
    preview_width = max(2, int(width * scale) // 2 * 2)
    preview_height = max(2, int(height * scale) // 2 * 2)
    return preview_width, preview_height


def ffmpeg_supports_encoder(encoder_name: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    return encoder_name in result.stdout


def preferred_video_codec(use_gpu: bool) -> tuple[str, list[str]]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg
    if use_gpu and ffmpeg_supports_encoder("h264_nvenc"):
        return "h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    return "libx264", ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def write_mp4_with_gpu_fallback(video, output_path: Path, fps: int, bitrate: str, use_gpu: bool, audio_codec: str | None = None) -> str:
    codec, ffmpeg_params = preferred_video_codec(use_gpu)
    kwargs = {
        "fps": fps,
        "codec": codec,
        "bitrate": bitrate,
        "preset": "medium",
        "ffmpeg_params": ffmpeg_params,
        "logger": None,
    }
    if audio_codec:
        kwargs["audio_codec"] = audio_codec
    else:
        kwargs["audio"] = False

    try:
        video.write_videofile(str(output_path), **kwargs)
        return codec
    except Exception:
        if codec != "h264_nvenc":
            raise
        if output_path.exists():
            output_path.unlink()
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["codec"] = "libx264"
        fallback_kwargs["ffmpeg_params"] = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        video.write_videofile(str(output_path), **fallback_kwargs)
        return "libx264"


def scaled_image(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    scaled_width = max(width, int(width * scale))
    scaled_height = max(height, int(height * scale))
    enlarged = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    left = (scaled_width - width) // 2
    top = (scaled_height - height) // 2
    return enlarged.crop((left, top, left + width, top + height))


def glowing_fade_frame(first: Image.Image, second: Image.Image, alpha: float, glow_strength: float) -> Image.Image:
    eased = smoothstep(alpha)
    wave = math.sin(math.pi * eased)
    zoom_first = scaled_image(first, 1 + 0.035 * eased)
    zoom_second = scaled_image(second, 1 + 0.035 * (1 - eased))
    blended = Image.blend(zoom_first, zoom_second, eased)

    blur_radius = max(0.1, glow_strength * 8 * wave)
    glow = blended.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    frame = Image.blend(blended, glow, min(0.45, 0.14 + 0.31 * wave))

    if wave > 0:
        tint = Image.new("RGB", frame.size, (80, 170, 255))
        frame = Image.blend(frame, tint, min(0.18, 0.18 * glow_strength * wave))
    return frame


def add_highlight_bloom(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return image
    arr = np.asarray(image, dtype=np.float32)
    luminance = arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722
    mask = np.clip((luminance - 135) / 120, 0, 1)[:, :, None]
    highlights = np.clip(arr * mask, 0, 255).astype(np.uint8)
    highlight_image = Image.fromarray(highlights, "RGB")
    bloom = highlight_image.filter(ImageFilter.GaussianBlur(radius=max(3, int(min(image.size) * 0.035 * strength))))
    return Image.blend(image, Image.blend(image, bloom, 0.72), min(0.65, strength * 0.42))


def add_vignette(image: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return image
    width, height = image.size
    y, x = np.ogrid[-1:1:height * 1j, -1:1:width * 1j]
    radius = np.sqrt((x * 0.82) ** 2 + (y * 1.05) ** 2)
    mask = np.clip((radius - 0.35) / 0.75, 0, 1)
    factor = 1 - mask * amount
    arr = np.asarray(image, dtype=np.float32)
    arr *= factor[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def ken_burns_frame(image: Image.Image, progress: float, scene_index: int, intensity: float) -> Image.Image:
    eased = smoothstep(progress)
    direction = -1 if scene_index % 2 else 1
    scale = 1 + intensity * (0.045 + 0.035 * eased)
    width, height = image.size
    scaled_width = max(width, int(width * scale))
    scaled_height = max(height, int(height * scale))
    enlarged = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    max_x = scaled_width - width
    max_y = scaled_height - height
    pan_x = int(max_x * (0.5 + direction * (eased - 0.5) * 0.34))
    pan_y = int(max_y * (0.5 + (0.5 - eased) * 0.18))
    pan_x = max(0, min(max_x, pan_x))
    pan_y = max(0, min(max_y, pan_y))
    return enlarged.crop((pan_x, pan_y, pan_x + width, pan_y + height))


def cinematic_morph_frame(
    first: Image.Image,
    second: Image.Image,
    alpha: float,
    scene_index: int,
    glow_strength: float,
) -> Image.Image:
    eased = smoothstep(alpha)
    wave = math.sin(math.pi * eased)
    first_motion = ken_burns_frame(first, eased, scene_index, 1.0)
    second_motion = ken_burns_frame(second, eased, scene_index + 1, 1.0)
    base = Image.blend(first_motion, second_motion, eased)
    base = add_highlight_bloom(base, glow_strength * (0.65 + wave * 0.9))

    haze = base.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(base.size) * 0.012 * (1 + wave)))))
    base = Image.blend(base, haze, min(0.38, 0.12 + wave * 0.28))
    warm_light = Image.new("RGB", base.size, (255, 224, 170))
    cool_shadow = Image.new("RGB", base.size, (32, 52, 78))
    base = Image.blend(base, cool_shadow, 0.04)
    base = Image.blend(base, warm_light, min(0.16, wave * 0.16 * glow_strength))
    return add_vignette(base, 0.22)


def render_morph_video(
    image_paths: list[Path],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    hold_seconds: float,
    transition_seconds: float,
    glow_strength: float,
    bitrate: str,
    use_gpu: bool,
    transition_style: str,
    fit_mode: str,
) -> Path:
    images = [prepare_canvas_image(path, width, height, fit_mode) for path in image_paths]
    if not images:
        raise ValueError("At least one image is required.")

    duration = (len(images) * hold_seconds) + (max(0, len(images) - 1) * transition_seconds)
    duration = max(1.0, duration)

    def make_frame(time_value: float):
        cursor = 0.0
        for index, image in enumerate(images):
            if time_value < cursor + hold_seconds or index == len(images) - 1:
                hold_progress = (time_value - cursor) / max(0.001, hold_seconds) if hold_seconds else 1.0
                if transition_style == "Cinematic morph":
                    return np.asarray(add_vignette(ken_burns_frame(image, hold_progress, index, 1.0), 0.18))
                return np.asarray(image)
            cursor += hold_seconds

            if time_value < cursor + transition_seconds:
                alpha = (time_value - cursor) / max(0.001, transition_seconds)
                if transition_style == "Cinematic morph":
                    frame = cinematic_morph_frame(image, images[index + 1], alpha, index, glow_strength)
                else:
                    frame = glowing_fade_frame(image, images[index + 1], alpha, glow_strength)
                return np.asarray(frame)
            cursor += transition_seconds
        return np.asarray(images[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video = VideoClip(make_frame, duration=duration)
    write_mp4_with_gpu_fallback(video, output_path, fps, bitrate, use_gpu)
    video.close()
    return output_path


def render_morph_section(image_files) -> None:
    WORK_DIR.mkdir(exist_ok=True)
    st.subheader("Image Morph Video")
    st.caption("Upload images in order. The default cinematic style uses slow push-in motion, highlight bloom, and a long luminous dissolve between scenes.")

    platform = st.selectbox("Platform", list(VIDEO_DIMENSIONS.keys()))
    orientations = list(VIDEO_DIMENSIONS[platform].keys())
    orientation = st.radio("Video orientation", orientations, horizontal=True)
    width, height, preset_description = VIDEO_DIMENSIONS[platform][orientation]
    preview_width, preview_height = preview_dimensions(width, height)
    transition_style = st.selectbox("Transition style", ["Cinematic morph", "Blue glow fade"], index=0)
    fit_mode = st.selectbox("Image fit", ["Fit with black bars", "Fit with blurred background", "Fill frame"], index=0)

    col1, col2 = st.columns(2)
    with col1:
        hold_seconds = st.slider("Hold per image", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        fps = st.selectbox("Frame rate", [24, 30, 60], index=1)
    with col2:
        transition_seconds = st.slider("Transition duration", min_value=0.5, max_value=6.0, value=3.0, step=0.1)
        glow_strength = st.slider("Glow strength", min_value=0.0, max_value=1.8, value=1.1, step=0.1)

    use_gpu = st.toggle("Use NVIDIA GPU encoding when available", value=True)
    gpu_ready = ffmpeg_supports_encoder("h264_nvenc")
    if use_gpu and gpu_ready:
        st.success("NVIDIA NVENC encoding is available and will be used for MP4 rendering.")
    elif use_gpu:
        st.warning("NVIDIA NVENC was not found in FFmpeg. Rendering will use CPU encoding.")
    else:
        st.info("GPU encoding is off. Rendering will use CPU encoding.")

    image_count = len(image_files) if image_files else 0
    if image_files:
        st.write("Render order:")
        for index, image_file in enumerate(image_files, start=1):
            st.write(f"{index}. {image_file.name}")

    if image_count < MORPH_MIN_IMAGES:
        st.info("Upload at least one image to create a video.")
        st.button("Create Preview", disabled=True, key="morph_preview_disabled_no_images")
        return
    if image_count > MORPH_MAX_IMAGES:
        st.error(f"Please upload no more than {MORPH_MAX_IMAGES} images.")
        st.button("Create Preview", disabled=True, key="morph_preview_disabled_too_many_images")
        return

    duration = (image_count * hold_seconds) + (max(0, image_count - 1) * transition_seconds)
    st.write(
        {
            "platform": platform,
            "orientation": orientation,
            "images": image_count,
            "final_resolution": f"{width}x{height}",
            "preview_resolution": f"{preview_width}x{preview_height}",
            "preset": preset_description,
            "transition": transition_style,
            "image_fit": fit_mode,
            "fps": fps,
            "duration_seconds": round(duration, 2),
            "format": "MP4",
            "encoder": "h264_nvenc" if use_gpu and gpu_ready else "libx264",
        }
    )

    token = uploaded_files_digest(
        image_files,
        platform,
        orientation,
        str(width),
        str(height),
        str(fps),
        str(hold_seconds),
        str(transition_seconds),
        str(glow_strength),
        str(use_gpu),
        transition_style,
        fit_mode,
    )
    temp_dir = WORK_DIR / f"morph_{token}"
    preview_path = temp_dir / "image_transition_preview.mp4"
    final_path = temp_dir / f"{platform.lower().replace(' / ', '_').replace(' ', '_')}_{orientation.lower()}_final.mp4"

    if st.button("Create Preview", type="primary", key="morph_create_preview"):
        with st.spinner("Rendering preview..."):
            image_paths = save_ordered_images(image_files, temp_dir / "source_images")
            render_morph_video(
                image_paths,
                preview_path,
                preview_width,
                preview_height,
                fps,
                hold_seconds,
                transition_seconds,
                glow_strength,
                "6000k",
                use_gpu,
                transition_style,
                fit_mode,
            )
        st.session_state["morph_preview_path"] = str(preview_path)
        st.session_state["morph_preview_token"] = token
        st.session_state.pop("morph_final_path", None)

    preview_value = st.session_state.get("morph_preview_path")
    current_preview = st.session_state.get("morph_preview_token") == token
    rendered_preview = Path(preview_value) if preview_value and current_preview else None
    if rendered_preview and rendered_preview.exists():
        st.subheader("Preview")
        st.video(str(rendered_preview))
        approved = st.checkbox("I approve this preview and want to render the final MP4", key="morph_approve_final")
        if approved and st.button("Generate Final MP4", key="morph_generate_final"):
            with st.spinner("Rendering final MP4... higher resolutions can take a while."):
                image_paths = save_ordered_images(image_files, temp_dir / "source_images")
                render_morph_video(
                    image_paths,
                    final_path,
                    width,
                    height,
                    fps,
                    hold_seconds,
                    transition_seconds,
                    glow_strength,
                    "45000k" if width >= MORPH_4K_WIDTH or height >= MORPH_4K_HEIGHT else "16000k",
                    use_gpu,
                    transition_style,
                    fit_mode,
                )
            st.session_state["morph_final_path"] = str(final_path)

    final_value = st.session_state.get("morph_final_path")
    rendered_final = Path(final_value) if final_value else None
    if rendered_final and rendered_final.exists():
        duration = probe_duration(rendered_final)
        file_size_mb = rendered_final.stat().st_size / (1024 * 1024)
        st.success("Final MP4 generated.")
        st.video(str(rendered_final))
        st.write(
            {
                "file": rendered_final.name,
                "resolution": f"{width}x{height}",
                "duration_seconds": round(duration or 0, 2),
                "file_size_mb": round(file_size_mb, 2),
            }
        )
        with rendered_final.open("rb") as file:
            st.download_button(
                "Download Final MP4",
                data=file,
                file_name=rendered_final.name,
                mime="video/mp4",
                key="morph_download_final",
            )


def probe_duration(video_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def format_excel_value(value) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%B %d, %Y")
    return str(value).strip()


def find_excel_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for key, original in normalized.items():
        if any(candidate in key for candidate in candidates):
            return original
    return None


def read_batch_rows(sheet_file, expected_count: int) -> tuple[list[VideoDetails], pd.DataFrame]:
    dataframe = pd.read_excel(sheet_file)
    dataframe = dataframe.dropna(how="all").reset_index(drop=True)
    date_col = find_excel_column(list(dataframe.columns), ["date", "current date"])
    verse_col = find_excel_column(list(dataframe.columns), ["verse", "verses", "bible verse", "verse text"])
    reference_col = find_excel_column(
        list(dataframe.columns),
        ["chapter", "chapter number", "chapter verse", "chapter and verse", "reference", "verse number"],
    )
    missing = [
        label
        for label, column in [("Date", date_col), ("Verses", verse_col), ("Chapter/verse number", reference_col)]
        if column is None
    ]
    if missing:
        raise ValueError(f"Missing required Excel column(s): {', '.join(missing)}")
    if len(dataframe) < expected_count:
        raise ValueError(f"The Excel sheet has {len(dataframe)} usable row(s), but {expected_count} video(s) were requested.")

    rows: list[VideoDetails] = []
    for _, row in dataframe.head(expected_count).iterrows():
        rows.append(
            VideoDetails(
                date_text=format_excel_value(row[date_col]),
                verse_reference=format_excel_value(row[reference_col]),
                verse_text=format_excel_value(row[verse_col]),
                duration=0,
            )
        )
    return rows, dataframe.head(expected_count).copy()


def display_batch_rows(rows: list[VideoDetails], completed_indexes: set[int]) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "Row": index + 1,
                "Date": row.date_text,
                "Verses": row.verse_text,
                "Chapter/Verse": row.verse_reference,
                "Status": "Completed" if index in completed_indexes else "Pending",
            }
            for index, row in enumerate(rows)
        ]
    )

    def highlight_completed(row):
        color = "background-color: #D1FADF" if row["Status"] == "Completed" else ""
        return [color] * len(row)

    st.dataframe(dataframe.style.apply(highlight_completed, axis=1), use_container_width=True)


def zip_video_files(video_paths: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for video_path in video_paths:
            archive.write(video_path, arcname=video_path.name)
    buffer.seek(0)
    return buffer.getvalue()


def style_digest(style: TextStyle) -> str:
    return "|".join(
        [
            style.font_family,
            style.text_color,
            style.glow_color,
            str(style.date_size),
            str(style.verse_size),
            str(style.reference_size),
            str(style.glow_strength),
            str(style.shadow_strength),
            str(style.show_date_box),
            str(style.show_verse_box),
            str(style.show_reference_box),
        ]
    )


def logo_style_digest(logo_style: LogoStyle | None) -> str:
    if not logo_style:
        return "no-logo"
    return "|".join([str(logo_style.size_percent), logo_style.position, str(logo_style.opacity)])


def render_section(
    details: VideoDetails,
    style: TextStyle,
    background_file,
    music_file,
    narration_file,
    caption_position: str,
    logo_file,
    logo_style: LogoStyle | None,
    include_background_video_audio: bool,
    use_gpu: bool,
) -> None:
    WORK_DIR.mkdir(exist_ok=True)
    token = file_digest(
        details.date_text,
        details.verse_reference,
        details.verse_text,
        str(details.duration),
        style_digest(style),
        background_file.name,
        str(background_file.size),
        music_file.name,
        str(music_file.size),
        narration_file.name if narration_file else "no-vo",
        str(narration_file.size) if narration_file else "0",
        caption_position,
        logo_file.name if logo_file else "no-logo",
        str(logo_file.size) if logo_file else "0",
        logo_style_digest(logo_style),
        str(include_background_video_audio),
        str(use_gpu),
    )
    temp_dir = WORK_DIR / token
    background_upload_path = save_upload(background_file, temp_dir / background_file.name)
    audio_path = save_upload(music_file, temp_dir / music_file.name)
    narration_path = save_upload(narration_file, temp_dir / narration_file.name) if narration_file else None
    if is_background_video(background_upload_path):
        background_path = background_upload_path
    else:
        background_path = create_background_frame(background_upload_path, temp_dir / "background.jpg")
    text_overlay_path = create_text_overlay(details, style, temp_dir / "text_overlay.png")
    logo_overlay_path = None
    if logo_file and logo_style:
        logo_upload_path = save_upload(logo_file, temp_dir / logo_file.name)
        logo_overlay_path = create_logo_overlay(logo_upload_path, logo_style, temp_dir / "logo_overlay.png")
    preview_path = temp_dir / "preview.mp4"
    final_path = temp_dir / "youtube_short_final.mp4"

    if st.button("Create Preview", type="primary", key="bible_create_preview"):
        with st.spinner("Rendering preview..."):
            render_video(
                background_path,
                text_overlay_path,
                logo_overlay_path,
                audio_path,
                narration_path,
                details.verse_text if narration_path else None,
                caption_position,
                preview_path,
                min(details.duration, PREVIEW_DURATION_SECONDS),
                "2500k",
                include_background_video_audio,
                use_gpu,
            )
        st.session_state["preview_path"] = str(preview_path)
        st.session_state["final_ready_token"] = token

    if (
        st.session_state.get("final_ready_token") == token
        and st.session_state.get("preview_path")
        and Path(st.session_state["preview_path"]).exists()
    ):
        st.subheader("Preview")
        st.video(st.session_state["preview_path"])

    if st.session_state.get("final_ready_token") == token:
        approved = st.checkbox("I approve this preview and want to generate the final MP4", key="bible_approve_final")
        if approved and st.button("Generate Final MP4", key="bible_generate_final"):
            with st.spinner("Rendering final YouTube Short..."):
                render_video(
                    background_path,
                    text_overlay_path,
                    logo_overlay_path,
                    audio_path,
                    narration_path,
                    details.verse_text if narration_path else None,
                    caption_position,
                    final_path,
                    details.duration,
                    "8000k",
                    include_background_video_audio,
                    use_gpu,
                )
            duration = probe_duration(final_path)
            file_size_mb = final_path.stat().st_size / (1024 * 1024)

            st.success("Final MP4 generated.")
            st.video(str(final_path))
            st.write(
                {
                    "file": final_path.name,
                    "format": "MP4",
                    "resolution": f"{WIDTH}x{HEIGHT}",
                    "aspect_ratio": "9:16",
                    "duration_seconds": round(duration or details.duration, 2),
                    "file_size_mb": round(file_size_mb, 2),
                }
            )
            with final_path.open("rb") as file:
                st.download_button(
                    "Download Final MP4",
                    data=file,
                    file_name="youtube_short_final.mp4",
                    mime="video/mp4",
                    key="bible_download_final",
                )


def repeated_file_for_index(files, index: int):
    return files[index % len(files)]


def render_batch_section(
    batch_rows: list[VideoDetails],
    style: TextStyle,
    background_files,
    music_files,
    narration_files,
    caption_position: str,
    logo_file,
    logo_style: LogoStyle | None,
    duration: int,
    include_background_video_audio: bool,
    use_gpu: bool,
) -> None:
    WORK_DIR.mkdir(exist_ok=True)
    batch_count = len(batch_rows)
    token = uploaded_files_digest(
        background_files,
        style_digest(style),
        logo_file.name if logo_file else "no-logo",
        str(logo_file.size) if logo_file else "0",
        str(duration),
        str(include_background_video_audio),
        str(use_gpu),
        *[f"bg:{file.name}:{file.size}" for file in background_files],
        *[f"music:{file.name}:{file.size}" for file in music_files],
        *[f"vo:{file.name}:{file.size}" for file in narration_files],
        caption_position,
        *[f"{row.date_text}|{row.verse_text}|{row.verse_reference}" for row in batch_rows],
    )
    temp_dir = WORK_DIR / f"batch_{token}"
    logo_overlay_path = None
    if logo_file and logo_style:
        logo_upload_path = save_upload(logo_file, temp_dir / logo_file.name)
        logo_overlay_path = create_logo_overlay(logo_upload_path, logo_style, temp_dir / "logo_overlay.png")

    preview_paths = [temp_dir / f"preview_{index + 1:02d}.mp4" for index in range(batch_count)]
    final_paths = [temp_dir / f"youtube_short_{index + 1:02d}.mp4" for index in range(batch_count)]

    preview_complete = st.session_state.get("batch_preview_token") == token
    final_complete = st.session_state.get("batch_final_token") == token
    completed_indexes = set(range(batch_count)) if final_complete else set()
    display_batch_rows(batch_rows, completed_indexes)

    if st.button("Create Batch Previews", type="primary", key="batch_create_previews"):
        with st.spinner(f"Rendering {batch_count} preview video(s)..."):
            source_dir = temp_dir / "backgrounds"
            music_dir = temp_dir / "music"
            narration_dir = temp_dir / "narration"
            source_dir.mkdir(parents=True, exist_ok=True)
            music_dir.mkdir(parents=True, exist_ok=True)
            narration_dir.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(batch_rows):
                background_file = repeated_file_for_index(background_files, index)
                music_file = repeated_file_for_index(music_files, index)
                narration_file = repeated_file_for_index(narration_files, index) if narration_files else None
                background_upload_path = save_upload(background_file, source_dir / f"{index + 1:02d}_{background_file.name}")
                audio_path = save_upload(music_file, music_dir / f"{index + 1:02d}_{music_file.name}")
                narration_path = save_upload(narration_file, narration_dir / f"{index + 1:02d}_{narration_file.name}") if narration_file else None
                if is_background_video(background_upload_path):
                    background_path = background_upload_path
                else:
                    background_path = create_background_frame(background_upload_path, temp_dir / f"background_{index + 1:02d}.jpg")
                row_details = VideoDetails(row.date_text, row.verse_reference, row.verse_text, duration)
                text_overlay_path = create_text_overlay(row_details, style, temp_dir / f"text_overlay_{index + 1:02d}.png")
                render_video(
                    background_path,
                    text_overlay_path,
                    logo_overlay_path,
                    audio_path,
                    narration_path,
                    row.verse_text if narration_path else None,
                    caption_position,
                    preview_paths[index],
                    min(duration, PREVIEW_DURATION_SECONDS),
                    "2500k",
                    include_background_video_audio,
                    use_gpu,
                )
        st.session_state["batch_preview_token"] = token
        st.session_state.pop("batch_final_token", None)

    preview_complete = st.session_state.get("batch_preview_token") == token
    if preview_complete and all(path.exists() for path in preview_paths):
        st.subheader("Batch Previews")
        for row_start in range(0, len(preview_paths), 4):
            columns = st.columns(4)
            for offset, column in enumerate(columns):
                index = row_start + offset
                if index >= len(preview_paths):
                    continue
                with column:
                    st.caption(f"{index + 1}. {batch_rows[index].date_text}")
                    st.video(str(preview_paths[index]))
                    if st.button("Open", key=f"batch_open_preview_{token}_{index}"):
                        st.session_state["batch_selected_preview"] = index

        selected_preview = st.session_state.get("batch_selected_preview")
        if selected_preview is not None and 0 <= selected_preview < len(preview_paths):
            st.subheader(f"Selected Preview {selected_preview + 1}")
            st.caption(f"{batch_rows[selected_preview].date_text} - {batch_rows[selected_preview].verse_reference}")
            st.video(str(preview_paths[selected_preview]))

        approved = st.checkbox("I approve these previews and want to render all final MP4 files", key="batch_approve_final")
        if approved and st.button("Generate Batch Final MP4s", key="batch_generate_final"):
            with st.spinner(f"Rendering {batch_count} final video(s)..."):
                source_dir = temp_dir / "backgrounds"
                music_dir = temp_dir / "music"
                narration_dir = temp_dir / "narration"
                for index, row in enumerate(batch_rows):
                    background_file = repeated_file_for_index(background_files, index)
                    music_file = repeated_file_for_index(music_files, index)
                    narration_file = repeated_file_for_index(narration_files, index) if narration_files else None
                    background_upload_path = source_dir / f"{index + 1:02d}_{background_file.name}"
                    audio_path = music_dir / f"{index + 1:02d}_{music_file.name}"
                    narration_path = narration_dir / f"{index + 1:02d}_{narration_file.name}" if narration_file else None
                    if not background_upload_path.exists():
                        background_upload_path = save_upload(background_file, background_upload_path)
                    if not audio_path.exists():
                        audio_path = save_upload(music_file, audio_path)
                    if narration_file and narration_path and not narration_path.exists():
                        narration_path = save_upload(narration_file, narration_path)
                    if is_background_video(background_upload_path):
                        background_path = background_upload_path
                    else:
                        background_path = create_background_frame(background_upload_path, temp_dir / f"background_{index + 1:02d}.jpg")
                    row_details = VideoDetails(row.date_text, row.verse_reference, row.verse_text, duration)
                    text_overlay_path = create_text_overlay(row_details, style, temp_dir / f"text_overlay_{index + 1:02d}.png")
                    render_video(
                        background_path,
                        text_overlay_path,
                        logo_overlay_path,
                        audio_path,
                        narration_path,
                        row.verse_text if narration_path else None,
                        caption_position,
                        final_paths[index],
                        duration,
                        "8000k",
                        include_background_video_audio,
                        use_gpu,
                    )
            st.session_state["batch_final_token"] = token

    final_complete = st.session_state.get("batch_final_token") == token
    if final_complete and all(path.exists() for path in final_paths):
        st.success("Batch final MP4 files generated.")
        display_batch_rows(batch_rows, set(range(batch_count)))
        st.write(
            {
                "videos": batch_count,
                "format": "MP4",
                "resolution": f"{WIDTH}x{HEIGHT}",
                "duration_seconds_each": duration,
            }
        )
        st.download_button(
            "Download All Final MP4s",
            data=zip_video_files(final_paths),
            file_name="youtube_shorts_batch.zip",
            mime="application/zip",
            key="batch_download_zip",
        )


def main() -> None:
    st.set_page_config(page_title="Python Video Builder", layout="centered")
    st.title("Python Video Builder")

    splitter_tab, morph_tab, bible_tab = st.tabs(["Image Splitter", "4K Image Morph", "Bible Short"])

    with splitter_tab:
        collage_file = st.file_uploader(
            "Upload collage image",
            type=["jpg", "jpeg", "png", "webp"],
            key="splitter_image_upload",
        )
        render_image_splitter_section(collage_file)

    with morph_tab:
        image_files = st.file_uploader(
            "Upload 1-10 images in sequence",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="morph_image_uploads",
        )
        render_morph_section(image_files)

    with bible_tab:
        st.caption("Creates vertical MP4 videos at 1080x1920, 9:16, up to 60 seconds.")

        batch_mode = st.toggle("Automate batch video process", value=False)
        batch_count = 1
        sheet_file = None
        background_files = None
        music_files = None
        narration_files = []
        background_file = None
        narration_file = None
        repeat_backgrounds = False
        repeat_music = False
        repeat_narration = False
        add_narration = st.toggle("Add narration VO audio", value=False)
        caption_position = "Bottom"
        if batch_mode:
            with st.expander("Batch automation inputs", expanded=True):
                st.info("Upload an Excel sheet with columns named Date, Verses, and Chapter number. One row will be used for each video.")
                batch_count = st.number_input("Number of videos to create", min_value=1, max_value=10, value=1, step=1)
                sheet_file = st.file_uploader(
                    "Upload Excel sheet (.xlsx/.xls)",
                    type=["xlsx", "xls"],
                    help="Required columns: Date, Verses, Chapter number",
                    key="batch_excel_upload",
                )
                repeat_backgrounds = st.toggle("Repeat uploaded backgrounds if fewer than video count", value=False)
                background_files = st.file_uploader(
                    f"Upload background image/video file(s){' to repeat' if repeat_backgrounds else f' ({batch_count} required)'}",
                    type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "m4v", "webm"],
                    accept_multiple_files=True,
                )
                repeat_music = st.toggle("Repeat uploaded background music if fewer than video count", value=False)
                music_files = st.file_uploader(
                    f"Upload background music file(s){' to repeat' if repeat_music else f' ({batch_count} required)'}",
                    type=["mp3", "wav", "m4a", "aac", "ogg"],
                    accept_multiple_files=True,
                )
                if add_narration:
                    repeat_narration = st.toggle("Repeat uploaded narration VO if fewer than video count", value=False)
                    narration_files = st.file_uploader(
                        f"Upload narration VO file(s){' to repeat' if repeat_narration else f' ({batch_count} required)'}",
                        type=["mp3", "wav", "m4a", "aac", "ogg"],
                        accept_multiple_files=True,
                    )
        else:
            background_file = st.file_uploader(
                "Upload background image or video",
                type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "m4v", "webm"],
            )
            music_file = st.file_uploader("Upload background music", type=["mp3", "wav", "m4a", "aac", "ogg"])
            if add_narration:
                narration_file = st.file_uploader("Upload narration VO audio", type=["mp3", "wav", "m4a", "aac", "ogg"])
        if batch_mode:
            music_file = None
        logo_file = st.file_uploader("Upload logo image", type=["png", "jpg", "jpeg", "webp"])

        if not batch_mode:
            date_text = st.text_input("Current date", placeholder="20 May 2026")
            verse_reference = st.text_input("Bible chapter and verse", placeholder="Psalm 23: 1")
            verse_text = st.text_area("Bible verse text", placeholder="The Lord is my shepherd; I shall not want.")
        else:
            date_text = ""
            verse_reference = ""
            verse_text = ""
        duration = st.slider("Video duration", min_value=5, max_value=MAX_DURATION_SECONDS, value=30, step=1)

        if add_narration:
            with st.expander("Closed captions", expanded=True):
                caption_position = st.selectbox("Caption position", CAPTION_POSITIONS, index=0)
                st.caption("Captions are generated from the verse text for each video and timed across the narration.")

        with st.expander("Text style", expanded=True):
            font_family = st.selectbox("Font", FONT_FAMILIES, index=0)
            text_color = st.color_picker("Text color", "#FFFFFF")
            glow_color = st.color_picker("Soft contrast color", "#FFF4D8")
            col1, col2, col3 = st.columns(3)
            with col1:
                date_size = st.slider("Date size", min_value=40, max_value=130, value=82, step=2)
            with col2:
                verse_size = st.slider("Verse size", min_value=44, max_value=120, value=82, step=2)
            with col3:
                reference_size = st.slider("Reference size", min_value=34, max_value=100, value=58, step=2)
            col4, col5 = st.columns(2)
            with col4:
                glow_strength = st.slider("Text clarity glow", min_value=0, max_value=6, value=2, step=1)
            with col5:
                shadow_strength = st.slider("Shadow strength", min_value=0, max_value=220, value=80, step=10)
            box_col1, box_col2, box_col3 = st.columns(3)
            with box_col1:
                show_date_box = st.toggle("Date box", value=True)
            with box_col2:
                show_verse_box = st.toggle("Verse box", value=False)
            with box_col3:
                show_reference_box = st.toggle("Reference box", value=True)

        with st.expander("Logo", expanded=False):
            logo_positions = [
                "Top Left",
                "Top Middle",
                "Top Right",
                "Middle Left",
                "Center",
                "Middle Right",
                "Bottom Left",
                "Bottom Middle",
                "Bottom Right",
            ]
            logo_size_percent = st.slider("Logo size (% of video width)", min_value=5, max_value=50, value=14, step=1)
            logo_position = st.selectbox("Logo position", logo_positions, index=2)
            logo_opacity = st.slider("Logo opacity", min_value=0, max_value=100, value=85, step=5)
            if logo_file is None:
                st.info("Upload a logo image to show it on the video.")

        with st.expander("Video rendering", expanded=False):
            if batch_mode:
                background_is_video = bool(background_files) and any(is_background_video(file.name) for file in background_files)
            else:
                background_is_video = background_file is not None and is_background_video(background_file.name)
            include_background_video_audio = st.toggle(
                "Unmute background video audio",
                value=False,
                disabled=not background_is_video,
                help="When enabled, the uploaded video's sound is mixed with the uploaded background music.",
            )
            use_gpu = st.toggle("Use NVIDIA GPU/CUDA encoding when available", value=True)
            gpu_ready = ffmpeg_supports_encoder("h264_nvenc")
            if use_gpu and gpu_ready:
                st.success("NVIDIA NVENC/CUDA encoding is available and will be used.")
            elif use_gpu:
                st.warning("NVIDIA NVENC/CUDA encoding was not found. The app will fall back to CPU encoding.")
            else:
                st.info("GPU encoding is off. The app will use CPU encoding.")

        style = TextStyle(
            font_family=font_family,
            text_color=text_color,
            glow_color=glow_color,
            date_size=date_size,
            verse_size=verse_size,
            reference_size=reference_size,
            glow_strength=glow_strength,
            shadow_strength=shadow_strength,
            show_date_box=show_date_box,
            show_verse_box=show_verse_box,
            show_reference_box=show_reference_box,
        )
        logo_style = (
            LogoStyle(
                size_percent=logo_size_percent,
                position=logo_position,
                opacity=logo_opacity,
            )
            if logo_file
            else None
        )

        if batch_mode:
            batch_rows: list[VideoDetails] = []
            batch_sheet_ready = False
            if sheet_file:
                try:
                    batch_rows, _ = read_batch_rows(sheet_file, int(batch_count))
                    batch_sheet_ready = True
                    display_batch_rows(batch_rows, set())
                except Exception as error:
                    st.error(str(error))
            background_count = len(background_files) if background_files else 0
            music_count = len(music_files) if music_files else 0
            narration_count = len(narration_files) if narration_files else 0
            if background_files and not repeat_backgrounds and background_count != int(batch_count):
                st.error(f"Please upload exactly {batch_count} background image/video file(s). You uploaded {background_count}.")
            if background_files and repeat_backgrounds and background_count < 1:
                st.error("Please upload at least one background image/video file to repeat.")
            if music_files and not repeat_music and music_count != int(batch_count):
                st.error(f"Please upload exactly {batch_count} background music file(s). You uploaded {music_count}.")
            if music_files and repeat_music and music_count < 1:
                st.error("Please upload at least one background music file to repeat.")
            if add_narration and narration_files and not repeat_narration and narration_count != int(batch_count):
                st.error(f"Please upload exactly {batch_count} narration VO file(s). You uploaded {narration_count}.")
            if add_narration and narration_files and repeat_narration and narration_count < 1:
                st.error("Please upload at least one narration VO file to repeat.")
            background_ready = bool(background_files) and (repeat_backgrounds or background_count == int(batch_count))
            music_ready = bool(music_files) and (repeat_music or music_count == int(batch_count))
            narration_ready = (not add_narration) or (bool(narration_files) and (repeat_narration or narration_count == int(batch_count)))
            ready = all([batch_sheet_ready, background_ready, music_ready, narration_ready])
            if ready:
                render_batch_section(
                    batch_rows,
                    style,
                    background_files,
                    music_files,
                    narration_files if add_narration else [],
                    caption_position,
                    logo_file,
                    logo_style,
                    duration,
                    include_background_video_audio,
                    use_gpu,
                )
            else:
                st.info("Complete the batch count, Excel sheet, background files, music files, and any requested VO files to begin. Exact counts are required unless repeat is enabled.")
                st.button("Create Batch Previews", disabled=True, key="batch_preview_disabled_incomplete")
        elif all([background_file, music_file, date_text.strip(), verse_reference.strip(), verse_text.strip()]):
            if add_narration and not narration_file:
                st.info("Upload narration VO audio or turn off narration.")
                st.button("Create Preview", disabled=True, key="bible_preview_disabled_no_vo")
                return
            details = VideoDetails(
                date_text=date_text.strip(),
                verse_reference=verse_reference.strip(),
                verse_text=verse_text.strip(),
                duration=duration,
            )
            render_section(
                details,
                style,
                background_file,
                music_file,
                narration_file if add_narration else None,
                caption_position,
                logo_file,
                logo_style,
                include_background_video_audio,
                use_gpu,
            )
        else:
            st.info("Complete all fields above. The Create Preview button will appear after the image, music, date, Bible reference, and verse text are provided.")
            st.button("Create Preview", disabled=True, key="bible_preview_disabled_incomplete")


if __name__ == "__main__":
    main()
