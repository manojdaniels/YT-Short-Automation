from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from moviepy import AudioFileClip, ImageClip
except ImportError:  # moviepy < 2
    from moviepy.editor import AudioFileClip, ImageClip


WIDTH = 1080
HEIGHT = 1920
MAX_DURATION_SECONDS = 60
PREVIEW_DURATION_SECONDS = 8
FPS = 30
WORK_DIR = Path("generated")
FONT_CACHE_DIR = WORK_DIR / "fonts"
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


def create_frame(image_path: Path, details: VideoDetails, style: TextStyle, output_path: Path) -> Path:
    frame = fit_image_to_short(image_path)
    scale = TEXT_RENDER_SCALE
    canvas_size = (WIDTH * scale, HEIGHT * scale)
    text_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    glow_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    glow_draw = ImageDraw.Draw(glow_layer)
    shadow_draw = ImageDraw.Draw(shadow_layer)

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
    verse_y = max(570 * scale, 930 * scale - (verse_height // 2))

    draw_centered_text_with_contrast(
        draw,
        glow_draw,
        shadow_draw,
        wrap_text(draw, details.date_text.upper(), date_font, 940 * scale),
        245 * scale,
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
        1690 * scale,
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
    overlay = Image.alpha_composite(shadow_layer, glow_layer)
    overlay = Image.alpha_composite(overlay, text_layer)
    overlay = overlay.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    composed = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path, quality=95)
    return output_path


def render_video(frame_path: Path, audio_path: Path, output_path: Path, duration: int, bitrate: str) -> Path:
    audio_clip = AudioFileClip(str(audio_path))
    render_duration = min(duration, MAX_DURATION_SECONDS, audio_clip.duration or MAX_DURATION_SECONDS)

    audio = audio_clip.subclipped(0, render_duration) if hasattr(audio_clip, "subclipped") else audio_clip.subclip(0, render_duration)
    video = ImageClip(str(frame_path))
    video = video.with_duration(render_duration) if hasattr(video, "with_duration") else video.set_duration(render_duration)
    video = video.with_audio(audio) if hasattr(video, "with_audio") else video.set_audio(audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate=bitrate,
        preset="medium",
        logger=None,
    )

    audio.close()
    audio_clip.close()
    video.close()
    return output_path


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
        ]
    )


def render_section(details: VideoDetails, style: TextStyle, background_file, music_file) -> None:
    WORK_DIR.mkdir(exist_ok=True)
    token = file_digest(
        details.date_text,
        details.verse_reference,
        details.verse_text,
        str(details.duration),
        style_digest(style),
    )
    temp_dir = WORK_DIR / token
    image_path = save_upload(background_file, temp_dir / background_file.name)
    audio_path = save_upload(music_file, temp_dir / music_file.name)
    frame_path = create_frame(image_path, details, style, temp_dir / "frame.jpg")
    preview_path = temp_dir / "preview.mp4"
    final_path = temp_dir / "youtube_short_final.mp4"

    if st.button("Create Preview", type="primary"):
        with st.spinner("Rendering preview..."):
            render_video(frame_path, audio_path, preview_path, min(details.duration, PREVIEW_DURATION_SECONDS), "2500k")
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
        approved = st.checkbox("I approve this preview and want to generate the final MP4")
        if approved and st.button("Generate Final MP4"):
            with st.spinner("Rendering final YouTube Short..."):
                render_video(frame_path, audio_path, final_path, details.duration, "8000k")
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
                )


def main() -> None:
    st.set_page_config(page_title="YouTube Short Bible Video Builder", layout="centered")
    st.title("YouTube Short Bible Video Builder")

    st.caption("Creates vertical MP4 videos at 1080x1920, 9:16, up to 60 seconds.")

    background_file = st.file_uploader("Upload background image", type=["jpg", "jpeg", "png", "webp"])
    music_file = st.file_uploader("Upload background music", type=["mp3", "wav", "m4a", "aac", "ogg"])

    date_text = st.text_input("Current date", placeholder="20 May 2026")
    verse_reference = st.text_input("Bible chapter and verse", placeholder="Psalm 23: 1")
    verse_text = st.text_area("Bible verse text", placeholder="The Lord is my shepherd; I shall not want.")
    duration = st.slider("Video duration", min_value=5, max_value=MAX_DURATION_SECONDS, value=30, step=1)

    with st.expander("Text style", expanded=True):
        font_family = st.selectbox("Font", FONT_FAMILIES, index=0)
        text_color = st.color_picker("Text color", "#5C2F05")
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

    style = TextStyle(
        font_family=font_family,
        text_color=text_color,
        glow_color=glow_color,
        date_size=date_size,
        verse_size=verse_size,
        reference_size=reference_size,
        glow_strength=glow_strength,
        shadow_strength=shadow_strength,
    )

    ready = all([background_file, music_file, date_text.strip(), verse_reference.strip(), verse_text.strip()])
    if ready:
        details = VideoDetails(
            date_text=date_text.strip(),
            verse_reference=verse_reference.strip(),
            verse_text=verse_text.strip(),
            duration=duration,
        )
        render_section(details, style, background_file, music_file)
    else:
        st.info("Complete all fields above. The Create Preview button will appear after the image, music, date, Bible reference, and verse text are provided.")
        st.button("Create Preview", disabled=True)


if __name__ == "__main__":
    main()
