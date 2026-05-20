from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

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


@dataclass(frozen=True)
class VideoDetails:
    date_text: str
    verse_reference: str
    verse_text: str
    duration: int


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


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int,
) -> int:
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (WIDTH - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=font, fill=fill)
        y += bbox[3] - bbox[1] + line_gap
    return y


def create_frame(image_path: Path, details: VideoDetails, output_path: Path) -> Path:
    frame = fit_image_to_short(image_path)
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Soft top and bottom panels keep the verse readable over busy photos.
    draw.rectangle((0, 0, WIDTH, 340), fill=(0, 0, 0, 112))
    draw.rectangle((0, 1190, WIDTH, HEIGHT), fill=(0, 0, 0, 148))

    date_font = load_font(48, bold=True)
    ref_font = load_font(76, bold=True)
    verse_font = load_font(58)

    draw_centered_text(
        draw,
        wrap_text(draw, details.date_text, date_font, 900),
        116,
        date_font,
        (255, 255, 255),
        12,
    )

    y = 1260
    y = draw_centered_text(
        draw,
        wrap_text(draw, details.verse_reference, ref_font, 900),
        y,
        ref_font,
        (255, 238, 178),
        18,
    )
    draw_centered_text(
        draw,
        wrap_text(draw, details.verse_text, verse_font, 920),
        y + 46,
        verse_font,
        (255, 255, 255),
        20,
    )

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


def render_section(details: VideoDetails, background_file, music_file) -> None:
    WORK_DIR.mkdir(exist_ok=True)
    token = file_digest(details.date_text, details.verse_reference, details.verse_text, str(details.duration))
    temp_dir = WORK_DIR / token
    image_path = save_upload(background_file, temp_dir / background_file.name)
    audio_path = save_upload(music_file, temp_dir / music_file.name)
    frame_path = create_frame(image_path, details, temp_dir / "frame.jpg")
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

    ready = all([background_file, music_file, date_text.strip(), verse_reference.strip(), verse_text.strip()])
    if ready:
        details = VideoDetails(
            date_text=date_text.strip(),
            verse_reference=verse_reference.strip(),
            verse_text=verse_text.strip(),
            duration=duration,
        )
        render_section(details, background_file, music_file)
    else:
        st.info("Upload the background image, background music, date, Bible verse, and chapter details to begin.")


if __name__ == "__main__":
    main()
