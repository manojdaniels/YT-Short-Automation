# Python Video Builder

Python Streamlit application for creating MP4 videos.

## Features

### 4K Image Morph

- Upload 1-10 images in sequence.
- Select platform and orientation presets for YouTube, TikTok, Facebook, Instagram, or Twitter / X.
- Render the final MP4 in the selected platform dimensions only.
- Create a lower-resolution preview first, approve it, then render the final MP4.
- Hold on each image and transition into the next image with a soft blue glow/fade effect.
- Control hold duration, transition duration, frame rate, and glow strength.
- Uses NVIDIA NVENC (`h264_nvenc`) for MP4 encoding when FFmpeg and the GPU support it, with CPU fallback.
- Download the generated MP4 from the app.

### Bible Short

- Upload a background image.
- Upload background music.
- Enter current date, Bible chapter and verse reference, and verse text.
- Generates a preview MP4 before final approval.
- Generates downloadable final MP4.
- Enforces YouTube Shorts basics:
  - 1080 x 1920 resolution
  - 9:16 aspect ratio
  - MP4 output
  - Maximum duration of 60 seconds

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

The rendered videos are stored under the `generated` folder.

## Streamlit Community Cloud

Deploy with these settings:

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`

Streamlit Community Cloud reads `requirements.txt` for Python packages and `packages.txt` for system packages. The `packages.txt` file installs FFmpeg, which is required for MP4 rendering.
"# YT-Short-Automation" 
"# YT-Short-Automation" 
