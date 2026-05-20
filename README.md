# YouTube Short Bible Video Builder

Python Streamlit application for creating Bible verse YouTube Shorts.

## Features

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
