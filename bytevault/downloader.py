"""
YouTube video downloader using yt-dlp.

Downloads the highest-quality available video stream (preferring 1080p MP4)
so that the decoder gets the best possible pixel fidelity.
"""

import os
from pathlib import Path
from typing import Optional


def download_video(
    url: str,
    output_dir: str = ".",
    quiet: bool = False,
    cookies_file: Optional[str] = None,
) -> str:
    """Download video at *url* to *output_dir* and return the local file path.

    Args:
        url:          YouTube watch URL or any yt-dlp-supported URL.
        output_dir:   Directory to save the downloaded file.
        quiet:        Suppress yt-dlp progress output.
        cookies_file: Path to a Netscape-format cookies.txt for private videos.

    Returns:
        Absolute path to the downloaded video file.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError(
            "yt-dlp not installed. Run: pip install yt-dlp"
        ) from e

    os.makedirs(output_dir, exist_ok=True)

    # Prefer 1080p MP4 for maximum fidelity; fall back gracefully
    format_spec = (
        "bestvideo[height=1080][ext=mp4]"
        "/bestvideo[height=1080]"
        "/bestvideo[height>=720][ext=mp4]"
        "/bestvideo[height>=720]"
        "/bestvideo[ext=mp4]"
        "/bestvideo"
        "/best[ext=mp4]"
        "/best"
    )

    downloaded_path: list[str] = []

    def progress_hook(d):
        if d["status"] == "finished":
            downloaded_path.append(d["filename"])

    ydl_opts = {
        "format": format_spec,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": quiet,
        "no_warnings": quiet,
        "progress_hooks": [progress_hook],
        # Merge video+audio into mp4 if needed (our videos are silent but just in case)
        "merge_output_format": "mp4",
        # Write video ID to make path predictable
        "restrictfilenames": False,
    }

    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        # yt-dlp may change extension after merge; prefer the hook result
        if downloaded_path:
            path = downloaded_path[-1]
        # If merged to mp4, extension may have changed
        if not os.path.exists(path):
            mp4_path = os.path.splitext(path)[0] + ".mp4"
            if os.path.exists(mp4_path):
                path = mp4_path

    path = os.path.abspath(path)
    if not quiet:
        size = Path(path).stat().st_size
        print(f"[download] → {path}  ({size:,} bytes)")
    return path
