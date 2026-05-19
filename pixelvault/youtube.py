"""
YouTube upload/download helpers for PixelVault.

Credentials:
  client_secret.json   — downloaded from Google Cloud Console (never commit)
  .youtube_token.json  — auto-created after first OAuth login (never commit)

Both files are searched in order:
  1. Current working directory
  2. ~/.pixelvault/
"""

import subprocess
from pathlib import Path

_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_SECRET_NAMES = ["client_secret.json"]
_TOKEN_NAME = ".youtube_token.json"
_PIXELVAULT_DIR = Path.home() / ".pixelvault"


def _find_file(names: list[str]) -> Path | None:
    for name in names:
        p = Path(name)
        if p.exists():
            return p
        p2 = _PIXELVAULT_DIR / name
        if p2.exists():
            return p2
    return None


def _token_path() -> Path:
    local = Path(_TOKEN_NAME)
    if local.exists():
        return local
    _PIXELVAULT_DIR.mkdir(parents=True, exist_ok=True)
    return _PIXELVAULT_DIR / _TOKEN_NAME


def get_credentials():
    """Return valid OAuth2 credentials, running the browser flow if needed."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_p = _token_path()
    creds = None

    if token_p.exists():
        creds = Credentials.from_authorized_user_file(str(token_p), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            secret_p = _find_file(_SECRET_NAMES)
            if secret_p is None:
                raise FileNotFoundError(
                    "client_secret.json not found.\n"
                    "Download it from Google Cloud Console and place it in:\n"
                    f"  {Path('client_secret.json').resolve()}\n"
                    f"  or {_PIXELVAULT_DIR / 'client_secret.json'}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secret_p), _SCOPES)
            creds = flow.run_local_server(port=0)

        token_p.write_text(creds.to_json())

    return creds


def upload(
    video_path: str,
    title: str | None = None,
    description: str = "Encoded with PixelVault",
    privacy: str = "unlisted",
    category_id: str = "22",
) -> str:
    """Upload *video_path* to YouTube and return the video ID.

    Args:
        video_path: Path to the MP4 to upload.
        title:      Video title. Defaults to the filename.
        description: Video description shown on YouTube.
        privacy:    "public", "unlisted", or "private". Default: "unlisted".
        category_id: YouTube category. 22 = People & Blogs (generic).

    Returns:
        The YouTube video ID (e.g. "dQw4w9WgXcQ").
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    video_path = str(video_path)
    if title is None:
        title = Path(video_path).stem
    # YouTube enforces a 100-character title limit
    if len(title) > 100:
        title = title[:97] + "..."

    creds = get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=8 * 1024 * 1024,  # 8 MB chunks
    )

    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    print(f"[youtube] uploading {Path(video_path).name} ({_human(Path(video_path).stat().st_size)}) ...")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"[youtube] {pct}% uploaded", end="\r", flush=True)

    print()
    video_id = response["id"]
    print(f"[youtube] done -> https://youtu.be/{video_id}")
    return video_id


def download(video_id_or_url: str, output_path: str, min_height: int = 1080) -> str:
    """Download a YouTube video to *output_path* using yt-dlp.

    Args:
        video_id_or_url: A YouTube video ID or full URL.
        output_path:     Destination file path (should end in .mp4).
        min_height:      Minimum video height to request. Use 2160 for 4K-encoded PixelVault videos.

    Returns:
        The path to the downloaded file.
    """
    if not video_id_or_url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={video_id_or_url}"
    else:
        url = video_id_or_url

    output_path = str(output_path)

    if min_height >= 2160:
        # 4K VP9/AV1 — must match the encode resolution exactly
        fmt = (
            "bestvideo[height=2160][ext=webm]+bestaudio[ext=m4a]"
            "/bestvideo[height=2160]+bestaudio"
            "/bestvideo[height>=2160]+bestaudio"
        )
        print(f"[youtube] downloading 4K (VP9) {url} ...")
    else:
        fmt = (
            "bestvideo[height>=1080]+bestaudio"
            "/bestvideo[height>=720]+bestaudio"
            "/bestvideo+bestaudio"
        )
        print(f"[youtube] downloading {url} ...")

    cmd = [
        "yt-dlp",
        "--format", fmt,
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    print(f"[youtube] saved -> {output_path}")
    return output_path


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
