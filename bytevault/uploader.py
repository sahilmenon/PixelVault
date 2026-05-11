"""
YouTube Data API v3 uploader.

Authentication flow:
  1. User downloads client_secrets.json from Google Cloud Console (OAuth 2.0 client ID).
  2. First run opens a browser for consent; token saved to ~/.bytevault/token.json.
  3. Subsequent runs refresh silently.

Required Google Cloud scopes: https://www.googleapis.com/auth/youtube.upload
"""

import os
import time
from pathlib import Path
from typing import Optional

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_DEFAULT_TOKEN = os.path.expanduser("~/.bytevault/token.json")


def get_youtube_service(
    client_secrets_path: str,
    token_path: str = _DEFAULT_TOKEN,
):
    """Return an authenticated YouTube API service object."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as e:
        raise ImportError(
            "Google API dependencies missing. "
            "Run: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
        ) from e

    creds: Optional[Credentials] = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, SCOPES)
            creds = flow.run_local_server(port=0)

        os.makedirs(os.path.dirname(os.path.abspath(token_path)), exist_ok=True)
        Path(token_path).write_text(creds.to_json())
        print(f"[upload] Token saved → {token_path}")

    return build("youtube", "v3", credentials=creds)


def upload_video(
    video_path: str,
    title: str,
    description: str = "Encoded by ByteVault Infinite",
    credentials_path: str = "client_secrets.json",
    token_path: str = _DEFAULT_TOKEN,
    privacy: str = "unlisted",
    tags: Optional[list] = None,
    category_id: str = "28",  # Science & Technology
) -> str:
    """Upload *video_path* to YouTube and return the watch URL.

    Args:
        video_path:         Path to the encoded .mp4 file.
        title:              YouTube video title.
        description:        YouTube video description.
        credentials_path:   Path to client_secrets.json from Google Cloud Console.
        token_path:         Where to store/load the OAuth token.
        privacy:            "public", "unlisted", or "private".
        tags:               Optional list of tag strings.
        category_id:        YouTube category ID (default 28 = Science & Technology).

    Returns:
        Full YouTube watch URL, e.g. "https://www.youtube.com/watch?v=XXXXXXXXXXX"
    """
    try:
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError
    except ImportError as e:
        raise ImportError(
            "google-api-python-client not installed. "
            "Run: pip install google-api-python-client"
        ) from e

    youtube = get_youtube_service(credentials_path, token_path)

    body = {
        "snippet": {
            "title": title[:100],           # YouTube title limit
            "description": description,
            "tags": tags or ["bytevault", "encoded"],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        chunksize=10 * 1024 * 1024,  # 10 MB chunks
        resumable=True,
    )

    print(f"[upload] Uploading {Path(video_path).name} → YouTube ({privacy})")
    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    response = None
    retry_count = 0
    max_retries = 10
    retry_exceptions = (Exception,)

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"\r[upload] {pct}%", end="", flush=True)
        except Exception as e:  # noqa: BLE001
            retry_count += 1
            if retry_count > max_retries:
                raise RuntimeError(f"Upload failed after {max_retries} retries: {e}") from e
            wait = 2 ** retry_count
            print(f"\n[upload] Error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print()  # newline after progress

    video_id = response["id"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"[upload] ✓ {url}")
    return url
