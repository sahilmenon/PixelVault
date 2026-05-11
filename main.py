#!/usr/bin/env python3
"""
ByteVault Infinite — The Eternal Encoder
CLI entry point.

Subcommands:
  encode  — convert any file into an MP4 video
  decode  — recover the original file from a local video or YouTube URL
  upload  — upload an already-encoded MP4 to YouTube
"""

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ffmpeg():
    import shutil
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print(
            "ERROR: ffmpeg / ffprobe not found in PATH.\n"
            "Install ffmpeg: https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(1)


def _mode_int(name: str) -> int:
    from bytevault.encoder import MODE_BINARY, MODE_RGB, MODE_PALETTE
    return {"binary": MODE_BINARY, "rgb": MODE_RGB, "palette": MODE_PALETTE}[name.lower()]


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def cmd_encode(args):
    _check_ffmpeg()
    from bytevault.encoder import encode_file, DEFAULT_BLOCK

    mode = _mode_int(args.mode)
    block_size = args.block_size or DEFAULT_BLOCK[mode]

    output = args.output or (Path(args.file).stem + ".mp4")

    video_path = encode_file(
        input_path=args.file,
        output_path=output,
        mode=mode,
        block_size=block_size,
        fps=args.fps,
        width=args.width,
        height=args.height,
        quiet=args.quiet,
    )

    if args.upload:
        if not args.credentials:
            print("ERROR: --credentials required for --upload", file=sys.stderr)
            sys.exit(1)
        from bytevault.uploader import upload_video
        title = args.title or Path(args.file).name
        url = upload_video(
            video_path=video_path,
            title=title,
            description=args.description or f"ByteVault encoded: {Path(args.file).name}",
            credentials_path=args.credentials,
            token_path=args.token,
            privacy=args.privacy,
        )
        print(url)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def cmd_decode(args):
    _check_ffmpeg()
    source = args.source

    # If the source looks like a URL, download it first
    if source.startswith("http://") or source.startswith("https://"):
        from bytevault.downloader import download_video
        print(f"[decode] Downloading {source}")
        video_path = download_video(
            url=source,
            output_dir=args.output_dir,
            quiet=args.quiet,
            cookies_file=args.cookies,
        )
    else:
        if not os.path.exists(source):
            print(f"ERROR: file not found: {source}", file=sys.stderr)
            sys.exit(1)
        video_path = source

    from bytevault.decoder import decode_file
    out = decode_file(
        video_path=video_path,
        output_dir=args.output_dir,
        quiet=args.quiet,
    )
    print(f"Recovered: {out}")


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------

def cmd_upload(args):
    if not os.path.exists(args.video):
        print(f"ERROR: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not args.credentials:
        print("ERROR: --credentials is required", file=sys.stderr)
        sys.exit(1)
    from bytevault.uploader import upload_video
    url = upload_video(
        video_path=args.video,
        title=args.title or Path(args.video).stem,
        description=args.description or "Encoded by ByteVault Infinite",
        credentials_path=args.credentials,
        token_path=args.token,
        privacy=args.privacy,
        tags=args.tags.split(",") if args.tags else None,
    )
    print(url)


# ---------------------------------------------------------------------------
# download (standalone)
# ---------------------------------------------------------------------------

def cmd_download(args):
    from bytevault.downloader import download_video
    path = download_video(
        url=args.url,
        output_dir=args.output_dir,
        quiet=args.quiet,
        cookies_file=args.cookies,
    )
    print(f"Downloaded: {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bytevault",
        description="ByteVault Infinite — encode any file into a YouTube video and back.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode a file (binary mode, auto-upload)
  python main.py encode secret.zip --upload --credentials client_secrets.json

  # Encode with palette mode at larger blocks for better compression resistance
  python main.py encode photo.png --mode palette --block-size 16

  # Decode directly from a YouTube URL
  python main.py decode https://www.youtube.com/watch?v=XXXXXXXXXXX

  # Decode a local video file
  python main.py decode encoded_video.mp4 --output-dir ./recovered

  # Upload an already-encoded video
  python main.py upload encoded_video.mp4 --credentials client_secrets.json --title "My Backup"

  # Download only (no decode)
  python main.py download https://www.youtube.com/watch?v=XXXXXXXXXXX
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- encode ----
    p_enc = sub.add_parser("encode", help="Convert a file into an MP4 video.")
    p_enc.add_argument("file", help="Path to the file to encode.")
    p_enc.add_argument(
        "--mode", choices=["binary", "rgb", "palette"], default="binary",
        help=(
            "Encoding mode:\n"
            "  binary  — 1 bit/pixel (black & white blocks). Most compression-resistant. "
            "~32× overhead. [default]\n"
            "  rgb     — 3 bytes/pixel (one byte per colour channel). Most space-efficient "
            "but fragile after YouTube re-encoding.\n"
            "  palette — 1 byte/pixel mapped to 256 maximally-separated colours. "
            "Good balance of density and robustness."
        ),
    )
    p_enc.add_argument(
        "--block-size", type=int, default=None, metavar="N",
        help=(
            "Pixels per logical-pixel edge. Larger = more YouTube-resistant, "
            "fewer bytes/frame.\n"
            "Defaults: binary=4, rgb=4, palette=8. "
            "Recommended: binary≥4, palette≥8."
        ),
    )
    p_enc.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24).")
    p_enc.add_argument("--width", type=int, default=1920, help="Frame width (default: 1920).")
    p_enc.add_argument("--height", type=int, default=1080, help="Frame height (default: 1080).")
    p_enc.add_argument("--output", "-o", default=None, help="Output .mp4 path.")
    p_enc.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output.")
    # Upload options
    p_enc.add_argument("--upload", action="store_true", help="Auto-upload to YouTube after encoding.")
    p_enc.add_argument("--credentials", default=None, metavar="PATH",
                       help="Path to client_secrets.json (required for --upload).")
    p_enc.add_argument("--token", default=os.path.expanduser("~/.bytevault/token.json"),
                       metavar="PATH", help="OAuth token cache path.")
    p_enc.add_argument("--title", default=None, help="YouTube video title.")
    p_enc.add_argument("--description", default=None, help="YouTube video description.")
    p_enc.add_argument(
        "--privacy", choices=["public", "unlisted", "private"], default="unlisted",
        help="YouTube privacy setting (default: unlisted).",
    )
    p_enc.set_defaults(func=cmd_encode)

    # ---- decode ----
    p_dec = sub.add_parser(
        "decode",
        help="Recover the original file from a local video or YouTube URL.",
    )
    p_dec.add_argument(
        "source",
        help="Path to local .mp4 file OR a YouTube watch URL.",
    )
    p_dec.add_argument("--output-dir", "-o", default=".", help="Directory to write recovered file.")
    p_dec.add_argument("--cookies", default=None, metavar="PATH",
                       help="Netscape cookies.txt for downloading private/age-gated videos.")
    p_dec.add_argument("--quiet", "-q", action="store_true")
    p_dec.set_defaults(func=cmd_decode)

    # ---- upload ----
    p_upl = sub.add_parser("upload", help="Upload an already-encoded video to YouTube.")
    p_upl.add_argument("video", help="Path to the .mp4 file to upload.")
    p_upl.add_argument("--credentials", default=None, metavar="PATH", required=True,
                       help="Path to client_secrets.json.")
    p_upl.add_argument("--token", default=os.path.expanduser("~/.bytevault/token.json"))
    p_upl.add_argument("--title", default=None)
    p_upl.add_argument("--description", default="Encoded by ByteVault Infinite")
    p_upl.add_argument("--tags", default=None, help="Comma-separated tags.")
    p_upl.add_argument(
        "--privacy", choices=["public", "unlisted", "private"], default="unlisted",
    )
    p_upl.set_defaults(func=cmd_upload)

    # ---- download ----
    p_dl = sub.add_parser("download", help="Download a YouTube video (no decode).")
    p_dl.add_argument("url", help="YouTube watch URL.")
    p_dl.add_argument("--output-dir", "-o", default=".")
    p_dl.add_argument("--cookies", default=None, metavar="PATH")
    p_dl.add_argument("--quiet", "-q", action="store_true")
    p_dl.set_defaults(func=cmd_download)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
