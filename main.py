#!/usr/bin/env python3
"""
ByteVault Infinite — The Eternal Encoder
CLI entry point.

Vault layout (auto-created on first use):
  vault/input/    ← drop source files here
  vault/encoded/  ← encoded MP4s land here
  vault/decoded/  ← recovered files land here

Subcommands:
  encode   — convert any file into an MP4 video
  decode   — recover the original file from a local video or YouTube URL/ID
  upload   — upload an already-encoded MP4 to YouTube
  download — download a YouTube video without decoding
"""

import argparse
import os
import sys
from pathlib import Path

from bytevault.vault import VAULT_ENCODED, VAULT_DECODED, VAULT_INPUT, ensure_dirs
from bytevault.encoder import WIDTH, HEIGHT, WIDTH_4K, HEIGHT_4K


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
    from bytevault.encoder import MODE_BINARY, MODE_RGB, MODE_PALETTE, MODE_RGB_BIN, MODE_NIBBLE, MODE_GRAY4
    return {"binary": MODE_BINARY, "rgb": MODE_RGB, "palette": MODE_PALETTE,
            "rgb_bin": MODE_RGB_BIN, "nibble": MODE_NIBBLE, "gray4": MODE_GRAY4}[name.lower()]


def _is_youtube(s: str) -> bool:
    return (
        s.startswith("https://") or
        s.startswith("http://") or
        s.startswith("yt:")
    )


def _yt_url(s: str) -> str:
    if s.startswith("yt:"):
        return f"https://www.youtube.com/watch?v={s[3:]}"
    return s


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def cmd_encode(args):
    _check_ffmpeg()
    ensure_dirs()
    from bytevault.encoder import encode_file, DEFAULT_BLOCK

    mode = _mode_int(args.mode)
    block_size = args.block_size or DEFAULT_BLOCK[mode]
    output = args.output or str(VAULT_ENCODED / (Path(args.file).stem + ".mp4"))
    w, h = (WIDTH_4K, HEIGHT_4K) if args.four_k else (WIDTH, HEIGHT)

    video_path = encode_file(
        input_path=args.file,
        output_path=output,
        mode=mode,
        block_size=block_size,
        fps=30,
        width=w,
        height=h,
        quiet=args.quiet,
        use_audio=args.audio,
        compress=args.compress,
        ecc_nsym=args.ecc_nsym,
        interleave=args.interleave,
        workers=args.workers,
        use_hw_encoder=not args.no_hw,
    )

    if args.upload:
        from bytevault import youtube as yt
        title = args.title or Path(args.file).name
        video_id = yt.upload(video_path, title=title, privacy=args.privacy)
        print(f"YouTube ID: {video_id}")
        print(f"Decode with: python main.py decode yt:{video_id}")


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def cmd_decode(args):
    _check_ffmpeg()
    ensure_dirs()
    source = args.source

    if _is_youtube(source):
        from bytevault import youtube as yt
        url = _yt_url(source)
        video_id = url.split("v=")[-1].split("&")[0]
        dl_path = str(VAULT_ENCODED / f"yt_{video_id}.mp4")
        min_h = 2160 if args.four_k else 1080
        video_path = yt.download(url, output_path=dl_path, min_height=min_h)
    else:
        if not os.path.exists(source):
            print(f"ERROR: file not found: {source}", file=sys.stderr)
            sys.exit(1)
        video_path = source

    from bytevault.decoder import decode_file
    out = decode_file(video_path=video_path, output_dir=args.output_dir, quiet=args.quiet,
                      workers=args.workers)
    print(f"Recovered: {out}")


# ---------------------------------------------------------------------------
# upload  (standalone — for already-encoded videos)
# ---------------------------------------------------------------------------

def cmd_upload(args):
    if not os.path.exists(args.video):
        print(f"ERROR: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    from bytevault import youtube as yt
    title = args.title or Path(args.video).stem
    video_id = yt.upload(
        args.video,
        title=title,
        description=args.description,
        privacy=args.privacy,
    )
    print(f"YouTube ID: {video_id}")
    print(f"Decode with: python main.py decode yt:{video_id}")


# ---------------------------------------------------------------------------
# download  (standalone — no decode)
# ---------------------------------------------------------------------------

def cmd_download(args):
    ensure_dirs()
    from bytevault import youtube as yt
    url = _yt_url(args.url)
    output = args.output or str(VAULT_ENCODED / "downloaded.mp4")
    min_h = 2160 if args.four_k else 1080
    path = yt.download(url, output_path=output, min_height=min_h)
    print(f"Downloaded: {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bytevault",
        description=(
            "ByteVault Infinite -- encode any file into a video and back.\n\n"
            "Vault folders (auto-created):\n"
            "  vault/input/    <- drop source files here\n"
            "  vault/encoded/  <- encoded MP4s land here (default output)\n"
            "  vault/decoded/  <- recovered files land here (default output)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── encode ──────────────────────────────────────────────────────────────
    p_enc = sub.add_parser("encode", help="Convert a file into an MP4 video.")
    p_enc.add_argument("file", help="File to encode (e.g. vault/input/myfile.pdf).")
    p_enc.add_argument("--mode", choices=["binary", "gray4", "rgb", "palette", "rgb_bin", "nibble"], default="binary")
    p_enc.add_argument("--block-size", type=int, default=None, metavar="N")
    p_enc.add_argument(
        "--output", "-o", default=None,
        help="Output .mp4 path (default: vault/encoded/<stem>.mp4).",
    )
    p_enc.add_argument("--audio", action="store_true", help="Use audio track for extra density (local use only — not YouTube-compatible).")
    p_enc.add_argument("--compress", action="store_true", help="zlib-compress the payload before encoding.")
    p_enc.add_argument(
        "--ecc", dest="ecc_nsym", type=int, nargs="?", const=16, default=16, metavar="NSYM",
        help="Reed-Solomon ECC symbols per block (default: 16). Use --ecc 0 to disable.",
    )
    p_enc.add_argument(
        "--interleave", action="store_true",
        help="Spread ECC bytes across frames to convert burst errors into uniform errors. Requires --ecc.",
    )
    p_enc.add_argument("--workers", type=int, default=0, metavar="N",
                       help="Frame-generation threads (default: auto).")
    p_enc.add_argument("--no-hw", action="store_true", help="Force libx264; disable NVENC/AMF/QSV.")
    p_enc.add_argument("--4k", dest="four_k", action="store_true",
                       help="Encode at 4K (3840×2160) for 4× data per frame. "
                            "YouTube serves 4K as VP9 — binary mode recommended.")
    p_enc.add_argument("--quiet", "-q", action="store_true")
    # YouTube upload
    p_enc.add_argument("--upload", action="store_true", help="Upload to YouTube after encoding.")
    p_enc.add_argument("--title", default=None, help="YouTube title (default: filename).")
    p_enc.add_argument("--privacy", choices=["public", "unlisted", "private"], default="unlisted")
    p_enc.set_defaults(func=cmd_encode)

    # ── decode ──────────────────────────────────────────────────────────────
    p_dec = sub.add_parser("decode", help="Recover the original file from a video or YouTube.")
    p_dec.add_argument("source", help="Local .mp4 path, YouTube URL, or yt:VIDEO_ID.")
    p_dec.add_argument(
        "--output-dir", "-o", default=str(VAULT_DECODED),
        help="Directory to write the recovered file (default: vault/decoded/).",
    )
    p_dec.add_argument("--4k", dest="four_k", action="store_true",
                       help="Download 4K (3840×2160) from YouTube. Required when the video was encoded with --4k.")
    p_dec.add_argument("--workers", type=int, default=0, metavar="N",
                       help="Frame-decoding threads (default: auto).")
    p_dec.add_argument("--quiet", "-q", action="store_true")
    p_dec.set_defaults(func=cmd_decode)

    # ── upload ──────────────────────────────────────────────────────────────
    p_upl = sub.add_parser("upload", help="Upload an already-encoded MP4 to YouTube.")
    p_upl.add_argument("video", help="Path to .mp4 file.")
    p_upl.add_argument("--title", default=None)
    p_upl.add_argument("--description", default="Encoded with ByteVault")
    p_upl.add_argument("--privacy", choices=["public", "unlisted", "private"], default="unlisted")
    p_upl.set_defaults(func=cmd_upload)

    # ── download ────────────────────────────────────────────────────────────
    p_dl = sub.add_parser("download", help="Download a YouTube video (no decode).")
    p_dl.add_argument("url", help="YouTube URL or yt:VIDEO_ID.")
    p_dl.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: vault/encoded/downloaded.mp4).",
    )
    p_dl.add_argument("--4k", dest="four_k", action="store_true",
                      help="Download 4K resolution (required for videos encoded with --4k).")
    p_dl.set_defaults(func=cmd_download)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
