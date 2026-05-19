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
import hashlib
import json
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

def _encode_kwargs(args, w, h):
    from bytevault.encoder import DEFAULT_BLOCK
    mode = _mode_int(args.mode)
    return dict(
        mode=mode,
        block_size=args.block_size or DEFAULT_BLOCK[mode],
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
        encrypt_password=args.password or None,
    )


def cmd_encode(args):
    _check_ffmpeg()
    ensure_dirs()
    from bytevault.encoder import encode_file

    w, h = (WIDTH_4K, HEIGHT_4K) if args.four_k else (WIDTH, HEIGHT)
    kwargs = _encode_kwargs(args, w, h)
    src = Path(args.file)

    chunk_bytes = (args.chunk_mb or 0) * 1024 * 1024

    if chunk_bytes > 0:
        _cmd_encode_chunked(args, src, w, h, kwargs, chunk_bytes)
        return

    output = args.output or str(VAULT_ENCODED / (src.stem + ".mp4"))
    video_path = encode_file(input_path=str(src), output_path=output, **kwargs)

    if args.upload:
        from bytevault import youtube as yt
        title = args.title or src.name
        video_id = yt.upload(video_path, title=title, description=args.description, privacy=args.privacy)
        print(f"YouTube ID: {video_id}")
        print(f"Decode with: python main.py decode yt:{video_id}")


def _cmd_encode_chunked(args, src: Path, w: int, h: int, kwargs: dict, chunk_bytes: int):
    """Split *src* into chunks, encode each, and write a .bvault manifest."""
    from bytevault.encoder import encode_file

    file_data = src.read_bytes()
    total_size = len(file_data)
    sha256 = hashlib.sha256(file_data).hexdigest()

    n_chunks = max(1, -(-total_size // chunk_bytes))
    pad = len(str(n_chunks - 1))
    out_dir = Path(args.output).parent if args.output else VAULT_ENCODED

    manifest = {
        "bvault_version": 1,
        "original_filename": src.name,
        "original_size": total_size,
        "sha256": sha256,
        "chunk_mb": args.chunk_mb,
        "four_k": args.four_k,
        "chunks": [],
    }
    manifest_path = out_dir / (src.stem + ".bvault")

    print(f"Chunking {src.name} ({total_size / 1e6:.1f} MB) into {n_chunks} chunk(s) of {args.chunk_mb} MB...")

    for i in range(n_chunks):
        start = i * chunk_bytes
        end = min(start + chunk_bytes, total_size)
        chunk_data = file_data[start:end]
        chunk_mp4 = out_dir / f"{src.stem}_chunk{i:0{pad}d}.mp4"

        print(f"  [{i+1}/{n_chunks}] encoding {len(chunk_data)/1e6:.1f} MB -> {chunk_mp4.name}")
        encode_file(
            input_path=str(src),
            output_path=str(chunk_mp4),
            raw_bytes=chunk_data,
            **kwargs,
        )

        entry = {
            "index": i,
            "offset": start,
            "size": len(chunk_data),
            "path": str(chunk_mp4),
            "video_id": None,
        }

        if args.upload:
            from bytevault import youtube as yt
            title = args.title or f"{src.name} [{i+1}/{n_chunks}]"
            if len(title) > 97:
                title = title[:97] + "..."
            video_id = yt.upload(str(chunk_mp4), title=title, privacy=args.privacy)
            entry["video_id"] = f"yt:{video_id}"
            print(f"    -> yt:{video_id}")

        manifest["chunks"].append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Manifest saved: {manifest_path}")
    if args.upload:
        print(f"Decode with: python main.py decode {manifest_path}")


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def cmd_decode(args):
    _check_ffmpeg()
    ensure_dirs()
    source = args.source

    if source.endswith(".bvault") or (os.path.exists(source) and source.endswith(".bvault")):
        _cmd_decode_manifest(args, source)
        return

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
                      workers=args.workers, decrypt_password=args.password or None)
    print(f"Recovered: {out}")


def _cmd_decode_manifest(args, manifest_path: str):
    """Decode a multi-chunk .bvault manifest: download+decode each chunk and reassemble."""
    from bytevault.decoder import decode_file

    manifest = json.loads(Path(manifest_path).read_text())
    chunks = sorted(manifest["chunks"], key=lambda c: c["index"])
    original_filename = manifest["original_filename"]
    original_size = manifest["original_size"]
    four_k = manifest.get("four_k", args.four_k)
    n = len(chunks)

    print(f"Reassembling '{original_filename}' from {n} chunk(s)...")

    chunk_parts: list[bytes] = []
    for c in chunks:
        idx = c["index"]
        video_id = c.get("video_id")
        local_path = c.get("path")

        if video_id and (video_id.startswith("yt:") or video_id.startswith("http")):
            from bytevault import youtube as yt
            url = _yt_url(video_id) if video_id.startswith("yt:") else video_id
            vid = url.split("v=")[-1].split("&")[0] if "v=" in url else video_id.lstrip("yt:")
            dl_path = str(VAULT_ENCODED / f"yt_{vid}.mp4")
            min_h = 2160 if four_k else 1080
            print(f"  [{idx+1}/{n}] downloading {video_id} ...")
            video_path = yt.download(url, output_path=dl_path, min_height=min_h)
        elif local_path and os.path.exists(local_path):
            video_path = local_path
            print(f"  [{idx+1}/{n}] decoding {Path(local_path).name} ...")
        else:
            print(f"ERROR: chunk {idx} has no valid path or video_id", file=sys.stderr)
            sys.exit(1)

        tmp_dir = str(VAULT_DECODED / f"_chunk_{idx:04d}")
        decode_file(video_path=video_path, output_dir=tmp_dir, quiet=args.quiet, workers=args.workers)
        chunk_files = list(Path(tmp_dir).glob("*"))
        if not chunk_files:
            print(f"ERROR: chunk {idx} decoded no files", file=sys.stderr)
            sys.exit(1)
        chunk_parts.append(chunk_files[0].read_bytes())

    reassembled = b"".join(chunk_parts)[:original_size]

    if "sha256" in manifest:
        actual = hashlib.sha256(reassembled).hexdigest()
        if actual != manifest["sha256"]:
            print(f"WARNING: SHA256 mismatch — expected {manifest['sha256']}, got {actual}", file=sys.stderr)
        else:
            print(f"SHA256 verified OK")

    out_path = Path(args.output_dir) / original_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(reassembled)
    print(f"Recovered: {out_path}  ({len(reassembled):,} bytes)")


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
    p_enc.add_argument(
        "--chunk-mb", type=int, default=None, metavar="MB",
        help=(
            "Split the input into MB-sized chunks and encode each as a separate MP4. "
            "A .bvault manifest is written next to the output files. "
            "Combine with --upload to store across multiple YouTube videos. "
            "Decode with: python main.py decode <file>.bvault"
        ),
    )
    p_enc.add_argument(
        "--password", default=None, metavar="PASS",
        help="Encrypt the payload with AES-256-GCM. Must also pass --password on decode.",
    )
    # YouTube upload
    p_enc.add_argument("--upload", action="store_true", help="Upload to YouTube after encoding.")
    p_enc.add_argument("--title", default=None, help="YouTube title (default: filename).")
    p_enc.add_argument("--description", default="Encoded with ByteVault", help="YouTube description.")
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
    p_dec.add_argument(
        "--password", default=None, metavar="PASS",
        help="Decryption password (required if the video was encoded with --password).",
    )
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
