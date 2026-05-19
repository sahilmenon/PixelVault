#!/usr/bin/env python3
"""
Full YouTube roundtrip bit-accuracy test for ByteVault.

Encodes a file, uploads to YouTube, downloads it back, decodes it,
and verifies byte-exact recovery through the full YouTube pipeline.

Vault layout used automatically:
  vault/encoded/<stem>_encoded.mp4   ← encoded MP4 (uploaded)
  vault/encoded/yt_<id>.mp4          ← downloaded from YouTube
  vault/decoded/<original name>      ← recovered file

Usage:
    python test_youtube_roundtrip.py <file>
    python test_youtube_roundtrip.py vault/input/myfile.pdf
    python test_youtube_roundtrip.py <file> --ecc --compress
    python test_youtube_roundtrip.py <file> --wait 120 --retries 8
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

from bytevault.vault import VAULT_ENCODED, VAULT_DECODED, ensure_dirs


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _step(n: int, label: str):
    print(f"\n[{n}/5] {label}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Full YouTube roundtrip bit-accuracy test for ByteVault."
    )
    parser.add_argument("file", help="File to test (e.g. vault/input/myfile.pdf).")
    parser.add_argument(
        "--mode", choices=["binary", "gray4", "rgb_bin", "palette", "nibble"], default="binary",
        help="Encoding mode (default: binary). gray4 encodes 2 bits/pixel using 4 luma levels — 2× binary density, YouTube-safe.",
    )
    parser.add_argument("--compress", action="store_true", help="zlib-compress the payload before encoding.")
    parser.add_argument(
        "--ecc", dest="ecc_nsym", type=int, nargs="?", const=16, default=16, metavar="NSYM",
        help="Reed-Solomon ECC symbols per block (default: 16). Use --ecc 0 to disable.",
    )
    parser.add_argument(
        "--wait", type=int, default=300,
        help="Seconds to wait for YouTube to finish processing before first download attempt (default: 300).",
    )
    parser.add_argument(
        "--retries", type=int, default=6,
        help="Number of download retry attempts if the video is not yet available (default: 6).",
    )
    parser.add_argument(
        "--workers", type=int, default=0, metavar="N",
        help="Frame-generation threads (default: auto).",
    )
    parser.add_argument(
        "--no-hw", action="store_true",
        help="Disable hardware encoder and force libx264.",
    )
    parser.add_argument("--4k", dest="four_k", action="store_true",
                        help="Encode and download at 4K (3840×2160) for 4× data density.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress encoder/decoder progress bars.")
    args = parser.parse_args()

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    ensure_dirs()

    original_data = input_path.read_bytes()
    original_hash = _sha256(original_data)

    print("=" * 64)
    print("  ByteVault — YouTube Roundtrip Bit-Accuracy Test")
    print("=" * 64)
    print(f"  File     : {input_path.name}")
    print(f"  Size     : {_human(len(original_data))} ({len(original_data):,} bytes)")
    print(f"  SHA-256  : {original_hash}")
    print(f"  Mode     : {args.mode}")
    print(f"  Res      : {'3840×2160 (4K)' if args.four_k else '1920×1080 (1080p)'}")
    print(f"  Compress : {args.compress}")
    print(f"  ECC      : nsym={args.ecc_nsym}" + (" (disabled)" if args.ecc_nsym == 0 else ""))
    print(f"  Audio    : off (does not survive YouTube re-encoding)")
    print(f"  Encoded  -> vault/encoded/")
    print(f"  Decoded  -> vault/decoded/")
    print("=" * 64)

    from bytevault.encoder import (
        encode_file, MODE_BINARY, MODE_PALETTE, MODE_RGB_BIN, MODE_NIBBLE, MODE_GRAY4,
        WIDTH, HEIGHT, WIDTH_4K, HEIGHT_4K,
    )
    from bytevault.decoder import decode_file
    from bytevault import youtube as yt

    mode_int = {"binary": MODE_BINARY, "gray4": MODE_GRAY4, "palette": MODE_PALETTE,
                "rgb_bin": MODE_RGB_BIN, "nibble": MODE_NIBBLE}[args.mode]
    enc_width, enc_height = (WIDTH_4K, HEIGHT_4K) if args.four_k else (WIDTH, HEIGHT)
    min_h = 2160 if args.four_k else 1080

    # ── 1. Encode ─────────────────────────────────────────────────────────────
    encoded_mp4 = VAULT_ENCODED / f"{input_path.stem}_encoded.mp4"
    _step(1, f"Encoding -> {encoded_mp4}")

    t0 = time.perf_counter()
    encode_file(
        input_path=str(input_path),
        output_path=str(encoded_mp4),
        mode=mode_int,
        fps=30,
        width=enc_width,
        height=enc_height,
        quiet=args.quiet,
        use_audio=False,
        compress=args.compress,
        ecc_nsym=args.ecc_nsym,
        workers=args.workers,
        use_hw_encoder=not args.no_hw,
    )
    encode_time = time.perf_counter() - t0
    print(f"     Done in {encode_time:.1f}s  ({_human(encoded_mp4.stat().st_size)})")

    # ── 2. Upload ─────────────────────────────────────────────────────────────
    title = input_path.name
    if len(title) > 100:
        title = title[:97] + "..."
    description = (
        f"ByteVault YouTube roundtrip test\n"
        f"SHA-256: {original_hash}\n"
        f"Mode: {args.mode}  Compress: {args.compress}  ECC nsym: {args.ecc_nsym or 0}"
    )

    _step(2, "Uploading to YouTube (privacy: unlisted) ...")
    t0 = time.perf_counter()
    video_id = yt.upload(
        str(encoded_mp4),
        title=title,
        description=description,
        privacy="unlisted",
    )
    upload_time = time.perf_counter() - t0
    print(f"     Done in {upload_time:.1f}s")
    print(f"     Video ID : {video_id}")
    print(f"     URL      : https://youtu.be/{video_id}")

    # ── 3. Wait for YouTube processing ────────────────────────────────────────
    _step(3, f"Waiting {args.wait}s for YouTube to process the video ...")
    wait_end = time.time() + args.wait
    while True:
        remaining = int(wait_end - time.time())
        if remaining <= 0:
            break
        print(f"     {remaining}s remaining ...", end="\r", flush=True)
        time.sleep(min(5, remaining))
    print()

    # ── 4. Download ───────────────────────────────────────────────────────────
    downloaded_mp4 = VAULT_ENCODED / f"yt_{video_id}.mp4"
    url = f"https://www.youtube.com/watch?v={video_id}"

    _step(4, f"Downloading from YouTube -> {downloaded_mp4.name} ...")
    for attempt in range(1, args.retries + 1):
        try:
            yt.download(url, output_path=str(downloaded_mp4), min_height=min_h)
            break
        except RuntimeError as e:
            if attempt < args.retries:
                retry_wait = 30 * attempt
                print(f"     Attempt {attempt} failed — retrying in {retry_wait}s ...")
                time.sleep(retry_wait)
            else:
                print(f"\nERROR: download failed after {args.retries} attempts:\n{e}", file=sys.stderr)
                sys.exit(1)

    # ── 5. Decode ─────────────────────────────────────────────────────────────
    _step(5, f"Decoding -> vault/decoded/ ...")
    t0 = time.perf_counter()
    recovered_path = decode_file(
        video_path=str(downloaded_mp4),
        output_dir=str(VAULT_DECODED),
        quiet=args.quiet,
    )
    decode_time = time.perf_counter() - t0
    recovered_data = Path(recovered_path).read_bytes()
    recovered_hash = _sha256(recovered_data)
    print(f"     Done in {decode_time:.1f}s  -> {Path(recovered_path).name}")

    # ── Verify ────────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  VERIFICATION")
    print(f"  Original  : {len(original_data):>12,} bytes")
    print(f"  Recovered : {len(recovered_data):>12,} bytes")
    print(f"  SHA-256 original  : {original_hash}")
    print(f"  SHA-256 recovered : {recovered_hash}")
    print()

    if recovered_data == original_data:
        print("  RESULT: PASS — byte-exact match through full YouTube roundtrip")
        exit_code = 0
    else:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(original_data, recovered_data)) if a != b),
            min(len(original_data), len(recovered_data)),
        )
        diff_count = sum(a != b for a, b in zip(original_data, recovered_data))

        print("  RESULT: FAIL — data mismatch detected")
        if len(original_data) != len(recovered_data):
            print(f"  Size mismatch    : expected {len(original_data):,}  got {len(recovered_data):,}")
        print(f"  First diff at    : byte offset {first_diff:,}  "
              f"(orig=0x{original_data[first_diff]:02X} got=0x{recovered_data[first_diff]:02X})")
        print(f"  Differing bytes  : {diff_count:,} / {min(len(original_data), len(recovered_data)):,}")

        lo = max(0, first_diff - 8)
        hi = min(len(original_data), first_diff + 24)
        print(f"\n  Original  [{lo}:{hi}]: {original_data[lo:hi].hex()}")
        hi2 = min(len(recovered_data), first_diff + 24)
        print(f"  Recovered [{lo}:{hi2}]: {recovered_data[lo:hi2].hex()}")
        exit_code = 1

    print("=" * 64)
    print()
    print(f"  Encoded MP4 : {encoded_mp4}")
    print(f"  Downloaded  : {downloaded_mp4}")
    print(f"  Recovered   : {recovered_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
