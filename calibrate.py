#!/usr/bin/env python3
"""
ByteVault YouTube calibration — measures real compression error rates and
recommends ECC settings for reliable transmission.

Workflow
--------
1. Generates a known pseudo-random test pattern (no ECC, no compression).
2. Encodes it to an MP4 with the chosen mode and block size.
3. Uploads to YouTube (unlisted).
4. Waits for processing, then downloads the re-encoded video.
5. Decodes without ECC to reveal raw compression errors.
6. Analyses per-block error counts (255-byte windows).
7. Prints recommendations for --ecc NSYM and --interleave.

Usage
-----
    python calibrate.py --mode palette
    python calibrate.py --mode binary --block-size 2
    python calibrate.py --mode rgb_bin --wait 120
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from bytevault.vault import VAULT_ENCODED, VAULT_DECODED, ensure_dirs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(n: int, label: str):
    print(f"\n[{n}] {label}", flush=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_test_pattern(n_bytes: int, seed: int = 42) -> bytes:
    """Deterministic pseudo-random bytes — reproducible across runs."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, n_bytes, dtype=np.uint8).tobytes()


def _recommend_nsym(max_errors: int, safety: int = 4) -> int:
    """Smallest even nsym that corrects *max_errors* errors, plus *safety* margin."""
    needed = 2 * max_errors + safety
    needed = needed + (needed % 2)          # round up to even
    return min(254, max(2, needed))


def _burst_length(original: bytes, recovered: bytes) -> int:
    """Longest run of consecutive wrong bytes."""
    n = min(len(original), len(recovered))
    best = cur = 0
    for i in range(n):
        if original[i] != recovered[i]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure YouTube compression error rates and recommend ECC settings."
    )
    parser.add_argument(
        "--mode", choices=["binary", "gray4", "rgb_bin", "palette", "nibble"], default="binary",
        help="Encoding mode to calibrate (default: binary).",
    )
    parser.add_argument("--block-size", type=int, default=None, metavar="N",
                        help="Block size in pixels (default: mode default).")
    parser.add_argument(
        "--size", type=int, default=512 * 1024, metavar="BYTES",
        help="Test payload size in bytes (default: 512 KB). Larger = more accurate statistics.",
    )
    parser.add_argument(
        "--wait", type=int, default=300,
        help="Seconds to wait for YouTube to process (default: 300).",
    )
    parser.add_argument(
        "--retries", type=int, default=6,
        help="Download retry attempts (default: 6).",
    )
    parser.add_argument("--no-hw", action="store_true", help="Force libx264 (disable NVENC/AMF/QSV).")
    parser.add_argument("--4k", dest="four_k", action="store_true",
                        help="Calibrate at 4K (3840×2160). YouTube serves 4K as VP9.")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    from bytevault.encoder import (
        encode_file, MODE_BINARY, MODE_GRAY4, MODE_RGB_BIN, MODE_PALETTE, MODE_NIBBLE,
        DEFAULT_BLOCK, WIDTH, HEIGHT, WIDTH_4K, HEIGHT_4K,
    )
    from bytevault.decoder import decode_file
    from bytevault import youtube as yt
    from bytevault import ecc as _ecc

    mode_map = {"binary": MODE_BINARY, "gray4": MODE_GRAY4, "rgb_bin": MODE_RGB_BIN, "palette": MODE_PALETTE, "nibble": MODE_NIBBLE}
    mode_int = mode_map[args.mode]
    block_size = args.block_size or DEFAULT_BLOCK[mode_int]
    enc_width, enc_height = (WIDTH_4K, HEIGHT_4K) if args.four_k else (WIDTH, HEIGHT)
    min_h = 2160 if args.four_k else 1080
    res_label = "3840×2160 (4K)" if args.four_k else "1920×1080"

    print("=" * 64)
    print("  ByteVault — YouTube Calibration")
    print("=" * 64)
    print(f"  Mode       : {args.mode}  (block={block_size})")
    print(f"  Resolution : {res_label}  fps=30")
    print(f"  Payload    : {args.size:,} bytes of pseudo-random data")
    print(f"  ECC        : OFF  (measuring raw error rates)")
    print("=" * 64)

    # ── 1. Generate test pattern ──────────────────────────────────────────────
    _step(1, "Generating test pattern ...")
    test_data = _make_test_pattern(args.size)
    orig_hash = _sha256(test_data)
    print(f"     SHA-256 : {orig_hash[:32]}...")

    # Write to a temp file for the encoder
    import tempfile, os
    fd, src_path = tempfile.mkstemp(suffix=".bin", prefix="bvcal_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(test_data)

        # ── 2. Encode (no ECC, no compression) ───────────────────────────────
        encoded_mp4 = VAULT_ENCODED / "calibrate_test.mp4"
        _step(2, f"Encoding → {encoded_mp4}")
        t0 = time.perf_counter()
        encode_file(
            input_path=src_path,
            output_path=str(encoded_mp4),
            mode=mode_int,
            block_size=block_size,
            fps=30,
            width=enc_width,
            height=enc_height,
            quiet=args.quiet,
            use_audio=False,
            compress=False,
            ecc_nsym=0,
            interleave=False,
            use_hw_encoder=not args.no_hw,
        )
        encode_time = time.perf_counter() - t0
        print(f"     Done in {encode_time:.1f}s  ({encoded_mp4.stat().st_size // 1024:,} KB)")

    finally:
        os.unlink(src_path)

    # ── 3. Upload ─────────────────────────────────────────────────────────────
    _step(3, "Uploading to YouTube (unlisted) ...")
    t0 = time.perf_counter()
    video_id = yt.upload(
        str(encoded_mp4),
        title="ByteVault calibration",
        description=f"Calibration run — mode={args.mode} block={block_size}",
        privacy="unlisted",
    )
    upload_time = time.perf_counter() - t0
    print(f"     Done in {upload_time:.1f}s  →  https://youtu.be/{video_id}")

    # ── 4. Wait ───────────────────────────────────────────────────────────────
    _step(4, f"Waiting {args.wait}s for YouTube to process ...")
    wait_end = time.time() + args.wait
    while True:
        remaining = int(wait_end - time.time())
        if remaining <= 0:
            break
        print(f"     {remaining}s remaining ...", end="\r", flush=True)
        time.sleep(min(5, remaining))
    print()

    # ── 5. Download ───────────────────────────────────────────────────────────
    downloaded_mp4 = VAULT_ENCODED / f"yt_{video_id}.mp4"
    url = f"https://www.youtube.com/watch?v={video_id}"
    _step(5, f"Downloading → {downloaded_mp4.name} ...")
    for attempt in range(1, args.retries + 1):
        try:
            yt.download(url, output_path=str(downloaded_mp4), min_height=min_h)
            break
        except RuntimeError as e:
            if attempt < args.retries:
                wait = 30 * attempt
                print(f"     Attempt {attempt} failed — retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"\nERROR: download failed: {e}", file=sys.stderr)
                return 1

    # ── 6. Decode (no ECC) ────────────────────────────────────────────────────
    _step(6, "Decoding (no ECC) ...")
    t0 = time.perf_counter()
    recovered_path = decode_file(
        video_path=str(downloaded_mp4),
        output_dir=str(VAULT_DECODED),
        quiet=args.quiet,
    )
    decode_time = time.perf_counter() - t0
    recovered_data = Path(recovered_path).read_bytes()
    print(f"     Done in {decode_time:.1f}s  →  {Path(recovered_path).name}")

    # ── 7. Analyse ────────────────────────────────────────────────────────────
    _step(7, "Analysing error patterns ...")

    orig = test_data[: min(len(test_data), len(recovered_data))]
    recv = recovered_data[: len(orig)]

    total_bytes = len(orig)
    wrong_bytes = sum(a != b for a, b in zip(orig, recv))
    error_rate = wrong_bytes / total_bytes if total_bytes else 0.0

    # Per-RS-block analysis (255-byte windows — the unit RS works on)
    block_errors = _ecc.block_error_counts(orig, recv, nsym=0)
    max_block_errs = max(block_errors) if block_errors else 0
    avg_block_errs = sum(block_errors) / len(block_errors) if block_errors else 0.0
    worst_blocks = sorted(enumerate(block_errors), key=lambda x: -x[1])[:5]

    burst = _burst_length(orig, recv)

    # Recommendations
    # Without interleaving: need nsym >= 2 * max_block_errs
    nsym_no_interleave = _recommend_nsym(max_block_errs)

    # With interleaving (depth ≈ n_data_frames): burst errors spread so each
    # 255-byte block receives at most ceil(255 / n_frames) bytes from any burst.
    # Conservatively estimate n_frames from the video length and bytes/frame.
    HEADER_SIZE_EST = 128
    from bytevault.encoder import _bytes_per_frame_video
    bpf = _bytes_per_frame_video(mode_int, enc_width // block_size, enc_height // block_size)
    n_frames_est = max(1, (total_bytes + HEADER_SIZE_EST) // bpf)
    # With interleave: a burst of `burst` bytes spreads across burst/bpf frames,
    # contributing burst/n_frames * 255 errors per block on average.
    effective_burst_errs = max(1, int((burst / max(n_frames_est, 1)) * 255))
    # Also consider uniform error rate
    effective_uniform_errs = max(1, int(error_rate * 255))
    nsym_interleaved = _recommend_nsym(max(effective_burst_errs, effective_uniform_errs))

    overhead_no_il = nsym_no_interleave / 255
    overhead_il = nsym_interleaved / 255

    print()
    print("=" * 64)
    print("  CALIBRATION RESULTS")
    print("=" * 64)
    if len(test_data) != len(recovered_data):
        print(f"  WARNING: size mismatch — orig={len(test_data):,}  got={len(recovered_data):,}")
    print(f"  Byte error rate      : {error_rate:.4%}  ({wrong_bytes:,} / {total_bytes:,} bytes wrong)")
    print(f"  Longest burst        : {burst:,} consecutive wrong bytes")
    print(f"  Avg errors / RS block: {avg_block_errs:.2f}  (255-byte window)")
    print(f"  Max errors / RS block: {max_block_errs}  ← critical for RS correction")
    print()
    print("  Worst RS blocks:")
    for idx, errs in worst_blocks:
        lo = idx * 255
        print(f"    block {idx:5d}  (bytes {lo:8,}–{lo+254:8,})  :  {errs} errors")
    print()
    print("─" * 64)
    print("  RECOMMENDATIONS")
    print("─" * 64)
    print()
    if wrong_bytes == 0:
        print("  No errors detected — mode is lossless through YouTube at this quality.")
        print("  ECC is unnecessary, but --ecc 8 adds cheap insurance.")
    else:
        print(f"  Without --interleave:")
        print(f"    python main.py encode --mode {args.mode} --ecc {nsym_no_interleave}")
        print(f"    overhead: +{overhead_no_il:.1%}  (corrects up to {nsym_no_interleave//2} errors/block)")
        print()
        print(f"  With --interleave  (recommended — converts bursts to uniform errors):")
        print(f"    python main.py encode --mode {args.mode} --ecc {nsym_interleaved} --interleave")
        print(f"    overhead: +{overhead_il:.1%}  (corrects up to {nsym_interleaved//2} errors/block)")
        print()
        if nsym_interleaved < nsym_no_interleave:
            savings = overhead_no_il - overhead_il
            print(f"  --interleave saves {savings:.1%} overhead by spreading burst errors.")
        print()
        print(f"  Binary mode is the most YouTube-robust choice (luma only, 0/255 threshold).")
        print(f"  Nibble mode (--mode nibble) is 4× denser and also YouTube-safe with ECC.")
    print("=" * 64)
    print()
    print(f"  Encoded MP4  : {encoded_mp4}")
    print(f"  Downloaded   : {downloaded_mp4}")
    print(f"  Decoded      : {recovered_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
