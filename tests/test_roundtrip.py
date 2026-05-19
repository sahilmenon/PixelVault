"""
Autonomous round-trip tests for PixelVault encoder/decoder.

Creates synthetic test files of various types and sizes, encodes each one
(video-only BVI\x01 and audio-extended BVI\x02), decodes, and asserts
byte-exact recovery.

Run:
    python test_roundtrip.py
"""

import hashlib
import os
import struct
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic test payloads
# ---------------------------------------------------------------------------

def _make_cases():
    rng = np.random.default_rng(42)

    cases = [
        # (name, bytes)
        ("empty_file",          b""),
        ("single_byte",         b"\xff"),
        ("ascii_text",          b"Hello, PixelVault!\nThis is a test of the encoder.\n" * 3),
        ("utf8_text",           "PixelVault — été à 中文\n".encode("utf-8") * 10),
        ("all_zeros",           b"\x00" * 1024),
        ("all_ones",            b"\xff" * 1024),
        ("alternating_bits",    bytes([0xAA, 0x55] * 512)),
        ("sequential_bytes",    bytes(range(256)) * 4),
        ("random_small",        rng.integers(0, 256, 512, dtype=np.uint8).tobytes()),
        ("random_medium",       rng.integers(0, 256, 8_192, dtype=np.uint8).tobytes()),
        ("random_large",        rng.integers(0, 256, 64_000, dtype=np.uint8).tobytes()),
        # A tiny fake PNG header + random body
        ("fake_png",            b"\x89PNG\r\n\x1a\n" + rng.integers(0, 256, 256, dtype=np.uint8).tobytes()),
        # A tiny fake ZIP local-file header
        ("fake_zip",            b"PK\x03\x04" + rng.integers(0, 256, 200, dtype=np.uint8).tobytes()),
        # Highly compressible (tests that padding/framing works even when data is trivial)
        ("run_of_zeros_large",  b"\x00" * 32_768),
        # File whose size sits right at a frame boundary (tests edge padding)
        ("boundary_size",       rng.integers(0, 256, 16_200, dtype=np.uint8).tobytes()),  # exactly 1 frame binary
    ]
    return cases


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

MODES = [
    ("binary",  0),
    ("palette", 2),
    # rgb: H.264 uses a lossy BGR↔YUV colour-space conversion even at CRF=0,
    # so RGB mode cannot round-trip through H.264. Documented as "local-only"
    # (use FFV1 or rawvideo for truly lossless RGB). Excluded from these tests.
]

# Use a small resolution to keep tests fast
WIDTH, HEIGHT, FPS = 320, 240, 24   # 320/4=80 lw, 240/4=60 lh → 4800 bits/frame binary

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def run_test(name, data, mode_name, mode_int, use_audio, compress, ecc_nsym, tmpdir):
    from pixelvault.encoder import encode_file, DEFAULT_BLOCK
    from pixelvault.decoder import decode_file

    block = DEFAULT_BLOCK[mode_int]
    if WIDTH % block != 0 or HEIGHT % block != 0:
        return "skip", f"res {WIDTH}x{HEIGHT} not divisible by block {block}"

    in_path = os.path.join(tmpdir, f"{name}.bin")
    Path(in_path).write_bytes(data)

    tag = ("audio" if use_audio else "video") + ("+zip" if compress else "") + (f"+ecc{ecc_nsym}" if ecc_nsym else "")
    out_video = os.path.join(tmpdir, f"{name}_{mode_name}_{tag}.mp4")
    out_dir   = os.path.join(tmpdir, f"dec_{name}_{mode_name}_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    try:
        encode_file(
            input_path=in_path,
            output_path=out_video,
            mode=mode_int,
            fps=FPS,
            width=WIDTH,
            height=HEIGHT,
            quiet=True,
            use_audio=use_audio,
            compress=compress,
            ecc_nsym=ecc_nsym,
        )
    except Exception as e:
        return "fail", f"encode error: {e}\n{traceback.format_exc()}"

    try:
        recovered_path = decode_file(out_video, output_dir=out_dir, quiet=True)
    except Exception as e:
        return "fail", f"decode error: {e}\n{traceback.format_exc()}"

    elapsed = time.perf_counter() - t0

    recovered = Path(recovered_path).read_bytes()
    if recovered != data:
        orig_h = sha256(data)
        recv_h = sha256(recovered)
        diff_at = next((i for i, (a, b) in enumerate(zip(data, recovered)) if a != b), -1)
        return "fail", (
            f"byte mismatch: orig={len(data)}B sha={orig_h}  "
            f"got={len(recovered)}B sha={recv_h}  "
            f"first diff at byte {diff_at}"
        )

    video_size = Path(out_video).stat().st_size
    return "pass", f"{elapsed:.1f}s  video={video_size//1024}KB"


def main():
    cases = _make_cases()

    results = []   # (label, status, detail)
    total = pass_count = fail_count = skip_count = 0

    with tempfile.TemporaryDirectory(prefix="pixelvault_test_") as tmpdir:
        for case_name, data in cases:
            for mode_name, mode_int in MODES:
                for use_audio in (False, True):
                    for compress in (False, True):
                        for ecc_nsym in (0, 16):
                            ecc_tag = f"+ecc{ecc_nsym}" if ecc_nsym else "      "
                            audio_tag = ("+audio" if use_audio else "video ") + ("+zip" if compress else "    ")
                            label = f"{case_name:<25}  {mode_name:<8}  {audio_tag}  {ecc_tag}  ({len(data):>7} B)"
                            total += 1

                            status, detail = run_test(
                                case_name, data, mode_name, mode_int, use_audio, compress, ecc_nsym, tmpdir
                            )

                            if status == "pass":
                                tag = PASS
                                pass_count += 1
                            elif status == "skip":
                                tag = SKIP
                                skip_count += 1
                            else:
                                tag = FAIL
                                fail_count += 1

                            line = f"  {tag}  {label}  {detail}"
                            print(line, flush=True)
                            results.append((label, status, detail))

    print()
    print("=" * 72)
    print(f"  Results: {pass_count} passed, {fail_count} failed, {skip_count} skipped  (total {total})")
    print("=" * 72)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
