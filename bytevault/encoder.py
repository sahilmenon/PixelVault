"""
File → video encoder.

Header layout (128 bytes, always at the start of the encoded payload):
  Offset  Size  Field
  0       4     Magic: b"BVI\\x01"
  4       1     Mode: 0=binary, 1=rgb, 2=palette
  5       1     Block size (pixels per logical-pixel edge)
  6       2     Filename length (uint16 LE)
  8       64    Filename (UTF-8, null-padded)
  72      8     Original file size in bytes (uint64 LE)
  80      4     Padding bytes appended to align payload (uint32 LE)
  84      44    Reserved (zeros)

The entire (header + file bytes) stream is encoded as logical pixels into frames,
then each logical pixel is rendered as a block_size × block_size square so that
YouTube's lossy codec cannot corrupt individual bytes.
"""

import os
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .palette import PALETTE_BGR

MAGIC = b"BVI\x01"
HEADER_SIZE = 128

MODE_BINARY = 0   # 1 bit per logical pixel (black=0, white=1)
MODE_RGB = 1      # 3 bytes per logical pixel (one byte per channel)
MODE_PALETTE = 2  # 1 byte per logical pixel (mapped to 256-colour palette)

_MODE_NAMES = {MODE_BINARY: "binary", MODE_RGB: "rgb", MODE_PALETTE: "palette"}

DEFAULT_BLOCK = {MODE_BINARY: 4, MODE_RGB: 4, MODE_PALETTE: 8}
WIDTH = 1920
HEIGHT = 1080


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _build_header(
    mode: int,
    block_size: int,
    filename: str,
    file_size: int,
    padding_bytes: int,
) -> bytes:
    fname_b = filename.encode("utf-8")[:64]
    header = struct.pack(
        "<4sBBH64sQI",
        MAGIC, mode, block_size,
        len(fname_b), fname_b.ljust(64, b"\x00"),
        file_size, padding_bytes,
    )
    return header + b"\x00" * (HEADER_SIZE - len(header))


def _calc_padding(payload_len: int, mode: int, lw: int, lh: int) -> int:
    if mode == MODE_BINARY:
        unit = lw * lh          # bits per frame
        total_bits = payload_len * 8
        rem = total_bits % unit
        pad_bits = (unit - rem) % unit
        return pad_bits // 8    # bytes of padding (may be fractional bits; negligible)
    elif mode == MODE_RGB:
        unit = lw * lh * 3
        rem = payload_len % unit
        return (unit - rem) % unit
    else:  # PALETTE
        unit = lw * lh
        rem = payload_len % unit
        return (unit - rem) % unit


# ---------------------------------------------------------------------------
# Frame generators  (streaming: yields one numpy frame at a time)
# ---------------------------------------------------------------------------

def _stream_binary(payload: bytes, lw: int, lh: int, bs: int):
    bpf = lw * lh  # bits per frame
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    rem = len(bits) % bpf
    if rem:
        bits = np.append(bits, np.zeros(bpf - rem, dtype=np.uint8))
    bit_frames = bits.reshape(-1, lh, lw)
    for bf in bit_frames:
        # np.repeat scales each logical pixel to a bs×bs block
        gray = np.repeat(np.repeat(bf, bs, axis=0), bs, axis=1) * 255
        yield np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


def _stream_rgb(payload: bytes, lw: int, lh: int, bs: int):
    bpf = lw * lh * 3
    arr = np.frombuffer(payload, dtype=np.uint8)
    rem = len(arr) % bpf
    if rem:
        arr = np.append(arr, np.zeros(bpf - rem, dtype=np.uint8))
    for chunk in arr.reshape(-1, lh, lw, 3):
        bgr = chunk[:, :, ::-1]  # RGB → BGR
        yield np.repeat(np.repeat(bgr, bs, axis=0), bs, axis=1).astype(np.uint8)


def _stream_palette(payload: bytes, lw: int, lh: int, bs: int):
    bpf = lw * lh
    arr = np.frombuffer(payload, dtype=np.uint8)
    rem = len(arr) % bpf
    if rem:
        arr = np.append(arr, np.zeros(bpf - rem, dtype=np.uint8))
    for chunk in arr.reshape(-1, lh, lw):
        bgr_small = PALETTE_BGR[chunk]  # (lh, lw, 3)
        yield np.repeat(np.repeat(bgr_small, bs, axis=0), bs, axis=1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_file(
    input_path: str,
    output_path: str,
    mode: int = MODE_BINARY,
    block_size: int = None,
    fps: int = 24,
    width: int = WIDTH,
    height: int = HEIGHT,
    quiet: bool = False,
) -> str:
    """Encode *input_path* into an MP4 video at *output_path*.

    Returns output_path.
    """
    if block_size is None:
        block_size = DEFAULT_BLOCK[mode]

    if width % block_size != 0 or height % block_size != 0:
        raise ValueError(
            f"Resolution {width}×{height} must be divisible by block_size={block_size}"
        )

    lw, lh = width // block_size, height // block_size
    input_path = Path(input_path)
    file_data = input_path.read_bytes()
    file_size = len(file_data)

    # Two-pass header: compute padding, then bake it in
    placeholder_header = _build_header(mode, block_size, input_path.name, file_size, 0)
    initial_payload_len = len(placeholder_header) + file_size
    padding = _calc_padding(initial_payload_len, mode, lw, lh)

    header = _build_header(mode, block_size, input_path.name, file_size, padding)
    payload = header + file_data + b"\x00" * padding

    mode_name = _MODE_NAMES[mode]
    if not quiet:
        print(f"[encode] {input_path.name}  {file_size:,} bytes")
        print(f"[encode] mode={mode_name}  block={block_size}  res={width}×{height}  fps={fps}")

    # Calculate total frames for progress bar
    if mode == MODE_BINARY:
        n_frames = len(np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).reshape(-1, lw * lh))
    elif mode == MODE_RGB:
        n_frames = (len(payload) + lw * lh * 3 - 1) // (lw * lh * 3)
    else:
        n_frames = (len(payload) + lw * lh - 1) // (lw * lh)

    if not quiet:
        print(f"[encode] frames={n_frames}  duration≈{n_frames / fps:.1f}s")

    # Pipe raw BGR frames directly into ffmpeg (no temp files)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-crf", "0",          # lossless H.264 — best source quality for YouTube upload
        "-preset", "ultrafast",
        "-pix_fmt", "yuv444p",  # no chroma subsampling
        output_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL if quiet else None,
    )

    if mode == MODE_BINARY:
        gen = _stream_binary(payload, lw, lh, block_size)
    elif mode == MODE_RGB:
        gen = _stream_rgb(payload, lw, lh, block_size)
    else:
        gen = _stream_palette(payload, lw, lh, block_size)

    try:
        for frame in tqdm(gen, total=n_frames, desc="Encoding", unit="fr", disable=quiet):
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    out_size = Path(output_path).stat().st_size
    if not quiet:
        print(f"[encode] → {output_path}  ({out_size:,} bytes, {out_size / max(file_size,1):.1f}× overhead)")

    return output_path
