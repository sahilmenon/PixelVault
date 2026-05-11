"""
Video → file decoder.

Auto-detects encoding parameters (mode, block_size) by trying each combination
against the first video frame and checking for the header magic bytes.
Frames are streamed from ffmpeg via stdout pipe to avoid large temp directories.
"""

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .encoder import MAGIC, HEADER_SIZE, MODE_BINARY, MODE_RGB, MODE_PALETTE
from .palette import bgr_array_to_bytes

_DETECTION_ORDER = [
    # (mode, block_size) — try most common / most reliable first
    (MODE_BINARY, 4),
    (MODE_BINARY, 8),
    (MODE_BINARY, 2),
    (MODE_RGB, 4),
    (MODE_RGB, 8),
    (MODE_RGB, 2),
    (MODE_PALETTE, 8),
    (MODE_PALETTE, 16),
    (MODE_PALETTE, 4),
]


# ---------------------------------------------------------------------------
# Video info
# ---------------------------------------------------------------------------

def _video_info(video_path: str) -> Tuple[int, int, int]:
    """Return (width, height, n_frames) via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    w, h = stream["width"], stream["height"]
    # nb_frames may be absent; fall back to duration × fps
    nb = stream.get("nb_frames")
    if nb and nb != "N/A":
        n = int(nb)
    else:
        fps_num, fps_den = map(int, stream.get("r_frame_rate", "24/1").split("/"))
        dur = float(stream.get("duration", 0))
        n = int(dur * fps_num / fps_den) if dur else 0
    return w, h, n


# ---------------------------------------------------------------------------
# Single-frame decoders (return raw bytes from one frame)
# ---------------------------------------------------------------------------

def _decode_frame_binary(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    c = bs // 2
    # Average inner 2×2 around centre for compression robustness (clamped for small blocks)
    lo = max(0, c - 1)
    region = gray[lo::bs, lo::bs][:lh, :lw].astype(np.uint16)
    if bs >= 4:
        r2 = gray[c::bs, lo::bs][:lh, :lw].astype(np.uint16)
        r3 = gray[lo::bs, c::bs][:lh, :lw].astype(np.uint16)
        r4 = gray[c::bs, c::bs][:lh, :lw].astype(np.uint16)
        region = (region + r2 + r3 + r4) // 4
    bits = (region > 127).astype(np.uint8).flatten()
    need_bits = max_bytes * 8
    if len(bits) < need_bits:
        bits = np.pad(bits, (0, need_bits - len(bits)))
    return np.packbits(bits[:need_bits]).tobytes()


def _decode_frame_rgb(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    c = bs // 2
    centers = frame_bgr[c::bs, c::bs, :][:lh, :lw, :]  # BGR
    rgb = centers[:, :, ::-1]                            # → RGB
    return rgb.flatten()[:max_bytes].tobytes()


def _decode_frame_palette(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    c = bs // 2
    centers = frame_bgr[c::bs, c::bs, :][:lh, :lw, :]  # (lh, lw, 3) BGR
    byte_vals = bgr_array_to_bytes(centers.reshape(-1, 3))
    return byte_vals[:max_bytes].tobytes()


def _decode_frame(frame_bgr, mode, lw, lh, bs, max_bytes) -> bytes:
    if mode == MODE_BINARY:
        return _decode_frame_binary(frame_bgr, lw, lh, bs, max_bytes)
    elif mode == MODE_RGB:
        return _decode_frame_rgb(frame_bgr, lw, lh, bs, max_bytes)
    else:
        return _decode_frame_palette(frame_bgr, lw, lh, bs, max_bytes)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _detect_params(first_frame: np.ndarray, w: int, h: int) -> Tuple[Optional[int], Optional[int]]:
    """Try every (mode, block_size) combo against the first frame header."""
    for mode, bs in _DETECTION_ORDER:
        if w % bs != 0 or h % bs != 0:
            continue
        lw, lh = w // bs, h // bs
        try:
            hdr = _decode_frame(first_frame, mode, lw, lh, bs, HEADER_SIZE)
            if hdr[:4] == MAGIC and hdr[4] == mode and hdr[5] == bs:
                return mode, bs
        except Exception:
            continue
    return None, None


# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------

def _parse_header(data: bytes):
    magic, mode, block_size, fname_len, fname_raw, file_size, padding_bytes = struct.unpack_from(
        "<4sBBH64sQI", data
    )
    if magic != MAGIC:
        raise ValueError(f"Bad magic bytes: {magic!r}")
    filename = fname_raw[:fname_len].decode("utf-8", errors="replace")
    return int(mode), int(block_size), filename, int(file_size), int(padding_bytes)


# ---------------------------------------------------------------------------
# Bytes-per-frame helpers
# ---------------------------------------------------------------------------

def _bytes_per_frame(mode: int, lw: int, lh: int) -> int:
    if mode == MODE_BINARY:
        return (lw * lh) // 8   # bits → bytes (floor; fractional tail handled later)
    elif mode == MODE_RGB:
        return lw * lh * 3
    else:
        return lw * lh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_file(video_path: str, output_dir: str = ".", quiet: bool = False) -> str:
    """Decode *video_path* back to the original file inside *output_dir*.

    Returns the path to the recovered file.
    """
    video_path = str(video_path)
    w, h, n_frames_est = _video_info(video_path)

    if not quiet:
        print(f"[decode] {Path(video_path).name}  {w}×{h}  ~{n_frames_est} frames")

    frame_size = w * h * 3  # bytes for one raw BGR frame

    # Open ffmpeg pipe
    proc = subprocess.Popen(
        [
            "ffmpeg", "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-vsync", "0",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else subprocess.PIPE,
    )

    def read_frame() -> Optional[np.ndarray]:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

    # --- Step 1: auto-detect params from first frame ---
    first_frame = read_frame()
    if first_frame is None:
        proc.wait()
        raise RuntimeError("Could not read first frame")

    mode, block_size = _detect_params(first_frame, w, h)
    if mode is None:
        proc.wait()
        raise RuntimeError(
            "Could not detect encoding parameters. "
            "Make sure the video was encoded by ByteVault."
        )

    lw, lh = w // block_size, h // block_size
    bpf = _bytes_per_frame(mode, lw, lh)

    if not quiet:
        mode_name = {MODE_BINARY: "binary", MODE_RGB: "rgb", MODE_PALETTE: "palette"}[mode]
        print(f"[decode] mode={mode_name}  block={block_size}  {lw}×{lh} logical pixels/frame")

    # --- Step 2: stream all frames and collect payload bytes ---
    payload_chunks: list[bytes] = []
    header_parsed = False
    file_size = 0
    total_needed = 0
    collected = 0

    # Decode first frame (already read)
    chunk = _decode_frame(first_frame, mode, lw, lh, block_size, bpf)
    payload_chunks.append(chunk)
    collected += len(chunk)

    # Parse header as soon as we have HEADER_SIZE bytes
    if collected >= HEADER_SIZE and not header_parsed:
        raw_so_far = b"".join(payload_chunks)
        _, _, filename, file_size, padding_bytes = _parse_header(raw_so_far[:HEADER_SIZE])
        total_needed = HEADER_SIZE + file_size
        header_parsed = True
        if not quiet:
            print(f"[decode] file={filename!r}  size={file_size:,} bytes")

    pbar = tqdm(
        desc="Decoding", unit="fr", total=n_frames_est or None, disable=quiet
    )
    pbar.update(1)

    while True:
        if header_parsed and collected >= total_needed:
            break
        frame = read_frame()
        if frame is None:
            break
        want = (total_needed - collected) if header_parsed else bpf
        chunk = _decode_frame(frame, mode, lw, lh, block_size, min(bpf, max(bpf, want)))
        payload_chunks.append(chunk)
        collected += len(chunk)

        if not header_parsed and collected >= HEADER_SIZE:
            raw_so_far = b"".join(payload_chunks)
            _, _, filename, file_size, padding_bytes = _parse_header(raw_so_far[:HEADER_SIZE])
            total_needed = HEADER_SIZE + file_size
            header_parsed = True
            if not quiet:
                print(f"[decode] file={filename!r}  size={file_size:,} bytes")

        pbar.update(1)

    pbar.close()
    proc.stdout.close()
    proc.wait()

    if not header_parsed:
        raise RuntimeError("Never found valid header — video may be corrupt or wrong format")

    payload = b"".join(payload_chunks)

    # For binary mode the bit-stream may shift; recover carefully
    if mode == MODE_BINARY:
        # payload_chunks contain bytes decoded via np.packbits from the bit stream.
        # Re-decode from the raw bit stream to get exact byte boundaries.
        # Collect all bits across frames then slice.
        proc2 = subprocess.Popen(
            [
                "ffmpeg", "-i", video_path,
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-vsync", "0", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        bits_needed = total_needed * 8
        all_bits: list[np.ndarray] = []
        bits_collected = 0
        while bits_collected < bits_needed:
            raw = proc2.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame_arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            gray = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2GRAY)
            c = block_size // 2
            lo = max(0, c - 1)
            region = gray[lo::block_size, lo::block_size][:lh, :lw].astype(np.uint16)
            if block_size >= 4:
                region = (region
                          + gray[c::block_size, lo::block_size][:lh, :lw].astype(np.uint16)
                          + gray[lo::block_size, c::block_size][:lh, :lw].astype(np.uint16)
                          + gray[c::block_size, c::block_size][:lh, :lw].astype(np.uint16)) // 4
            bits_frame = (region > 127).astype(np.uint8).flatten()
            all_bits.append(bits_frame)
            bits_collected += len(bits_frame)
        proc2.stdout.close()
        proc2.wait()

        all_bits_arr = np.concatenate(all_bits)[:bits_needed]
        if len(all_bits_arr) < bits_needed:
            all_bits_arr = np.pad(all_bits_arr, (0, bits_needed - len(all_bits_arr)))
        payload = np.packbits(all_bits_arr).tobytes()

    file_data = payload[HEADER_SIZE: HEADER_SIZE + file_size]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    # Avoid clobbering existing files
    if os.path.exists(out_path):
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(out_path):
            out_path = os.path.join(output_dir, f"{base}_{i}{ext}")
            i += 1

    Path(out_path).write_bytes(file_data)

    if not quiet:
        print(f"[decode] → {out_path}  ({len(file_data):,} bytes)")

    return out_path
