"""
Video → file decoder.

Auto-detects encoding parameters (mode, block_size) by trying each combination
against the first video frame and checking for the header magic bytes.
Frames are streamed from ffmpeg via stdout pipe to avoid large temp directories.

Supports both BVI\\x01 (video-only) and BVI\\x02 (audio-extended) formats.
For BVI\\x02, the audio track is extracted and decoded to recover the tail bytes
of the original file.
"""

import json
import os
import struct
import subprocess
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

from . import ecc as _ecc

import cv2
import numpy as np
from tqdm import tqdm

from .encoder import (
    MAGIC_V1, MAGIC_V2, HEADER_SIZE, HEADER_STRUCT,
    MODE_BINARY, MODE_RGB, MODE_PALETTE, MODE_RGB_BIN, MODE_NIBBLE, MODE_GRAY4,
    ENC_NONE, ENC_AES_GCM, _decrypt_payload,
)
from .palette import bgr_array_to_bytes, bgr_array_to_nibbles
from . import audio as _audio

_MAGIC_ALL = {MAGIC_V1, MAGIC_V2}

_DETECTION_ORDER = [
    (MODE_BINARY, 2),
    (MODE_BINARY, 4),
    (MODE_BINARY, 8),
    (MODE_BINARY, 1),
    (MODE_GRAY4, 2),
    (MODE_GRAY4, 4),
    (MODE_GRAY4, 8),
    (MODE_RGB_BIN, 2),
    (MODE_RGB_BIN, 4),
    (MODE_RGB, 2),
    (MODE_RGB, 1),
    (MODE_RGB, 4),
    (MODE_RGB, 8),
    (MODE_NIBBLE, 4),
    (MODE_NIBBLE, 8),
    (MODE_NIBBLE, 16),
    (MODE_PALETTE, 4),
    (MODE_PALETTE, 8),
    (MODE_PALETTE, 16),
    (MODE_PALETTE, 2),
]


# ---------------------------------------------------------------------------
# Video info
# ---------------------------------------------------------------------------

def _video_info(video_path: str) -> Tuple[int, int, int, int]:
    """Return (width, height, n_frames, fps) via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    w, h = stream["width"], stream["height"]
    fps_num, fps_den = map(int, stream.get("r_frame_rate", "24/1").split("/"))
    fps = fps_num // fps_den
    nb = stream.get("nb_frames")
    if nb and nb != "N/A":
        n = int(nb)
    else:
        dur = float(stream.get("duration", 0))
        n = int(dur * fps_num / fps_den) if dur else 0
    return w, h, n, fps


# ---------------------------------------------------------------------------
# Single-frame decoders
# ---------------------------------------------------------------------------

def _decode_frame_binary(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    c = bs // 2
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


def _decode_frame_gray4(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    """Decode a gray4-mode frame: 4 luma levels → 2 bits per logical pixel."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    c = bs // 2
    lo = max(0, c - 1)
    region = gray[lo::bs, lo::bs][:lh, :lw].astype(np.uint16)
    if bs >= 4:
        region = (region
                  + gray[c::bs, lo::bs][:lh, :lw].astype(np.uint16)
                  + gray[lo::bs, c::bs][:lh, :lw].astype(np.uint16)
                  + gray[c::bs, c::bs][:lh, :lw].astype(np.uint16)) // 4
    # Thresholds between levels 0/85/170/255 are at 42, 127, 213
    dibits = np.zeros(region.shape, dtype=np.uint8)
    dibits[region >= 42]  = 1
    dibits[region >= 127] = 2
    dibits[region >= 213] = 3
    flat = dibits.flatten()
    n_pixels = len(flat)
    n_bytes = n_pixels // 4
    result = np.zeros(n_bytes, dtype=np.uint8)
    result |= (flat[0::4] & 0x3) << 6
    result |= (flat[1::4] & 0x3) << 4
    result |= (flat[2::4] & 0x3) << 2
    result |=  flat[3::4] & 0x3
    return result[:max_bytes].tobytes()


def _decode_frame_rgb(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    c = bs // 2
    centers = frame_bgr[c::bs, c::bs, :][:lh, :lw, :]
    rgb = centers[:, :, ::-1]
    return rgb.flatten()[:max_bytes].tobytes()


def _decode_frame_palette(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    c = bs // 2
    centers = frame_bgr[c::bs, c::bs, :][:lh, :lw, :]
    byte_vals = bgr_array_to_bytes(centers.reshape(-1, 3))
    return byte_vals[:max_bytes].tobytes()


def _decode_frame_nibble(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    """Decode a nibble-mode frame: classify each centre pixel to the nearest
    of the 16 YCbCr palette colours, then pack nibble pairs into bytes."""
    c = bs // 2
    centers = frame_bgr[c::bs, c::bs, :][:lh, :lw, :]   # (lh, lw, 3) BGR
    nibbles = bgr_array_to_nibbles(centers.reshape(-1, 3))  # (lh*lw,)
    # Pack pairs: nibble[0] → high bits, nibble[1] → low bits
    n = len(nibbles)
    if n % 2:
        nibbles = np.append(nibbles, np.uint8(0))
    result = ((nibbles[0::2] << 4) | nibbles[1::2]).astype(np.uint8)
    return result[:max_bytes].tobytes()


def _decode_frame_rgb_bin(frame_bgr: np.ndarray, lw: int, lh: int, bs: int, max_bytes: int) -> bytes:
    """Decode a frame encoded in rgb_bin mode (3 bits per logical pixel, one per channel)."""
    c = bs // 2
    lo = max(0, c - 1)
    centers = frame_bgr[lo::bs, lo::bs, :][:lh, :lw, :].astype(np.uint16)
    if bs >= 4:
        c2 = frame_bgr[c::bs, lo::bs, :][:lh, :lw, :].astype(np.uint16)
        c3 = frame_bgr[lo::bs, c::bs, :][:lh, :lw, :].astype(np.uint16)
        c4 = frame_bgr[c::bs, c::bs, :][:lh, :lw, :].astype(np.uint16)
        centers = (centers + c2 + c3 + c4) // 4
    # BGR frame: axis-2 = [B, G, R]. Encoder stored [R_bit, G_bit, B_bit] then flipped to BGR.
    # Reverse: bgr[:,:,2]=R_bit, bgr[:,:,1]=G_bit, bgr[:,:,0]=B_bit.
    bgr_bits = (centers > 127).astype(np.uint8)
    # Reorder to [R_bit, G_bit, B_bit] per pixel, same order as encoder input.
    rgb_bits = bgr_bits[:, :, ::-1].flatten()    # [R0, G0, B0, R1, G1, B1, ...]
    n_bits = min(len(rgb_bits), max_bytes * 8)
    return np.packbits(rgb_bits[:n_bits]).tobytes()[:max_bytes]


def _frame_to_bits(frame_bgr: np.ndarray, mode: int, lw: int, lh: int, bs: int) -> np.ndarray:
    """Extract a flat uint8 bit array from one binary or rgb_bin frame."""
    c = bs // 2
    lo = max(0, c - 1)
    if mode == MODE_BINARY:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        region = gray[lo::bs, lo::bs][:lh, :lw].astype(np.uint16)
        if bs >= 4:
            region = (region
                      + gray[c::bs, lo::bs][:lh, :lw].astype(np.uint16)
                      + gray[lo::bs, c::bs][:lh, :lw].astype(np.uint16)
                      + gray[c::bs, c::bs][:lh, :lw].astype(np.uint16)) // 4
        return (region > 127).astype(np.uint8).flatten()
    else:  # MODE_RGB_BIN
        centers = frame_bgr[lo::bs, lo::bs, :][:lh, :lw, :].astype(np.uint16)
        if bs >= 4:
            centers = (centers
                       + frame_bgr[c::bs, lo::bs, :][:lh, :lw, :].astype(np.uint16)
                       + frame_bgr[lo::bs, c::bs, :][:lh, :lw, :].astype(np.uint16)
                       + frame_bgr[c::bs, c::bs, :][:lh, :lw, :].astype(np.uint16)) // 4
        return (centers[:, :, ::-1] > 127).astype(np.uint8).flatten()


def _decode_frame(frame_bgr, mode, lw, lh, bs, max_bytes) -> bytes:
    if mode == MODE_BINARY:
        return _decode_frame_binary(frame_bgr, lw, lh, bs, max_bytes)
    elif mode == MODE_GRAY4:
        return _decode_frame_gray4(frame_bgr, lw, lh, bs, max_bytes)
    elif mode == MODE_RGB:
        return _decode_frame_rgb(frame_bgr, lw, lh, bs, max_bytes)
    elif mode == MODE_RGB_BIN:
        return _decode_frame_rgb_bin(frame_bgr, lw, lh, bs, max_bytes)
    elif mode == MODE_NIBBLE:
        return _decode_frame_nibble(frame_bgr, lw, lh, bs, max_bytes)
    else:
        return _decode_frame_palette(frame_bgr, lw, lh, bs, max_bytes)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _detect_params(first_frame: np.ndarray, w: int, h: int,
                   verbose: bool = False) -> Tuple[Optional[int], Optional[int]]:
    mode_names = {MODE_BINARY: "binary", MODE_GRAY4: "gray4", MODE_RGB: "rgb",
                  MODE_PALETTE: "palette", MODE_RGB_BIN: "rgb_bin", MODE_NIBBLE: "nibble"}
    for mode, bs in _DETECTION_ORDER:
        if w % bs != 0 or h % bs != 0:
            continue
        lw, lh = w // bs, h // bs
        try:
            hdr = _decode_frame(first_frame, mode, lw, lh, bs, HEADER_SIZE)
            got_magic = hdr[:4]
            got_mode  = hdr[4] if len(hdr) > 4 else None
            got_bs    = hdr[5] if len(hdr) > 5 else None
            match = (got_magic in _MAGIC_ALL and got_mode == mode and got_bs == bs)
            if verbose:
                tag = "✓ MATCH" if match else "✗"
                print(f"  [{tag}] mode={mode_names.get(mode, mode):<8} bs={bs}"
                      f"  magic={got_magic!r}  hdr[4]={got_mode}  hdr[5]={got_bs}")
            if match:
                return mode, bs
        except Exception as exc:
            if verbose:
                print(f"  [!] mode={mode_names.get(mode, mode):<8} bs={bs}  exception: {exc}")
            continue

    # Extra diagnostics: show raw luma values at the gray4 header pixels
    if verbose:
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        print(f"\n  First-frame luma values at block_size=2 positions (first 32 pixels):")
        samples = gray[0::2, 0::2].flatten()[:32]
        print(f"  {list(int(v) for v in samples)}")
        print(f"\n  Expected gray4 dibits for magic BVI\\x01 (first 16 pixels):")
        magic_bytes = b"BVI\x01" + bytes([MODE_GRAY4, 2])  # mode=5, bs=2
        expected_dibits = []
        for byte in magic_bytes[:4]:
            expected_dibits.extend([(byte >> 6) & 3, (byte >> 4) & 3, (byte >> 2) & 3, byte & 3])
        levels = [0, 85, 170, 255]
        print(f"  dibits={expected_dibits}  → Y values={[levels[d] for d in expected_dibits]}")
        classified = []
        for v in samples[:16]:
            if v < 42:   classified.append(0)
            elif v < 127: classified.append(1)
            elif v < 213: classified.append(2)
            else:         classified.append(3)
        print(f"  decoded dibits from frame: {classified}")
        print(f"  expected dibits:           {expected_dibits[:16]}")
        mismatches = sum(a != b for a, b in zip(classified, expected_dibits[:16]))
        print(f"  mismatches in first 16 pixels: {mismatches}/16")

    return None, None


# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------

def _parse_header(data: bytes):
    (magic, mode, block_size, fname_len, fname_raw,
     file_size, padding_bytes,
     audio_bytes, audio_n_levels, audio_block,
     flags, compressed_size, ecc_nsym,
     interleave_depth, enc_type, nonce) = struct.unpack_from(HEADER_STRUCT, data)

    if magic not in _MAGIC_ALL:
        raise ValueError(f"Bad magic bytes: {magic!r}")
    filename = fname_raw[:fname_len].decode("utf-8", errors="replace")
    return (
        int(mode), int(block_size), filename,
        int(file_size), int(padding_bytes),
        int(audio_bytes), int(audio_n_levels), int(audio_block),
        int(flags), int(compressed_size), int(ecc_nsym),
        int(interleave_depth), int(enc_type), bytes(nonce),
    )


# ---------------------------------------------------------------------------
# Bytes-per-frame helpers
# ---------------------------------------------------------------------------

def _bytes_per_frame(mode: int, lw: int, lh: int) -> int:
    if mode == MODE_BINARY:
        return (lw * lh) // 8
    elif mode == MODE_GRAY4:
        return (lw * lh) // 4
    elif mode == MODE_RGB:
        return lw * lh * 3
    elif mode == MODE_RGB_BIN:
        return lw * lh * 3 // 8
    elif mode == MODE_NIBBLE:
        return lw * lh // 2
    else:
        return lw * lh


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _extract_audio_bytes(
    video_path: str,
    audio_n_levels: int,
    audio_block: int,
    audio_bytes: int,
    n_frames: int,
    fps: int,
    quiet: bool,
) -> bytes:
    """Extract raw PCM from the video's audio track and decode it."""
    sample_rate = _audio.SAMPLE_RATE
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-f", "s16le", "-ac", "2", "-ar", str(sample_rate),
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError("ffmpeg failed to extract audio track")

    pcm = np.frombuffer(result.stdout, dtype=np.int16)

    # AAC encoding pads to multiples of 1024 samples; trim to the exact stereo
    # sample count that was written so the L/R channel boundary is correct.
    expected_stereo = n_frames * (sample_rate // fps) * 2
    if len(pcm) > expected_stereo:
        pcm = pcm[:expected_stereo]

    if not quiet:
        print(f"[decode] audio track: {len(pcm) // 2:,} stereo samples → decoding {audio_bytes:,} B")
    return _audio.decode(pcm, audio_n_levels, audio_block, audio_bytes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_file(video_path: str, output_dir: str = ".", quiet: bool = False,
                workers: int = 0, decrypt_password: str | None = None) -> str:
    """Decode *video_path* back to the original file inside *output_dir*.

    Handles both BVI\\x01 (video-only) and BVI\\x02 (audio-extended) formats.
    Returns the path to the recovered file.

    *workers* controls how many threads decode frames in parallel (0 = auto).
    """
    video_path = str(video_path)
    w, h, n_frames_est, video_fps = _video_info(video_path)

    if not quiet:
        print(f"[decode] {Path(video_path).name}  {w}×{h}  ~{n_frames_est} frames")

    frame_size = w * h * 3
    n_workers = workers or max(1, os.cpu_count() or 1)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-threads", "0",
            "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-vsync", "0",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    def read_frame() -> Optional[np.ndarray]:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

    # --- Step 1: auto-detect params from first frame ---
    first_frame = read_frame()
    if first_frame is None:
        proc.kill(); proc.wait()
        raise RuntimeError("Could not read first frame")

    mode, block_size = _detect_params(first_frame, w, h, verbose=not quiet)
    if mode is None:
        proc.kill(); proc.wait()
        raise RuntimeError(
            "Could not detect encoding parameters. "
            "Make sure the video was encoded by ByteVault."
        )

    lw, lh = w // block_size, h // block_size
    bpf = _bytes_per_frame(mode, lw, lh)

    if not quiet:
        mode_name = {MODE_BINARY: "binary", MODE_GRAY4: "gray4", MODE_RGB: "rgb",
                     MODE_PALETTE: "palette", MODE_RGB_BIN: "rgb_bin", MODE_NIBBLE: "nibble"}[mode]
        print(f"[decode] mode={mode_name}  block={block_size}  {lw}×{lh} logical pixels/frame")

    # --- Step 2: collect payload ---

    if mode in (MODE_BINARY, MODE_RGB_BIN):
        # Single-pass bit collection — no second ffmpeg process.
        # The first frame always holds ≥ HEADER_SIZE bytes at any supported block size.
        first_bits = _frame_to_bits(first_frame, mode, lw, lh, block_size)
        (_, _, filename, file_size, _padding,
         audio_bytes, audio_n_levels, audio_block_size,
         hdr_flags, hdr_compressed_size, hdr_ecc_nsym,
         hdr_interleave_depth, hdr_enc_type, hdr_nonce) = _parse_header(np.packbits(first_bits[:HEADER_SIZE * 8]).tobytes())
        _ps = hdr_compressed_size if (hdr_flags & 3) else file_size
        _vfb = _ecc.encoded_size(_ps, hdr_ecc_nsym) - audio_bytes
        _vfb_padded_bits = (-(-_vfb // hdr_interleave_depth) * hdr_interleave_depth
                            if hdr_interleave_depth > 1 else _vfb)
        bits_needed = (HEADER_SIZE + _vfb_padded_bits) * 8
        if not quiet:
            print(f"[decode] file={filename!r}  size={file_size:,} bytes")
            if hdr_flags & 1:
                print(f"[decode] compressed payload: {hdr_compressed_size:,} bytes (zlib)")
            if hdr_flags & 2:
                print(f"[decode] encrypted (AES-256-GCM)")
            if hdr_ecc_nsym > 0:
                print(f"[decode] ECC nsym={hdr_ecc_nsym}  correcting errors...")
            if hdr_interleave_depth > 1:
                print(f"[decode] interleave_depth={hdr_interleave_depth}  deinterleaving...")
            if audio_bytes > 0:
                print(f"[decode] audio portion: {audio_bytes:,} bytes (BVI\\x02)")

        # Pack each frame's bits to bytes immediately so we accumulate ~8x less RAM.
        # bits_needed is always a multiple of 8 (it's HEADER_SIZE + payload_bytes converted to bits).
        bytes_needed = bits_needed >> 3
        all_payload_bytes: list[bytes] = [np.packbits(first_bits).tobytes()]
        bytes_collected = len(all_payload_bytes[0])
        pbar = tqdm(desc="Decoding", unit="fr", total=n_frames_est or None, disable=quiet)
        pbar.update(1)

        window = n_workers * 2
        pending: deque = deque()
        exhausted = False

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            while bytes_collected < bytes_needed:
                # Fill the sliding window with submitted decode tasks.
                while len(pending) < window and not exhausted:
                    frame = read_frame()
                    if frame is None:
                        exhausted = True
                        break
                    pending.append(ex.submit(_frame_to_bits, frame, mode, lw, lh, block_size))

                if not pending:
                    break

                bits = pending.popleft().result()
                chunk_bytes = np.packbits(bits).tobytes()
                all_payload_bytes.append(chunk_bytes)
                bytes_collected += len(chunk_bytes)
                pbar.update(1)

            for fut in pending:
                fut.cancel()

        pbar.close()
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()

        raw = b"".join(all_payload_bytes)
        if len(raw) < bytes_needed:
            raw = raw + b"\x00" * (bytes_needed - len(raw))
        payload = raw[:bytes_needed]

    else:
        # Byte-based collection for palette and rgb modes.
        payload_chunks: list[bytes] = []
        header_parsed = False
        file_size = 0
        audio_bytes = 0
        audio_n_levels = 0
        audio_block_size = 0
        total_needed = 0
        collected = 0

        chunk = _decode_frame(first_frame, mode, lw, lh, block_size, bpf)
        payload_chunks.append(chunk)
        collected += len(chunk)

        if collected >= HEADER_SIZE:
            raw_so_far = b"".join(payload_chunks)
            (_, _, filename, file_size, _padding,
             audio_bytes, audio_n_levels, audio_block_size,
             hdr_flags, hdr_compressed_size, hdr_ecc_nsym,
             hdr_interleave_depth, hdr_enc_type, hdr_nonce) = _parse_header(raw_so_far[:HEADER_SIZE])
            _ps = hdr_compressed_size if (hdr_flags & 3) else file_size
            _vfb = _ecc.encoded_size(_ps, hdr_ecc_nsym) - audio_bytes
            _vfb_padded = (-(-_vfb // hdr_interleave_depth) * hdr_interleave_depth
                           if hdr_interleave_depth > 1 else _vfb)
            total_needed = HEADER_SIZE + _vfb_padded
            header_parsed = True
            if not quiet:
                print(f"[decode] file={filename!r}  size={file_size:,} bytes")
                if hdr_flags & 1:
                    print(f"[decode] compressed payload: {hdr_compressed_size:,} bytes (zlib)")
                if hdr_flags & 2:
                    print(f"[decode] encrypted (AES-256-GCM)")
                if hdr_ecc_nsym > 0:
                    print(f"[decode] ECC nsym={hdr_ecc_nsym}  correcting errors...")
                if hdr_interleave_depth > 1:
                    print(f"[decode] interleave_depth={hdr_interleave_depth}  deinterleaving...")
                if audio_bytes > 0:
                    print(f"[decode] audio portion: {audio_bytes:,} bytes (BVI\\x02)")

        pbar = tqdm(desc="Decoding", unit="fr", total=n_frames_est or None, disable=quiet)
        pbar.update(1)

        window = n_workers * 2
        pending: deque = deque()
        exhausted = False

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            while True:
                if header_parsed and collected >= total_needed:
                    break

                # Fill the sliding window.
                while len(pending) < window and not exhausted:
                    frame = read_frame()
                    if frame is None:
                        exhausted = True
                        break
                    pending.append(ex.submit(_decode_frame, frame, mode, lw, lh, block_size, bpf))

                if not pending:
                    break

                chunk = pending.popleft().result()
                payload_chunks.append(chunk)
                collected += len(chunk)

                if not header_parsed and collected >= HEADER_SIZE:
                    raw_so_far = b"".join(payload_chunks)
                    (_, _, filename, file_size, _padding,
                     audio_bytes, audio_n_levels, audio_block_size,
                     hdr_flags, hdr_compressed_size, hdr_ecc_nsym,
                     hdr_interleave_depth, hdr_enc_type, hdr_nonce) = _parse_header(raw_so_far[:HEADER_SIZE])
                    _ps = hdr_compressed_size if (hdr_flags & 3) else file_size
                    _vfb = _ecc.encoded_size(_ps, hdr_ecc_nsym) - audio_bytes
                    _vfb_padded = (-(-_vfb // hdr_interleave_depth) * hdr_interleave_depth
                                   if hdr_interleave_depth > 1 else _vfb)
                    total_needed = HEADER_SIZE + _vfb_padded
                    header_parsed = True
                    if not quiet:
                        print(f"[decode] file={filename!r}  size={file_size:,} bytes")
                        if hdr_flags & 1:
                            print(f"[decode] compressed payload: {hdr_compressed_size:,} bytes (zlib)")
                        if hdr_flags & 2:
                            print(f"[decode] encrypted (AES-256-GCM)")
                        if hdr_ecc_nsym > 0:
                            print(f"[decode] ECC nsym={hdr_ecc_nsym}  correcting errors...")
                        if hdr_interleave_depth > 1:
                            print(f"[decode] interleave_depth={hdr_interleave_depth}  deinterleaving...")
                        if audio_bytes > 0:
                            print(f"[decode] audio portion: {audio_bytes:,} bytes (BVI\\x02)")

                pbar.update(1)

            for fut in pending:
                fut.cancel()

        pbar.close()
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()

        if not header_parsed:
            raise RuntimeError("Never found valid header — video may be corrupt or wrong format")

        payload = b"".join(payload_chunks)

    payload_size = hdr_compressed_size if (hdr_flags & 3) else file_size
    ecc_size = _ecc.encoded_size(payload_size, hdr_ecc_nsym)
    video_file_bytes = ecc_size - audio_bytes
    _vfb_padded = (-(-video_file_bytes // hdr_interleave_depth) * hdr_interleave_depth
                   if hdr_interleave_depth > 1 else video_file_bytes)
    video_file_data_raw = payload[HEADER_SIZE: HEADER_SIZE + _vfb_padded]

    # --- Step 3: deinterleave video data (reverses burst-error spreading) ---
    if hdr_interleave_depth > 1:
        video_file_data = _ecc.deinterleave(video_file_data_raw, hdr_interleave_depth,
                                             video_file_bytes)
    else:
        video_file_data = video_file_data_raw[:video_file_bytes]

    # --- Step 4: extract audio portion if present ---
    if audio_bytes > 0:
        audio_file_data = _extract_audio_bytes(
            video_path, audio_n_levels, audio_block_size, audio_bytes,
            n_frames=n_frames_est, fps=video_fps, quiet=quiet,
        )
        ecc_data = video_file_data + audio_file_data
    else:
        ecc_data = video_file_data

    # --- Step 5: ECC error correction ---
    if hdr_ecc_nsym > 0:
        raw_payload = _ecc.decode(ecc_data, hdr_ecc_nsym, payload_size,
                                  workers=os.cpu_count() or 1)
    else:
        raw_payload = ecc_data[:payload_size]

    # --- Step 6: decrypt if needed ---
    if hdr_enc_type == ENC_AES_GCM:
        if not decrypt_password:
            raise ValueError(
                "This video is encrypted. Provide the password with --password."
            )
        if not quiet:
            print("[decode] decrypting (AES-256-GCM) ...")
        try:
            raw_payload = _decrypt_payload(raw_payload, decrypt_password, hdr_nonce)
        except Exception as exc:
            raise ValueError(
                "Decryption failed — wrong password or corrupted data."
            ) from exc

    # --- Step 7: decompress if needed ---
    if hdr_flags & 1:
        zlib_size = hdr_compressed_size - (16 if hdr_enc_type == ENC_AES_GCM else 0)
        file_data = zlib.decompress(raw_payload[:zlib_size])
    else:
        file_data = raw_payload

    # Sanity trim
    file_data = file_data[:file_size]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

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
