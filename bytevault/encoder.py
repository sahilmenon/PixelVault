"""
File → video encoder.

Header layout (128 bytes, always encoded in the first video frame):
  Offset  Size  Field
  0       4     Magic: b"BVI\\x01" (video-only) or b"BVI\\x02" (audio-extended)
  4       1     Mode: 0=binary, 1=rgb, 2=palette
  5       1     Video block size (pixels per logical-pixel edge)
  6       2     Filename length (uint16 LE)
  8       64    Filename (UTF-8, null-padded)
  72      8     Original file size in bytes (uint64 LE)  — uncompressed, full file
  80      4     Video padding bytes (uint32 LE)
  84      4     Audio payload bytes (uint32 LE)  — 0 for BVI\\x01
  88      1     Audio n_levels                   — 0 for BVI\\x01
  89      1     Audio block size                 — 0 for BVI\\x01
  90      1     Flags (uint8): bit 0 = zlib-compressed payload
  91      8     Compressed payload size (uint64 LE); 0 when not compressed
  99      1     ECC nsym (uint8): Reed-Solomon ECC symbols per 255-byte block; 0 = no ECC
  100     28    Reserved (zeros)

BVI\\x01 (video-only): audio fields are zero; old decoders read zeros harmlessly.
BVI\\x02 (audio-extended): video carries header + payload[:video_file_bytes];
  audio carries payload[video_file_bytes:] (last audio_bytes bytes of payload).
  When compressed (flags bit 0 set), payload is the zlib-compressed file data.
"""

import itertools
import math
import os
import struct
import subprocess
import tempfile
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Encryption helpers (AES-256-GCM via cryptography package)
# ---------------------------------------------------------------------------

ENC_NONE    = 0
ENC_AES_GCM = 1
_NONCE_SIZE = 12  # AES-GCM standard nonce


def _derive_key(password: str, nonce: bytes) -> bytes:
    """Derive a 32-byte AES-256 key from password + nonce (PBKDF2-SHA256)."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=nonce + b"ByteVault", iterations=200_000)
    return kdf.derive(password.encode("utf-8"))


def _encrypt_payload(data: bytes, password: str) -> tuple[bytes, bytes]:
    """Return (nonce, ciphertext_with_16_byte_tag)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(_NONCE_SIZE)
    key   = _derive_key(password, nonce)
    ct    = AESGCM(key).encrypt(nonce, data, None)
    return nonce, ct


def _decrypt_payload(ct_with_tag: bytes, password: str, nonce: bytes) -> bytes:
    """Decrypt and authenticate; raises InvalidTag on wrong password/corruption."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = _derive_key(password, nonce)
    return AESGCM(key).decrypt(nonce, ct_with_tag, None)

from .palette import PALETTE_BGR, NIBBLE_PALETTE_BGR
from . import audio as _audio
from . import ecc as _ecc

MAGIC_V1 = b"BVI\x01"   # video-only (legacy)
MAGIC_V2 = b"BVI\x02"   # audio-extended

HEADER_SIZE = 128
HEADER_STRUCT = "<4sBBH64sQIIBBBQBIB12s"  # 117 bytes; + 11 zeros = 128
# Fields: magic, mode, block_size, fname_len, fname, file_size, padding_bytes,
#         audio_bytes, audio_n_levels, audio_block, flags, compressed_size,
#         ecc_nsym, interleave_depth, enc_type, nonce

MODE_BINARY = 0   # 1 bit per logical pixel (black=0, white=1)
MODE_RGB = 1      # 3 bytes per logical pixel (one byte per channel)
MODE_PALETTE = 2  # 1 byte per logical pixel (mapped to 256-colour palette)
MODE_RGB_BIN = 3  # 3 bits per logical pixel (one bit per R/G/B channel, each 0 or 255)
MODE_NIBBLE = 4   # 4 bits per logical pixel (one of 16 YCbCr-designed colours); local only
MODE_GRAY4 = 5    # 2 bits per logical pixel (4 luma levels: 0/85/170/255); YouTube-safe, 2× binary

_MODE_NAMES = {
    MODE_BINARY: "binary", MODE_RGB: "rgb", MODE_PALETTE: "palette",
    MODE_RGB_BIN: "rgb_bin", MODE_NIBBLE: "nibble", MODE_GRAY4: "gray4",
}

DEFAULT_BLOCK = {MODE_BINARY: 2, MODE_RGB: 2, MODE_PALETTE: 4, MODE_RGB_BIN: 2, MODE_NIBBLE: 4, MODE_GRAY4: 4}
WIDTH    = 1920   # default 1080p
HEIGHT   = 1080
WIDTH_4K = 3840   # 4K UHD — 4× data per frame, YouTube serves as VP9
HEIGHT_4K = 2160

# ---------------------------------------------------------------------------
# Hardware encoder detection
# ---------------------------------------------------------------------------

# False = not yet probed; None = none available; str = encoder name
_hw_encoder_cache: str | None | bool = False


def _detect_hw_encoder() -> str | None:
    """Probe ffmpeg for the first available hardware H.264 encoder."""
    global _hw_encoder_cache
    if _hw_encoder_cache is not False:
        return _hw_encoder_cache

    candidates = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p1"]),
        ("h264_amf",   ["-c:v", "h264_amf"]),
        ("h264_qsv",   ["-c:v", "h264_qsv"]),
    ]
    for name, enc_opts in candidates:
        try:
            r = subprocess.run(
                ["ffmpeg", "-y",
                 "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1",
                 "-vframes", "1"] + enc_opts + ["-f", "null", "-"],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                _hw_encoder_cache = name
                return name
        except Exception:
            continue

    _hw_encoder_cache = None
    return None


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _build_header(
    mode: int,
    block_size: int,
    filename: str,
    file_size: int,
    padding_bytes: int,
    audio_bytes: int = 0,
    audio_n_levels: int = 0,
    audio_block: int = 0,
    flags: int = 0,
    compressed_size: int = 0,
    ecc_nsym: int = 0,
    interleave_depth: int = 0,
    enc_type: int = ENC_NONE,
    nonce: bytes = b"\x00" * _NONCE_SIZE,
) -> bytes:
    # If the filename is too long, trim the stem but always keep the extension
    # so the decoded file gets the right type (e.g. .pdf, .zip).
    fname_full = filename.encode("utf-8")
    if len(fname_full) > 64:
        ext = Path(filename).suffix.encode("utf-8")   # e.g. b".pdf"
        fname_b = fname_full[:64 - len(ext)] + ext
    else:
        fname_b = fname_full
    fname_b = fname_b[:64]  # hard cap in case ext itself is unusually long
    magic = MAGIC_V2 if audio_bytes > 0 else MAGIC_V1
    raw = struct.pack(
        HEADER_STRUCT,
        magic, mode, block_size,
        len(fname_b), fname_b.ljust(64, b"\x00"),
        file_size, padding_bytes,
        audio_bytes, audio_n_levels, audio_block,
        flags, compressed_size, ecc_nsym,
        interleave_depth,
        enc_type, nonce.ljust(_NONCE_SIZE, b"\x00")[:_NONCE_SIZE],
    )
    return raw + b"\x00" * (HEADER_SIZE - len(raw))


def _bytes_per_frame_video(mode: int, lw: int, lh: int) -> int:
    """Bytes of payload per video frame (approximate; binary/rgb_bin/gray4 are bit-level)."""
    if mode == MODE_BINARY:
        return lw * lh // 8
    elif mode == MODE_RGB:
        return lw * lh * 3
    elif mode == MODE_RGB_BIN:
        return lw * lh * 3 // 8
    elif mode == MODE_NIBBLE:
        return lw * lh // 2  # 4 bits per pixel = 0.5 bytes per pixel
    elif mode == MODE_GRAY4:
        return lw * lh // 4  # 2 bits per pixel = 0.25 bytes per pixel
    else:
        return lw * lh


def _calc_padding(payload_len: int, mode: int, lw: int, lh: int) -> int:
    if mode == MODE_BINARY:
        unit = lw * lh
        total_bits = payload_len * 8
        rem = total_bits % unit
        pad_bits = (unit - rem) % unit
        return pad_bits // 8
    elif mode == MODE_RGB_BIN:
        bits_per_frame = lw * lh * 3
        total_bits = payload_len * 8
        rem = total_bits % bits_per_frame
        pad_bits = (bits_per_frame - rem) % bits_per_frame
        return (pad_bits + 7) // 8  # round up to byte boundary
    elif mode == MODE_RGB:
        unit = lw * lh * 3
        rem = payload_len % unit
        return (unit - rem) % unit
    elif mode == MODE_NIBBLE:
        unit = lw * lh // 2   # bytes per frame
        rem = payload_len % unit
        return (unit - rem) % unit
    elif mode == MODE_GRAY4:
        unit = lw * lh        # bits per frame
        total_bits = payload_len * 8
        rem = total_bits % (unit * 2)   # 2 bits per pixel
        pad_bits = (unit * 2 - rem) % (unit * 2)
        return pad_bits // 8
    else:
        unit = lw * lh
        rem = payload_len % unit
        return (unit - rem) % unit


def _split_payload(
    file_size: int,
    mode: int,
    lw: int,
    lh: int,
    fps: int,
    audio_sample_rate: int,
    audio_n_levels: int,
    audio_block: int,
) -> tuple[int, int]:
    """Return (video_file_bytes, audio_file_bytes) for optimal split.

    Minimises total frame count by distributing data across both tracks.
    audio_file_bytes == 0 when the file fits entirely in the video.
    """
    B_v = _bytes_per_frame_video(mode, lw, lh)
    B_a = _audio.bytes_per_frame(fps, audio_sample_rate, audio_n_levels, audio_block)
    if B_a == 0:
        return file_size, 0

    total = HEADER_SIZE + file_size
    N = math.ceil(total / (B_v + B_a))
    video_file_cap = N * B_v - HEADER_SIZE
    if video_file_cap >= file_size:
        return file_size, 0
    return max(0, int(video_file_cap)), file_size - max(0, int(video_file_cap))


# ---------------------------------------------------------------------------
# Frame generators
# ---------------------------------------------------------------------------

_BATCH = 64   # frames per generation batch (1080p); scaled down for 4K


def _adaptive_batch(width: int, height: int) -> int:
    """Batch size that keeps peak RAM around 400 MB regardless of resolution."""
    frame_bytes = width * height * 3
    return max(8, min(_BATCH, (400 * 1024 * 1024) // frame_bytes))


def _write_parallel(make_batch_fn, n_frames: int, pipe, pbar, n_workers: int,
                    batch: int = _BATCH):
    """Write all frames to *pipe*, generating batches in parallel.

    Uses a sliding window of futures so at most *n_workers* batches are
    in-flight at once (bounds peak RAM), while the main thread writes the
    already-finished batch to ffmpeg — overlapping CPU and I/O.
    numpy releases the GIL for C-level ops, so threads run truly in parallel.
    """
    batch_starts = range(0, n_frames, batch)
    if n_workers <= 1:
        for start in batch_starts:
            data, count = make_batch_fn(start)
            pipe.write(data)
            pbar.update(count)
        return

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        pending: deque = deque()
        it = iter(batch_starts)

        for start in itertools.islice(it, n_workers):
            pending.append(executor.submit(make_batch_fn, start))

        for start in it:
            data, count = pending.popleft().result()
            pipe.write(data)
            pbar.update(count)
            pending.append(executor.submit(make_batch_fn, start))

        while pending:
            data, count = pending.popleft().result()
            pipe.write(data)
            pbar.update(count)


def _write_binary(payload: bytes, lw: int, lh: int, bs: int,
                  n_frames: int, pipe, pbar, n_workers: int = 1, batch: int = _BATCH):
    bpf = lw * lh          # bits per frame
    Bpf = bpf >> 3         # bytes per frame (standard res: lw*lh always divisible by 8)
    payload_arr = np.frombuffer(payload, dtype=np.uint8)

    def make_batch(start):
        end = min(start + batch, n_frames)
        frames = end - start
        b_start, b_end = start * Bpf, min(end * Bpf, len(payload_arr))
        need = frames * bpf
        if b_start < len(payload_arr):
            raw = np.unpackbits(payload_arr[b_start:b_end])
            if len(raw) < need:
                raw = np.append(raw, np.zeros(need - len(raw), dtype=np.uint8))
        else:
            raw = np.zeros(need, dtype=np.uint8)
        logical = raw.reshape(frames, lh, lw)
        scaled = np.repeat(np.repeat(logical[..., None] * 255, bs, axis=1), bs, axis=2)
        return np.broadcast_to(scaled, (*scaled.shape[:3], 3)).copy().astype(np.uint8).tobytes(), frames

    _write_parallel(make_batch, n_frames, pipe, pbar, n_workers, batch)


_GRAY4_LEVELS = np.array([0, 85, 170, 255], dtype=np.uint8)


def _write_gray4(payload: bytes, lw: int, lh: int, bs: int,
                 n_frames: int, pipe, pbar, n_workers: int = 1, batch: int = _BATCH):
    """2 bits per logical pixel using 4 luma levels: 0, 85, 170, 255."""
    ppf = lw * lh          # pixels (dibits) per frame
    Bpf = ppf >> 2         # bytes per frame (4 pixels per payload byte)
    payload_arr = np.frombuffer(payload, dtype=np.uint8)

    def make_batch(start):
        end = min(start + batch, n_frames)
        frames = end - start
        b_start, b_end = start * Bpf, min(end * Bpf, len(payload_arr))
        need = frames * ppf
        if b_start < len(payload_arr):
            chunk = payload_arr[b_start:b_end]
            raw = np.empty(len(chunk) * 4, dtype=np.uint8)
            raw[0::4] = (chunk >> 6) & 0x3
            raw[1::4] = (chunk >> 4) & 0x3
            raw[2::4] = (chunk >> 2) & 0x3
            raw[3::4] =  chunk       & 0x3
            if len(raw) < need:
                raw = np.append(raw, np.zeros(need - len(raw), dtype=np.uint8))
        else:
            raw = np.zeros(need, dtype=np.uint8)
        logical = raw[:need].reshape(frames, lh, lw)
        gray = _GRAY4_LEVELS[logical]
        frame = np.stack([gray, gray, gray], axis=-1)
        scaled = np.repeat(np.repeat(frame, bs, axis=1), bs, axis=2)
        return scaled.tobytes(), frames

    _write_parallel(make_batch, n_frames, pipe, pbar, n_workers, batch)


def _write_rgb(payload: bytes, lw: int, lh: int, bs: int,
               n_frames: int, pipe, pbar, n_workers: int = 1, batch: int = _BATCH):
    Bpf = lw * lh * 3     # bytes per frame
    payload_arr = np.frombuffer(payload, dtype=np.uint8)

    def make_batch(start):
        end = min(start + batch, n_frames)
        frames = end - start
        b_start, b_end = start * Bpf, min(end * Bpf, len(payload_arr))
        need = frames * Bpf
        if b_start < len(payload_arr):
            raw = payload_arr[b_start:b_end]
            if len(raw) < need:
                raw = np.append(raw, np.zeros(need - len(raw), dtype=np.uint8))
        else:
            raw = np.zeros(need, dtype=np.uint8)
        chunk = raw.reshape(frames, lh, lw, 3)[:, :, :, ::-1]  # RGB→BGR
        return np.repeat(np.repeat(chunk, bs, axis=1), bs, axis=2).tobytes(), frames

    _write_parallel(make_batch, n_frames, pipe, pbar, n_workers, batch)


def _write_palette(payload: bytes, lw: int, lh: int, bs: int,
                   n_frames: int, pipe, pbar, n_workers: int = 1, batch: int = _BATCH):
    Bpf = lw * lh          # bytes per frame (1 byte per logical pixel)
    payload_arr = np.frombuffer(payload, dtype=np.uint8)

    def make_batch(start):
        end = min(start + batch, n_frames)
        frames = end - start
        b_start, b_end = start * Bpf, min(end * Bpf, len(payload_arr))
        need = frames * Bpf
        if b_start < len(payload_arr):
            raw = payload_arr[b_start:b_end]
            if len(raw) < need:
                raw = np.append(raw, np.zeros(need - len(raw), dtype=np.uint8))
        else:
            raw = np.zeros(need, dtype=np.uint8)
        logical = raw.reshape(frames, lh, lw)
        bgr_small = PALETTE_BGR[logical]
        return np.repeat(np.repeat(bgr_small, bs, axis=1), bs, axis=2).astype(np.uint8).tobytes(), frames

    _write_parallel(make_batch, n_frames, pipe, pbar, n_workers, batch)


def _write_nibble(payload: bytes, lw: int, lh: int, bs: int,
                  n_frames: int, pipe, pbar, n_workers: int = 1, batch: int = _BATCH):
    """4 bits per logical pixel: one of 16 YCbCr-designed colours."""
    ppf = lw * lh          # pixels per frame
    Bpf = ppf >> 1         # bytes per frame (2 nibbles per payload byte)
    payload_arr = np.frombuffer(payload, dtype=np.uint8)

    def make_batch(start):
        end = min(start + batch, n_frames)
        frames = end - start
        b_start, b_end = start * Bpf, min(end * Bpf, len(payload_arr))
        need = frames * ppf
        if b_start < len(payload_arr):
            chunk = payload_arr[b_start:b_end]
            raw = np.empty(len(chunk) * 2, dtype=np.uint8)
            raw[0::2] = (chunk >> 4) & 0x0F
            raw[1::2] =  chunk       & 0x0F
            if len(raw) < need:
                raw = np.append(raw, np.zeros(need - len(raw), dtype=np.uint8))
        else:
            raw = np.zeros(need, dtype=np.uint8)
        logical = raw[:need].reshape(frames, lh, lw)
        bgr_small = NIBBLE_PALETTE_BGR[logical]
        return np.repeat(np.repeat(bgr_small, bs, axis=1), bs, axis=2).tobytes(), frames

    _write_parallel(make_batch, n_frames, pipe, pbar, n_workers, batch)


def _write_rgb_bin(payload: bytes, lw: int, lh: int, bs: int,
                   n_frames: int, pipe, pbar, n_workers: int = 1, batch: int = _BATCH):
    """3 bits per logical pixel: one bit per R/G/B channel (0 or 255)."""
    bpf = lw * lh * 3     # bits per frame
    Bpf = bpf >> 3         # bytes per frame
    payload_arr = np.frombuffer(payload, dtype=np.uint8)

    def make_batch(start):
        end = min(start + batch, n_frames)
        frames = end - start
        b_start, b_end = start * Bpf, min(end * Bpf, len(payload_arr))
        need = frames * bpf
        if b_start < len(payload_arr):
            raw = np.unpackbits(payload_arr[b_start:b_end])
            if len(raw) < need:
                raw = np.append(raw, np.zeros(need - len(raw), dtype=np.uint8))
        else:
            raw = np.zeros(need, dtype=np.uint8)
        logical = raw[:need].reshape(frames, lh, lw, 3)
        bgr = logical[:, :, :, ::-1] * 255   # RGB→BGR, scale to 0/255
        return np.repeat(np.repeat(bgr, bs, axis=1), bs, axis=2).astype(np.uint8).tobytes(), frames

    _write_parallel(make_batch, n_frames, pipe, pbar, n_workers, batch)


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
    use_audio: bool = False,
    audio_n_levels: int = _audio.N_LEVELS,
    audio_block: int = _audio.BLOCK_SIZE,
    audio_sample_rate: int = _audio.SAMPLE_RATE,
    compress: bool = False,
    ecc_nsym: int = 16,
    interleave: bool = False,
    workers: int = 0,
    use_hw_encoder: bool = True,
    raw_bytes: bytes | None = None,
    encrypt_password: str | None = None,
) -> str:
    """Encode *input_path* into an MP4 video at *output_path*.

    *raw_bytes* overrides reading *input_path* — useful for chunked encoding
    where the caller slices the source file and passes each slice directly.
    The filename stored in the video header is always taken from *input_path*.

    When *use_audio* is True, part of the file payload is stored in the audio
    track (amplitude-level PCM, AAC-encoded) and the remainder in video frames.
    This produces a shorter video for the same file size.

    When *compress* is True, the payload is zlib-compressed before encoding if
    doing so reduces its size (e.g. text, logs, sparse binary data).

    *ecc_nsym* controls Reed-Solomon ECC: 0 = off, 16 = default (corrects up to
    8 byte errors per 255-byte block), 32 = stronger (up to 16 errors/block).

    Returns output_path.
    """
    if block_size is None:
        block_size = DEFAULT_BLOCK[mode]

    if width % block_size != 0 or height % block_size != 0:
        raise ValueError(
            f"Resolution {width}×{height} must be divisible by block_size={block_size}"
        )
    if mode == MODE_RGB_BIN and block_size < 2:
        raise ValueError(
            "rgb_bin mode requires block_size >= 2; block_size=1 corrupts the B channel "
            "via yuv420p chroma subsampling when adjacent pixels differ."
        )

    n_workers = workers if workers > 0 else min(4, os.cpu_count() or 1)
    batch = _adaptive_batch(width, height)
    # Hardware encoding: safe for binary and rgb_bin (block-based 0/255 values survive CRF=18).
    # palette/RGB need lossless CRF=0 which hw encoders don't reliably support.
    hw_enc = _detect_hw_encoder() if (use_hw_encoder and mode in (MODE_BINARY, MODE_GRAY4, MODE_RGB_BIN)) else None

    lw, lh = width // block_size, height // block_size
    input_path = Path(input_path)
    file_data = raw_bytes if raw_bytes is not None else input_path.read_bytes()
    file_size = len(file_data)

    # --- Optional zlib compression ---
    flags = 0
    compressed_size = 0
    payload_data = file_data
    zlib_out_size = None
    if compress and file_size > 0:
        candidate = zlib.compress(file_data, level=6)
        if len(candidate) < len(file_data):
            payload_data = candidate
            flags |= 1
            zlib_out_size = len(candidate)

    # --- Optional AES-256-GCM encryption ---
    enc_type = ENC_NONE
    nonce = b"\x00" * _NONCE_SIZE
    if encrypt_password:
        nonce, payload_data = _encrypt_payload(payload_data, encrypt_password)
        enc_type = ENC_AES_GCM
        flags |= 2

    # compressed_size stores the pre-ECC payload size when any transform was applied.
    # When only compress: len(zlib_data). When encrypt (±compress): len(ciphertext+tag).
    if flags & 2:
        compressed_size = len(payload_data)
    elif flags & 1:
        compressed_size = zlib_out_size

    # --- Optional Reed-Solomon ECC ---
    # ECC is applied to payload_data; the video+audio carry ecc_data
    ecc_data = _ecc.encode(payload_data, ecc_nsym, workers=n_workers)
    ecc_size = len(ecc_data)  # what actually goes into frames

    # --- Determine video/audio split (based on ECC-expanded size) ---
    if use_audio:
        video_file_bytes, audio_file_bytes = _split_payload(
            ecc_size, mode, lw, lh, fps,
            audio_sample_rate, audio_n_levels, audio_block,
        )
    else:
        video_file_bytes, audio_file_bytes = ecc_size, 0

    video_data = ecc_data[:video_file_bytes]
    audio_data = ecc_data[video_file_bytes:]  # may be empty

    # --- Optional interleaving of video_data (after ECC, before framing) ---
    # Interleave depth = estimated number of data frames, so each frame carries
    # bytes spaced depth apart in the original ECC stream.  A burst of K corrupt
    # frames contributes at most ceil(255/depth)*K errors per RS block.
    interleave_depth = 0
    if interleave and ecc_nsym > 0 and len(video_data) > 0:
        bpf_v = _bytes_per_frame_video(mode, lw, lh)
        interleave_depth = max(2, math.ceil((HEADER_SIZE + len(video_data)) / bpf_v))
        video_data = _ecc.interleave(video_data, interleave_depth)
        # video_data may now be slightly longer (padded to multiple of depth)

    # Two-pass header: compute padding based on (possibly-padded) video_data,
    # then bake the final padding value into the header.
    placeholder_header = _build_header(
        mode, block_size, input_path.name, file_size, 0,
        audio_file_bytes, audio_n_levels if use_audio else 0,
        audio_block if use_audio else 0,
        flags, compressed_size, ecc_nsym, interleave_depth,
        enc_type, nonce,
    )
    initial_payload_len = len(placeholder_header) + len(video_data)
    padding = _calc_padding(initial_payload_len, mode, lw, lh)

    header = _build_header(
        mode, block_size, input_path.name, file_size, padding,
        audio_file_bytes, audio_n_levels if use_audio else 0,
        audio_block if use_audio else 0,
        flags, compressed_size, ecc_nsym, interleave_depth,
        enc_type, nonce,
    )
    video_payload = header + video_data + b"\x00" * padding

    mode_name = _MODE_NAMES[mode]
    if not quiet:
        print(f"[encode] {input_path.name}  {file_size:,} bytes")
        if flags & 1:
            ratio = (zlib_out_size or 0) / file_size if file_size else 1
            print(f"[encode] compressed {file_size:,} -> {zlib_out_size:,} bytes ({ratio:.1%})")
        if flags & 2:
            print(f"[encode] encrypted (AES-256-GCM)  payload {len(payload_data):,} bytes")
        if ecc_nsym > 0:
            pre_ecc = len(payload_data)
            overhead = (ecc_size - pre_ecc) / max(pre_ecc, 1)
            print(f"[encode] ECC nsym={ecc_nsym}  overhead +{overhead:.1%}  ({ecc_size:,} bytes encoded)")
        print(f"[encode] mode={mode_name}  block={block_size}  res={width}×{height}  fps={fps}")
        print(f"[encode] encoder={hw_enc or 'libx264'}  workers={n_workers}"
              + (f"  interleave_depth={interleave_depth}" if interleave_depth else ""))
        if use_audio:
            print(
                f"[encode] audio={audio_file_bytes:,} B in audio track  "
                f"video={len(video_data):,} B in video frames  "
                f"(levels={audio_n_levels} block={audio_block})"
            )

    # Trailing blank frames.
    # With ECC nsym>=8 active, RS correction handles last-frame compression
    # artefacts (up to 8 wrong bytes/block for nsym=16). Without ECC, the final
    # GOP receives heavier quantisation; 30 blank tail frames ensure no data
    # frame sits in that compressed region when uploading to YouTube.
    TAIL_PAD_FRAMES = 5 if ecc_nsym >= 8 else 30

    # Minimum total frames: keep the container well-formed (≥ 5 frames) but
    # do NOT enforce an fps-based floor.  A 10-frame 1080p video at any FPS
    # is accepted by YouTube/VLC/etc.  The old fps*5 = 120 min frames turned
    # a 5-frame payload into a 120-frame encode — dominating encode time for
    # small files.  Large files have many data frames and are unaffected.
    MIN_FRAMES = max(TAIL_PAD_FRAMES, 5)
    if mode == MODE_BINARY:
        n_data_frames = len(
            np.unpackbits(np.frombuffer(video_payload, dtype=np.uint8)).reshape(-1, lw * lh)
        )
    elif mode == MODE_RGB_BIN:
        n_data_frames = math.ceil(len(video_payload) * 8 / (lw * lh * 3))
    elif mode == MODE_RGB:
        n_data_frames = math.ceil(len(video_payload) / (lw * lh * 3))
    elif mode == MODE_NIBBLE:
        n_data_frames = math.ceil(len(video_payload) * 2 / (lw * lh))
    elif mode == MODE_GRAY4:
        n_data_frames = math.ceil(len(video_payload) * 4 / (lw * lh))
    else:
        n_data_frames = math.ceil(len(video_payload) / (lw * lh))

    n_frames = max(n_data_frames + TAIL_PAD_FRAMES, MIN_FRAMES)
    n_pad_frames = n_frames - n_data_frames

    if not quiet:
        print(f"[encode] frames={n_frames}  duration~{n_frames / fps:.1f}s"
              + (f"  (pad={n_pad_frames})" if n_pad_frames else ""))

    # --- Build ffmpeg command ---
    audio_tmp = None
    if use_audio and audio_file_bytes > 0:
        # Generate PCM audio for exactly n_frames frames
        pcm = _audio.encode(
            audio_data, n_frames, fps, audio_sample_rate, audio_n_levels, audio_block
        )
        # Write raw PCM to a temp file so ffmpeg can mux it alongside the video pipe
        fd, audio_tmp = tempfile.mkstemp(suffix=".raw")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(pcm.tobytes())
        except Exception:
            os.close(fd)
            raise

    # Binary/rgb_bin: luma-only or 1-bit-per-channel → CRF 18 is fine; ECC covers residual errors.
    # Nibble/palette/RGB: chroma palette must survive two lossy passes (local + YouTube) → lossless
    # CRF 0 so only YouTube's single re-encode introduces error, staying within palette margins.
    crf = "18" if mode in (MODE_BINARY, MODE_GRAY4, MODE_RGB_BIN) else "0"

    # Build video codec options.
    # Hardware encoders are only used for binary mode (lossless mode not needed).
    if hw_enc == "h264_nvenc":
        enc_opts = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", crf, "-bf", "0"]
    elif hw_enc == "h264_amf":
        enc_opts = ["-c:v", "h264_amf", "-quality", "quality",
                    "-rc", "cqp", "-qp_i", crf, "-qp_p", crf]
    elif hw_enc == "h264_qsv":
        enc_opts = ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", crf]
    else:
        enc_opts = ["-c:v", "libx264", "-crf", crf, "-preset", "ultrafast", "-threads", "0"]

    try:
        if audio_tmp:
            # Data audio channel — use ALAC (lossless) for local roundtrip
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-f", "s16le", "-ac", "2", "-ar", str(audio_sample_rate),
                "-i", audio_tmp,
            ] + enc_opts + [
                "-pix_fmt", "yuv420p",
                "-c:a", "alac",
                output_path,
            ]
        else:
            # No data audio — mux a silent AAC track so YouTube accepts the file
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-f", "lavfi",
                "-i", f"anullsrc=r={audio_sample_rate}:cl=stereo",
            ] + enc_opts + [
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "32k",
                "-shortest",
                output_path,
            ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL if quiet else None,
        )

        pbar = tqdm(total=n_frames, desc="Encoding", unit="fr", disable=quiet)
        try:
            if mode == MODE_BINARY:
                _write_binary(video_payload, lw, lh, block_size, n_frames, proc.stdin, pbar, n_workers, batch)
            elif mode == MODE_GRAY4:
                _write_gray4(video_payload, lw, lh, block_size, n_frames, proc.stdin, pbar, n_workers, batch)
            elif mode == MODE_RGB:
                _write_rgb(video_payload, lw, lh, block_size, n_frames, proc.stdin, pbar, n_workers, batch)
            elif mode == MODE_RGB_BIN:
                _write_rgb_bin(video_payload, lw, lh, block_size, n_frames, proc.stdin, pbar, n_workers, batch)
            elif mode == MODE_NIBBLE:
                _write_nibble(video_payload, lw, lh, block_size, n_frames, proc.stdin, pbar, n_workers, batch)
            else:
                _write_palette(video_payload, lw, lh, block_size, n_frames, proc.stdin, pbar, n_workers, batch)
        finally:
            pbar.close()
            proc.stdin.close()
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    finally:
        if audio_tmp and os.path.exists(audio_tmp):
            os.unlink(audio_tmp)

    out_size = Path(output_path).stat().st_size
    if not quiet:
        print(
            f"[encode] -> {output_path}  "
            f"({out_size:,} bytes, {out_size / max(file_size, 1):.1f}x overhead)"
        )

    return output_path
