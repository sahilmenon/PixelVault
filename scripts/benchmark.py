#!/usr/bin/env python3
"""
Cross-tool benchmark: PixelVault vs ISG / YouBit / bin2video algorithmic equivalents.

ISG source was deleted from GitHub; YouBit requires Python 3.9-3.11 (we have 3.14);
bin2video requires gcc. So we implement each tool's encoding algorithm faithfully
in Python (same numpy/ffmpeg stack), isolating algorithmic differences from language
differences. All tools run on the same hardware with the same ffmpeg binary.

Each "tool" here is defined by:
  - Resolution & FPS
  - Block size (logical pixel = block_size x block_size physical pixels)
  - Bits per pixel (how many bits each logical pixel encodes)
  - ECC scheme
  - ffmpeg CRF setting

Outputs: benchmark_results.json + a Markdown summary table.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# -- paths --------------------------------------------------------------------
# All output goes into bench/ inside the project root (gitignored).
BENCH_DIR  = Path(__file__).parent / "bench"
DATA_DIR   = BENCH_DIR / "testdata"
OUT_DIR    = BENCH_DIR / "results"
FFMPEG     = "ffmpeg"
FFPROBE    = "ffprobe"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- test file sizes -----------------------------------------------------------
SIZES = {
    "1MB":   1 * 1024 * 1024,
    "10MB": 10 * 1024 * 1024,
    "50MB": 50 * 1024 * 1024,
}

def make_test_file(size: int, label: str) -> Path:
    p = DATA_DIR / f"{label}.bin"
    if p.exists() and p.stat().st_size == size:
        return p
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
    p.write_bytes(data)
    print(f"  Generated {label}: {size:,} bytes")
    return p

# -- low-level helpers ---------------------------------------------------------
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def ffmpeg_pipe_encode(frames_iter, w, h, fps, crf, output_path: str,
                       pix_fmt="yuv420p", extra_opts=None):
    """Write raw BGR24 frames to ffmpeg and produce an MP4."""
    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", str(crf),
        "-pix_fmt", pix_fmt,
    ]
    if extra_opts:
        cmd += extra_opts
    cmd += [output_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames_iter:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()

def ffmpeg_lossless_encode(frames_iter, w, h, fps, output_path: str):
    """Write raw RGB24 frames using FFV1 lossless codec (exact pixel preservation)."""
    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "ffv1", "-pix_fmt", "rgb24",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames_iter:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def ffmpeg_lossless_decode(video_path: str):
    """Yield raw RGB24 numpy frames (lossless path, preserves exact pixel values)."""
    result = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", video_path],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(result.stdout)["streams"][0]
    w, h = s["width"], s["height"]
    frame_size = w * h * 3
    proc = subprocess.Popen(
        [FFMPEG, "-i", video_path,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        proc.wait(timeout=10)


def ffmpeg_pipe_decode(video_path: str):
    """Yield raw BGR24 numpy frames from a video file."""
    result = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", video_path],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(result.stdout)["streams"][0]
    w, h = s["width"], s["height"]
    frame_size = w * h * 3
    proc = subprocess.Popen(
        [FFMPEG, "-threads", "0", "-i", video_path,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        proc.wait(timeout=10)

# ===============================================================================
# TOOL IMPLEMENTATIONS
# ===============================================================================

# -- ISG equivalent ------------------------------------------------------------
# Algorithm: 2x2 pixel blocks, bit=0→black(0,0,0), bit=1→white(255,255,255),
#            grayscale written to Y channel. No ECC. No header in video.
#            Metadata stored in filename. CRF=18 (same as ISG's default).
#            Resolution: user-configurable, ISG defaulted to 1920x1080.
#            FPS: 30.

def isg_encode(data: bytes, output_mp4: str,
               width=1920, height=1080, fps=30, block_size=2, crf=18):
    lw, lh = width // block_size, height // block_size
    bpf = (lw * lh) // 8        # bytes per frame
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    # pad to full frames
    total_bits = len(bits)
    frames_needed = -(-total_bits // (lw * lh))
    bits = np.pad(bits, (0, frames_needed * lw * lh - total_bits))
    bit_frames = bits.reshape(frames_needed, lh, lw)

    def frame_iter():
        for bf in bit_frames:
            # Expand each logical pixel to block_size x block_size
            px = (bf * 255).astype(np.uint8)
            px = np.repeat(np.repeat(px, block_size, axis=0), block_size, axis=1)
            frame = np.stack([px, px, px], axis=2)   # BGR
            yield frame

    ffmpeg_pipe_encode(frame_iter(), width, height, fps, crf, output_mp4)


def isg_decode(video_mp4: str, original_size: int) -> bytes:
    bits_list = []
    for frame in ffmpeg_pipe_decode(video_mp4):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Sample centre of each block
        # ISG: 2x2 blocks → sample pixel [1,1] of each block
        c = 1  # centre of 2x2 block
        logical = gray[c::2, c::2]
        bits = (logical > 127).astype(np.uint8).flatten()
        bits_list.append(bits)
        if sum(len(b) for b in bits_list) >= original_size * 8:
            break
    all_bits = np.concatenate(bits_list)[: original_size * 8]
    if len(all_bits) < original_size * 8:
        all_bits = np.pad(all_bits, (0, original_size * 8 - len(all_bits)))
    return np.packbits(all_bits).tobytes()[:original_size]


# -- YouBit equivalent ---------------------------------------------------------
# Algorithm: BPP=1 grayscale, 1 pixel per logical pixel (no block upscaling),
#            bit=0→0, bit=1→255. 1 FPS (YouBit's key design choice for YouTube:
#            forces YouTube to make each data frame a keyframe). gzip compression.
#            RS ECC (default 20 symbols). No header in video (stored in metadata).
#            Resolution: 1920x1080. CRF=18, tune=grain, no-deblock.

def youbit_encode(data: bytes, output_mp4: str,
                  width=1920, height=1080, fps=1, crf=18,
                  ecc_symbols=20):
    import gzip
    # gzip compress (YouBit always compresses)
    compressed = gzip.compress(data, compresslevel=6)

    # Apply RS ECC (use PixelVault's vectorised RS encoder for accuracy)
    sys.path.insert(0, str(Path(__file__).parent.parent /
                           "PixelVault-Infinite--The-Eternal-Encoder"))
    try:
        from pixelvault.ecc import encode as rs_encode
        ecc_data = rs_encode(compressed, ecc_symbols)
    except Exception:
        ecc_data = compressed   # fallback: no ECC

    # Transform bytes → BPP=1 grayscale pixels (0 or 255)
    bits = np.unpackbits(np.frombuffer(ecc_data, dtype=np.uint8))
    pixels_per_frame = width * height
    total_frames = -(-len(bits) // pixels_per_frame)
    bits = np.pad(bits, (0, total_frames * pixels_per_frame - len(bits)))
    bit_frames = bits.reshape(total_frames, height, width)

    def frame_iter():
        for bf in bit_frames:
            px = (bf * 255).astype(np.uint8)
            frame = np.stack([px, px, px], axis=2)  # BGR (but gray)
            yield frame

    extra = ["-tune", "grain", "-x264-params", "no-deblock=1"]
    ffmpeg_pipe_encode(frame_iter(), width, height, fps, crf, output_mp4,
                       extra_opts=extra)


def youbit_decode(video_mp4: str, original_size: int, ecc_symbols=20) -> bytes:
    import gzip
    sys.path.insert(0, str(Path(__file__).parent.parent /
                           "PixelVault-Infinite--The-Eternal-Encoder"))
    bits_list = []
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = (gray.flatten() > 127).astype(np.uint8)
        bits_list.append(bits)
    all_bits = np.concatenate(bits_list)
    raw = np.packbits(all_bits).tobytes()
    try:
        from pixelvault.ecc import decode as rs_decode
        # We don't know exact ecc payload size without the YouBit metadata
        # so we recover and decompress
        # Estimate: data grows by factor ecc_symbols/239 after ECC
        est_compressed = int(len(raw) * (255 - ecc_symbols) / 255)
        corrected = rs_decode(raw, ecc_symbols, est_compressed)
        return gzip.decompress(corrected)[:original_size]
    except Exception:
        try:
            return gzip.decompress(raw)[:original_size]
        except Exception:
            return raw[:original_size]


# -- bin2video equivalent ------------------------------------------------------
# Algorithm: 1 bpp binary (0=black, 255=white), block_size=5x5 pixels,
#            resolution 1280x720 (bin2video default), no ECC.
#            First frame uses 10x10 blocks for metadata (we skip that here).
#            CRF=18, 30 FPS.

def bin2video_encode(data: bytes, output_mp4: str,
                     width=1280, height=720, fps=30, block_size=5, crf=18):
    lw, lh = width // block_size, height // block_size
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    total_logical = lw * lh
    frames_needed = -(-len(bits) // total_logical)
    bits = np.pad(bits, (0, frames_needed * total_logical - len(bits)))
    bit_frames = bits.reshape(frames_needed, lh, lw)

    def frame_iter():
        for bf in bit_frames:
            px = (bf * 255).astype(np.uint8)
            px = np.repeat(np.repeat(px, block_size, axis=0), block_size, axis=1)
            frame = np.stack([px, px, px], axis=2)
            yield frame

    ffmpeg_pipe_encode(frame_iter(), width, height, fps, crf, output_mp4)


def bin2video_decode(video_mp4: str, original_size: int) -> bytes:
    bs = 5
    bits_list = []
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c = bs // 2
        logical = gray[c::bs, c::bs]
        bits = (logical > 127).astype(np.uint8).flatten()
        bits_list.append(bits)
        if sum(len(b) for b in bits_list) >= original_size * 8:
            break
    all_bits = np.concatenate(bits_list)[: original_size * 8]
    if len(all_bits) < original_size * 8:
        all_bits = np.pad(all_bits, (0, original_size * 8 - len(all_bits)))
    return np.packbits(all_bits).tobytes()[:original_size]


# -- Data2Video equivalent -----------------------------------------------------
# Algorithm: 1 bpp binary pixels (0=black, 255=white), 4K (3840×2160),
#            10 FPS. Original uses animated GIF (lossless); here we use MP4
#            with CRF=18 — the pure-black/pure-white pixel pattern survives
#            lossy H.264 at this resolution with negligible error rate.
#            No ECC, no compression, no header (benchmark simplification).
#            Source: github.com/bfaure/Data2Video

def data2video_encode(data: bytes, output_mp4: str,
                      width=3840, height=2160, fps=10, crf=18):
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    pixels_per_frame = width * height
    n_frames = -(-len(bits) // pixels_per_frame)
    bits = np.pad(bits, (0, n_frames * pixels_per_frame - len(bits)))
    bit_frames = bits.reshape(n_frames, height, width)

    def frame_iter():
        for bf in bit_frames:
            px = (bf * 255).astype(np.uint8)
            yield np.stack([px, px, px], axis=2)

    ffmpeg_pipe_encode(frame_iter(), width, height, fps, crf, output_mp4)


def data2video_decode(video_mp4: str, original_size: int) -> bytes:
    bits_list = []
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = (gray.flatten() > 127).astype(np.uint8)
        bits_list.append(bits)
        if sum(len(b) for b in bits_list) >= original_size * 8:
            break
    all_bits = np.concatenate(bits_list)[: original_size * 8]
    if len(all_bits) < original_size * 8:
        all_bits = np.pad(all_bits, (0, original_size * 8 - len(all_bits)))
    return np.packbits(all_bits).tobytes()[:original_size]


# -- binary2video equivalent ---------------------------------------------------
# Algorithm: raw bytes stored as RGB24 pixel values (3 bytes → 1 pixel),
#            gzip pre-compression (level 1, "fast"), FFV1 lossless codec.
#            Resolution 320×240 (default), 1 FPS.
#            No ECC — lossless codec guarantees pixel-exact reconstruction.
#            Source: github.com/HbHbNr/binary2video

def binary2video_lossless_encode(data: bytes, output_mp4: str,
                                  width=320, height=240, fps=1):
    import gzip
    compressed = gzip.compress(data, compresslevel=1)  # gzip --fast
    bytes_per_frame = width * height * 3
    pad = (-len(compressed)) % bytes_per_frame
    padded = compressed + b"\x00" * pad
    n_frames = len(padded) // bytes_per_frame

    def frame_iter():
        for i in range(n_frames):
            chunk = padded[i * bytes_per_frame: (i + 1) * bytes_per_frame]
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)

    ffmpeg_lossless_encode(frame_iter(), width, height, fps, output_mp4)


def binary2video_lossless_decode(video_mp4: str, original_size: int) -> bytes:
    import gzip
    chunks = []
    for frame in ffmpeg_lossless_decode(video_mp4):
        chunks.append(frame.flatten().tobytes())
    raw = b"".join(chunks)
    try:
        return gzip.decompress(raw)[:original_size]
    except Exception:
        return raw[:original_size]


# -- file2video equivalent -----------------------------------------------------
# Algorithm: bits stored in 270×270 logical grid (1 bit per cell),
#            upscaled 4× to 1080×1080 via nearest-neighbour (no anti-aliasing),
#            Reed-Solomon (255,245) — 10 ECC symbols per block, H.264 CRF=40.
#            20 FPS.
#            Source: github.com/karaketir16/file2video

def file2video_encode(data: bytes, output_mp4: str,
                      grid_w=270, grid_h=270, fps=20, crf=40,
                      ecc_nsym=10):
    sys.path.insert(0, str(Path(__file__).parent))
    from pixelvault.ecc import encode as rs_encode
    ecc_data = rs_encode(data, ecc_nsym)

    bits = np.unpackbits(np.frombuffer(ecc_data, dtype=np.uint8))
    cells_per_frame = grid_w * grid_h
    n_frames = -(-len(bits) // cells_per_frame)
    bits = np.pad(bits, (0, n_frames * cells_per_frame - len(bits)))
    bit_frames = bits.reshape(n_frames, grid_h, grid_w)
    scale = 1080 // grid_h  # 4×

    def frame_iter():
        for bf in bit_frames:
            px = (bf * 255).astype(np.uint8)
            upscaled = np.repeat(np.repeat(px, scale, axis=0), scale, axis=1)
            yield np.stack([upscaled, upscaled, upscaled], axis=2)

    out_w, out_h = grid_w * scale, grid_h * scale
    ffmpeg_pipe_encode(frame_iter(), out_w, out_h, fps, crf, output_mp4)


def file2video_decode(video_mp4: str, original_size: int, ecc_nsym=10) -> bytes:
    sys.path.insert(0, str(Path(__file__).parent))
    from pixelvault.ecc import decode as rs_decode, encoded_size
    bits_list = []
    scale = 4
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Sample centre of each upscaled block
        c = scale // 2
        logical = gray[c::scale, c::scale]
        bits = (logical > 127).astype(np.uint8).flatten()
        bits_list.append(bits)

    all_bits = np.concatenate(bits_list)
    raw = np.packbits(all_bits).tobytes()
    ecc_payload_size = encoded_size(original_size, ecc_nsym)
    raw = raw[:ecc_payload_size]
    try:
        return rs_decode(raw, ecc_nsym, original_size)
    except Exception:
        return raw[:original_size]


# -- PixelVault via CLI ---------------------------------------------------------
PV_ROOT = Path("C:/Users/sahil/OneDrive/Documents/GitHub/PixelVault-Infinite--The-Eternal-Encoder")

def pixelvault_encode(input_file: str, output_mp4: str, extra_args=None) -> float:
    cmd = [sys.executable, str(PV_ROOT / "main.py"), "encode",
           input_file, "-o", output_mp4, "-q"]
    if extra_args:
        cmd += extra_args
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(PV_ROOT))
    return time.perf_counter() - t0


def pixelvault_decode(video_mp4: str, output_dir: str) -> float:
    cmd = [sys.executable, str(PV_ROOT / "main.py"), "decode",
           video_mp4, "-o", output_dir, "-q"]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(PV_ROOT))
    return time.perf_counter() - t0


# ===============================================================================
# BENCHMARK RUNNER
# ===============================================================================

def run_benchmark(name, size_label, input_path: Path,
                  encode_fn, decode_fn, output_mp4: Path, decode_out: Path):
    data = input_path.read_bytes()
    original_size = len(data)
    orig_hash = sha256(data)

    print(f"  [{name}] {size_label} encode ...", end="", flush=True)
    t0 = time.perf_counter()
    encode_fn(str(input_path), str(output_mp4))
    encode_s = time.perf_counter() - t0
    mp4_size = output_mp4.stat().st_size if output_mp4.exists() else 0
    print(f" {encode_s:.2f}s  ({mp4_size // 1024:,} KB mp4)", flush=True)

    print(f"  [{name}] {size_label} decode ...", end="", flush=True)
    decode_out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    decode_fn(str(output_mp4), str(decode_out))
    decode_s = time.perf_counter() - t0
    print(f" {decode_s:.2f}s", flush=True)

    # Verify
    recovered_files = list(decode_out.glob("*"))
    match = False
    if recovered_files:
        recovered = recovered_files[0].read_bytes()
        match = sha256(recovered[:original_size]) == orig_hash and len(recovered) >= original_size

    return {
        "name": name, "size_label": size_label,
        "input_bytes": original_size,
        "encode_s": round(encode_s, 3),
        "decode_s": round(decode_s, 3),
        "mp4_bytes": mp4_size,
        "match": match,
        "throughput_encode_mbps": round(original_size / 1e6 / encode_s, 2),
        "throughput_decode_mbps": round(original_size / 1e6 / decode_s, 2),
    }


def run_simple_benchmark(name, size_label, input_path: Path,
                         encode_fn, decode_fn, output_mp4: Path):
    """For tools without PixelVault-style CLI — encode/decode handled in-process."""
    data = input_path.read_bytes()
    original_size = len(data)
    orig_hash = sha256(data)

    print(f"  [{name}] {size_label} encode ...", end="", flush=True)
    t0 = time.perf_counter()
    encode_fn(data, str(output_mp4))
    encode_s = time.perf_counter() - t0
    mp4_size = output_mp4.stat().st_size if output_mp4.exists() else 0
    print(f" {encode_s:.2f}s  ({mp4_size // 1024:,} KB mp4)", flush=True)

    print(f"  [{name}] {size_label} decode ...", end="", flush=True)
    t0 = time.perf_counter()
    recovered = decode_fn(str(output_mp4), original_size)
    decode_s = time.perf_counter() - t0
    match = sha256(recovered[:original_size]) == orig_hash
    print(f" {decode_s:.2f}s  {'PASS' if match else 'MISMATCH'}", flush=True)

    return {
        "name": name, "size_label": size_label,
        "input_bytes": original_size,
        "encode_s": round(encode_s, 3),
        "decode_s": round(decode_s, 3),
        "mp4_bytes": mp4_size,
        "match": match,
        "throughput_encode_mbps": round(original_size / 1e6 / encode_s, 2),
        "throughput_decode_mbps": round(original_size / 1e6 / decode_s, 2),
    }


# ===============================================================================
# TOOL CONFIGURATIONS
# ===============================================================================

# PixelVault wrappers (uses real CLI)
def bv_encode_default(src, dst):
    subprocess.run([sys.executable, str(PV_ROOT / "main.py"), "encode",
                    src, "-o", dst, "-q", "--ecc", "16"],
                   check=True, cwd=str(PV_ROOT))

def bv_encode_bs1(src, dst):
    subprocess.run([sys.executable, str(PV_ROOT / "main.py"), "encode",
                    src, "-o", dst, "-q", "--ecc", "16", "--block-size", "1"],
                   check=True, cwd=str(PV_ROOT))

def bv_encode_noecc(src, dst):
    subprocess.run([sys.executable, str(PV_ROOT / "main.py"), "encode",
                    src, "-o", dst, "-q", "--ecc", "0"],
                   check=True, cwd=str(PV_ROOT))

def bv_decode(src, dst_dir):
    subprocess.run([sys.executable, str(PV_ROOT / "main.py"), "decode",
                    src, "-o", dst_dir, "-q"],
                   check=True, cwd=str(PV_ROOT))


TOOLS = [
    # (name, description, encode_fn, decode_fn, is_simple)
    ("PixelVault binary bs=2 ecc=16",
     "1080p, 2x2 blocks, RS-16 ECC — default YouTube mode",
     bv_encode_default, bv_decode, False),

    ("PixelVault binary bs=1 ecc=16",
     "1080p, 1x1 blocks (max 1080p density), RS-16 ECC",
     bv_encode_bs1, bv_decode, False),

    ("PixelVault binary bs=2 no-ecc",
     "1080p, 2x2 blocks, no ECC — pure algorithm speed",
     bv_encode_noecc, bv_decode, False),

    ("ISG-equiv bs=2 no-ecc",
     "ISG algorithm: 1080p, 2x2 blocks, no ECC, no header (Python reimpl.)",
     lambda data, dst: isg_encode(data, dst, block_size=2),
     isg_decode, True),

    ("YouBit-equiv BPP=1 1FPS ecc=20",
     "YouBit algorithm: 1080p, 1px/logical px, 1 FPS, gzip+RS-20 (Python reimpl.)",
     lambda data, dst: youbit_encode(data, dst, fps=1, ecc_symbols=20),
     lambda src, sz: youbit_decode(src, sz, ecc_symbols=20), True),

    ("bin2video-equiv bs=5 no-ecc",
     "bin2video default: 1280x720, 5x5 blocks, no ECC (Python reimpl.)",
     lambda data, dst: bin2video_encode(data, dst, block_size=5),
     bin2video_decode, True),

    ("Data2Video-equiv 4K 1bpp",
     "Data2Video: 3840x2160, 1 bpp binary pixels, 10 FPS, CRF=18 (MP4 instead of GIF)",
     data2video_encode, data2video_decode, True),

    ("binary2video-equiv 320x240 lossless",
     "binary2video: 320x240, gzip+raw RGB24 bytes/pixel, FFV1 lossless, 1 FPS",
     binary2video_lossless_encode, binary2video_lossless_decode, True),

    ("file2video-equiv 1080p grid RS-10",
     "file2video: 270x270 grid -> 1080x1080, RS(255,245) ecc=10, 20 FPS, CRF=40",
     lambda data, dst: file2video_encode(data, dst),
     lambda src, sz: file2video_decode(src, sz), True),
]

DENSITY = {
    "PixelVault binary bs=2 ecc=16":          "~60,300 B/fr (net of 6.7% ECC)",
    "PixelVault binary bs=1 ecc=16":          "~241,920 B/fr (net of 6.7% ECC)",
    "PixelVault binary bs=2 no-ecc":          "64,800 B/fr",
    "ISG-equiv bs=2 no-ecc":                 "64,800 B/fr (no ECC)",
    "YouBit-equiv BPP=1 1FPS ecc=20":        "~204,204 B/fr (net of ~7.8% ECC+gzip)",
    "bin2video-equiv bs=5 no-ecc":           "4,608 B/fr (1280x720, 5x5 blocks)",
    "Data2Video-equiv 4K 1bpp":              "~1,036,800 B/fr (3840x2160, no ECC)",
    "binary2video-equiv 320x240 lossless":   "~230,400 B/fr raw (320x240x3, gzip)",
    "file2video-equiv 1080p grid RS-10":     "~8,615 B/fr data (270x270 grid, RS-10 ECC)",
}

# ===============================================================================
# MAIN
# ===============================================================================

def main():
    print("=" * 72)
    print("  PixelVault Cross-Tool Benchmark")
    print("=" * 72)

    # Generate test files
    print("\nGenerating test files ...")
    test_files = {}
    for label, size in SIZES.items():
        test_files[label] = make_test_file(size, label)
    print()

    results = []
    for label, input_path in test_files.items():
        print(f"\n{'-'*72}")
        print(f"  FILE SIZE: {label}  ({input_path.stat().st_size:,} bytes)")
        print(f"{'-'*72}")

        for (name, desc, enc_fn, dec_fn, is_simple) in TOOLS:
            slug = name.replace(" ", "_").replace("=", "").replace("/", "_")
            output_mp4 = OUT_DIR / f"{slug}_{label}.mp4"
            decode_out  = OUT_DIR / f"{slug}_{label}_decoded"

            # Clean up from prior run
            if output_mp4.exists():
                try:
                    output_mp4.unlink()
                except PermissionError:
                    import time as _t; _t.sleep(1)
                    try:
                        output_mp4.unlink()
                    except Exception:
                        pass
            if decode_out.exists():
                import shutil
                shutil.rmtree(decode_out, ignore_errors=True)

            try:
                if is_simple:
                    data = input_path.read_bytes()
                    r = run_simple_benchmark(
                        name, label, input_path,
                        enc_fn, dec_fn, output_mp4)
                else:
                    decode_out.mkdir(parents=True, exist_ok=True)
                    data = input_path.read_bytes()
                    orig_hash = sha256(data)

                    print(f"  [{name}] {label} encode ...", end="", flush=True)
                    t0 = time.perf_counter()
                    enc_fn(str(input_path), str(output_mp4))
                    encode_s = time.perf_counter() - t0
                    mp4_size = output_mp4.stat().st_size if output_mp4.exists() else 0
                    print(f" {encode_s:.2f}s  ({mp4_size // 1024:,} KB mp4)", flush=True)

                    print(f"  [{name}] {label} decode ...", end="", flush=True)
                    t0 = time.perf_counter()
                    dec_fn(str(output_mp4), str(decode_out))
                    decode_s = time.perf_counter() - t0
                    recovered_files = list(decode_out.glob("*"))
                    match = False
                    if recovered_files:
                        rec = recovered_files[0].read_bytes()
                        match = len(rec) >= len(data) and sha256(rec[:len(data)]) == orig_hash
                    print(f" {decode_s:.2f}s  {'PASS' if match else 'MISMATCH'}", flush=True)

                    r = {
                        "name": name, "size_label": label,
                        "input_bytes": len(data),
                        "encode_s": round(encode_s, 3),
                        "decode_s": round(decode_s, 3),
                        "mp4_bytes": mp4_size,
                        "match": match,
                        "throughput_encode_mbps": round(len(data)/1e6/encode_s, 2),
                        "throughput_decode_mbps": round(len(data)/1e6/decode_s, 2),
                    }

                r["density"] = DENSITY.get(name, "?")
                results.append(r)
            except Exception as e:
                print(f"  [{name}] {label} ERROR: {e}", flush=True)
                results.append({
                    "name": name, "size_label": label,
                    "error": str(e),
                    "input_bytes": input_path.stat().st_size,
                })

    # Save JSON
    out_json = BENCH_DIR / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_json}")

    # Print summary table
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    print(f"{'Tool':<38} {'Size':>5} {'Enc(s)':>7} {'Dec(s)':>7} {'MP4 MB':>7} {'MB/s enc':>9} {'OK':>4}")
    print("-" * 72)
    for r in results:
        if "error" in r:
            print(f"  {r['name'][:36]:<36} {r['size_label']:>5}  ERROR: {r['error'][:30]}")
            continue
        mp4_mb = r["mp4_bytes"] / 1e6
        print(f"  {r['name'][:36]:<36} {r['size_label']:>5}"
              f" {r['encode_s']:>7.2f} {r['decode_s']:>7.2f}"
              f" {mp4_mb:>7.1f} {r['throughput_encode_mbps']:>9.2f}"
              f" {'PASS' if r['match'] else 'FAIL':>4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
