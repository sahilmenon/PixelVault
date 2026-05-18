#!/usr/bin/env python3
"""
ByteVault Cross-Tool Benchmark — Jellyfish Edition

Downloads real-world Jellyfish video files as binary test data and benchmarks
9 file-to-video encoding tools. Each (tool, file) pair runs 3 times.
README is updated after each file size completes.

Faithful per-tool configurations:
  ISG-equiv        2x2 blocks, 1080p, 30 FPS, NO ECC, NO compression
  YouBit-equiv     1x1 pixels, 1080p, 1 FPS,  gzip(6) + RS-20 ECC
  bin2video-equiv  5x5 blocks,  720p, 30 FPS, NO ECC, NO compression
  Data2Video-equiv 1x1 pixels,   4K,  10 FPS, NO ECC, NO compression
  binary2video     raw RGB24, 320x240, 1 FPS,  gzip(1), FFV1 lossless
  file2video       270x270 grid->1080p, 20 FPS, RS-10 ECC, CRF=40
  ByteVault bs=2 ecc=16 1080p | bs=1 ecc=16 1080p | bs=2 ecc=16 4K
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

# ── config ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
BENCH_DIR    = ROOT / "bench"
DATA_DIR     = BENCH_DIR / "jellyfish"
OUT_DIR      = BENCH_DIR / "jf_results"
RESULTS_JSON = BENCH_DIR / "jellyfish_results.json"
README       = ROOT / "README.md"
FFMPEG       = "ffmpeg"
FFPROBE      = "ffprobe"
BV_ROOT      = ROOT

RUNS        = 3
ENC_TIMEOUT = 1800   # 30 min
DEC_TIMEOUT = 1800   # 30 min

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── jellyfish test files ──────────────────────────────────────────────────────
BASE_URL = "http://larmoire.org/jellyfish/media/"

JELLYFISH = [
    ("~15 MB HD",  "jellyfish-3-mbps-hd-h264.mkv"),
    ("~52 MB HD",  "jellyfish-10-mbps-hd-h264.mkv"),
    ("~150 MB HD", "jellyfish-40-mbps-hd-h264.mkv"),
    ("~375 MB HD", "jellyfish-100-mbps-hd-h264.mkv"),
    ("~450 MB 4K", "jellyfish-120-mbps-4k-uhd-h264.mkv"),
    ("~1.4 GB 4K", "jellyfish-400-mbps-4k-uhd-hevc-10bit.mkv"),
]

PREDOWNLOADED = {
    "jellyfish-400-mbps-4k-uhd-hevc-10bit.mkv":
        Path(r"C:\Users\sahil\Downloads\jellyfish-400-mbps-4k-uhd-hevc-10bit.mkv"),
}

# ── per-thread subprocess tracking (for timeout kills) ───────────────────────
# Each encode/decode function registers its active subprocess so the timeout
# handler can kill it when the thread is abandoned.

_active_procs: dict = {}   # {thread_ident: subprocess.Popen}
_procs_lock = threading.Lock()

def _reg_proc(proc):
    with _procs_lock:
        _active_procs[threading.get_ident()] = proc

def _unreg_proc():
    with _procs_lock:
        _active_procs.pop(threading.get_ident(), None)

def _kill_thread_proc(thread_ident: int):
    with _procs_lock:
        proc = _active_procs.pop(thread_ident, None)
    if proc is None:
        return
    try:
        # Use taskkill /T to also kill ffmpeg grandchildren
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False, timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def ffmpeg_pipe_encode(frames_iter, w, h, fps, crf, output_path,
                       pix_fmt="yuv420p", extra_opts=None):
    cmd = [FFMPEG, "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0",
           "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", pix_fmt]
    if extra_opts:
        cmd += extra_opts
    cmd.append(str(output_path))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _reg_proc(proc)
    try:
        for frame in frames_iter:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
    finally:
        _unreg_proc()


def ffmpeg_lossless_encode(frames_iter, w, h, fps, output_path):
    cmd = [FFMPEG, "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0",
           "-c:v", "ffv1", "-pix_fmt", "rgb24", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _reg_proc(proc)
    try:
        for frame in frames_iter:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
    finally:
        _unreg_proc()


def ffmpeg_lossless_decode(video_path):
    r = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", str(video_path)],
        capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    w, h = s["width"], s["height"]
    frame_size = w * h * 3
    proc = subprocess.Popen(
        [FFMPEG, "-i", str(video_path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=30)


def ffmpeg_pipe_decode(video_path):
    r = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", str(video_path)],
        capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    w, h = s["width"], s["height"]
    frame_size = w * h * 3
    proc = subprocess.Popen(
        [FFMPEG, "-threads", "0", "-i", str(video_path),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=30)


# ── tool implementations ──────────────────────────────────────────────────────
# encode(input_path: str, output_mp4: str) -> None   (lazy, no full pre-alloc)
# decode(video_mp4: str, original_size: int) -> bytes

def _isg_encode(input_path: str, output_mp4: str,
                width=1920, height=1080, fps=30, block_size=2, crf=18):
    lw, lh = width // block_size, height // block_size
    Bpf = (lw * lh) >> 3
    payload = np.frombuffer(Path(input_path).read_bytes(), dtype=np.uint8)
    n_frames = -(-len(payload) // Bpf)
    bpf = lw * lh

    def frame_iter():
        for i in range(n_frames):
            b0, b1 = i * Bpf, min((i + 1) * Bpf, len(payload))
            bits = np.unpackbits(payload[b0:b1])
            if len(bits) < bpf:
                bits = np.append(bits, np.zeros(bpf - len(bits), np.uint8))
            px = (bits.reshape(lh, lw) * 255).astype(np.uint8)
            px = np.repeat(np.repeat(px, block_size, 0), block_size, 1)
            yield np.stack([px, px, px], axis=2)

    ffmpeg_pipe_encode(frame_iter(), width, height, fps, crf, output_mp4)


def _isg_decode(video_mp4: str, original_size: int) -> bytes:
    chunks, collected = [], 0
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = (gray[1::2, 1::2] > 127).astype(np.uint8).flatten()
        chunks.append(np.packbits(bits).tobytes())
        collected += len(chunks[-1])
        if collected >= original_size:
            break
    return b"".join(chunks)[:original_size]


def _youbit_encode(input_path: str, output_mp4: str,
                   width=1920, height=1080, fps=1, crf=18, ecc_symbols=20):
    import gzip
    data = Path(input_path).read_bytes()
    compressed = gzip.compress(data, compresslevel=6)
    try:
        sys.path.insert(0, str(ROOT))
        from bytevault.ecc import encode as rs_encode
        ecc_data = rs_encode(compressed, ecc_symbols)
    except Exception:
        ecc_data = compressed
    ppf = width * height
    Bpf = ppf >> 3
    payload = np.frombuffer(ecc_data, dtype=np.uint8)
    n_frames = -(-len(payload) // Bpf)

    def frame_iter():
        for i in range(n_frames):
            b0, b1 = i * Bpf, min((i + 1) * Bpf, len(payload))
            bits = np.unpackbits(payload[b0:b1])
            if len(bits) < ppf:
                bits = np.append(bits, np.zeros(ppf - len(bits), np.uint8))
            px = (bits.reshape(height, width) * 255).astype(np.uint8)
            yield np.stack([px, px, px], axis=2)

    ffmpeg_pipe_encode(frame_iter(), width, height, fps, crf, output_mp4,
                       extra_opts=["-tune", "grain", "-x264-params", "no-deblock=1"])


def _youbit_decode(video_mp4: str, original_size: int, ecc_symbols=20) -> bytes:
    import gzip
    sys.path.insert(0, str(ROOT))
    chunks = []
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chunks.append(np.packbits((gray.flatten() > 127).astype(np.uint8)).tobytes())
    raw = b"".join(chunks)
    try:
        from bytevault.ecc import decode as rs_decode
        corrected = rs_decode(raw, ecc_symbols, int(len(raw) * (255 - ecc_symbols) / 255))
        return gzip.decompress(corrected)[:original_size]
    except Exception:
        try:
            return gzip.decompress(raw)[:original_size]
        except Exception:
            return raw[:original_size]


def _bin2video_encode(input_path: str, output_mp4: str,
                      width=1280, height=720, fps=30, block_size=5, crf=18):
    lw, lh = width // block_size, height // block_size
    Bpf = (lw * lh) >> 3
    payload = np.frombuffer(Path(input_path).read_bytes(), dtype=np.uint8)
    n_frames = -(-len(payload) // Bpf)
    bpf = lw * lh

    def frame_iter():
        for i in range(n_frames):
            b0, b1 = i * Bpf, min((i + 1) * Bpf, len(payload))
            bits = np.unpackbits(payload[b0:b1])
            if len(bits) < bpf:
                bits = np.append(bits, np.zeros(bpf - len(bits), np.uint8))
            px = (bits.reshape(lh, lw) * 255).astype(np.uint8)
            px = np.repeat(np.repeat(px, block_size, 0), block_size, 1)
            yield np.stack([px, px, px], axis=2)

    ffmpeg_pipe_encode(frame_iter(), width, height, fps, crf, output_mp4)


def _bin2video_decode(video_mp4: str, original_size: int) -> bytes:
    bs, c = 5, 2
    chunks, collected = [], 0
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = (gray[c::bs, c::bs] > 127).astype(np.uint8).flatten()
        chunks.append(np.packbits(bits).tobytes())
        collected += len(chunks[-1])
        if collected >= original_size:
            break
    return b"".join(chunks)[:original_size]


def _data2video_encode(input_path: str, output_mp4: str,
                       width=3840, height=2160, fps=10, crf=18):
    ppf = width * height
    Bpf = ppf >> 3
    payload = np.frombuffer(Path(input_path).read_bytes(), dtype=np.uint8)
    n_frames = -(-len(payload) // Bpf)

    def frame_iter():
        for i in range(n_frames):
            b0, b1 = i * Bpf, min((i + 1) * Bpf, len(payload))
            bits = np.unpackbits(payload[b0:b1])
            if len(bits) < ppf:
                bits = np.append(bits, np.zeros(ppf - len(bits), np.uint8))
            px = (bits.reshape(height, width) * 255).astype(np.uint8)
            yield np.stack([px, px, px], axis=2)

    ffmpeg_pipe_encode(frame_iter(), width, height, fps, crf, output_mp4)


def _data2video_decode(video_mp4: str, original_size: int) -> bytes:
    chunks, collected = [], 0
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chunks.append(np.packbits((gray.flatten() > 127).astype(np.uint8)).tobytes())
        collected += len(chunks[-1])
        if collected >= original_size:
            break
    return b"".join(chunks)[:original_size]


def _binary2video_encode(input_path: str, output_mp4: str,
                         width=320, height=240, fps=1):
    import gzip
    compressed = gzip.compress(Path(input_path).read_bytes(), compresslevel=1)
    Bpf = width * height * 3
    pad = (-len(compressed)) % Bpf
    padded = compressed + b"\x00" * pad
    n_frames = len(padded) // Bpf

    def frame_iter():
        for i in range(n_frames):
            chunk = padded[i * Bpf:(i + 1) * Bpf]
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)

    ffmpeg_lossless_encode(frame_iter(), width, height, fps, output_mp4)


def _binary2video_decode(video_mp4: str, original_size: int) -> bytes:
    import gzip
    chunks = []
    for frame in ffmpeg_lossless_decode(video_mp4):
        chunks.append(frame.flatten().tobytes())
    raw = b"".join(chunks)
    try:
        return gzip.decompress(raw)[:original_size]
    except Exception:
        return raw[:original_size]


def _file2video_encode(input_path: str, output_mp4: str,
                       grid_w=270, grid_h=270, fps=20, crf=40, ecc_nsym=10):
    sys.path.insert(0, str(ROOT))
    from bytevault.ecc import encode as rs_encode
    data = Path(input_path).read_bytes()
    ecc_data = rs_encode(data, ecc_nsym)
    Bpf = (grid_w * grid_h) >> 3
    payload = np.frombuffer(ecc_data, dtype=np.uint8)
    n_frames = -(-len(payload) // Bpf)
    bpf = grid_w * grid_h
    scale = 1080 // grid_h
    out_w, out_h = grid_w * scale, grid_h * scale

    def frame_iter():
        for i in range(n_frames):
            b0, b1 = i * Bpf, min((i + 1) * Bpf, len(payload))
            bits = np.unpackbits(payload[b0:b1])
            if len(bits) < bpf:
                bits = np.append(bits, np.zeros(bpf - len(bits), np.uint8))
            px = (bits.reshape(grid_h, grid_w) * 255).astype(np.uint8)
            up = np.repeat(np.repeat(px, scale, 0), scale, 1)
            yield np.stack([up, up, up], axis=2)

    ffmpeg_pipe_encode(frame_iter(), out_w, out_h, fps, crf, output_mp4)


def _file2video_decode(video_mp4: str, original_size: int, ecc_nsym=10) -> bytes:
    sys.path.insert(0, str(ROOT))
    from bytevault.ecc import decode as rs_decode, encoded_size
    scale, c = 4, 2
    chunks = []
    for frame in ffmpeg_pipe_decode(video_mp4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = (gray[c::scale, c::scale] > 127).astype(np.uint8).flatten()
        chunks.append(np.packbits(bits).tobytes())
    raw = b"".join(chunks)[:encoded_size(original_size, ecc_nsym)]
    try:
        return rs_decode(raw, ecc_nsym, original_size)
    except Exception:
        return raw[:original_size]


def _bv_encode(input_path: str, output_mp4: str, extra_args=None):
    cmd = [sys.executable, str(BV_ROOT / "main.py"), "encode",
           str(input_path), "-o", str(output_mp4), "-q"]
    if extra_args:
        cmd += extra_args
    proc = subprocess.Popen(cmd, cwd=str(BV_ROOT))
    _reg_proc(proc)
    try:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        _unreg_proc()


def _bv_decode_to_bytes(video_mp4: str, original_size: int) -> bytes:
    out_dir = OUT_DIR / ("bv_dec_" + Path(video_mp4).stem)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(BV_ROOT / "main.py"), "decode",
           str(video_mp4), "-o", str(out_dir), "-q"]
    proc = subprocess.Popen(cmd, cwd=str(BV_ROOT))
    _reg_proc(proc)
    try:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        _unreg_proc()
    files = list(out_dir.glob("*"))
    return files[0].read_bytes()[:original_size] if files else b""


# Named top-level wrappers (lambdas aren't picklable across mp.Process on Windows)
def _bv_enc_bs2_ecc16_1080p(s, d): _bv_encode(s, d, ["--ecc", "16"])
def _bv_enc_bs1_ecc16_1080p(s, d): _bv_encode(s, d, ["--ecc", "16", "--block-size", "1"])
def _bv_enc_bs2_ecc16_4k(s, d):    _bv_encode(s, d, ["--ecc", "16", "--4k"])
def _bv_dec_local(v, sz):           return _bv_decode_to_bytes(v, sz)


# ── tool registry ─────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    name:         str
    desc:         str
    encode_fn:    Callable
    decode_fn:    Callable
    resolution:   str
    density:      str
    ecc:          str
    codec:        str
    youtube_safe: bool = False


TOOLS: List[ToolSpec] = [
    ToolSpec(
        name="ByteVault bs=2 ecc=16 1080p",
        desc="2x2 blocks, RS-16 ECC, 1920x1080. Default YouTube mode.",
        encode_fn=_bv_enc_bs2_ecc16_1080p, decode_fn=_bv_dec_local,
        resolution="1920x1080", density="~60,300 B/fr (net ECC)",
        ecc="RS nsym=16", codec="H.264 HW/libx264", youtube_safe=True,
    ),
    ToolSpec(
        name="ByteVault bs=1 ecc=16 1080p",
        desc="1x1 blocks (max 1080p density), RS-16 ECC, 1920x1080.",
        encode_fn=_bv_enc_bs1_ecc16_1080p, decode_fn=_bv_dec_local,
        resolution="1920x1080", density="~241,920 B/fr (net ECC)",
        ecc="RS nsym=16", codec="H.264 HW/libx264", youtube_safe=True,
    ),
    ToolSpec(
        name="ByteVault bs=2 ecc=16 4K",
        desc="2x2 blocks, RS-16 ECC, 3840x2160. 4x data density vs 1080p bs=2.",
        encode_fn=_bv_enc_bs2_ecc16_4k, decode_fn=_bv_dec_local,
        resolution="3840x2160", density="~241,920 B/fr (net ECC)",
        ecc="RS nsym=16", codec="H.264 HW/libx264", youtube_safe=True,
    ),
    ToolSpec(
        name="ISG-equiv bs=2 no-ecc",
        desc="ISG algo: 2x2 blocks, 1080p, 30 FPS, NO ECC, NO header.",
        encode_fn=_isg_encode, decode_fn=_isg_decode,
        resolution="1920x1080", density="64,800 B/fr",
        ecc="None (faithful to ISG)", codec="H.264 CRF=18", youtube_safe=True,
    ),
    ToolSpec(
        name="YouBit-equiv BPP=1 1FPS ecc=20",
        desc="YouBit: 1x1 px, 1080p, 1 FPS, gzip(6)+RS-20. tune=grain no-deblock.",
        encode_fn=_youbit_encode, decode_fn=_youbit_decode,
        resolution="1920x1080", density="~204,204 B/fr (net gzip+ECC)",
        ecc="gzip + RS nsym=20", codec="H.264 CRF=18 tune=grain", youtube_safe=True,
    ),
    ToolSpec(
        name="bin2video-equiv bs=5 no-ecc",
        desc="bin2video default: 5x5 blocks, 1280x720, 30 FPS, NO ECC.",
        encode_fn=_bin2video_encode, decode_fn=_bin2video_decode,
        resolution="1280x720", density="4,608 B/fr (256x144 logical px)",
        ecc="None (faithful to bin2video)", codec="H.264 CRF=18", youtube_safe=True,
    ),
    ToolSpec(
        name="Data2Video-equiv 4K 1bpp",
        desc="Data2Video: 1x1 px, 3840x2160, 10 FPS, NO ECC. MP4 vs original GIF.",
        encode_fn=_data2video_encode, decode_fn=_data2video_decode,
        resolution="3840x2160", density="~1,036,800 B/fr (1x1 px NOT H.264-safe)",
        ecc="None (faithful to Data2Video)", codec="H.264 CRF=18 (orig: GIF)",
        youtube_safe=False,
    ),
    ToolSpec(
        name="binary2video-equiv 320x240 lossless",
        desc="binary2video: raw bytes as RGB24 pixels, 320x240, gzip(1), FFV1 lossless.",
        encode_fn=_binary2video_encode, decode_fn=_binary2video_decode,
        resolution="320x240", density="~230,400 B/fr raw (gzip reduces output)",
        ecc="None (FFV1 lossless = no corruption)", codec="FFV1 lossless",
        youtube_safe=False,
    ),
    ToolSpec(
        name="file2video-equiv 1080p RS-10",
        desc="file2video: 270x270 grid->1080x1080 (4x NN), RS-10 ECC, 20 FPS, CRF=40.",
        encode_fn=_file2video_encode, decode_fn=_file2video_decode,
        resolution="1080x1080", density="~8,615 B/fr data (270x270 grid, RS-10)",
        ecc="RS(255,245) nsym=10", codec="H.264 CRF=40", youtube_safe=True,
    ),
]


# ── threading-based runner (no spawn overhead; threads share loaded cv2/numpy) ─

def _run_timed(fn, args, timeout_s):
    """Run fn(*args) in a daemon thread. Returns (result, elapsed_s, error_str).
    On timeout: kills the registered subprocess for that thread, returns TIMEOUT.
    MemoryError is caught and reported as OOM without crashing the main process.
    """
    result_box = [None]
    error_box  = [None]
    elapsed_box = [0.0]
    done_event = threading.Event()

    def worker():
        t0 = time.perf_counter()
        try:
            result_box[0] = fn(*args)
        except MemoryError:
            error_box[0] = "OOM: out of memory"
        except Exception as e:
            error_box[0] = f"ERROR: {str(e)[:200]}"
        finally:
            elapsed_box[0] = time.perf_counter() - t0
            done_event.set()

    t = threading.Thread(target=worker, daemon=True)
    t0_wall = time.perf_counter()
    t.start()
    finished = done_event.wait(timeout=timeout_s)
    wall_elapsed = time.perf_counter() - t0_wall

    if not finished:
        # Kill the subprocess the thread is currently waiting on
        _kill_thread_proc(t.ident)
        return None, wall_elapsed, f"TIMEOUT (>{timeout_s // 60:.0f} min)"

    elapsed = elapsed_box[0] or wall_elapsed
    if error_box[0]:
        return None, elapsed, error_box[0]
    return result_box[0], elapsed, None


# ── sha256 of file (streamed, handles large files) ────────────────────────────

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── download helper ───────────────────────────────────────────────────────────

def download_jellyfish() -> dict:
    print("\nChecking / downloading Jellyfish test files...")
    paths = {}
    for label, fname in JELLYFISH:
        dest = DATA_DIR / fname
        key = (label, fname)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  OK  {fname}  ({dest.stat().st_size // 1_048_576} MB)")
            paths[key] = dest
            continue
        if fname in PREDOWNLOADED and PREDOWNLOADED[fname].exists():
            print(f"  Copying {fname} from Downloads...", end="", flush=True)
            shutil.copy2(PREDOWNLOADED[fname], dest)
            print(f" done ({dest.stat().st_size // 1_048_576} MB)")
            paths[key] = dest
            continue
        url = BASE_URL + fname
        print(f"  Downloading {fname}...", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f" done ({dest.stat().st_size // 1_048_576} MB)")
            paths[key] = dest
        except Exception as e:
            print(f" FAILED: {e}")
    return paths


# ── per-(tool, file, run) benchmark ──────────────────────────────────────────

def run_one(tool: ToolSpec, input_path: Path, file_label: str, run_idx: int) -> dict:
    slug = (tool.name.replace(" ", "_").replace("x", "x")
            .replace("=", "").replace("/", "_").replace(".", ""))
    fs = Path(input_path).stem[:18]
    output_mp4 = OUT_DIR / f"{slug}_{fs}_r{run_idx}.mp4"
    if output_mp4.exists():
        try:
            output_mp4.unlink()
        except PermissionError:
            pass

    original_size = input_path.stat().st_size
    orig_sha256   = _sha256_file(str(input_path))

    r = dict(tool=tool.name, file=input_path.name, label=file_label,
             size_bytes=original_size, run=run_idx,
             encode_s=None, decode_s=None, mp4_bytes=None,
             passed=None, status="pending", notes="")

    # Encode
    print(f"      run {run_idx} encode ...", end="", flush=True)
    _, enc_s, enc_err = _run_timed(
        tool.encode_fn, (str(input_path), str(output_mp4)), ENC_TIMEOUT)
    if enc_err:
        print(f" {enc_err}")
        r.update(encode_s=round(enc_s, 2), status=enc_err)
        return r

    mp4_sz = output_mp4.stat().st_size if output_mp4.exists() else 0
    print(f" {enc_s:.1f}s  ({mp4_sz // 1_048_576} MB mp4)", flush=True)
    r.update(encode_s=round(enc_s, 2), mp4_bytes=mp4_sz)

    if not output_mp4.exists() or mp4_sz == 0:
        r["status"] = "ERROR: no output file"
        return r

    # Decode — fn(video_mp4, original_size) -> bytes; verify in-thread
    def _decode_and_verify():
        data = tool.decode_fn(str(output_mp4), original_size)
        return hashlib.sha256(data[:original_size]).hexdigest() == orig_sha256

    print(f"      run {run_idx} decode ...", end="", flush=True)
    match, dec_s, dec_err = _run_timed(_decode_and_verify, (), DEC_TIMEOUT)
    if dec_err:
        print(f" {dec_err}")
        r.update(decode_s=round(dec_s, 2), status=dec_err)
        return r

    verdict = "PASS" if match else "MISMATCH"
    print(f" {dec_s:.1f}s  [{verdict}]", flush=True)
    r.update(decode_s=round(dec_s, 2), passed=match,
             status="PASS" if match else "MISMATCH")
    return r


def aggregate(runs: list) -> dict:
    ok = [r for r in runs if r["status"] == "PASS"]
    base = dict(tool=runs[0]["tool"], file=runs[0]["file"], label=runs[0]["label"],
                size_bytes=runs[0]["size_bytes"], n_runs=len(runs), runs=runs)
    if ok:
        enc = [r["encode_s"] for r in ok]
        dec = [r["decode_s"] for r in ok if r["decode_s"] is not None]
        mp4 = [r["mp4_bytes"] for r in ok if r["mp4_bytes"]]
        base.update(
            enc_avg=round(sum(enc)/len(enc), 2),
            enc_min=round(min(enc), 2),
            enc_max=round(max(enc), 2),
            dec_avg=round(sum(dec)/len(dec), 2) if dec else None,
            dec_min=round(min(dec), 2) if dec else None,
            dec_max=round(max(dec), 2) if dec else None,
            mp4_bytes_avg=int(sum(mp4)/len(mp4)) if mp4 else None,
            passed=True, status="PASS",
        )
    else:
        first_err = next((r["status"] for r in runs if r["status"] not in ("pending",)), "ERROR")
        base.update(enc_avg=None, enc_min=None, enc_max=None,
                    dec_avg=None, dec_min=None, dec_max=None,
                    mp4_bytes_avg=None, passed=False, status=first_err)
    return base


# ── README updater ────────────────────────────────────────────────────────────

_BENCH_RE = re.compile(
    r'(### Benchmark results\n)(.*?)(\n---\n\n### Detailed analysis)',
    re.DOTALL,
)


def _fmt_s(v, lo, hi):
    if v is None:
        return "—"
    if lo is not None and hi is not None and abs(hi - lo) > 0.5:
        return f"{v:.1f} ({lo:.1f}–{hi:.1f})"
    return f"{v:.1f}"


def _fmt_mb(b):
    if b is None:
        return "—"
    if b >= 1_073_741_824:
        return f"{b/1_073_741_824:.2f} GB"
    return f"{b/1_048_576:.0f} MB"


def _fmt_mbps(size_bytes, avg_s):
    if avg_s and avg_s > 0:
        return f"{size_bytes / 1e6 / avg_s:.2f}"
    return "—"


def generate_readme_section(all_results: list, completed_labels: list) -> str:
    """Build the full benchmark results section (between the h3 markers)."""
    lines = []

    lines.append("> **Machine:** Intel Core i7-1065G7 @ 1.30 GHz · Windows 11 · "
                 "Python 3.14 · ffmpeg 2026-05-06 · 4 cores  ")
    lines.append("> **Test data:** Jellyfish video files (larmoire.org) — real-world "
                 "compressed video used as high-entropy binary blobs.  ")
    lines.append("> **Methodology:** 3 runs per (tool, file). Enc/Dec columns show "
                 "avg (min–max) when spread > 0.5 s. 30-min timeout per run. "
                 "All tools use the same ffmpeg binary.  ")
    lines.append("> **Faithfulness:** ISG/bin2video have NO ECC (none in original). "
                 "YouBit uses gzip+RS-20 (same as original). "
                 "ByteVault uses RS-16 (its default). "
                 "Data2Video uses MP4 instead of GIF (original is GIF-only, not YouTube-compatible).  ")
    lines.append("")

    # group by file label (preserving JELLYFISH order)
    order = [fname for _, fname in JELLYFISH]
    label_map = {fname: lbl for lbl, fname in JELLYFISH}

    by_file: dict[str, list] = {}
    for r in all_results:
        by_file.setdefault(r["file"], []).append(r)

    for fname in order:
        if fname not in by_file:
            continue
        lbl = label_map[fname]
        file_results = by_file[fname]
        sz = file_results[0]["size_bytes"]
        sz_str = (f"{sz/1_073_741_824:.2f} GB" if sz >= 1_073_741_824
                  else f"{sz/1_048_576:.0f} MB")

        lines.append(f"#### {fname}  ({lbl} — {sz_str})")
        lines.append("")
        lines.append("| Tool | Resolution | Density | ECC | Codec | "
                     "Enc avg (s) | Dec avg (s) | MP4 size | MB/s enc | Result |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        for r in file_results:
            enc_str  = _fmt_s(r.get("enc_avg"), r.get("enc_min"), r.get("enc_max"))
            dec_str  = _fmt_s(r.get("dec_avg"), r.get("dec_min"), r.get("dec_max"))
            mp4_str  = _fmt_mb(r.get("mp4_bytes_avg"))
            mbps_str = _fmt_mbps(sz, r.get("enc_avg"))
            status   = r.get("status", "?")
            if status == "PASS":
                result_cell = "✅ PASS"
            elif "TIMEOUT" in status:
                result_cell = f"⏱ {status}"
            elif "OOM" in status:
                result_cell = "💾 OOM (out of memory)"
            elif "MISMATCH" in status:
                result_cell = "❌ MISMATCH"
            elif "SKIPPED" in status:
                result_cell = f"⏭ {status}"
            else:
                result_cell = f"❌ {status[:50]}"

            # find tool spec for density/ecc/codec
            spec = next((t for t in TOOLS if t.name == r["tool"]), None)
            res  = spec.resolution if spec else "—"
            dens = spec.density    if spec else "—"
            ecc  = spec.ecc        if spec else "—"
            cod  = spec.codec      if spec else "—"

            lines.append(f"| {r['tool']} | {res} | {dens} | {ecc} | {cod} | "
                         f"{enc_str} | {dec_str} | {mp4_str} | {mbps_str} | {result_cell} |")

        lines.append("")

    # Per-run detail tables (collapsed)
    lines.append("<details>")
    lines.append("<summary>Per-run timing detail</summary>")
    lines.append("")
    for fname in order:
        if fname not in by_file:
            continue
        lbl = label_map[fname]
        lines.append(f"**{fname}** ({lbl})")
        lines.append("")
        lines.append("| Tool | Run | Enc (s) | Dec (s) | MP4 size | Result |")
        lines.append("|---|---|---|---|---|---|")
        for agg in by_file[fname]:
            for run in agg.get("runs", []):
                enc = f"{run['encode_s']:.1f}" if run.get("encode_s") else "—"
                dec = f"{run['decode_s']:.1f}" if run.get("decode_s") else "—"
                mp4 = _fmt_mb(run.get("mp4_bytes"))
                st  = run.get("status", "?")
                if st == "PASS":
                    res = "✅"
                elif "TIMEOUT" in st:
                    res = f"⏱ {st}"
                elif "OOM" in st:
                    res = "💾 OOM"
                elif "MISMATCH" in st:
                    res = "❌ mismatch"
                elif "SKIPPED" in st:
                    res = f"⏭ skipped"
                else:
                    res = f"❌ {st[:35]}"
                lines.append(f"| {run['tool']} | {run['run']} | {enc} | {dec} | {mp4} | {res} |")
        lines.append("")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


def update_readme(all_results: list, completed_labels: list):
    text = README.read_text(encoding="utf-8")
    new_body = generate_readme_section(all_results, completed_labels)
    replaced, n = _BENCH_RE.subn(
        lambda m: m.group(1) + "\n" + new_body + m.group(3),
        text,
    )
    if n == 0:
        print("  WARNING: benchmark section markers not found in README — skipping update")
        return
    README.write_text(replaced, encoding="utf-8")
    last = completed_labels[-1] if completed_labels else "?"
    print(f"  README updated through {last}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 output on Windows (avoids cp1252 encode errors for box chars)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 72)
    print("  ByteVault Jellyfish Benchmark")
    print("=" * 72)

    # Statuses that are definitive — never re-run these.
    # OOM/ERROR/MISMATCH/SKIPPED will be retried (threading may succeed where spawn failed).
    _SKIP_STATUSES = {"PASS", "TIMEOUT"}

    def _is_done(status: str) -> bool:
        if status in _SKIP_STATUSES:
            return True
        if status.startswith("TIMEOUT"):
            return True
        return False

    # Load existing results (resume support)
    all_results: list = []
    done_keys: set = set()   # (tool_name, fname, run_idx) that are definitively done
    if RESULTS_JSON.exists():
        try:
            saved = json.loads(RESULTS_JSON.read_text())
            all_results = saved.get("aggregated", [])
            skipped = retrying = 0
            for agg in all_results:
                for run in agg.get("runs", []):
                    key = (run["tool"], run["file"], run["run"])
                    if _is_done(run.get("status", "")):
                        done_keys.add(key)
                        skipped += 1
                    else:
                        retrying += 1
            print(f"  Resuming — {skipped} done, {retrying} will be retried.")
        except Exception:
            pass

    # Write existing results to README immediately before starting new runs
    if all_results:
        existing_labels = list(dict.fromkeys(r["label"] for r in all_results))
        print(f"  Writing existing results to README ({len(existing_labels)} file size(s))...")
        update_readme(all_results, existing_labels)

    file_paths = download_jellyfish()

    # Track which file-size labels have had all tools run (for README updates)
    completed_labels = list(dict.fromkeys(r["label"] for r in all_results))
    for (label, fname), input_path in file_paths.items():
        print(f"\n{'-'*72}")
        print(f"  FILE: {fname}  ({label}  /  {input_path.stat().st_size // 1_048_576} MB)")
        print(f"{'-'*72}")

        file_agg_results = []  # aggregated per tool for this file
        for tool in TOOLS:
            print(f"\n  [{tool.name}]")
            runs = []
            first_timed_out = False
            for run_idx in range(1, RUNS + 1):
                key = (tool.name, fname, run_idx)
                if key in done_keys:
                    # Definitively done (PASS or TIMEOUT) — load and skip
                    existing = next(
                        (r for agg in all_results
                         for r in agg.get("runs", [])
                         if (r["tool"], r["file"], r["run"]) == key),
                        None,
                    )
                    if existing:
                        print(f"      run {run_idx} — done ({existing['status']}), skipping")
                        runs.append(existing)
                        continue

                if first_timed_out:
                    # Skip remaining runs after first timeout
                    runs.append(dict(
                        tool=tool.name, file=fname, label=label,
                        size_bytes=input_path.stat().st_size,
                        run=run_idx, encode_s=None, decode_s=None,
                        mp4_bytes=None, passed=None,
                        status="SKIPPED (prior run timed out)", notes=""))
                    print(f"      run {run_idx} — skipped (prior timeout)")
                    continue

                r = run_one(tool, input_path, label, run_idx)
                runs.append(r)
                if "TIMEOUT" in r["status"]:
                    first_timed_out = True

                # Checkpoint after every run
                agg = aggregate(runs)
                # Update or add this tool's entry for this file
                existing_idx = next(
                    (i for i, a in enumerate(all_results)
                     if a["tool"] == tool.name and a["file"] == fname),
                    None)
                if existing_idx is not None:
                    all_results[existing_idx] = agg
                else:
                    all_results.append(agg)
                RESULTS_JSON.write_text(json.dumps(
                    {"aggregated": all_results}, indent=2))

            file_agg_results.append(aggregate(runs))

        completed_labels.append(label)
        print(f"\n  Updating README after {label}...")
        update_readme(all_results, completed_labels)

    print("\n" + "=" * 72)
    print("  BENCHMARK COMPLETE")
    print("=" * 72)
    print(f"  Results: {RESULTS_JSON}")
    print(f"  README : {README}")

    # Print summary
    print(f"\n{'Tool':<38} {'File':<14} {'Enc avg':>9} {'Dec avg':>9} {'Status':>14}")
    print("-" * 87)
    for r in all_results:
        enc = f"{r['enc_avg']:.1f}s" if r.get("enc_avg") else "—"
        dec = f"{r['dec_avg']:.1f}s" if r.get("dec_avg") else "—"
        print(f"  {r['tool'][:36]:<36} {r['label']:<14} {enc:>9} {dec:>9} {r['status']:>12}")


if __name__ == "__main__":
    main()
