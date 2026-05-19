#!/usr/bin/env python3
"""
PixelVault Comprehensive Test Suite — 23 file types × multiple size tiers.

Usage:
  python test_comprehensive.py                     # 1 KB + 100 KB, all types, binary+palette, +/-audio
  python test_comprehensive.py --medium            # add 1 MB tier
  python test_comprehensive.py --large             # add 10 MB tier
  python test_comprehensive.py --xlarge            # add 100 MB tier  (slow)
  python test_comprehensive.py --huge              # add 512 MB tier  (very slow)
  python test_comprehensive.py --max               # add 1 GB + 2 GB  (benchmark only)
  python test_comprehensive.py --types txt,pdf,jpg # filter file types
  python test_comprehensive.py --no-audio          # skip audio variants
  python test_comprehensive.py --no-palette        # binary mode only
  python test_comprehensive.py --compress-variants # also test --compress flag
  python test_comprehensive.py --ecc-variants      # also test --ecc 16 flag
  python test_comprehensive.py --csv results.csv   # export metrics to CSV
  python test_comprehensive.py --width 1920 --height 1080 --fps 24
"""

import argparse
import csv as csv_module
import io
import json as _json
import os
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic file generators
# Each returns bytes of approximately the requested size.
# The exact size may differ slightly (especially ZIP-based formats).
# ─────────────────────────────────────────────────────────────────────────────

_rng = np.random.default_rng(0xBEEF_CAFE)


def _rand(n: int) -> bytes:
    if n <= 0:
        return b""
    return bytes(_rng.integers(0, 256, n, dtype=np.uint8))


def _rep(line: bytes, n: int) -> bytes:
    return (line * (n // len(line) + 1))[:n]


# ── Text / XML formats (highly compressible) ─────────────────────────────────

def gen_txt(n):
    return _rep(b"The quick brown fox jumps over the lazy dog. PixelVault test.\n", n)

def gen_csv(n):
    h = b"id,name,value,flag,description\n"
    r = b'42,"Widget A",3.14,true,"PixelVault synthetic CSV test row"\n'
    return (h + _rep(r, n - len(h)))[:n]

def gen_json(n):
    chunk = b'{"id":1,"key":"value","num":42,"arr":[1,2,3],"ok":true}\n'
    return (b"[\n" + _rep(chunk, n - 3) + b"\n]")[:n]

def gen_yaml(n):
    return _rep(b"- key: value\n  num: 42\n  active: true\n  tags: [a, b, c]\n", n)

def gen_html(n):
    h = b"<!DOCTYPE html><html><body>\n"
    f = b"</body></html>\n"
    p = b"<p>The quick brown fox jumps over the lazy dog.</p>\n"
    return (h + _rep(p, n - len(h) - len(f)) + f)[:n]

def gen_svg(n):
    h = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n'
    e = b'<circle cx="50" cy="50" r="40" fill="red" stroke="black"/>\n'
    f = b"</svg>\n"
    return (h + _rep(e, n - len(h) - len(f)) + f)[:n]

def gen_obj(n):
    h = b"# PixelVault OBJ test\n"
    v = b"v 0.123 4.567 8.901\n"
    fa = b"f 1 2 3\n"
    return (h + _rep(v + fa, n - len(h)))[:n]

def gen_urdf(n):
    h = b'<?xml version="1.0"?><robot name="bot">\n'
    lk = b'  <link name="l"><visual><geometry><box size="1 1 1"/></geometry></visual></link>\n'
    f = b"</robot>\n"
    return (h + _rep(lk, n - len(h) - len(f)) + f)[:n]


# ── Binary formats (magic bytes + random body — incompressible) ───────────────

def gen_pdf(n):
    h = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
    f = b"\n%%EOF\n"
    return (h + _rand(max(0, n - len(h) - len(f))) + f)[:n]

def gen_jpg(n):
    h = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    return (h + _rand(max(0, n - len(h) - 2)) + b"\xff\xd9")[:n]

def gen_png(n):
    magic = b"\x89PNG\r\n\x1a\n"
    return (magic + _rand(max(0, n - len(magic))))[:n]

def gen_gif(n):
    h = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x00\x00\x00\x00\x00\x2c"
    return (h + _rand(max(0, n - len(h) - 1)) + b"\x3b")[:n]

def gen_webp(n):
    body_n = max(4, n - 12)
    h = b"RIFF" + struct.pack("<I", body_n) + b"WEBP"
    return (h + _rand(max(0, n - len(h))))[:n]

def gen_mp3(n):
    id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00"
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
    return (id3 + _rep(frame, max(0, n - len(id3))))[:n]

def gen_wav(n):
    ds = max(0, n - 44)
    h = (b"RIFF" + struct.pack("<I", ds + 36) + b"WAVE" +
         b"fmt " + struct.pack("<IHHIIHH", 16, 1, 2, 44100, 176400, 4, 16) +
         b"data" + struct.pack("<I", ds))
    return (h + _rand(ds))[:n]

def gen_mp4(n):
    ftyp = struct.pack(">I", 20) + b"ftypisom\x00\x00\x00\x00isomiso2"
    return (ftyp + _rand(max(0, n - len(ftyp))))[:n]

def gen_mov(n):
    ftyp = struct.pack(">I", 20) + b"ftypqt  \x00\x00\x00\x00qt      "
    return (ftyp + _rand(max(0, n - len(ftyp))))[:n]

def gen_exe(n):
    dos = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 56 + struct.pack("<I", 64)
    pe = b"PE\x00\x00"
    return (dos + pe + _rand(max(0, n - len(dos) - len(pe))))[:n]

def gen_parquet(n):
    return (b"PAR1" + _rand(max(0, n - 8)) + b"PAR1")[:n]

def gen_stl(n):
    n_tri = max(1, (n - 84) // 50)
    hdr = b"PixelVault STL synthetic test file" + b" " * 47
    tri = struct.pack("<fff", 0.0, 0.0, 1.0) + struct.pack("<" + "fff" * 3, 0, 0, 0, 1, 0, 0, 0, 1, 0) + b"\x00\x00"
    return (hdr + struct.pack("<I", n_tri) + tri * n_tri)[:n]


def _zip_of_size(n: int) -> bytes:
    """Generate a ZIP file as close to n bytes as possible."""
    buf = io.BytesIO()
    content_size = max(0, n - 200)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("data.bin", _rand(content_size))
    return buf.getvalue()

def gen_zip(n):   return _zip_of_size(n)
def gen_docx(n):  return _zip_of_size(n)   # simplified — ZIP with random body
def gen_xlsx(n):  return _zip_of_size(n)   # simplified — ZIP with random body


# ── Registry ──────────────────────────────────────────────────────────────────

FILE_GENERATORS: Dict[str, Callable[[int], bytes]] = {
    "txt":     gen_txt,
    "csv":     gen_csv,
    "json":    gen_json,
    "yaml":    gen_yaml,
    "html":    gen_html,
    "svg":     gen_svg,
    "obj":     gen_obj,
    "urdf":    gen_urdf,
    "pdf":     gen_pdf,
    "jpg":     gen_jpg,
    "png":     gen_png,
    "gif":     gen_gif,
    "webp":    gen_webp,
    "mp3":     gen_mp3,
    "wav":     gen_wav,
    "mp4":     gen_mp4,
    "mov":     gen_mov,
    "exe":     gen_exe,
    "parquet": gen_parquet,
    "stl":     gen_stl,
    "zip":     gen_zip,
    "docx":    gen_docx,
    "xlsx":    gen_xlsx,
}

TEXT_TYPES = {"txt", "csv", "json", "yaml", "html", "svg", "obj", "urdf"}

# ─────────────────────────────────────────────────────────────────────────────
# Size tiers
# ─────────────────────────────────────────────────────────────────────────────

SIZE_TIERS = {
    "1KB":   1_024,
    "100KB": 100 * 1_024,
    "1MB":   1_024 * 1_024,
    "10MB":  10 * 1_024 * 1_024,
    "100MB": 100 * 1_024 * 1_024,
    "512MB": 512 * 1_024 * 1_024,
    "1GB":   1_024 * 1_024 * 1_024,
    "2GB":   2 * 1_024 * 1_024 * 1_024,
}

TIER_ORDER = ["1KB", "100KB", "1MB", "10MB", "100MB", "512MB", "1GB", "2GB"]

# ─────────────────────────────────────────────────────────────────────────────
# Large-file writer (chunked to avoid holding GBs in RAM)
# ─────────────────────────────────────────────────────────────────────────────

_PAD_TEXT   = b"PixelVault large-file padding content line. " * 24 + b"\n"  # ~1 KB
_CHUNK_SIZE = 4 * 1_024 * 1_024  # 4 MB


def write_synthetic(path: Path, file_type: str, target: int) -> int:
    """Write a synthetic file and return actual byte count written."""
    if target <= 32 * 1_024 * 1_024:
        data = FILE_GENERATORS[file_type](target)
        path.write_bytes(data)
        return len(data)

    # Large file: write magic header then stream padding
    header = FILE_GENERATORS[file_type](min(4096, target))
    is_text = file_type in TEXT_TYPES
    written = 0
    with open(path, "wb") as fh:
        fh.write(header)
        written = len(header)
        while written < target:
            n = min(_CHUNK_SIZE, target - written)
            if is_text:
                fh.write((_PAD_TEXT * (n // len(_PAD_TEXT) + 1))[:n])
            else:
                fh.write(bytes(_rng.integers(0, 256, n, dtype=np.uint8)))
            written += n
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Test result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    file_type:   str
    size_label:  str
    file_size:   int
    mode:        str
    use_audio:   bool
    compress:    bool
    ecc_nsym:    int
    status:      str        # "pass" | "fail" | "skip"
    error:       str = ""
    enc_s:       float = 0.0
    dec_s:       float = 0.0
    video_bytes: int   = 0
    n_frames:    int   = 0
    fps:         int   = 24

    @property
    def total_s(self)  -> float: return self.enc_s + self.dec_s

    @property
    def enc_MBs(self)  -> float:
        mb = self.file_size / 1_048_576
        return mb / self.enc_s if self.enc_s > 0 and mb > 0 else 0.0

    @property
    def dec_MBs(self)  -> float:
        mb = self.file_size / 1_048_576
        return mb / self.dec_s if self.dec_s > 0 and mb > 0 else 0.0

    @property
    def video_MB(self) -> float: return self.video_bytes / 1_048_576

    @property
    def overhead(self) -> float:
        return self.video_bytes / self.file_size if self.file_size > 0 else 0.0

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    def as_csv_row(self) -> dict:
        return {
            "file_type":   self.file_type,
            "size_label":  self.size_label,
            "file_size_B": self.file_size,
            "mode":        self.mode,
            "audio":       self.use_audio,
            "compress":    self.compress,
            "ecc_nsym":    self.ecc_nsym,
            "status":      self.status,
            "enc_s":       round(self.enc_s, 3),
            "dec_s":       round(self.dec_s, 3),
            "total_s":     round(self.total_s, 3),
            "enc_MB_s":    round(self.enc_MBs, 3),
            "dec_MB_s":    round(self.dec_MBs, 3),
            "video_MB":    round(self.video_MB, 3),
            "overhead_x":  round(self.overhead, 2),
            "n_frames":    self.n_frames,
            "duration_s":  round(self.duration_s, 1),
            "error":       self.error[:200],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Single test runner
# ─────────────────────────────────────────────────────────────────────────────

_MODE_INT = {"binary": 0, "palette": 2}


def _frame_count(vid_path: Path) -> int:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", str(vid_path)],
            capture_output=True, text=True, check=True,
        )
        s = _json.loads(r.stdout)["streams"][0]
        nb = s.get("nb_frames")
        if nb and nb != "N/A":
            return int(nb)
        dur = float(s.get("duration", 0))
        fr  = s.get("r_frame_rate", "24/1").split("/")
        fps = int(fr[0]) / max(1, int(fr[1]))
        return int(dur * fps)
    except Exception:
        return 0


def run_test(
    file_type: str,
    size_label: str,
    target: int,
    mode: str,
    use_audio: bool,
    compress: bool,
    ecc_nsym: int,
    tmpdir: str,
    width: int,
    height: int,
    fps: int,
) -> TestResult:
    from pixelvault.encoder import encode_file, DEFAULT_BLOCK
    from pixelvault.decoder import decode_file

    r = TestResult(
        file_type=file_type, size_label=size_label, file_size=0,
        mode=mode, use_audio=use_audio, compress=compress,
        ecc_nsym=ecc_nsym, status="fail", fps=fps,
    )

    mode_int = _MODE_INT[mode]
    block = DEFAULT_BLOCK[mode_int]
    if width % block != 0 or height % block != 0:
        r.status = "skip"
        r.error = f"{width}x{height} not divisible by block {block}"
        return r

    slug = (f"{file_type}_{size_label}_{mode}"
            f"_{'a' if use_audio else 'v'}"
            f"{'z' if compress else ''}"
            f"{'e' if ecc_nsym else ''}")
    # Prefix input with "src_" so .mp4-type files don't collide with the output video
    in_p   = Path(tmpdir) / f"src_{slug}.{file_type}"
    vid_p  = Path(tmpdir) / f"{slug}.mp4"
    out_d  = Path(tmpdir) / f"dec_{slug}"
    out_d.mkdir(exist_ok=True)

    # 1. Generate file
    try:
        r.file_size = write_synthetic(in_p, file_type, target)
    except Exception as e:
        r.error = f"gen: {e}"
        return r

    # 2. Encode
    try:
        t0 = time.perf_counter()
        encode_file(
            input_path=str(in_p),
            output_path=str(vid_p),
            mode=mode_int, fps=fps,
            width=width, height=height,
            quiet=True,
            use_audio=use_audio,
            compress=compress,
            ecc_nsym=ecc_nsym,
        )
        r.enc_s      = time.perf_counter() - t0
        r.video_bytes = vid_p.stat().st_size
        r.n_frames   = _frame_count(vid_p)
    except Exception as e:
        r.error = f"encode: {e}"
        return r

    # 3. Decode
    try:
        t0 = time.perf_counter()
        rec_path = decode_file(str(vid_p), output_dir=str(out_d), quiet=True)
        r.dec_s = time.perf_counter() - t0
    except Exception as e:
        r.error = f"decode: {e}"
        return r

    # 4. Verify
    try:
        orig = in_p.read_bytes()
        rec  = Path(rec_path).read_bytes()
        if orig == rec:
            r.status = "pass"
        else:
            diff = next((i for i, (a, b) in enumerate(zip(orig, rec)) if a != b), len(min(orig, rec, key=len)))
            r.status = "fail"
            r.error  = f"mismatch orig={len(orig)}B rec={len(rec)}B first_diff@{diff}"
    except Exception as e:
        r.error = f"verify: {e}"

    # 5. Clean up large files to free disk space
    try:
        if r.file_size > 50 * 1_024 * 1_024:
            in_p.unlink(missing_ok=True)
            vid_p.unlink(missing_ok=True)
    except Exception:
        pass

    return r


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_SKIP = "\033[33mSKIP\033[0m"
_ST   = {"pass": _PASS, "fail": _FAIL, "skip": _SKIP}

_HDR = (
    f"{'Type':<9} {'Size':>6} {'Mode':<8} {'Aud':>3} {'Cmp':>3} {'ECC':>3}  "
    f"  Status  {'Enc(s)':>7} {'Dec(s)':>7} "
    f"{'EncMB/s':>8} {'DecMB/s':>8} {'VidMB':>7} {'Ovrhd':>6} {'Frames':>7}"
)
_SEP = "-" * len(_HDR)


def _fmt(r: TestResult) -> str:
    st = _ST[r.status]
    return (
        f"{r.file_type:<9} {r.size_label:>6} {r.mode:<8} "
        f"{'Y' if r.use_audio else 'N':>3} "
        f"{'Y' if r.compress else 'N':>3} "
        f"{str(r.ecc_nsym) if r.ecc_nsym else '-':>3}  "
        f"  {st}  "
        f"{r.enc_s:>7.2f} {r.dec_s:>7.2f} "
        f"{r.enc_MBs:>8.2f} {r.dec_MBs:>8.2f} "
        f"{r.video_MB:>7.2f} {r.overhead:>5.1f}x {r.n_frames:>7}"
    )


def print_header():
    print(_HDR)
    print(_SEP)


def print_row(r: TestResult):
    print(_fmt(r))
    if r.status == "fail" and r.error:
        print(f"  {'':9} {'':6} {'':8}     +-- {r.error[:110]}")
    sys.stdout.flush()


def print_summary(results: List[TestResult]):
    passed  = [r for r in results if r.status == "pass"]
    failed  = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]

    print()
    print("=" * 80)
    print(f"  RESULTS: {len(passed)} passed  {len(failed)} failed  {len(skipped)} skipped  "
          f"(total {len(results)})")
    print("=" * 80)

    if not passed:
        if failed:
            for r in failed:
                print(f"  FAIL {r.file_type} {r.size_label} {r.mode}: {r.error[:80]}")
        return

    # Per mode+audio summary
    print(f"\n  {'Mode':<8} {'Audio':<6} {'Tests':>5}  "
          f"{'Avg Enc MB/s':>13} {'Avg Dec MB/s':>13}  "
          f"{'Avg Overhead':>13}  {'Avg Frames':>11}")
    print("  " + "-" * 70)
    for mode in ("binary", "palette"):
        for audio in (False, True):
            sub = [r for r in passed if r.mode == mode and r.use_audio == audio]
            if not sub:
                continue
            avg_enc = sum(r.enc_MBs   for r in sub) / len(sub)
            avg_dec = sum(r.dec_MBs   for r in sub) / len(sub)
            avg_ovh = sum(r.overhead  for r in sub) / len(sub)
            avg_frm = sum(r.n_frames  for r in sub) / len(sub)
            print(f"  {mode:<8} {'Y' if audio else 'N':<6} {len(sub):>5}  "
                  f"{avg_enc:>13.2f} {avg_dec:>13.2f}  "
                  f"{avg_ovh:>12.1f}x  {avg_frm:>11.0f}")

    # Per size tier
    print(f"\n  {'Tier':>6}  {'Tests':>5}  {'Avg Enc MB/s':>13} {'Avg Dec MB/s':>13}  "
          f"{'Avg Total(s)':>13}")
    print("  " + "-" * 60)
    for tier in TIER_ORDER:
        sub = [r for r in passed if r.size_label == tier]
        if not sub:
            continue
        print(f"  {tier:>6}  {len(sub):>5}  "
              f"{sum(r.enc_MBs for r in sub)/len(sub):>13.2f} "
              f"{sum(r.dec_MBs for r in sub)/len(sub):>13.2f}  "
              f"{sum(r.total_s for r in sub)/len(sub):>13.2f}")

    # Top-3 throughput
    top_enc = sorted(passed, key=lambda r: r.enc_MBs, reverse=True)[:3]
    top_dec = sorted(passed, key=lambda r: r.dec_MBs, reverse=True)[:3]
    print(f"\n  Top encode throughput:")
    for r in top_enc:
        print(f"    {r.file_type:<9} {r.size_label:>6} {r.mode:<8}: {r.enc_MBs:.2f} MB/s")
    print(f"  Top decode throughput:")
    for r in top_dec:
        print(f"    {r.file_type:<9} {r.size_label:>6} {r.mode:<8}: {r.dec_MBs:.2f} MB/s")

    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for r in failed:
            print(f"    {r.file_type} {r.size_label} {r.mode} "
                  f"audio={r.use_audio}: {r.error[:80]}")
    print()


def export_csv(results: List[TestResult], path: str):
    rows = [r.as_csv_row() for r in results]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv_module.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] Results exported -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Test plan builder
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TIERS  = ["1KB", "100KB"]
OPTIONAL_TIERS = {
    "medium": ["1MB"],
    "large":  ["10MB"],
    "xlarge": ["100MB"],
    "huge":   ["512MB"],
    "max":    ["1GB", "2GB"],
}


def build_plan(args) -> List[Tuple]:
    tiers = list(DEFAULT_TIERS)
    for flag, extra in OPTIONAL_TIERS.items():
        if getattr(args, flag, False):
            tiers.extend(extra)

    all_types = list(FILE_GENERATORS)
    if args.types:
        requested = [t.strip() for t in args.types.split(",")]
        all_types = [t for t in requested if t in FILE_GENERATORS]
        unknown   = [t for t in requested if t not in FILE_GENERATORS]
        if unknown:
            print(f"Warning: unknown types ignored: {unknown}", file=sys.stderr)

    all_modes = ["binary"] if args.no_palette else ["binary", "palette"]
    if args.modes:
        all_modes = [m for m in args.modes.split(",") if m in _MODE_INT]

    audio_opts   = [False] if args.no_audio else [False, True]
    compress_opts = [False, True] if args.compress_variants else [False]
    ecc_opts      = [0, 16]      if args.ecc_variants      else [0]

    plan = []
    for tier in tiers:
        target = SIZE_TIERS[tier]
        # For very large tiers, limit to binary only to keep time manageable
        modes = ["binary"] if target >= 100 * 1_024 * 1_024 else all_modes
        modes = [m for m in modes if m in all_modes]
        for ft in all_types:
            for mode in modes:
                for audio in audio_opts:
                    for compress in compress_opts:
                        for ecc in ecc_opts:
                            plan.append((ft, tier, target, mode, audio, compress, ecc))
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Estimated time warning
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_minutes(plan, width, height, fps) -> float:
    from pixelvault.encoder import DEFAULT_BLOCK
    total = 0.0
    for ft, tier, target, mode, audio, compress, ecc in plan:
        mode_int = _MODE_INT[mode]
        block = DEFAULT_BLOCK[mode_int]
        lw, lh = width // block, height // block
        if mode == "binary":
            bpf = lw * lh // 8
        else:
            bpf = lw * lh
        frames = max(1, (target + 128) // bpf)
        # ~60 fps encode, ~60 fps decode (rough heuristic)
        total += frames / 60 * 2
    return total / 60


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="PixelVault comprehensive benchmark — 23 file types × multiple size tiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Size tiers
    p.add_argument("--medium",  action="store_true", help="Add 1 MB tier")
    p.add_argument("--large",   action="store_true", help="Add 10 MB tier")
    p.add_argument("--xlarge",  action="store_true", help="Add 100 MB tier (slow)")
    p.add_argument("--huge",    action="store_true", help="Add 512 MB tier (very slow)")
    p.add_argument("--max",     action="store_true", help="Add 1 GB + 2 GB tiers (benchmark only)")
    # Filters
    p.add_argument("--types",      default="", metavar="T1,T2",
                   help=f"File types to test. Available: {','.join(FILE_GENERATORS)}")
    p.add_argument("--modes",      default="", metavar="M1,M2",
                   help="Modes: binary,palette (default: both)")
    p.add_argument("--no-audio",   dest="no_audio",   action="store_true",
                   help="Skip audio variants")
    p.add_argument("--no-palette", dest="no_palette", action="store_true",
                   help="Binary mode only")
    # Feature variants
    p.add_argument("--compress-variants", dest="compress_variants", action="store_true",
                   help="Also test with --compress flag")
    p.add_argument("--ecc-variants", dest="ecc_variants", action="store_true",
                   help="Also test with --ecc 16 flag")
    # Encode settings
    p.add_argument("--width",  type=int, default=1920, help="Encode width  (default 1920)")
    p.add_argument("--height", type=int, default=1080, help="Encode height (default 1080)")
    p.add_argument("--fps",    type=int, default=24,   help="Frames/second (default 24)")
    # Output
    p.add_argument("--csv",    default="", metavar="PATH",
                   help="Export full results to CSV")
    p.add_argument("--tmpdir", default="", metavar="PATH",
                   help="Custom temp directory (use a path with enough disk space for large tests)")
    p.add_argument("--quiet",  action="store_true",
                   help="Suppress per-test rows; show summary only")
    args = p.parse_args()

    plan = build_plan(args)
    est  = _estimate_minutes(plan, args.width, args.height, args.fps)

    print(f"\nPixelVault Comprehensive Test Suite")
    print(f"  Tests:      {len(plan)}")
    print(f"  Resolution: {args.width}x{args.height} @ {args.fps} fps")
    print(f"  Types:      {sorted(set(t[0] for t in plan))}")
    print(f"  Tiers:      {[t for t in TIER_ORDER if any(x[1] == t for x in plan)]}")
    print(f"  Est. time:  ~{est:.0f} min  (wall clock, varies with hardware)")
    print()

    if not args.quiet:
        print_header()

    results: List[TestResult] = []
    tmpdir_ctx = (
        tempfile.TemporaryDirectory(prefix="bv_bench_", dir=args.tmpdir or None)
        if not args.tmpdir else None
    )
    tmpdir = args.tmpdir if args.tmpdir else None

    try:
        with (tmpdir_ctx if tmpdir_ctx else _null_ctx()) as td:
            use_dir = tmpdir or td
            for i, (ft, tier, target, mode, audio, compress, ecc) in enumerate(plan, 1):
                r = run_test(
                    ft, tier, target, mode, audio, compress, ecc,
                    use_dir, args.width, args.height, args.fps,
                )
                results.append(r)
                if not args.quiet:
                    print_row(r)
    finally:
        print_summary(results)
        if args.csv:
            export_csv(results, args.csv)

    return 0 if all(r.status != "fail" for r in results) else 1


class _null_ctx:
    """Context manager that does nothing (used when tmpdir is user-supplied)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    sys.exit(main())
