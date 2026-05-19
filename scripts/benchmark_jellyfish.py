#!/usr/bin/env python3
"""
PixelVault Jellyfish Benchmark — Python 3 programs only, real GitHub code

Tests only Python 3 file-to-video tools using their actual code unmodified
(or with minimal non-functional changes: CLI wiring, import compatibility):

  PixelVault (this project)
      python main.py encode / decode

  file2video  (github.com/karaketir16/file2video)
      python file2video.py --encode / --decode
      Cloned automatically on first run into third_party/file2video/

  YouBit  (github.com/MeViMo/youbit)
      youbit.encode.Encoder / youbit.decode.decode_local
      Non-functional changes: replaced Cython ECC with reedsolo, numba with numpy,
      relaxed version constraints. Local decode path added for non-YouTube videos.
      Located at third_party/youbit/

Excluded Python tools and reasons:
  DataToVideoEncoderDecoder — pixel_size mismatch (enc=5, dec=10) means roundtrip
                              always fails. OOM on large files (all frames in RAM).
  qStore             — requires ffmpeg installed system-wide; QR encode/decode is
                       extremely slow and file must be a .zip/.tar.gz.

Each (tool, file-size) pair runs RUNS times.  Results are checkpointed to
bench/jellyfish_results.json after every individual run so the script can
resume from a crash.  README is updated after every file-size group.
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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ── paths & config ────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
BENCH_DIR    = ROOT / "bench"
DATA_DIR     = BENCH_DIR / "jellyfish"
OUT_DIR      = BENCH_DIR / "jf_results"
RESULTS_JSON = BENCH_DIR / "jellyfish_results.json"
README       = ROOT / "README.md"
PV_ROOT      = ROOT
THIRD_PARTY  = ROOT / "third_party"
F2V_DIR      = THIRD_PARTY / "file2video"
YB_DIR       = THIRD_PARTY / "youbit"

RUNS        = 3
ENC_TIMEOUT = 1800   # 30 min per encode
DEC_TIMEOUT = 1800   # 30 min per decode

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)
THIRD_PARTY.mkdir(parents=True, exist_ok=True)

# ── Jellyfish test files ──────────────────────────────────────────────────────
BASE_URL = "http://larmoire.org/jellyfish/media/"
JELLYFISH = [
    ("~15 MB HD",   "jellyfish-3-mbps-hd-h264.mkv"),
    ("~52 MB HD",   "jellyfish-10-mbps-hd-h264.mkv"),
    ("~150 MB HD",  "jellyfish-40-mbps-hd-h264.mkv"),
    ("~375 MB HD",  "jellyfish-100-mbps-hd-h264.mkv"),
    ("~450 MB 4K",  "jellyfish-120-mbps-4k-uhd-h264.mkv"),
    ("~1.4 GB 4K",  "jellyfish-400-mbps-4k-uhd-hevc-10bit.mkv"),
]

# If the large file already exists locally, copy instead of download
PREDOWNLOADED = {
    "jellyfish-400-mbps-4k-uhd-hevc-10bit.mkv":
        Path(r"C:\Users\sahil\Downloads\jellyfish-400-mbps-4k-uhd-hevc-10bit.mkv"),
}

# ── per-thread subprocess tracking (for timeout kill) ────────────────────────
_active_procs: dict = {}
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
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False, timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

# ── timed runner (threading — shares already-loaded numpy/cv2, zero spawn cost) ──
def _run_timed(fn, args, timeout_s):
    result_box  = [None]
    error_box   = [None]
    elapsed_box = [0.0]
    done_event  = threading.Event()

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
        _kill_thread_proc(t.ident)
        return None, wall_elapsed, f"TIMEOUT (>{timeout_s // 60:.0f} min)"

    elapsed = elapsed_box[0] or wall_elapsed
    if error_box[0]:
        return None, elapsed, error_box[0]
    return result_box[0], elapsed, None

# ── download helpers ──────────────────────────────────────────────────────────
def _download(url: str, dest: Path):
    print(f"  Downloading {dest.name}...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"done ({dest.stat().st_size // 1_000_000} MB)")
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        raise

def ensure_jellyfish():
    print("\nChecking / downloading Jellyfish test files...")
    for label, fname in JELLYFISH:
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"  OK  {fname}  ({dest.stat().st_size // 1_000_000} MB)")
            continue
        if fname in PREDOWNLOADED and PREDOWNLOADED[fname].exists():
            src = PREDOWNLOADED[fname]
            print(f"  Copying {fname} from {src.parent.name}...", end=" ", flush=True)
            shutil.copy2(src, dest)
            print(f"done ({dest.stat().st_size // 1_000_000} MB)")
        else:
            _download(BASE_URL + fname, dest)

# ── third-party tool setup ────────────────────────────────────────────────────
def setup_file2video():
    """Clone file2video and install its dependencies (once)."""
    if not F2V_DIR.exists():
        print("  Cloning file2video from GitHub...")
        r = subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/karaketir16/file2video.git", str(F2V_DIR)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{r.stderr}")
        print(f"  Cloned -> {F2V_DIR}")
    else:
        print(f"  file2video already present at {F2V_DIR.relative_to(ROOT)}")

    req = F2V_DIR / "requirements.txt"
    if req.exists():
        print("  Installing file2video dependencies (pip)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req),
             "--quiet", "--disable-pip-version-check"],
            check=False,   # non-fatal; some extras (customtkinter) may be optional
        )

def setup_youbit():
    """Ensure YouBit is importable (installed editable in third_party/youbit)."""
    try:
        import youbit  # noqa: F401
        print(f"  YouBit already importable.")
    except ImportError:
        print("  Installing YouBit...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(YB_DIR),
             "--ignore-requires-python", "--no-build-isolation", "--quiet"],
            check=False,
        )

def setup_tools():
    print("\nSetting up third-party tools...")
    setup_file2video()
    setup_youbit()
    print("  All tools ready.\n")

# ── PixelVault wrappers ────────────────────────────────────────────────────────
def _bv_encode(input_path: str, output_mp4: str, extra_args=None):
    cmd = [sys.executable, str(PV_ROOT / "main.py"), "encode",
           str(input_path), "-o", str(output_mp4), "-q"]
    if extra_args:
        cmd += extra_args
    proc = subprocess.Popen(cmd, cwd=str(PV_ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    cmd = [sys.executable, str(PV_ROOT / "main.py"), "decode",
           str(video_mp4), "-o", str(out_dir), "-q"]
    proc = subprocess.Popen(cmd, cwd=str(PV_ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _reg_proc(proc)
    try:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        _unreg_proc()
    files = list(out_dir.glob("*"))
    return files[0].read_bytes()[:original_size] if files else b""

def _bv_enc_bs2_ecc16_1080p(s, d): _bv_encode(s, d, ["--ecc", "16"])
def _bv_enc_bs1_ecc16_1080p(s, d): _bv_encode(s, d, ["--ecc", "16", "--block-size", "1"])
def _bv_enc_bs2_ecc16_4k(s, d):    _bv_encode(s, d, ["--ecc", "16", "--4k"])

# ── file2video wrappers ───────────────────────────────────────────────────────
def _f2v_encode(input_path: str, output_mp4: str):
    """Call file2video.py --encode <input> <output> (real GitHub code, unmodified)."""
    cmd = [sys.executable, str(F2V_DIR / "file2video.py"),
           "--encode", str(input_path), str(output_mp4)]
    proc = subprocess.Popen(cmd, cwd=str(F2V_DIR),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _reg_proc(proc)
    try:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        _unreg_proc()

def _f2v_decode(video_mp4: str, original_size: int) -> bytes:
    """Call file2video.py --decode <video> <outdir> and return recovered bytes."""
    out_dir = OUT_DIR / ("f2v_dec_" + Path(video_mp4).stem)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(F2V_DIR / "file2video.py"),
           "--decode", str(video_mp4), str(out_dir)]
    proc = subprocess.Popen(cmd, cwd=str(F2V_DIR),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _reg_proc(proc)
    try:
        proc.wait()
    finally:
        _unreg_proc()
    files = list(out_dir.glob("*"))
    return files[0].read_bytes()[:original_size] if files else b""

# ── YouBit wrappers ───────────────────────────────────────────────────────────
def _yb_encode(input_path: str, output_mp4: str, settings=None):
    """Encode using YouBit Encoder.encode_local, extract video.mp4 from zip."""
    import sys as _sys
    if str(YB_DIR) not in _sys.path:
        _sys.path.insert(0, str(YB_DIR))
    from youbit.encode import Encoder
    from youbit.settings import Settings

    src = Path(input_path)
    mp4_out = Path(output_mp4)
    tmp_dir = OUT_DIR / ("yb_enc_tmp_" + mp4_out.stem)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    enc_settings = settings or Settings()
    encoder = Encoder(src, enc_settings)
    zip_no_ext = encoder.encode_local(tmp_dir)
    zip_path = Path(str(zip_no_ext) + ".zip")

    # Extract video.mp4 and save b64 metadata alongside output_mp4
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir / "extracted")
    video_src = tmp_dir / "extracted" / "video.mp4"
    readme = (tmp_dir / "extracted" / "README.txt").read_text()
    b64_meta = readme.split()[-1]

    shutil.copy2(video_src, mp4_out)
    Path(str(mp4_out) + ".b64meta").write_text(b64_meta)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _yb_decode(video_mp4: str, original_size: int) -> bytes:
    """Decode using YouBit decode_local, return recovered bytes."""
    import sys as _sys
    if str(YB_DIR) not in _sys.path:
        _sys.path.insert(0, str(YB_DIR))
    from youbit.decode import decode_local
    from youbit.metadata import Metadata

    mp4_path = Path(video_mp4)
    b64_path = Path(str(mp4_path) + ".b64meta")
    b64_meta = b64_path.read_text().strip()
    metadata = Metadata.create_from_base64(b64_meta)

    dec_dir = OUT_DIR / ("yb_dec_" + mp4_path.stem)
    if dec_dir.exists():
        shutil.rmtree(dec_dir, ignore_errors=True)
    dec_dir.mkdir(parents=True, exist_ok=True)

    out_path = decode_local(mp4_path, dec_dir, metadata)
    return out_path.read_bytes()[:original_size]


def _yb_enc_bpp1(s, d): _yb_encode(s, d)
def _yb_enc_bpp2(s, d):
    from youbit.settings import Settings, BitsPerPixel
    _yb_encode(s, d, Settings(bits_per_pixel=BitsPerPixel.TWO))

# ── tool registry ─────────────────────────────────────────────────────────────
@dataclass
class ToolSpec:
    name: str
    desc: str
    encode_fn: Callable
    decode_fn: Callable
    resolution: str
    density: str
    ecc: str
    codec: str
    source: str           # GitHub URL of real code
    youtube_safe: bool

TOOLS = [
    ToolSpec(
        name="PixelVault bs=2 ecc=16 1080p",
        desc="PixelVault default: 2×2 blocks, 1920×1080, RS nsym=16 ECC.",
        encode_fn=_bv_enc_bs2_ecc16_1080p,
        decode_fn=_bv_decode_to_bytes,
        resolution="1920×1080", density="~64,800 B/fr",
        ecc="RS nsym=16", codec="H.264 NVENC/libx264",
        source="https://github.com/sahilmenon/PixelVault-Infinite--The-Eternal-Encoder",
        youtube_safe=True,
    ),
    ToolSpec(
        name="PixelVault bs=1 ecc=16 1080p",
        desc="PixelVault max density: 1×1 blocks, 1920×1080, RS nsym=16 ECC.",
        encode_fn=_bv_enc_bs1_ecc16_1080p,
        decode_fn=_bv_decode_to_bytes,
        resolution="1920×1080", density="~259,200 B/fr",
        ecc="RS nsym=16", codec="H.264 NVENC/libx264",
        source="https://github.com/sahilmenon/PixelVault-Infinite--The-Eternal-Encoder",
        youtube_safe=True,
    ),
    ToolSpec(
        name="PixelVault bs=2 ecc=16 4K",
        desc="PixelVault 4K: 2×2 blocks, 3840×2160, RS nsym=16 ECC.",
        encode_fn=_bv_enc_bs2_ecc16_4k,
        decode_fn=_bv_decode_to_bytes,
        resolution="3840×2160", density="~259,200 B/fr",
        ecc="RS nsym=16", codec="H.264 NVENC/libx264",
        source="https://github.com/sahilmenon/PixelVault-Infinite--The-Eternal-Encoder",
        youtube_safe=True,
    ),
    ToolSpec(
        name="file2video RS-10",
        desc="file2video: 270×270 grid→1080p (4× NN), RS nsym=10 ECC, 20 FPS, CRF=40.",
        encode_fn=_f2v_encode,
        decode_fn=_f2v_decode,
        resolution="1080×1080", density="~9,112 B/fr (270×270 grid, RS-10)",
        ecc="RS(255,245) nsym=10", codec="H.264 CRF=40",
        source="https://github.com/karaketir16/file2video",
        youtube_safe=True,
    ),
    ToolSpec(
        name="YouBit bpp=1 default",
        desc="YouBit: 1920×1080, 1 bit/pixel, RS ECC, gzip. Local roundtrip.",
        encode_fn=_yb_enc_bpp1,
        decode_fn=_yb_decode,
        resolution="1920×1080", density="~259,200 B/fr (1 bpp)",
        ecc="RS (youbit default)", codec="H.264 libx264 CRF=18 grain",
        source="https://github.com/MeViMo/youbit",
        youtube_safe=False,
    ),
    ToolSpec(
        name="YouBit bpp=2 default",
        desc="YouBit: 1920×1080, 2 bits/pixel, RS ECC, gzip. Local roundtrip.",
        encode_fn=_yb_enc_bpp2,
        decode_fn=_yb_decode,
        resolution="1920×1080", density="~518,400 B/fr (2 bpp)",
        ecc="RS (youbit default)", codec="H.264 libx264 CRF=18 grain",
        source="https://github.com/MeViMo/youbit",
        youtube_safe=False,
    ),
]

# ── checkpoint helpers ────────────────────────────────────────────────────────
_SKIP_STATUSES = {"PASS", "TIMEOUT"}

def _is_done(status: str) -> bool:
    if status in _SKIP_STATUSES:
        return True
    if status.startswith("TIMEOUT"):
        return True
    return False

def _load_results() -> list:
    if RESULTS_JSON.exists():
        try:
            d = json.loads(RESULTS_JSON.read_text())
            return d.get("aggregated", [])
        except Exception:
            return []
    return []

def _save_results(results: list):
    RESULTS_JSON.write_text(json.dumps({"aggregated": results}, indent=2))

def _find_result(results: list, tool: str, fname: str) -> Optional[dict]:
    for r in results:
        if r["tool"] == tool and r["file"] == fname:
            return r
    return None

def _find_run(agg: dict, run_n: int) -> Optional[dict]:
    for r in agg.get("runs", []):
        if r["run"] == run_n:
            return r
    return None

# ── README update ─────────────────────────────────────────────────────────────
_BENCH_RE = re.compile(
    r'(### Benchmark results\n)(.*?)(\n---\n\n### Detailed analysis)',
    re.DOTALL,
)

def _human_time(s):
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}m"

def _human_mb(b):
    if b is None:
        return "—"
    return f"{b/1e6:.0f} MB"

def _mb_s(size_bytes, seconds):
    if not seconds or not size_bytes:
        return "—"
    return f"{size_bytes / seconds / 1e6:.2f}"

def update_readme(results: list, file_labels: list):
    """Rebuild the benchmark section of README.md from checkpoint data."""
    if not results:
        return

    lines = []
    lines.append("")

    # Excluded tools note
    lines.append("> **Scope:** Python 3 programs only, using real GitHub code with minimal non-functional changes.")
    lines.append("> Excluded: DataToVideoEncoderDecoder (pixel_size enc/dec mismatch, OOM on large files) ·")
    lines.append("> qStore (requires system ffmpeg, QR encoding extremely slow).")
    lines.append("")

    detail_lines = ["<details>"]
    detail_lines.append("<summary>Per-run timing detail</summary>")
    detail_lines.append("")

    for label in file_labels:
        label_results = [r for r in results if r["label"] == label]
        if not label_results:
            continue

        file_bytes = label_results[0]["size_bytes"]

        lines.append(f"#### {label} ({file_bytes / 1e6:.0f} MB raw)")
        lines.append("")
        lines.append("| Tool | Resolution | Density | ECC | Codec | Enc avg | Dec avg | MP4 size | Enc MB/s | Result |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        for tool in TOOLS:
            r = next((x for x in label_results if x["tool"] == tool.name), None)
            if r is None:
                lines.append(f"| {tool.name} | {tool.resolution} | {tool.density} | {tool.ecc} | {tool.codec} | — | — | — | — | pending |")
                continue
            status = r["status"]
            enc = _human_time(r.get("enc_avg"))
            dec = _human_time(r.get("dec_avg"))
            mp4 = _human_mb(r.get("mp4_bytes_avg"))
            mbs = _mb_s(file_bytes, r.get("enc_avg"))
            result_cell = "✅ PASS" if status == "PASS" else f"❌ {status}"
            lines.append(f"| {tool.name} | {tool.resolution} | {tool.density} | {tool.ecc} | {tool.codec} | {enc} | {dec} | {mp4} | {mbs} | {result_cell} |")

        lines.append("")

        # Per-run detail
        detail_lines.append(f"##### {label}")
        detail_lines.append("")
        detail_lines.append("| Tool | Run | Enc (s) | Dec (s) | MP4 size | Status |")
        detail_lines.append("|---|---|---|---|---|---|")
        for tool in TOOLS:
            r = next((x for x in label_results if x["tool"] == tool.name), None)
            if r is None:
                detail_lines.append(f"| {tool.name} | — | — | — | — | pending |")
                continue
            for run in r.get("runs", []):
                e = f"{run['encode_s']:.1f}" if run.get("encode_s") else "—"
                d = f"{run['decode_s']:.1f}" if run.get("decode_s") else "—"
                m = f"{run['mp4_bytes']/1e6:.0f} MB" if run.get("mp4_bytes") else "—"
                s = "✅" if run.get("status") == "PASS" else f"❌ {run.get('status','?')}"
                detail_lines.append(f"| {tool.name} | {run['run']} | {e} | {d} | {m} | {s} |")
        detail_lines.append("")

    detail_lines.append("</details>")
    lines.extend(detail_lines)

    new_body = "\n".join(lines)
    txt = README.read_text(encoding="utf-8")
    new_txt, n = _BENCH_RE.subn(
        lambda m: m.group(1) + new_body + m.group(3),
        txt,
    )
    if n:
        README.write_text(new_txt, encoding="utf-8")
    else:
        print("  [WARN] README benchmark section pattern not matched")

# ── run one (tool, file, run-number) ─────────────────────────────────────────
def run_one(tool: ToolSpec, src_path: Path, label: str, run_n: int,
            all_results: list, file_bytes: int):

    stem = f"{tool.name.replace(' ', '_').replace('/', '_')}_{src_path.stem}_r{run_n}"
    mp4_path = OUT_DIR / (stem + ".mp4")

    # ── encode ──
    print(f"      run {run_n} encode ...", end=" ", flush=True)
    _, enc_s, enc_err = _run_timed(tool.encode_fn, (str(src_path), str(mp4_path)), ENC_TIMEOUT)

    if enc_err:
        print(f"{enc_s:.1f}s  [{enc_err}]")
        return {"tool": tool.name, "file": src_path.name, "label": label,
                "size_bytes": file_bytes, "run": run_n,
                "encode_s": enc_s, "decode_s": None, "mp4_bytes": None,
                "passed": None, "status": enc_err, "notes": ""}

    mp4_bytes = mp4_path.stat().st_size if mp4_path.exists() else None
    print(f"{enc_s:.1f}s  ({mp4_bytes // 1_000_000 if mp4_bytes else '?'} MB mp4)")

    # ── decode ──
    print(f"      run {run_n} decode ...", end=" ", flush=True)
    recovered, dec_s, dec_err = _run_timed(
        tool.decode_fn, (str(mp4_path), file_bytes), DEC_TIMEOUT
    )

    if dec_err:
        print(f"{dec_s:.1f}s  [{dec_err}]")
        return {"tool": tool.name, "file": src_path.name, "label": label,
                "size_bytes": file_bytes, "run": run_n,
                "encode_s": enc_s, "decode_s": dec_s, "mp4_bytes": mp4_bytes,
                "passed": None, "status": dec_err, "notes": ""}

    # ── verify ──
    original = src_path.read_bytes()
    matched = (recovered == original)
    status = "PASS" if matched else "MISMATCH"
    print(f"{dec_s:.1f}s  [{status}]")
    return {"tool": tool.name, "file": src_path.name, "label": label,
            "size_bytes": file_bytes, "run": run_n,
            "encode_s": enc_s, "decode_s": dec_s, "mp4_bytes": mp4_bytes,
            "passed": matched, "status": status, "notes": ""}

# ── aggregate runs into a summary entry ──────────────────────────────────────
def _aggregate(tool_name: str, fname: str, label: str, size_bytes: int,
               runs: list) -> dict:
    pass_runs = [r for r in runs if r.get("status") == "PASS"]
    enc_times = [r["encode_s"] for r in pass_runs if r.get("encode_s")]
    dec_times = [r["decode_s"] for r in pass_runs if r.get("decode_s")]
    mp4_sizes = [r["mp4_bytes"] for r in pass_runs if r.get("mp4_bytes")]
    statuses  = [r["status"] for r in runs]
    final_status = "PASS" if statuses and all(s == "PASS" for s in statuses) \
        else (statuses[-1] if statuses else "pending")
    return {
        "tool": tool_name, "file": fname, "label": label,
        "size_bytes": size_bytes, "n_runs": len(runs), "runs": runs,
        "enc_avg":  round(sum(enc_times) / len(enc_times), 2) if enc_times else None,
        "enc_min":  round(min(enc_times), 2) if enc_times else None,
        "enc_max":  round(max(enc_times), 2) if enc_times else None,
        "dec_avg":  round(sum(dec_times) / len(dec_times), 2) if dec_times else None,
        "dec_min":  round(min(dec_times), 2) if dec_times else None,
        "dec_max":  round(max(dec_times), 2) if dec_times else None,
        "mp4_bytes_avg": int(sum(mp4_sizes) / len(mp4_sizes)) if mp4_sizes else None,
        "passed": final_status == "PASS",
        "status": final_status,
    }

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 72)
    print("  PixelVault Jellyfish Benchmark  (Python 3 programs only)")
    print("=" * 72)

    setup_tools()
    ensure_jellyfish()

    all_results = _load_results()
    done_count  = sum(
        1 for r in all_results
        for run in r.get("runs", [])
        if _is_done(run.get("status", ""))
    )
    retry_count = sum(
        1 for r in all_results
        for run in r.get("runs", [])
        if not _is_done(run.get("status", ""))
    )
    need_count = sum(
        (RUNS - sum(1 for run in (
            (_find_result(all_results, t.name, f) or {}).get("runs", [])
        )))
        for _, f in JELLYFISH
        for t in TOOLS
    )

    if all_results:
        print(f"  Resuming - {done_count} done, {retry_count + need_count} to run.")
        existing_labels = list(dict.fromkeys(r["label"] for r in all_results))
        print(f"  Writing existing results to README ({len(existing_labels)} file size(s))...")
        update_readme(all_results, existing_labels)
        print(f"  README updated through {existing_labels[-1] if existing_labels else '—'}")
    else:
        print("  Starting fresh benchmark.")

    file_labels_done = []

    for label, fname in JELLYFISH:
        src_path = DATA_DIR / fname
        if not src_path.exists():
            print(f"\n  [SKIP] {fname} not found, skipping.")
            continue

        file_bytes = src_path.stat().st_size
        print(f"\n{'-' * 72}")
        print(f"  FILE: {fname}  ({label}  /  {file_bytes // 1_000_000} MB)")
        print(f"{'-' * 72}")

        for tool in TOOLS:
            print(f"\n  [{tool.name}]")
            agg = _find_result(all_results, tool.name, fname)
            completed_runs = agg.get("runs", []) if agg else []

            new_runs = list(completed_runs)
            for run_n in range(1, RUNS + 1):
                existing_run = _find_run(agg, run_n) if agg else None
                if existing_run and _is_done(existing_run.get("status", "")):
                    print(f"      run {run_n} - done ({existing_run['status']}), skipping")
                    continue

                run_result = run_one(tool, src_path, label, run_n, all_results, file_bytes)

                # Replace or append the run entry
                new_runs = [r for r in new_runs if r.get("run") != run_n]
                new_runs.append(run_result)
                new_runs.sort(key=lambda r: r["run"])

                new_agg = _aggregate(tool.name, fname, label, file_bytes, new_runs)
                all_results = [r for r in all_results
                               if not (r["tool"] == tool.name and r["file"] == fname)]
                all_results.append(new_agg)
                _save_results(all_results)

        file_labels_done.append(label)
        print(f"\n  Updating README after {label}...")
        update_readme(all_results, file_labels_done)
        print(f"  README updated through {label}")

    print(f"\n{'=' * 72}")
    print("  Benchmark complete.")
    print(f"{'=' * 72}")

if __name__ == "__main__":
    main()
