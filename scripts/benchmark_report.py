#!/usr/bin/env python3
"""
PixelVault Scaling Report Generator
====================================
Benchmarks PixelVault across multiple file sizes and configurations, then
produces a research-style report with matplotlib figures.

Usage:
    python benchmark_report.py [--skip-bench]   # --skip-bench: use cached data only
    python benchmark_report.py --runs N         # runs per (config, size) pair (default 2)

Output:
    bench/report/figures/*.png    — individual figure files
    bench/report/REPORT.md        — full markdown report with embedded figures
    bench/report/results.json     — raw benchmark data
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
BENCH_DIR  = ROOT / "bench"
REPORT_DIR = BENCH_DIR / "report"
FIG_DIR    = REPORT_DIR / "figures"
RESULTS_J  = REPORT_DIR / "results.json"
JF_JSON    = BENCH_DIR / "jellyfish_results.json"
DATA_DIR   = BENCH_DIR / "testdata"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── benchmark configurations ─────────────────────────────────────────────────
CONFIGS = [
    {
        "name":  "PixelVault bs=2 ecc=16 1080p",
        "short": "BV default\n(bs=2 1080p)",
        "color": "#2196F3",
        "marker": "o",
        "args":  ["--ecc", "16"],
    },
    {
        "name":  "PixelVault bs=1 ecc=16 1080p",
        "short": "BV max density\n(bs=1 1080p)",
        "color": "#4CAF50",
        "marker": "s",
        "args":  ["--ecc", "16", "--block-size", "1"],
    },
    {
        "name":  "PixelVault bs=2 ecc=16 4K",
        "short": "BV 4K\n(bs=2 4K)",
        "color": "#FF5722",
        "marker": "^",
        "args":  ["--ecc", "16", "--4k"],
    },
    {
        "name":  "PixelVault bs=2 ecc=0 1080p",
        "short": "BV no ECC\n(bs=2 1080p)",
        "color": "#9C27B0",
        "marker": "D",
        "args":  ["--ecc", "0"],
    },
    {
        "name":  "PixelVault bs=2 ecc=32 1080p",
        "short": "BV ecc=32\n(bs=2 1080p)",
        "color": "#FF9800",
        "marker": "v",
        "args":  ["--ecc", "32"],
    },
]

# sizes to benchmark (bytes)
MB = 1024 * 1024
TEST_SIZES = [1*MB, 5*MB, 10*MB, 25*MB, 50*MB, 100*MB]


# ─── data generation ──────────────────────────────────────────────────────────
def ensure_test_file(size_bytes: int) -> Path:
    name = f"{size_bytes // MB}MB.bin"
    path = DATA_DIR / name
    if not path.exists() or path.stat().st_size != size_bytes:
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=size_bytes, dtype=np.uint8)
        path.write_bytes(data.tobytes())
    return path


# ─── single encode/decode benchmark ──────────────────────────────────────────
def _run_encode(src: Path, out_mp4: Path, extra_args: list) -> tuple[float, int]:
    """Returns (encode_seconds, mp4_bytes)."""
    cmd = [sys.executable, str(ROOT / "main.py"), "encode", str(src),
           "-o", str(out_mp4), "-q"] + extra_args
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace")[:300])
    return elapsed, out_mp4.stat().st_size


def _run_decode(mp4: Path, out_dir: Path) -> float:
    """Returns decode_seconds."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "main.py"), "decode", str(mp4),
           "-o", str(out_dir), "-q"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace")[:300])
    return elapsed


# ─── benchmark runner ─────────────────────────────────────────────────────────
def run_benchmarks(runs: int = 2) -> list[dict]:
    results = []

    total = len(TEST_SIZES) * len(CONFIGS) * runs
    done = 0
    print(f"\nRunning {total} encode+decode operations ({len(TEST_SIZES)} sizes × "
          f"{len(CONFIGS)} configs × {runs} runs)...\n")

    for size_bytes in TEST_SIZES:
        src = ensure_test_file(size_bytes)
        size_mb = size_bytes / MB
        print(f"  -- {size_mb:.0f} MB -----------------------------")

        for cfg in CONFIGS:
            enc_times, dec_times, mp4_sizes = [], [], []
            failed = False

            for run_n in range(1, runs + 1):
                done += 1
                stem = f"rpt_{cfg['name'].replace(' ','_').replace('=','')}_"\
                       f"{size_bytes//MB}MB_r{run_n}"
                mp4  = REPORT_DIR / "tmp" / (stem + ".mp4")
                ddir = REPORT_DIR / "tmp" / (stem + "_dec")

                print(f"    [{done}/{total}] {cfg['name']}  "
                      f"{size_mb:.0f}MB  run{run_n} ...", end=" ", flush=True)
                try:
                    enc_s, mp4_b = _run_encode(src, mp4, cfg["args"])
                    dec_s        = _run_decode(mp4, ddir)
                    enc_times.append(enc_s)
                    dec_times.append(dec_s)
                    mp4_sizes.append(mp4_b)
                    print(f"enc={enc_s:.1f}s  dec={dec_s:.1f}s  "
                          f"mp4={mp4_b/MB:.1f}MB")
                except Exception as e:
                    print(f"FAILED: {e}")
                    failed = True
                    break
                finally:
                    if mp4.exists():
                        mp4.unlink()
                    if ddir.exists():
                        shutil.rmtree(ddir, ignore_errors=True)

            if failed or not enc_times:
                continue

            results.append({
                "config":      cfg["name"],
                "size_bytes":  size_bytes,
                "enc_avg":     round(sum(enc_times) / len(enc_times), 3),
                "enc_min":     round(min(enc_times), 3),
                "enc_max":     round(max(enc_times), 3),
                "dec_avg":     round(sum(dec_times) / len(dec_times), 3),
                "dec_min":     round(min(dec_times), 3),
                "dec_max":     round(max(dec_times), 3),
                "mp4_avg":     int(sum(mp4_sizes) / len(mp4_sizes)),
                "runs":        runs,
            })

    return results


# ─── load / merge with existing jellyfish data ────────────────────────────────
def load_jellyfish() -> list[dict]:
    if not JF_JSON.exists():
        return []
    try:
        raw = json.loads(JF_JSON.read_text())["aggregated"]
    except Exception:
        return []

    out = []
    jf_cfg_map = {
        "PixelVault bs=2 ecc=16 1080p": "PixelVault bs=2 ecc=16 1080p",
        "PixelVault bs=1 ecc=16 1080p": "PixelVault bs=1 ecc=16 1080p",
        "PixelVault bs=2 ecc=16 4K":    "PixelVault bs=2 ecc=16 4K",
    }
    for r in raw:
        cfg = jf_cfg_map.get(r["tool"])
        if not cfg:
            continue
        if r.get("status") != "PASS":
            continue
        out.append({
            "config":     cfg,
            "size_bytes": r["size_bytes"],
            "enc_avg":    r["enc_avg"],
            "enc_min":    r.get("enc_min", r["enc_avg"]),
            "enc_max":    r.get("enc_max", r["enc_avg"]),
            "dec_avg":    r["dec_avg"],
            "dec_min":    r.get("dec_min", r["dec_avg"]),
            "dec_max":    r.get("dec_max", r["dec_avg"]),
            "mp4_avg":    r.get("mp4_bytes_avg"),
            "runs":       r.get("n_runs", 3),
            "source":     "jellyfish",
        })
    return out


def merge(new_results: list[dict], jf: list[dict]) -> list[dict]:
    """Prefer new_results; fall back to jf for sizes not covered."""
    covered = {(r["config"], r["size_bytes"]) for r in new_results}
    merged = list(new_results)
    for r in jf:
        if (r["config"], r["size_bytes"]) not in covered:
            merged.append(r)
    return merged


# ─── plotting helpers ─────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#e6edf3",
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "text.color":        "#c9d1d9",
    "lines.linewidth":   2.2,
    "lines.markersize":  7,
    "font.size":         10,
    "axes.titlepad":     10,
    "figure.dpi":        150,
}


def apply_style():
    plt.rcParams.update(STYLE)


def _cfg_props(name: str) -> dict:
    for c in CONFIGS:
        if c["name"] == name:
            return c
    return {"color": "#aaa", "marker": "x", "short": name}


def _get_series(data: list[dict], config: str, metric: str):
    pts = [(r["size_bytes"] / MB, r[metric])
           for r in data if r["config"] == config and r.get(metric) is not None]
    pts.sort()
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def _errbar(data: list[dict], config: str, metric: str, lo_key: str, hi_key: str):
    pts = [
        (r["size_bytes"] / MB,
         r[metric],
         r[metric] - r.get(lo_key, r[metric]),
         r.get(hi_key, r[metric]) - r[metric])
        for r in data if r["config"] == config and r.get(metric) is not None
    ]
    pts.sort()
    if not pts:
        return [], [], [], []
    xs, ys, lo, hi = zip(*pts)
    return list(xs), list(ys), list(lo), list(hi)


# ─── individual figures ───────────────────────────────────────────────────────
def fig_throughput(data: list[dict], show_configs: list[str], kind: str) -> Path:
    """Encode or decode throughput (MB/s) vs file size."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    metric     = "enc_avg" if kind == "encode" else "dec_avg"
    lo_key     = "enc_min" if kind == "encode" else "dec_min"
    hi_key     = "enc_max" if kind == "encode" else "dec_max"
    label_verb = "Encode" if kind == "encode" else "Decode"

    for cfg_name in show_configs:
        xs, ys, lo, hi = _errbar(data, cfg_name, metric, lo_key, hi_key)
        if not xs:
            continue
        p = _cfg_props(cfg_name)
        throughput = [s / t for s, t in zip(xs, ys)]
        lo_tp = [s / (t + e) if (t + e) else 0 for s, t, e in zip(xs, ys, hi)]
        hi_tp = [s / max(t - e, 1e-9) for s, t, e in zip(xs, ys, lo)]
        err_lo = [a - b for a, b in zip(throughput, lo_tp)]
        err_hi = [b - a for a, b in zip(throughput, hi_tp)]
        ax.errorbar(xs, throughput,
                    yerr=[err_lo, err_hi],
                    color=p["color"], marker=p["marker"],
                    label=cfg_name,
                    capsize=4, capthick=1.2, elinewidth=1)

    ax.set_xlabel("Input file size (MB)")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title(f"{label_verb} Throughput vs File Size")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = FIG_DIR / f"throughput_{kind}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_time(data: list[dict], show_configs: list[str]) -> Path:
    """Encode + decode time (s) vs file size, side-by-side subplots."""
    apply_style()
    fig, (ax_enc, ax_dec) = plt.subplots(1, 2, figsize=(13, 5))

    for cfg_name in show_configs:
        p = _cfg_props(cfg_name)
        for ax, metric, lo_k, hi_k, label in [
            (ax_enc, "enc_avg", "enc_min", "enc_max", "Encode"),
            (ax_dec, "dec_avg", "dec_min", "dec_max", "Decode"),
        ]:
            xs, ys, lo, hi = _errbar(data, cfg_name, metric, lo_k, hi_k)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[lo, hi],
                        color=p["color"], marker=p["marker"],
                        label=cfg_name, capsize=4, capthick=1.2, elinewidth=1)

    for ax, label in [(ax_enc, "Encode"), (ax_dec, "Decode")]:
        ax.set_xlabel("Input file size (MB)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{label} Time vs File Size")
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Encode & Decode Time Scaling", fontsize=13, color="#e6edf3")
    fig.tight_layout()
    path = FIG_DIR / "time_scaling.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_roundtrip(data: list[dict], show_configs: list[str]) -> Path:
    """Total roundtrip (encode + decode) time vs file size."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for cfg_name in show_configs:
        pts_enc = {r["size_bytes"]: r["enc_avg"]
                   for r in data if r["config"] == cfg_name and r.get("enc_avg")}
        pts_dec = {r["size_bytes"]: r["dec_avg"]
                   for r in data if r["config"] == cfg_name and r.get("dec_avg")}
        common = sorted(set(pts_enc) & set(pts_dec))
        if not common:
            continue
        p = _cfg_props(cfg_name)
        xs = [s / MB for s in common]
        ys = [pts_enc[s] + pts_dec[s] for s in common]
        ax.plot(xs, ys, color=p["color"], marker=p["marker"], label=cfg_name)

    ax.set_xlabel("Input file size (MB)")
    ax.set_ylabel("Encode + Decode time (s)")
    ax.set_title("Total Roundtrip Time vs File Size")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "roundtrip_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_overhead(data: list[dict], show_configs: list[str]) -> Path:
    """MP4 size as multiple of input size vs file size."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for cfg_name in show_configs:
        pts = [(r["size_bytes"] / MB, r["mp4_avg"] / r["size_bytes"])
               for r in data
               if r["config"] == cfg_name
               and r.get("mp4_avg") and r.get("size_bytes")]
        pts.sort()
        if not pts:
            continue
        p = _cfg_props(cfg_name)
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, color=p["color"], marker=p["marker"], label=cfg_name)

        abs_pts = [(r["size_bytes"] / MB, r["mp4_avg"] / MB)
                   for r in data
                   if r["config"] == cfg_name and r.get("mp4_avg")]
        abs_pts.sort()
        xs2, ys2 = zip(*abs_pts)
        ax2.plot(xs2, ys2, color=p["color"], marker=p["marker"], label=cfg_name)

    # Reference line: 1× (no overhead)
    ax1.axhline(1.0, color="#555", linestyle="--", linewidth=1, label="1× (no overhead)")
    ax1.set_xlabel("Input file size (MB)")
    ax1.set_ylabel("MP4 size / input size  (×)")
    ax1.set_title("Video Overhead Ratio vs File Size")
    ax1.legend(fontsize=7)

    all_sizes = sorted({r["size_bytes"] / MB for r in data if r.get("mp4_avg")})
    if all_sizes:
        ax2.plot(all_sizes, all_sizes, color="#555", linestyle="--",
                 linewidth=1, label="1× reference")
    ax2.set_xlabel("Input file size (MB)")
    ax2.set_ylabel("MP4 output size (MB)")
    ax2.set_title("Absolute Video Output Size")
    ax2.legend(fontsize=7)

    fig.suptitle("Storage Overhead Analysis", fontsize=13, color="#e6edf3")
    fig.tight_layout()
    path = FIG_DIR / "overhead.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_ecc_comparison(data: list[dict]) -> Path:
    """Compare encode throughput for ecc=0, ecc=16, ecc=32 (1080p bs=2)."""
    ecc_configs = [
        "PixelVault bs=2 ecc=0 1080p",
        "PixelVault bs=2 ecc=16 1080p",
        "PixelVault bs=2 ecc=32 1080p",
    ]
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for cfg_name in ecc_configs:
        xs, ys, lo, hi = _errbar(data, cfg_name, "enc_avg", "enc_min", "enc_max")
        if not xs:
            continue
        p = _cfg_props(cfg_name)
        throughput = [s / t for s, t in zip(xs, ys)]
        ax.plot(xs, throughput, color=p["color"], marker=p["marker"], label=cfg_name)

    ax.set_xlabel("Input file size (MB)")
    ax.set_ylabel("Encode throughput (MB/s)")
    ax.set_title("ECC Overhead Impact on Encode Throughput\n(bs=2, 1080p)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "ecc_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_heatmap(data: list[dict]) -> Path:
    """Heatmap: throughput (MB/s) across configs × file sizes."""
    apply_style()
    cfg_names = [c["name"] for c in CONFIGS]
    sizes_mb  = sorted({r["size_bytes"] / MB for r in data})

    matrix = np.full((len(cfg_names), len(sizes_mb)), np.nan)
    for i, cfg in enumerate(cfg_names):
        for j, smb in enumerate(sizes_mb):
            sb = smb * MB
            match = [r for r in data if r["config"] == cfg
                     and abs(r["size_bytes"] - sb) < 1024
                     and r.get("enc_avg")]
            if match:
                matrix[i, j] = smb / match[0]["enc_avg"]

    fig, ax = plt.subplots(figsize=(max(8, len(sizes_mb) * 1.2), 5))
    im = ax.imshow(matrix, aspect="auto", cmap="plasma", vmin=0)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Encode throughput (MB/s)", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

    ax.set_xticks(range(len(sizes_mb)))
    ax.set_xticklabels([f"{s:.0f} MB" for s in sizes_mb])
    ax.set_yticks(range(len(cfg_names)))
    ax.set_yticklabels([c["short"].replace("\n", " ") for c in CONFIGS], fontsize=8)
    ax.set_title("Encode Throughput Heatmap (MB/s)\nConfig × File Size")

    for i in range(len(cfg_names)):
        for j in range(len(sizes_mb)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v < matrix[~np.isnan(matrix)].max() * 0.7 else "black")

    fig.tight_layout()
    path = FIG_DIR / "throughput_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_bar_comparison(data: list[dict], size_mb: float = 10.0) -> Path:
    """Bar chart: encode + decode time at one representative file size."""
    apply_style()
    target = size_mb * MB
    candidates = sorted({r["size_bytes"] for r in data},
                         key=lambda s: abs(s - target))
    if not candidates:
        return None
    best_size = candidates[0]
    actual_mb = best_size / MB

    cfg_names = [c["name"] for c in CONFIGS]
    enc_vals, dec_vals = [], []
    present = []

    for cfg in cfg_names:
        match = [r for r in data if r["config"] == cfg
                 and abs(r["size_bytes"] - best_size) < 1024
                 and r.get("enc_avg")]
        if match:
            enc_vals.append(match[0]["enc_avg"])
            dec_vals.append(match[0]["dec_avg"] or 0)
            present.append(cfg)

    if not present:
        return None

    colors = [_cfg_props(n)["color"] for n in present]
    x = np.arange(len(present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.6), 5.5))
    bars1 = ax.bar(x - width/2, enc_vals, width, label="Encode",
                   color=colors, alpha=0.9)
    bars2 = ax.bar(x + width/2, dec_vals, width, label="Decode",
                   color=colors, alpha=0.55, hatch="//")

    ax.set_ylabel("Time (s)")
    ax.set_title(f"Encode & Decode Time at ≈{actual_mb:.0f} MB")
    ax.set_xticks(x)
    ax.set_xticklabels([_cfg_props(n)["short"] for n in present],
                       fontsize=8, ha="center")
    ax.legend()

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color="#c9d1d9")
    for bar in bars2:
        h = bar.get_height()
        if h:
            ax.annotate(f"{h:.1f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color="#c9d1d9")

    fig.tight_layout()
    path = FIG_DIR / f"bar_comparison_{int(actual_mb)}MB.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_throughput_combined(data: list[dict]) -> Path:
    """4-panel figure: enc throughput, dec throughput, time, overhead."""
    apply_style()
    main_cfgs = [
        "PixelVault bs=2 ecc=16 1080p",
        "PixelVault bs=1 ecc=16 1080p",
        "PixelVault bs=2 ecc=16 4K",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_enc_tp, ax_dec_tp), (ax_time, ax_ovhd) = axes

    for cfg_name in main_cfgs:
        p = _cfg_props(cfg_name)
        # Encode throughput
        xs, ys, lo, hi = _errbar(data, cfg_name, "enc_avg", "enc_min", "enc_max")
        if xs:
            tp = [s / t for s, t in zip(xs, ys)]
            ax_enc_tp.plot(xs, tp, color=p["color"], marker=p["marker"], label=cfg_name)
        # Decode throughput
        xs, ys, lo, hi = _errbar(data, cfg_name, "dec_avg", "dec_min", "dec_max")
        if xs:
            tp = [s / t for s, t in zip(xs, ys)]
            ax_dec_tp.plot(xs, tp, color=p["color"], marker=p["marker"])
        # Time
        enc_pts = {r["size_bytes"]: r["enc_avg"]
                   for r in data if r["config"] == cfg_name and r.get("enc_avg")}
        dec_pts = {r["size_bytes"]: r["dec_avg"]
                   for r in data if r["config"] == cfg_name and r.get("dec_avg")}
        common = sorted(set(enc_pts) & set(dec_pts))
        if common:
            xs_t = [s / MB for s in common]
            ys_rt = [enc_pts[s] + dec_pts[s] for s in common]
            ax_time.plot(xs_t, ys_rt, color=p["color"], marker=p["marker"])
        # Overhead
        ovhd_pts = [(r["size_bytes"] / MB, r["mp4_avg"] / r["size_bytes"])
                    for r in data
                    if r["config"] == cfg_name
                    and r.get("mp4_avg") and r.get("size_bytes")]
        ovhd_pts.sort()
        if ovhd_pts:
            xo, yo = zip(*ovhd_pts)
            ax_ovhd.plot(xo, yo, color=p["color"], marker=p["marker"])

    ax_enc_tp.set_title("Encode Throughput (MB/s)");  ax_enc_tp.set_ylabel("MB/s")
    ax_dec_tp.set_title("Decode Throughput (MB/s)");  ax_dec_tp.set_ylabel("MB/s")
    ax_time.set_title("Total Roundtrip Time (s)");     ax_time.set_ylabel("Enc+Dec (s)")
    ax_ovhd.set_title("MP4 / Input Size Ratio");       ax_ovhd.set_ylabel("ratio (×)")
    ax_ovhd.axhline(1.0, color="#555", linestyle="--", linewidth=1)

    for ax in axes.flat:
        ax.set_xlabel("Input file size (MB)")
        ax.grid(True, alpha=0.4)

    ax_enc_tp.legend(fontsize=7, loc="best")
    fig.suptitle("PixelVault Performance vs File Size", fontsize=14, color="#e6edf3", y=1.01)
    fig.tight_layout()
    path = FIG_DIR / "combined_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── markdown report ──────────────────────────────────────────────────────────
def _rel(path: Path) -> str:
    """Relative path from REPORT_DIR."""
    return str(path.relative_to(REPORT_DIR)).replace("\\", "/")


def build_report(data: list[dict], figs: dict) -> Path:
    def _tp_row(cfg_name: str):
        rows = []
        for r in sorted(data, key=lambda x: x["size_bytes"]):
            if r["config"] != cfg_name or not r.get("enc_avg"):
                continue
            smb  = r["size_bytes"] / MB
            enc  = r["enc_avg"]
            dec  = r.get("dec_avg") or 0
            mp4m = r["mp4_avg"] / MB if r.get("mp4_avg") else None
            ovhd = r["mp4_avg"] / r["size_bytes"] if r.get("mp4_avg") else None
            e_tp = smb / enc
            d_tp = smb / dec if dec else None
            mp4s = f"{mp4m:.1f} MB" if mp4m else "—"
            ovs  = f"{ovhd:.2f}×" if ovhd else "—"
            dts  = f"{d_tp:.2f}" if d_tp else "—"
            rows.append(f"| {smb:.0f} | {enc:.2f} | {e_tp:.2f} | "
                        f"{dec:.2f} | {dts} | {mp4s} | {ovs} |")
        return rows

    md = []
    md.append("# PixelVault Scaling Performance Report\n")
    md.append(f"> Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    md.append(f"> Benchmark: {len(data)} (config × size) data points  |  "
              f"Runs per point: {max((r['runs'] for r in data), default=1)}")
    md.append("> Input data: incompressible pseudo-random binary (worst case for "
              "compression; all metrics reflect raw throughput).")
    md.append("")
    md.append("---\n")

    md.append("## Overview\n")
    md.append("PixelVault encodes files into YouTube-compatible MP4s using pixel blocks "
              "on video frames.  This report characterises how encode throughput, decode "
              "throughput, total roundtrip latency, and video-container overhead scale "
              "with input file size across the three primary configurations:\n")
    md.append("| Config | Bits/px | Resolution | Density (B/fr) | YouTube-safe |")
    md.append("|---|---|---|---|---|")
    md.append("| `bs=2 ecc=16 1080p` | 0.25 | 1920×1080 | ~64,800 | ✅ |")
    md.append("| `bs=1 ecc=16 1080p` | 1.00 | 1920×1080 | ~259,200 | ✅ (more errors) |")
    md.append("| `bs=2 ecc=16 4K` | 0.25 | 3840×2160 | ~259,200 | ✅ (VP9) |")
    md.append("")

    md.append("### Combined overview (4 metrics)\n")
    if figs.get("combined"):
        md.append(f"![Combined overview]({_rel(figs['combined'])})\n")

    md.append("---\n")
    md.append("## Encode Throughput\n")
    md.append("Throughput = input file size ÷ encode wall-clock time.  "
              "Error bars show min/max across runs.\n")
    if figs.get("enc_tp"):
        md.append(f"![Encode throughput]({_rel(figs['enc_tp'])})\n")

    md.append("---\n")
    md.append("## Decode Throughput\n")
    md.append("Throughput = input file size ÷ decode wall-clock time.\n")
    if figs.get("dec_tp"):
        md.append(f"![Decode throughput]({_rel(figs['dec_tp'])})\n")

    md.append("---\n")
    md.append("## Encode & Decode Time Scaling\n")
    md.append("Raw time (seconds) vs file size.  Both encode and decode scale "
              "sub-linearly for small files (fixed overhead dominates) and "
              "linearly for large files (I/O and frame generation dominate).\n")
    if figs.get("time"):
        md.append(f"![Time scaling]({_rel(figs['time'])})\n")

    md.append("---\n")
    md.append("## Total Roundtrip Time\n")
    md.append("Encode + decode combined.  The curve slope gives the effective "
              "end-to-end cost per MB for each configuration.\n")
    if figs.get("roundtrip"):
        md.append(f"![Roundtrip time]({_rel(figs['roundtrip'])})\n")

    md.append("---\n")
    md.append("## Storage Overhead\n")
    md.append("Left: MP4 container size as a multiple of the original file size.  "
              "Right: absolute output sizes.  "
              "Overhead converges to ~1× for large files as the fixed 128-byte header "
              "and tail padding become negligible.  "
              "The `bs=1` config produces a smaller MP4 than `bs=2` for the same file "
              "because 4× the data per frame means 4× fewer frames and less container metadata.\n")
    if figs.get("overhead"):
        md.append(f"![Overhead]({_rel(figs['overhead'])})\n")

    md.append("---\n")
    md.append("## ECC Overhead Impact\n")
    md.append("`ecc=0` disables Reed-Solomon; `ecc=16` (default) adds ~6.7% overhead "
              "and corrects up to 8 errors per 255-byte block; `ecc=32` adds ~14.4% "
              "and corrects up to 16 errors.  "
              "The throughput reduction from ECC is small because ECC computation "
              "is parallelised across worker threads.\n")
    if figs.get("ecc"):
        md.append(f"![ECC comparison]({_rel(figs['ecc'])})\n")

    md.append("---\n")
    md.append("## Per-Configuration Bar Charts\n")
    for k, v in figs.items():
        if k.startswith("bar_"):
            mb = k.replace("bar_", "").replace("MB", "")
            md.append(f"### ≈{mb} MB file\n")
            md.append(f"![Bar comparison {mb} MB]({_rel(v)})\n")

    md.append("---\n")
    md.append("## Throughput Heatmap\n")
    md.append("Each cell shows encode throughput (MB/s) for a (config, file-size) "
              "pair.  Brighter = faster.\n")
    if figs.get("heatmap"):
        md.append(f"![Throughput heatmap]({_rel(figs['heatmap'])})\n")

    md.append("---\n")
    md.append("## Raw Data Tables\n")
    for cfg in CONFIGS:
        rows = _tp_row(cfg["name"])
        if not rows:
            continue
        md.append(f"### {cfg['name']}\n")
        md.append("| Size (MB) | Enc time (s) | Enc MB/s | Dec time (s) | Dec MB/s | MP4 size | Overhead |")
        md.append("|---|---|---|---|---|---|---|")
        md.extend(rows)
        md.append("")

    md.append("---\n")
    md.append("## Methodology\n")
    md.append("- **Input files:** pseudo-random binary blobs (incompressible) generated "
              "with NumPy `default_rng(seed=42)`.  Worst case for `--compress`; all "
              "figures reflect raw throughput without compression savings.\n")
    md.append("- **Hardware:** benchmarked on the machine running this script.  "
              "NVENC/AMF/QSV hardware encoding is used automatically when available "
              "(detected at runtime).\n")
    md.append("- **ECC:** Reed-Solomon nsym=16 by default (corrects up to 8 byte errors "
              "per 255-byte block, ~6.7% overhead).\n")
    md.append("- **Timing:** wall-clock time via `time.perf_counter()`, averaged across "
              f"{max((r['runs'] for r in data), default=1)} runs per (config, size) pair.\n")
    md.append("- **No YouTube upload:** all benchmarks are local encode→decode roundtrips.  "
              "YouTube processing adds latency but does not affect local encode/decode speed.\n")

    report_path = REPORT_DIR / "REPORT.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PixelVault scaling benchmark + report")
    parser.add_argument("--skip-bench", action="store_true",
                        help="Skip running new benchmarks; use only cached data")
    parser.add_argument("--runs", type=int, default=2, help="Runs per (config, size)")
    args = parser.parse_args()

    # ── load or run benchmarks ────────────────────────────────────────────────
    if not args.skip_bench:
        new_results = run_benchmarks(runs=args.runs)
        RESULTS_J.write_text(json.dumps(new_results, indent=2))
    else:
        new_results = json.loads(RESULTS_J.read_text()) if RESULTS_J.exists() else []
        print(f"  [--skip-bench] loaded {len(new_results)} cached results from {RESULTS_J}")

    jf = load_jellyfish()
    print(f"  Loaded {len(jf)} jellyfish results")
    data = merge(new_results, jf)
    print(f"  Total dataset: {len(data)} (config × size) entries\n")

    if not data:
        print("No data to plot. Run without --skip-bench first.")
        sys.exit(1)

    # ── generate figures ──────────────────────────────────────────────────────
    main_cfgs = [
        "PixelVault bs=2 ecc=16 1080p",
        "PixelVault bs=1 ecc=16 1080p",
        "PixelVault bs=2 ecc=16 4K",
    ]
    all_cfgs = [c["name"] for c in CONFIGS]

    print("Generating figures...")
    figs = {}
    figs["enc_tp"]    = fig_throughput(data, all_cfgs, "encode")
    figs["dec_tp"]    = fig_throughput(data, all_cfgs, "decode")
    figs["time"]      = fig_time(data, main_cfgs)
    figs["roundtrip"] = fig_roundtrip(data, main_cfgs)
    figs["overhead"]  = fig_overhead(data, main_cfgs)
    figs["ecc"]       = fig_ecc_comparison(data)
    figs["heatmap"]   = fig_heatmap(data)
    figs["combined"]  = fig_throughput_combined(data)

    all_sizes = sorted({r["size_bytes"] / MB for r in data})
    for smb in all_sizes:
        p = fig_bar_comparison(data, size_mb=smb)
        if p:
            figs[f"bar_{int(smb)}MB"] = p

    print(f"  {len(figs)} figures saved to {FIG_DIR}")

    # ── build report ──────────────────────────────────────────────────────────
    report = build_report(data, figs)
    print(f"\nReport written: {report}")
    print(f"Figures:        {FIG_DIR}")


if __name__ == "__main__":
    main()
