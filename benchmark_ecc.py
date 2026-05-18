#!/usr/bin/env python3
"""
Benchmark: reedsolo (current) vs vectorised numpy RS encoder.

The numpy approach processes ALL blocks simultaneously in each of the k=block_data
inner-loop steps instead of one block at a time in pure Python.
"""

import time
import numpy as np
import reedsolo

# ---------------------------------------------------------------------------
# GF(2^8) tables  (same prim poly 0x11d and alpha=2 as reedsolo defaults)
# ---------------------------------------------------------------------------

def _build_gf(prim: int = 0x11d):
    exp = np.zeros(510, dtype=np.uint8)   # 2*255: allows log-space addition without wrap
    log = np.zeros(256, dtype=np.uint8)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x = ((x << 1) ^ prim if x & 0x80 else x << 1) & 0xFF
    exp[255:510] = exp[:255]
    return exp, log

_EXP, _LOG = _build_gf()

# (256, 256) multiplication table: _MUL[a, b] = a * b in GF(2^8)
def _build_mul_table() -> np.ndarray:
    a = np.arange(256, dtype=np.uint16)
    b = np.arange(256, dtype=np.uint16)
    la = _LOG[a].astype(np.uint16)
    lb = _LOG[b].astype(np.uint16)
    s = (la[:, None] + lb[None, :]) % 255
    t = _EXP[s].copy()
    t[0, :] = 0   # 0 * anything = 0
    t[:, 0] = 0
    return t

_MUL = _build_mul_table()

# ---------------------------------------------------------------------------
# Generator polynomial (cached by nsym)
# ---------------------------------------------------------------------------

_GEN_CACHE: dict[int, np.ndarray] = {}

def _generator(nsym: int) -> np.ndarray:
    if nsym in _GEN_CACHE:
        return _GEN_CACHE[nsym]
    g = [1]
    for i in range(nsym):
        # multiply g by (x + alpha^i)
        ai = int(_EXP[i])
        new_g = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            new_g[j] ^= c
            if c and ai:
                new_g[j + 1] ^= int(_EXP[(int(_LOG[c]) + i) % 255])
        g = new_g
    result = np.array(g[1:], dtype=np.uint8)   # drop leading 1
    _GEN_CACHE[nsym] = result
    return result

# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def encode_reedsolo(data: bytes, nsym: int) -> bytes:
    block_data = 255 - nsym
    pad = (block_data - len(data) % block_data) % block_data
    padded = data + b"\x00" * pad
    rs = reedsolo.RSCodec(nsym)
    out = bytearray()
    for i in range(0, len(padded), block_data):
        out.extend(rs.encode(padded[i : i + block_data]))
    return bytes(out)


def encode_numpy(data: bytes, nsym: int) -> bytes:
    """Vectorised RS: all N blocks encoded simultaneously in k=block_data passes."""
    if not data:
        return data
    block_data = 255 - nsym
    pad = (block_data - len(data) % block_data) % block_data
    padded = np.frombuffer(data + b"\x00" * pad, dtype=np.uint8)
    N = len(padded) // block_data
    msg = padded.reshape(N, block_data)       # (N, block_data)
    gen = _generator(nsym)                    # (nsym,)

    remainder = np.zeros((N, nsym), dtype=np.uint8)
    for i in range(block_data):
        # feedback coefficient for each block
        coef = msg[:, i] ^ remainder[:, 0]   # (N,)
        # shift remainder left (no allocation: slice assignment is safe here
        # because source cols 1..nsym-1 are never written by the destination)
        remainder[:, :-1] = remainder[:, 1:]
        remainder[:, -1] = 0
        # GF multiply-accumulate over all blocks and all generator coefficients at once
        remainder ^= _MUL[coef[:, None], gen[None, :]]   # (N, nsym)

    out = np.empty((N, 255), dtype=np.uint8)
    out[:, :block_data] = msg
    out[:, block_data:] = remainder
    return out.tobytes()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench(fn, data, nsym, n_runs: int = 3) -> tuple[bytes, float]:
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(data, nsym)
        times.append(time.perf_counter() - t0)
    return result, min(times)


def _mbps(n_bytes: int, t: float) -> float:
    return n_bytes / 1024 ** 2 / t

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- correctness check ---
    print("Correctness check (numpy output must match reedsolo byte-for-byte):")
    test = bytes(rng.integers(0, 256, 2048, dtype=np.uint8))
    for nsym in [8, 16, 32, 64]:
        r1 = encode_reedsolo(test, nsym)
        r2 = encode_numpy(test, nsym)
        ok = r1 == r2
        print(f"  nsym={nsym:2d}  {'PASS' if ok else 'FAIL'}", end="")
        if not ok:
            for i, (a, b) in enumerate(zip(r1, r2)):
                if a != b:
                    print(f"  (first diff: byte {i}, reedsolo={a}, numpy={b})", end="")
                    break
        print()
    print()

    # --- speed benchmark ---
    print(f"{'Size':>8}  {'nsym':>4}  {'reedsolo':>12}  {'numpy':>10}  {'speedup':>8}")
    print("-" * 56)
    for size_mb in [0.1, 1, 10]:
        data = bytes(rng.integers(0, 256, int(size_mb * 1024 ** 2), dtype=np.uint8))
        for nsym in [16, 32]:
            n_runs_rs = 1 if size_mb >= 10 else 2   # reedsolo is very slow; limit runs
            _, t_rs = _bench(encode_reedsolo, data, nsym, n_runs=n_runs_rs)
            _, t_np = _bench(encode_numpy,    data, nsym, n_runs=3)
            s_rs = _mbps(len(data), t_rs)
            s_np = _mbps(len(data), t_np)
            print(f"  {size_mb:5.1f} MB  {nsym:4d}  "
                  f"{t_rs:7.2f}s ({s_rs:5.1f} MB/s)  "
                  f"{t_np:5.3f}s ({s_np:5.0f} MB/s)  "
                  f"{t_rs/t_np:6.1f}×")
    print()
    print("numpy RS is drop-in compatible with reedsolo (same GF(2^8) prim=0x11d, fcr=0).")
    print("Encoded output is identical → reedsolo can still decode it.")
