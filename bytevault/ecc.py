"""
Reed-Solomon error correction + byte interleaving for ByteVault payload data.

Pipeline (encode):
  file → [zlib] → [RS encode] → [interleave] → video+audio frames

Pipeline (decode):
  frames → [deinterleave] → [RS decode+correct] → [zlib decompress] → file

Reed-Solomon
------------
Encoding uses a vectorised NumPy implementation that processes all N blocks
simultaneously.  The Python loop runs only k = (255-nsym) iterations regardless
of file size, giving a 20-35x speedup over the pure-Python reedsolo encoder.
reedsolo is still used for *decoding* (error correction requires the
Berlekamp-Massey algorithm which is harder to vectorise).

  nsym=8  → ~3.2% overhead, corrects up to  4 byte errors/block
  nsym=16 → ~6.7% overhead, corrects up to  8 byte errors/block  (default)
  nsym=32 → ~14.4% overhead, corrects up to 16 byte errors/block
  nsym=64 → ~33.5% overhead, corrects up to 32 byte errors/block

Interleaving
------------
Without interleaving, a burst of B consecutive corrupted bytes hits at most
two RS blocks and can exhaust their correction capacity entirely.

With interleaving (depth D), the ECC data is permuted so that each video
frame carries bytes spaced D positions apart in the original ECC stream.
A burst of K corrupt frames therefore contributes at most ceil(255/D)*K
wrong bytes to each RS block — typically just 1-2 per block, well within
correction capacity even at low nsym.

Recommended settings for YouTube roundtrip:
  --mode palette --ecc 32 --interleave   (safe, ~14% overhead)
  --mode palette --ecc 16 --interleave   (after calibration shows <8 errors/block)
"""

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import reedsolo

_SYNDROME_ROOTS: dict[int, np.ndarray] = {}   # nsym → alpha^j roots array


# ---------------------------------------------------------------------------
# GF(2^8) tables  (prim=0x11d, fcr=0, generator=2 — matches reedsolo defaults)
# ---------------------------------------------------------------------------

def _build_gf(prim: int = 0x11d):
    exp = np.zeros(510, dtype=np.uint8)   # 2*255: log-space addition stays in range
    log = np.zeros(256, dtype=np.uint8)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x = ((x << 1) ^ prim if x & 0x80 else x << 1) & 0xFF
    exp[255:510] = exp[:255]
    return exp, log

_GF_EXP, _GF_LOG = _build_gf()

# (256, 256) uint8 table: _GF_MUL[a, b] = a * b in GF(2^8)
def _build_mul_table() -> np.ndarray:
    a = np.arange(256, dtype=np.uint16)
    b = np.arange(256, dtype=np.uint16)
    s = (_GF_LOG[a].astype(np.uint16)[:, None] + _GF_LOG[b].astype(np.uint16)[None, :]) % 255
    t = _GF_EXP[s].copy()
    t[0, :] = 0   # 0 * x = 0
    t[:, 0] = 0
    return t

_GF_MUL = _build_mul_table()

# Generator polynomial coefficients (one cache entry per nsym value)
_GEN_CACHE: dict[int, np.ndarray] = {}

def _generator(nsym: int) -> np.ndarray:
    """Return the nsym generator polynomial coefficients (leading 1 dropped)."""
    if nsym in _GEN_CACHE:
        return _GEN_CACHE[nsym]
    g = [1]
    for i in range(nsym):
        ai = int(_GF_EXP[i])
        new_g = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            new_g[j] ^= c
            if c and ai:
                new_g[j + 1] ^= int(_GF_EXP[(int(_GF_LOG[c]) + i) % 255])
        g = new_g
    result = np.array(g[1:], dtype=np.uint8)
    _GEN_CACHE[nsym] = result
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encoded_size(payload_size: int, nsym: int) -> int:
    """Return the byte count after RS encoding *payload_size* bytes with *nsym* ECC symbols."""
    if nsym == 0:
        return payload_size
    block_data = 255 - nsym
    n_blocks = (payload_size + block_data - 1) // block_data
    return n_blocks * 255


def encode(data: bytes, nsym: int, workers: int = 1) -> bytes:
    """RS-encode *data*, returning data+ECC in 255-byte blocks.

    Uses a vectorised NumPy implementation: all N blocks are processed
    simultaneously in k = (255-nsym) Python-loop iterations, regardless
    of file size.  ~20-35x faster than the reedsolo pure-Python encoder.

    Output is byte-for-byte identical to reedsolo, so reedsolo can decode it.
    The *workers* parameter is accepted for API compatibility but ignored —
    NumPy already parallelises across all blocks internally.
    """
    if nsym == 0 or not data:
        return data

    block_data = 255 - nsym
    pad = (block_data - len(data) % block_data) % block_data
    padded = np.frombuffer(data + b"\x00" * pad, dtype=np.uint8)
    N = len(padded) // block_data
    msg = padded.reshape(N, block_data)     # (N, block_data)
    gen = _generator(nsym)                  # (nsym,)

    # Polynomial long division over GF(2^8) for all N blocks simultaneously.
    # Each of the k=block_data iterations updates an (N, nsym) remainder matrix.
    # Source cols 1..nsym-1 are never a destination, so the shift is safe in-place.
    remainder = np.zeros((N, nsym), dtype=np.uint8)
    for i in range(block_data):
        coef = msg[:, i] ^ remainder[:, 0]          # (N,) feedback
        remainder[:, :-1] = remainder[:, 1:]         # shift left
        remainder[:, -1] = 0
        remainder ^= _GF_MUL[coef[:, None], gen[None, :]]   # (N, nsym)

    out = np.empty((N, 255), dtype=np.uint8)
    out[:, :block_data] = msg
    out[:, block_data:] = remainder
    return out.tobytes()


def _syndrome_no_error_mask(blocks: np.ndarray, nsym: int) -> np.ndarray:
    """Return boolean mask — True where the 255-byte block has no RS errors.

    Uses Horner's rule vectorised over all N blocks and all *nsym* roots
    simultaneously.  For valid codewords every syndrome is zero by definition,
    so we can skip Berlekamp-Massey entirely for clean blocks.

    blocks : (N, 255) uint8
    returns: (N,) bool
    """
    if nsym not in _SYNDROME_ROOTS:
        _SYNDROME_ROOTS[nsym] = np.array(
            [int(_GF_EXP[j]) for j in range(nsym)], dtype=np.uint8
        )
    roots = _SYNDROME_ROOTS[nsym]          # (nsym,)

    s = np.zeros((len(blocks), nsym), dtype=np.uint8)
    for coef in blocks.T:                  # 255 numpy iterations
        # s[n, j] = GF_MUL(s[n,j], roots[j]) XOR coef[n]
        s = _GF_MUL[s, roots[np.newaxis, :]] ^ coef[:, np.newaxis]

    return np.all(s == 0, axis=1)          # True ↔ all syndromes zero


# Module-level for ProcessPoolExecutor pickling (Windows/macOS spawn).
def _decode_chunk(args: tuple) -> bytes:
    chunk, nsym = args
    rs = reedsolo.RSCodec(nsym)
    block_total = 255
    block_data = 255 - nsym
    out = bytearray()
    for i in range(0, len(chunk), block_total):
        block = chunk[i : i + block_total]
        if len(block) < block_total:
            break
        try:
            corrected = rs.decode(block)[0]
        except reedsolo.ReedSolomonError:
            corrected = bytearray(block_data)
        out.extend(corrected)
    return bytes(out)


def decode(data: bytes, nsym: int, payload_size: int, workers: int = 1) -> bytes:
    """RS-decode *data*, correcting errors, returning exactly *payload_size* bytes.

    Three-tier decode strategy:

    1. **Syndrome fast-path** — compute syndromes for every block in a single
       vectorised numpy pass (no Python loop per block).  For clean blocks
       (syndrome == 0) the data portion is extracted directly, skipping
       Berlekamp-Massey entirely.  Local roundtrips with no errors complete
       in ~10-50 ms regardless of file size.

    2. **reedsolo per-block correction** — only blocks whose syndromes are
       non-zero run through BM, keeping error-correction overhead proportional
       to the *number of bad blocks* rather than the total file size.

    3. **ProcessPoolExecutor** (workers > 1) — falls back to parallel
       reedsolo when all blocks need correction (e.g. heavily corrupted data).
    """
    if nsym == 0:
        return data[:payload_size]

    block_total = 255
    block_data = 255 - nsym
    arr = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(arr) // block_total
    if n_blocks == 0:
        return b""

    blocks = arr[:n_blocks * block_total].reshape(n_blocks, block_total)

    # --- Stage 1: vectorised syndrome check for all blocks ---
    no_error = _syndrome_no_error_mask(blocks, nsym)

    if no_error.all():
        # Fast path: no errors anywhere — strip ECC symbols and return
        return blocks[:, :block_data].flatten().tobytes()[:payload_size]

    # --- Stage 2: correct only blocks that need it ---
    # Start with the data portion of every block (uncorrected as default).
    out = blocks[:, :block_data].copy()
    bad_idx = np.where(~no_error)[0]

    if workers > 1 and len(bad_idx) > 256:
        # Many bad blocks: parallelise across cores
        bad_data = blocks[bad_idx].flatten().tobytes()
        per = max(255 * 256, (len(bad_data) // workers // 255) * 255)
        chunks = [bad_data[i : i + per] for i in range(0, len(bad_data), per)]
        try:
            with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
                corrected_bytes = b"".join(
                    ex.map(_decode_chunk, [(c, nsym) for c in chunks])
                )
            corrected_arr = np.frombuffer(corrected_bytes, dtype=np.uint8)
            out[bad_idx] = corrected_arr[: len(bad_idx) * block_data].reshape(
                len(bad_idx), block_data
            )
            return out.flatten().tobytes()[:payload_size]
        except Exception:
            pass  # fall through to single-threaded

    rs = reedsolo.RSCodec(nsym)
    for i in bad_idx:
        try:
            corrected = rs.decode(blocks[i].tobytes())[0]
            out[i] = np.frombuffer(bytes(corrected), dtype=np.uint8)[:block_data]
        except reedsolo.ReedSolomonError:
            out[i] = np.zeros(block_data, dtype=np.uint8)

    return out.flatten().tobytes()[:payload_size]


# ---------------------------------------------------------------------------
# Byte interleaving
# ---------------------------------------------------------------------------

def interleave(data: bytes, depth: int) -> bytes:
    """Permute *data* so that consecutive bytes in the output come from positions
    *depth* apart in the input.

    After interleaving, each video frame carries bytes spaced *depth* positions
    apart in the original ECC stream.  A burst of K corrupt frames therefore
    corrupts at most ceil(255/depth) bytes per RS block — converting burst errors
    into near-uniform errors that RS handles efficiently.

    The output length is rounded up to the nearest multiple of *depth* (zero-
    padded); callers trim with *original_len* on the decode side.
    """
    if depth <= 1:
        return data
    arr = np.frombuffer(data, dtype=np.uint8)
    pad = int(math.ceil(len(arr) / depth)) * depth - len(arr)
    if pad:
        arr = np.append(arr, np.zeros(pad, dtype=np.uint8))
    width = len(arr) // depth
    return arr.reshape(depth, width).T.flatten().tobytes()


def deinterleave(data: bytes, depth: int, original_len: int) -> bytes:
    """Reverse of :func:`interleave`.  Returns exactly *original_len* bytes."""
    if depth <= 1:
        return data[:original_len]
    arr = np.frombuffer(data, dtype=np.uint8)
    pad = int(math.ceil(len(arr) / depth)) * depth - len(arr)
    if pad:
        arr = np.append(arr, np.zeros(pad, dtype=np.uint8))
    width = len(arr) // depth
    return arr.reshape(width, depth).T.flatten().tobytes()[:original_len]


def block_error_counts(original: bytes, recovered: bytes, nsym: int) -> list[int]:
    """Return per-RS-block wrong-byte counts (useful for calibration).

    Compares *original* and *recovered* byte-by-byte in 255-byte windows
    (or (255-nsym)-byte data windows when nsym > 0, since that is how the
    decoder slices blocks).  The returned list has one entry per block.
    """
    block_size = (255 - nsym) if nsym > 0 else 255
    counts = []
    n = min(len(original), len(recovered))
    for i in range(0, n, block_size):
        o = original[i: i + block_size]
        r = recovered[i: i + block_size]
        counts.append(sum(a != b for a, b in zip(o, r)))
    return counts
