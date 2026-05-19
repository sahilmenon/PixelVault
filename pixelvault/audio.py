"""
Audio-track encoder/decoder for PixelVault.

Data is encoded as amplitude-level-modulated PCM samples: each "logical
sample" maps to one of N_LEVELS discrete amplitude values.  The audio track
is stored as ALAC (Apple Lossless) inside the MP4, so reconstruction is
bit-perfect — no lossy-codec tolerance margin is required.

Default parameters (N_LEVELS=4, BLOCK_SIZE=64):
  - 4 levels (2 bits per logical sample), stereo → 4 bits/logical-sample-pair
  - Each logical sample is held for BLOCK_SIZE real samples; middle-50%
    averaging in decode guards against player EQ / minor processing.
  - Stereo: left and right channels carry independent data → ×2 density.
  - At 48 000 Hz, 24 fps: 31 logical samples/ch/frame → 15 B/frame.
"""

import numpy as np

SAMPLE_RATE = 48_000
N_LEVELS = 4     # 2 bits per logical sample; step ≈ 18 667, safe against AAC
BLOCK_SIZE = 64  # real samples per logical sample


def _levels(n: int) -> np.ndarray:
    """Return n evenly-spaced int32 amplitude levels between ±28 000."""
    return np.linspace(-28_000, 28_000, n, dtype=np.float64).astype(np.int32)


def bytes_per_frame(
    fps: int,
    sample_rate: int = SAMPLE_RATE,
    n_levels: int = N_LEVELS,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Return whole bytes of payload encoded per video frame via the audio track."""
    bpl = int(np.log2(n_levels))          # bits per logical sample
    mono_per_frame = sample_rate // fps   # real samples per frame per channel
    logical_per_ch = mono_per_frame // block_size
    return logical_per_ch * 2 * bpl // 8  # stereo, rounded down to whole bytes


def encode(
    data: bytes,
    n_frames: int,
    fps: int,
    sample_rate: int = SAMPLE_RATE,
    n_levels: int = N_LEVELS,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Encode *data* into a stereo int16 PCM array sized for *n_frames* video frames.

    Returns an interleaved (L0, R0, L1, R1, …) int16 array.
    Only the first bytes_per_frame(fps,…)*n_frames bytes of *data* are used;
    any remainder is zero-padded.
    """
    bpl = int(np.log2(n_levels))
    lvls = _levels(n_levels)

    mono_samples = n_frames * (sample_rate // fps)
    logical_per_ch = mono_samples // block_size
    total_logical = logical_per_ch * 2       # L channel then R channel
    total_bits = total_logical * bpl
    total_bytes = total_bits // 8

    padded = (data + b"\x00" * total_bytes)[:total_bytes]
    bits = np.unpackbits(np.frombuffer(padded, dtype=np.uint8))
    if len(bits) < total_bits:
        bits = np.pad(bits, (0, total_bits - len(bits)))
    bits = bits[:total_bits].reshape(total_logical, bpl)

    pows = 1 << np.arange(bpl - 1, -1, -1)
    indices = (bits.astype(np.int32) * pows).sum(axis=1)  # (total_logical,)
    amplitudes = lvls[indices].astype(np.int16)

    left_log = amplitudes[:logical_per_ch]
    right_log = amplitudes[logical_per_ch:]

    # Expand logical samples to real samples; zero-pad tail to match mono_samples exactly
    left_exp = np.zeros(mono_samples, dtype=np.int16)
    right_exp = np.zeros(mono_samples, dtype=np.int16)
    used = logical_per_ch * block_size
    left_exp[:used] = np.repeat(left_log, block_size)
    right_exp[:used] = np.repeat(right_log, block_size)

    pcm = np.empty(mono_samples * 2, dtype=np.int16)
    pcm[0::2] = left_exp
    pcm[1::2] = right_exp
    return pcm


def decode(
    pcm: np.ndarray,
    n_levels: int,
    block_size: int,
    n_bytes: int,
) -> bytes:
    """Decode a stereo int16 PCM array back to *n_bytes* of payload bytes."""
    if n_bytes == 0:
        return b""

    bpl = int(np.log2(n_levels))
    lvls = _levels(n_levels)

    left = pcm[0::2].astype(np.int32)
    right = pcm[1::2].astype(np.int32)

    logical_per_ch = len(left) // block_size

    # Average the middle 50% of each block for robustness against AAC window
    # overlap/add artefacts that corrupt samples near block boundaries.
    quarter = max(1, block_size // 4)
    n_avg = block_size - 2 * quarter  # samples in middle half

    left_blocks  = left[:logical_per_ch * block_size].reshape(logical_per_ch, block_size)
    right_blocks = right[:logical_per_ch * block_size].reshape(logical_per_ch, block_size)
    left_c  = left_blocks[:, quarter: quarter + n_avg].mean(axis=1).astype(np.int32)
    right_c = right_blocks[:, quarter: quarter + n_avg].mean(axis=1).astype(np.int32)

    combined = np.concatenate([left_c, right_c])  # L-then-R, matches encode order

    dists = np.abs(combined[:, None] - lvls[None, :])
    indices = dists.argmin(axis=1).astype(np.uint8)

    bit_groups = np.zeros((len(indices), bpl), dtype=np.uint8)
    for i in range(bpl):
        bit_groups[:, bpl - 1 - i] = (indices >> i) & 1
    bits = bit_groups.flatten()

    n_bits = n_bytes * 8
    bits = bits[:n_bits]
    if len(bits) % 8:
        bits = np.pad(bits, (0, 8 - len(bits) % 8))

    return np.packbits(bits).tobytes()[:n_bytes]
