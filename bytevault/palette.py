"""
Color palettes for ByteVault encoding modes.

256-color palette (MODE_PALETTE):
  Layout: R=3 bits, G=3 bits, B=2 bits → 8×8×4 = 256 colors.
  Minimum channel step: R/G ≈ 36, B = 85.

16-color YCbCr palette (MODE_NIBBLE):
  4 luma levels × 4 chroma quadrants, designed in BT.601 YCbCr space.
  Y  ∈ {32, 96, 160, 224} — spacing 64
  Cb ∈ {64, 192}           — spacing 128
  Cr ∈ {64, 192}           — spacing 128
  Minimum distance to any neighbour: 64 (in Y).
  YouTube's typical chroma noise is ±20–30, safely below the ±64 decision
  boundary, so most pixels survive intact and ECC corrects the rest.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 256-colour palette (MODE_PALETTE)
# ---------------------------------------------------------------------------

_R = np.array([round(i * 255 / 7) for i in range(8)], dtype=np.uint8)
_G = np.array([round(i * 255 / 7) for i in range(8)], dtype=np.uint8)
_B = np.array([0, 85, 170, 255], dtype=np.uint8)

# PALETTE[i] = [R, G, B] for byte value i
PALETTE: np.ndarray = np.array(
    [[_R[(i >> 5) & 0x07], _G[(i >> 2) & 0x07], _B[i & 0x03]] for i in range(256)],
    dtype=np.uint8,
)

# BGR version for OpenCV (which uses BGR channel order)
PALETTE_BGR: np.ndarray = PALETTE[:, ::-1].copy()


def bgr_array_to_bytes(pixels_bgr: np.ndarray) -> np.ndarray:
    """Map an (N, 3) BGR uint8 array to (N,) palette-index uint8 array."""
    p = PALETTE_BGR.astype(np.int32)
    x = pixels_bgr.reshape(-1, 3).astype(np.int32)
    diffs = x[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = (diffs * diffs).sum(axis=2)
    return dists.argmin(axis=1).astype(np.uint8)


# ---------------------------------------------------------------------------
# 16-colour YCbCr nibble palette (MODE_NIBBLE)
# ---------------------------------------------------------------------------
# Index layout: bits [3:2] = Y level (0–3), bit [1] = Cb level, bit [0] = Cr level.
# BT.601 full-range YCbCr → RGB:
#   R = Y + 1.402*(Cr-128)
#   G = Y - 0.344136*(Cb-128) - 0.714136*(Cr-128)
#   B = Y + 1.772*(Cb-128)
# All values clamped to [0, 255].

def _ycbcr_to_bgr(Y: int, Cb: int, Cr: int):
    R = max(0, min(255, round(Y + 1.402   * (Cr - 128))))
    G = max(0, min(255, round(Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))))
    B = max(0, min(255, round(Y + 1.772   * (Cb - 128))))
    return (B, G, R)


_Y_LEVELS  = (32,  96, 160, 224)
_CB_LEVELS = (64, 192)
_CR_LEVELS = (64, 192)

# NIBBLE_PALETTE_BGR[nibble] = (B, G, R) — 16 entries
NIBBLE_PALETTE_BGR: np.ndarray = np.array(
    [
        _ycbcr_to_bgr(Y, Cb, Cr)
        for Y in _Y_LEVELS
        for Cb in _CB_LEVELS
        for Cr in _CR_LEVELS
    ],
    dtype=np.uint8,
)  # shape (16, 3)

# YCbCr target points for nearest-neighbour decoding
NIBBLE_PALETTE_YCBCR: np.ndarray = np.array(
    [
        (Y, Cb, Cr)
        for Y in _Y_LEVELS
        for Cb in _CB_LEVELS
        for Cr in _CR_LEVELS
    ],
    dtype=np.float32,
)  # shape (16, 3)


def bgr_array_to_nibbles(pixels_bgr: np.ndarray) -> np.ndarray:
    """Map an (N, 3) BGR uint8 array to (N,) nibble-index (0–15) uint8 array.

    Classification is done in YCbCr space where the palette is equidistant.
    """
    x = pixels_bgr.reshape(-1, 3).astype(np.float32)
    B, G, R = x[:, 0], x[:, 1], x[:, 2]
    Y  =  0.299   * R + 0.587    * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.500 * B + 128.0
    Cr =  0.500   * R - 0.418688 * G - 0.081312 * B + 128.0
    ycbcr = np.stack([Y, Cb, Cr], axis=1)          # (N, 3)
    p = NIBBLE_PALETTE_YCBCR                        # (16, 3)
    diffs = ycbcr[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = (diffs * diffs).sum(axis=2)             # (N, 16)
    return dists.argmin(axis=1).astype(np.uint8)
