"""
256-color palette for palette encoding mode.

Layout: R=3 bits (8 levels), G=3 bits (8 levels), B=2 bits (4 levels) → 8×8×4 = 256 colors.
Minimum channel step: R/G ≈ 36, B = 85. Large enough that YouTube VP9/H.264 compression
(which introduces ±10–20 per channel) cannot push a color to the wrong palette entry.
"""

import numpy as np

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
    """Map an (N, 3) BGR uint8 array to (N,) palette-index uint8 array.

    Uses vectorised nearest-colour search in Euclidean RGB space.
    """
    p = PALETTE_BGR.astype(np.int32)           # (256, 3)
    x = pixels_bgr.reshape(-1, 3).astype(np.int32)  # (N, 3)
    # (N, 256, 3) → sum of squares → (N, 256) → argmin → (N,)
    diffs = x[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = (diffs * diffs).sum(axis=2)
    return dists.argmin(axis=1).astype(np.uint8)
