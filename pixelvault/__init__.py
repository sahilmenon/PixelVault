__version__ = "1.0.0"

from . import audio, youtube
from .decoder import decode_file
from .encoder import (
    HEIGHT,
    HEIGHT_4K,
    MODE_BINARY,
    MODE_GRAY4,
    MODE_NIBBLE,
    MODE_PALETTE,
    MODE_RGB,
    MODE_RGB_BIN,
    WIDTH,
    WIDTH_4K,
    encode_file,
)

__all__ = [
    "__version__",
    "encode_file", "decode_file",
    "audio", "youtube",
    "MODE_BINARY", "MODE_RGB", "MODE_PALETTE", "MODE_RGB_BIN", "MODE_NIBBLE", "MODE_GRAY4",
    "WIDTH", "HEIGHT", "WIDTH_4K", "HEIGHT_4K",
]
