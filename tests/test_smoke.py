"""Fast import-level smoke tests — no ffmpeg required, runs in CI."""
import subprocess
import sys

import pytest


def test_package_importable():
    import pixelvault
    assert pixelvault.__version__


def test_public_api_present():
    import pixelvault
    assert callable(pixelvault.encode_file)
    assert callable(pixelvault.decode_file)


def test_mode_constants():
    from pixelvault import MODE_BINARY, MODE_GRAY4, MODE_PALETTE, MODE_RGB, MODE_RGB_BIN, MODE_NIBBLE
    modes = {MODE_BINARY, MODE_GRAY4, MODE_PALETTE, MODE_RGB, MODE_RGB_BIN, MODE_NIBBLE}
    assert len(modes) == 6, "all mode constants must be distinct"


def test_frame_dimensions():
    from pixelvault import WIDTH, HEIGHT, WIDTH_4K, HEIGHT_4K
    assert WIDTH == 1920 and HEIGHT == 1080
    assert WIDTH_4K == 3840 and HEIGHT_4K == 2160


def test_ecc_encode_decode_roundtrip():
    """RS encode→decode is lossless for a small payload."""
    from pixelvault.ecc import encode as ecc_encode, decode as ecc_decode
    data = b"PixelVault ECC sanity check" * 4
    encoded = ecc_encode(data, nsym=16)
    decoded = ecc_decode(encoded, nsym=16, payload_size=len(data))
    assert decoded == data


def test_ecc_corrects_errors():
    """RS decoder corrects up to nsym/2 byte errors per block."""
    from pixelvault.ecc import encode as ecc_encode, decode as ecc_decode
    data = bytes(range(100))
    encoded = bytearray(ecc_encode(data, nsym=16))
    # corrupt 8 bytes (= nsym/2, within correction capacity)
    for i in range(0, 80, 10):
        encoded[i] ^= 0xFF
    decoded = ecc_decode(bytes(encoded), nsym=16, payload_size=len(data))
    assert decoded == data


def test_ecc_interleave_roundtrip():
    """Interleave → deinterleave is the identity."""
    from pixelvault.ecc import interleave, deinterleave
    data = bytes(range(256))
    interleaved = interleave(data, depth=8)
    recovered = deinterleave(interleaved, depth=8, original_len=len(data))
    assert recovered == data


def test_cli_help_exits_cleanly():
    """CLI --help must exit 0 and mention subcommands."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "encode" in result.stdout
    assert "decode" in result.stdout
