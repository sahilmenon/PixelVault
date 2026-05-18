"""
Vault directory layout for ByteVault.

  vault/
    input/    ← drop files here to encode
    encoded/  ← encoded MP4s land here
    decoded/  ← recovered files land here

All three are created automatically on first use.
Paths are resolved relative to cwd (normally the project root).
"""

from pathlib import Path

VAULT_ROOT = Path("vault")
VAULT_INPUT = VAULT_ROOT / "input"
VAULT_ENCODED = VAULT_ROOT / "encoded"
VAULT_DECODED = VAULT_ROOT / "decoded"


def ensure_dirs() -> None:
    """Create vault directories if they don't exist."""
    for d in (VAULT_INPUT, VAULT_ENCODED, VAULT_DECODED):
        d.mkdir(parents=True, exist_ok=True)
