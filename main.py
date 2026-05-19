#!/usr/bin/env python3
"""Backward-compatible shim — CLI logic lives in pixelvault/_cli.py."""
from pixelvault._cli import main

if __name__ == "__main__":
    main()
