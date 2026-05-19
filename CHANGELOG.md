# Changelog

All notable changes to PixelVault are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] — 2026-05-19

### Added
- **Six encoding modes**: `binary`, `gray4`, `palette`, `nibble`, `rgb_bin`, `rgb`
- **Reed-Solomon ECC** (GF(2^8), configurable `--ecc NSYM`, default 16 symbols / 255-byte block)
- **Byte interleaving** (`--interleave`) — matrix-transpose permutation converts burst YouTube corruption into uniform errors RS can correct
- **AES-256-GCM encryption** (`--password`) — encrypt-after-compress, before ECC
- **zlib compression** (`--compress`) — applied before encryption and ECC
- **4K encoding** (`--4k`) — 3840×2160 VP9, 4× data density vs 1080p H.264
- **Hardware encoder detection** — probes NVENC/AMF/QSV, falls back to libx264
- **Streaming encoder pipeline** — sliding-window `ThreadPoolExecutor`, bounded RAM usage (~400 MB peak)
- **Three-tier ECC decoder** — syndrome fast-path → per-block Berlekamp-Massey → `ProcessPoolExecutor` for correctable errors
- **Self-describing 128-byte video header** — stores mode, block size, original filename/size; enables auto-detection on decode
- **Chunked encoding** (`--chunk-mb`) — split large files across multiple videos with a `.pvault` manifest
- **YouTube upload/download** — OAuth 2.0 via Google API; `--upload` flag on encode, `download` subcommand
- **Silent AAC audio track** — always muxed (YouTube rejects audio-free uploads)
- **Calibration tool** (`calibrate.py`) — measures your YouTube pipeline's luma error budget
- **Comprehensive benchmark suite** (`scripts/`) — jellyfish 4K reference, ECC stress tests, roundtrip reports

### Changed
- Project renamed from **ByteVault** to **PixelVault**
- Package renamed from `bytevault` to `pixelvault`

[Unreleased]: https://github.com/sahilmenon01/pixelvault/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/sahilmenon01/pixelvault/releases/tag/v1.0.0
