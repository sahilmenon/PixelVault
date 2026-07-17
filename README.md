# PixelVault

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![ffmpeg](https://img.shields.io/badge/requires-ffmpeg-orange)

**Encode any file into a YouTube video and recover it perfectly later — using YouTube as infinite cloud storage.**

> Inspired by [Infinite-Storage-Glitch](https://github.com/DvorakDwarf/Infinite-Storage-Glitch) and [YouTubeDrive](https://github.com/dzhang314/YouTubeDrive). A proof-of-concept showcase of compression-resistant binary encoding. For educational and personal use; respect YouTube's Terms of Service.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [The Enemy: YouTube's Re-Encoder](#the-enemy-youtubes-re-encoder)
  - [The Core Idea: Logical Pixels as Uniform Blocks](#the-core-idea-logical-pixels-as-uniform-blocks)
  - [Why Binary Mode Is the Only Viable YouTube Option](#why-binary-mode-is-the-only-viable-youtube-option)
  - [The Full Encode Pipeline](#the-full-encode-pipeline)
  - [The Full Decode Pipeline](#the-full-decode-pipeline)
  - [How the Layers Interact](#how-the-layers-interact)
  - [The NumPy Vectorisation Strategy](#the-numpy-vectorisation-strategy)
  - [The Three-Tier RS Decode Strategy](#the-three-tier-rs-decode-strategy)
- [Encoding Modes](#encoding-modes)
- [Calibration](#calibration)
- [Installation](#installation)
- [YouTube API Setup](#youtube-api-setup)
- [Full CLI Reference](#full-cli-reference)
- [Self-Describing Videos](#self-describing-videos)
- [Limitations & Notes](#limitations--notes)
- [Architecture](#architecture)
- [Benchmarks & Comparison](#benchmarks--comparison)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

**Prerequisites:** Python 3.10+, [ffmpeg](https://ffmpeg.org/download.html) on PATH.

```bash
# Install Python dependencies
pip install -r requirements.txt

# Encode a file → video (1080p, ECC on by default)
python main.py encode secret.zip

# Encode at 4K for 4× data density
python main.py encode secret.zip --4k

# Decode a local video → original file
python main.py decode vault/encoded/secret.zip.mp4
```

For YouTube upload/download, set up OAuth credentials first — see [YouTube API Setup](#youtube-api-setup).

```bash
# Encode and upload to YouTube
python main.py encode secret.zip --upload
# → YouTube ID: dQw4w9WgXcQ

# Decode directly from YouTube
python main.py decode yt:dQw4w9WgXcQ

# 4K roundtrip (4× density, ~10–20 min YouTube processing)
python main.py encode secret.zip --4k --upload
python main.py decode yt:dQw4w9WgXcQ --4k
```

### Recommended YouTube workflow
```bash
# 1. Calibrate once per account (measures actual error rates for your region)
python calibrate.py --mode binary

# 2. Encode + upload (ECC nsym=16 on by default)
python main.py encode secret.zip --upload         # 1080p: ~64,800 B/frame
python main.py encode secret.zip --4k --upload    # 4K:   ~259,200 B/frame (4× density)

# 3. Large files: split into chunks across multiple videos
python main.py encode big_archive.tar.gz --chunk-mb 512 --upload
# Saves a manifest; decode reassembles automatically:
python main.py decode vault/encoded/big_archive.pvault
```

---

## How It Works

### The Enemy: YouTube's Re-Encoder

Every upload to YouTube is re-transcoded. The video track is re-encoded as H.264 (1080p, ~4–8 Mbps) or VP9 (4K, ~35–45 Mbps). The audio track is re-encoded as AAC. Both are lossy.

The critical constraint is **DCT compression**. H.264 and VP9 divide frames into 4×4 or 8×8 blocks and apply the Discrete Cosine Transform to each, then quantise the coefficients. This introduces **ringing artefacts** — oscillations of ±44–67 luma units near hard edges (like the boundary between a black pixel block and a white one). A pixel that was encoded as 85 can come back as anywhere between 18 and 152. This is a fundamental property of block-DCT codecs and cannot be avoided at any reasonable bitrate.

PixelVault's entire encoding strategy is a response to this constraint.

### The Core Idea: Logical Pixels as Uniform Blocks

Instead of using 1×1 pixels, PixelVault encodes each logical data pixel as an **N×N uniform square** (`block_size=N`). The block interior is a single flat colour. When H.264 applies its DCT, ringing only affects the *edges* of the square — the *centre* of a large uniform region remains accurate.

The decoder reads only the block centre (or averages a few nearby centre samples for large blocks) and ignores the edges entirely. As long as the centre is far enough from any decision boundary, the correct bit is recovered.

This is why the default block size for YouTube uploads is 2×2: at 1080p, a 2×2 block has a 1×1 pixel centre, which sits roughly 1 pixel away from the nearest edge. At block_size=1, there is no interior — every pixel is an edge — so DCT ringing corrupts too many bits for practical use at 1080p bitrates.

At 4K, YouTube allocates roughly 4× more bits per pixel (via VP9), so 2×2 blocks at 4K are as robust as — or more robust than — 2×2 blocks at 1080p, while delivering 4× the data per frame.

### Why Binary Mode Is the Only Viable YouTube Option

The math: H.264's DCT ringing at block_size=2 introduces ±44–67 luma units of error. Binary mode encodes `0` as luma=0 and `1` as luma=255. The decision threshold is at 127. Even a worst-case ±67 error cannot move a 0 above 67 or a 255 below 188 — both stay on the correct side of 127. **The ±127 margin absorbs the full DCT ringing budget.**

Any multi-level scheme (gray4, nibble, palette) needs smaller decision margins. Gray4 uses four luma levels spaced 85 apart with thresholds at 42, 127, and 213 — but the ±44–67 ringing exceeds the ±42 margin of the lowest threshold, causing misclassifications. To recover the full ±42 margin, you'd need block_size≥4, which halves your data density per frame. Nibble and palette modes fail on chroma: YouTube's YUV420p chroma subsampling and quantisation introduce ±20–30 Cb/Cr noise, which exceeds any practical palette margin for 4+ bits per pixel. In practice, all multi-level modes fail on YouTube and are local-only.

**The fundamental ceiling:** binary mode achieves the highest data density per frame for any YouTube-safe configuration. Multi-level schemes can only match it by using larger blocks, giving the same density or worse. Binary is simultaneously the simplest and the optimal encoding for YouTube.

### The Full Encode Pipeline

```
File bytes
    │
    ├─ [Optional] zlib compress (level 6)
    │     Skip silently if compressed > original (e.g. already-compressed files like JPEG or ZIP)
    │
    ├─ [Optional] AES-256-GCM encrypt
    │     PBKDF2-SHA256 key derivation (200k iterations), random 12-byte nonce
    │     Nonce stored in the video header
    │
    ├─ Reed-Solomon ECC encode (default nsym=16)
    │     GF(2^8) polynomial codes, 255-byte blocks
    │     Vectorised NumPy: all N blocks processed in a single matrix operation
    │     ~20–35× faster than the reedsolo pure-Python encoder
    │
    ├─ [Optional] Byte interleave (depth = estimated frame count)
    │     Permutes the ECC byte stream so consecutive video frames
    │     carry bytes from positions D apart in the ECC stream.
    │     Burst of K corrupt frames → ≤ ceil(255/D) errors/RS block
    │     Must happen AFTER ECC so RS blocks span the full stream
    │
    ├─ Split: header (128 bytes) + video payload [+ optional audio tail]
    │
    ▼
128-byte self-describing header
    ├─ Magic bytes, mode, block size, filename, file size, flags,
    │  ECC nsym, interleave depth, encryption type, nonce
    └─ Written as the very first pixels of the first frame
    
Video frames (NumPy → ffmpeg pipe)
    ├─ Each byte → N×N uniform coloured block
    ├─ Batches generated in parallel (ThreadPoolExecutor)
    ├─ Streamed to ffmpeg stdin — no full materialisation in RAM
    ├─ ffmpeg encodes: H.264 NVENC/AMF/QSV (hw) or libx264 (sw)
    │     binary/rgb_bin: CRF=18 (ECC covers residual error)
    │     palette/nibble/rgb: CRF=0 lossless (palette must survive exactly)
    ├─ Silent AAC audio track always muxed (YouTube requires an audio stream)
    └─ [Optional] ALAC lossless audio track (data channel, local use only)
```

### The Full Decode Pipeline

```
MP4 (local file or yt:VIDEO_ID)
    │
    ├─ [If YouTube] yt-dlp download (with retry logic for 4K processing delay)
    │
    ├─ ffprobe: read resolution, frame count, fps
    │
    ├─ ffmpeg → bgr24 raw frames via stdout pipe (streaming, no temp dir)
    │
    ├─ Auto-detection from first frame
    │     Try each (mode, block_size) combination from a fixed priority list
    │     Read 128 bytes of header pixels from the first frame
    │     If magic bytes match AND header's stored mode/block_size match what was tried
    │     → confirmed. Otherwise try next combination.
    │     Detection takes one frame; no extra ffmpeg pass needed.
    │
    ├─ Parse header → learn: filename, file size, ECC nsym, interleave depth,
    │     encryption type, nonce, compression flag, audio bytes
    │
    ├─ Collect exactly the bytes needed (stop reading frames early)
    │     ThreadPoolExecutor with sliding window: n_workers×2 frames in flight
    │     Decoding tasks overlap I/O from the ffmpeg pipe
    │
    ├─ [Optional] Deinterleave (reverses the encode permutation)
    │
    ├─ [Optional] Append audio-track bytes (for BVI\x02 files)
    │
    ├─ Reed-Solomon ECC decode
    │     Stage 1: vectorised syndrome check on all N blocks simultaneously
    │       If all syndromes zero → data is clean, extract without BM
    │       Local roundtrips with no errors complete in ~10–50 ms
    │     Stage 2: run Berlekamp-Massey only on blocks with non-zero syndromes
    │       Proportional to number of bad blocks, not total file size
    │     Stage 3: if >256 bad blocks, parallelize via ProcessPoolExecutor
    │
    ├─ [Optional] AES-256-GCM decrypt (authenticated — wrong password → error)
    │
    └─ [Optional] zlib decompress → original file bytes
```

### How the Layers Interact

The four transforms — compression, encryption, ECC, interleaving — must happen in a specific order, and the order is not arbitrary:

**Compression before ECC.** Compressed data is smaller, so it generates fewer RS blocks and less ECC overhead. Encrypting first or ECC'ing first would inflate the payload before compression, wasting space. zlib is skipped silently if it doesn't help (e.g., already-compressed inputs like JPEG or ZIP).

**Encryption before ECC.** ECC must protect the ciphertext (the thing that will actually be stored in the video), not the plaintext. Encrypting after ECC would mean RS blocks span plaintext and the correction would reconstruct plaintext — but we need to correct whatever bytes came out of the frame pixels, which is ciphertext. Doing ECC on plaintext and encryption on the result would also require a separate corruption-recovery mechanism. So: encrypt first, ECC second.

**ECC before interleaving.** Interleaving only makes sense on ECC data, not raw payload. The RS encoder produces 255-byte codewords where each codeword's ECC symbols protect the data bytes within the same block. Interleaving then permutes the *entire* ECC stream across frames, so a burst of corrupt frames scatters errors across many different RS blocks rather than destroying one block entirely. If you interleaved raw bytes before RS encoding, the RS blocks would be incoherent — each block would contain bytes from many different parts of the file and the ECC relationship would be meaningless.

**Interleaving depth = estimated frame count.** The interleave depth is set to the number of data frames in the video. This ensures that each frame carries exactly one byte from each RS block (one byte per 255-byte block per frame). A single corrupt frame then adds exactly one error to each RS block it touches — the maximum the ECC can distribute.

**The header is stored in the first frame's pixels.** This matters because the decoder reads the header to know *how many bytes to collect* from subsequent frames. The header must come first, before any payload, so the decoder never has to read further than it needs to. The 128-byte header fits easily in the first frame at any supported mode and block size: at 1080p binary bs=2, one frame carries 64,800 bytes — far more than 128. The header's magic bytes (`BVI\x01` or `BVI\x02`) plus the stored mode and block_size values allow the decoder to auto-detect parameters with no external hints.

**The silent audio track.** YouTube's ingest pipeline refuses videos without an audio stream. PixelVault always muxes a silent 32 kbps AAC track alongside the video data. This track carries no payload data. When `--audio` is used, the data portion of the audio is stored in a *separate* ALAC lossless track — but since YouTube re-encodes audio to AAC (lossy), the ALAC data channel only survives local roundtrips. For YouTube uploads, all payload must go through the video track.

**CRF=18 vs CRF=0.** Binary and rgb_bin modes use CRF=18 (high quality, lossy) for the local H.264 encode. This is correct because ECC will fix any residual pixel errors the local encode introduces, and the mode's ±127 luma margin absorbs the small additional noise. Palette, nibble, and RGB modes use CRF=0 (lossless). These modes depend on exact colour values for correct classification — a ±1 channel error from a lossy local encode would corrupt the palette lookup, and the ECC margin is too tight to tolerate two rounds of lossy encoding (local + YouTube). CRF=0 ensures that only YouTube's single re-encode introduces colour error.

### The NumPy Vectorisation Strategy

Frame generation is the performance bottleneck. For a 50 MB file in binary mode at 1080p, that's ~770 frames × 1920×1080×3 bytes = ~4.5 GB of raw pixel data to generate and pipe to ffmpeg.

PixelVault avoids a Python loop over frames by treating all N frames in a batch as a single 4D NumPy array `(frames, height, width, channels)`. The byte→pixel expansion is:

```python
# payload_arr is the raw byte array (e.g. 50 MB of ECC-encoded data)
raw = np.unpackbits(payload_arr[b_start:b_end])        # flat bit array
logical = raw.reshape(frames, lh, lw)                   # (frames, lh, lw) bits
scaled = np.repeat(np.repeat(logical[..., None] * 255,  # expand N×N blocks
                              bs, axis=1), bs, axis=2)
```

NumPy releases the GIL for all C-level operations, so threads in the `ThreadPoolExecutor` run in true parallel. A sliding window of futures keeps at most `n_workers × batch` frames in RAM at any time, bounding peak memory to ~400 MB regardless of file size. While one batch is being written to the ffmpeg pipe, the next batch is being computed in the thread pool — overlapping CPU and I/O.

### The Three-Tier RS Decode Strategy

Reed-Solomon correction speed matters because a 50 MB file produces ~6,700 RS blocks to check. Berlekamp-Massey (the standard RS error locator algorithm) is slow in pure Python.

**Stage 1 — vectorised syndrome check.** For every valid RS codeword, all `nsym` syndromes are zero by definition. PixelVault computes syndromes for all N blocks simultaneously using Horner's rule as a numpy matrix operation (255 iterations over the coefficient axis, updating an `(N, nsym)` syndrome matrix). Blocks where all syndromes are zero are clean — no BM needed. For a local roundtrip with no corruption, this covers 100% of blocks and the entire decode completes in ~10–50 ms.

**Stage 2 — per-block BM for bad blocks only.** If some syndromes are non-zero, only *those* blocks go to the full BM algorithm via `reedsolo`. Because YouTube typically corrupts a small fraction of blocks (burst artefacts), most blocks still pass the syndrome check and skip BM.

**Stage 3 — ProcessPoolExecutor.** If the file is heavily corrupted (>256 bad blocks), BM is parallelised across CPU cores. Each process gets a chunk of bad blocks and runs reedsolo independently. This path is rare on normal YouTube roundtrips.

---

## Encoding Modes

### Video track

| Mode | Bits/logical px | Default block | Bytes/frame (1080p) | Bytes/frame (4K) | YouTube resistance |
|------|----------------|---------------|---------------------|------------------|--------------------|
| `binary` | 1 bit | 2×2 px | ~64,800 B | ~259,200 B | ★★★ Best — 0/255 luma threshold ±127, absorbs DCT ringing |
| `gray4` | 2 bits | 2×2 px | ~129,600 B (local) | — | ✗ block_size=2 local only — DCT ringing ±44–67 exceeds ±42 threshold |
| `rgb_bin` | 3 bits | 2×2 px | ~194,400 B | — | ✗ Local only — untested on YouTube |
| `nibble` | 4 bits | 4×4 px | ~64,800 B | — | ✗ Local only — chroma compression exceeds ±64 Cb/Cr margin |
| `palette` | 8 bits | 4×4 px | ~129,600 B | — | ✗ Local only — chroma compression exceeds ±18-channel margin |
| `rgb` | 24 bits | 2×2 px | — | — | ✗ Local only — YUV rounding corrupts header |

> **For YouTube uploads**, use `binary` only. It is the maximum achievable density for YouTube: H.264's 4×4 DCT ringing at block_size=2 is ±44–67 luma units, so only a ±127 threshold (binary) survives it. Any multi-level scheme needs block_size≥4 to avoid ringing, but that halves or worse the density — making it less useful than binary. All chroma modes fail because chroma quantization errors exceed any feasible palette margin.
>
> **4K mode (`--4k`)** uploads at 3840×2160. YouTube re-encodes 4K as VP9 at ~35–45 Mbps (vs ~4–8 Mbps for 1080p H.264), giving 4× the data per frame — **~259,200 bytes/frame** vs 64,800 at 1080p. Binary's ±127 luma threshold comfortably survives VP9 at these bitrates. Use `--4k` on both encode and decode. Note: YouTube takes ~10–20 minutes to process 4K (vs ~5 min for 1080p).

### Mode design rationale

**`gray4`** uses luma levels 0, 85, 170, 255 — the widest possible 4-level spacing in [0,255]. Decision thresholds sit at 42, 127, 213 (midpoints). At block_size=2, DCT ringing of ±44–67 exceeds the ±42 margin, causing errors at the 0/85 boundary. At block_size=4, the wider block interior absorbs the ringing and `gray4` becomes reliable — but at 4× block area, density is the same as binary at block_size=2. It's a wash for YouTube, with the added complication of a more fragile threshold.

**`nibble`** designs its 16 colours in YCbCr space rather than RGB. YouTube quantises luma and chroma independently (YUV420p); a palette designed in RGB space would have irregular, unpredictable margins in YCbCr. By placing the 16 colours at Y ∈ {32,96,160,224} and Cb/Cr ∈ {64,192} — a 64-unit Y spacing and 128-unit Cb/Cr spacing — the nearest-neighbour decision boundaries are at ±32 in Y and ±64 in Cb/Cr. YouTube's chroma noise of ±20–30 should sit inside the ±64 margin, but in practice YUV420p's 4:2:0 chroma averaging (each Cb/Cr sample covers a 2×2 pixel area) causes additional quantisation at block boundaries that exceeds the margin. ECC is required; `--ecc 32` recommended.

**`palette`** encodes 8 bits per pixel as one of 256 RGB colours with minimum channel steps of 36 (R/G) and 85 (B). The intent was that YouTube's ±10–20 channel noise would stay below the 36/85 decision boundaries. In practice, YUV420p's chroma subsampling also affects this mode — the ±18-channel boundary is exceeded by blended chroma at block edges, making palette local-only.

**`rgb`** stores 3 bytes per logical pixel with no palette constraint. It fails on YouTube because the very first bytes (the header magic `BVI`) land in pixels where RGB→YUV→RGB rounding introduces ±1–2 channel errors, which corrupts the magic-byte check. Even if the header survived, 24-bit colour space has no margin at all.

### Audio track (optional — `--audio`, local use only)

| Parameter | Value |
|-----------|-------|
| Sample rate | 48 000 Hz |
| Levels | 4 (2 bits/sample) |
| Channels | Stereo (independent L/R) |
| Codec | ALAC (lossless) |
| Yield | ~15 B/frame at 30 fps |

The audio data channel encodes payload bytes as amplitude levels in PCM. ALAC stores them losslessly — so a local roundtrip through ffmpeg recovers every bit. YouTube's AAC re-encoder is not amplitude-preserving; omit `--audio` for YouTube uploads.

The audio track adds a modest number of bytes per frame (the bit budget is just stereo PCM bandwidth ÷ fps), but it does reduce total frame count for small files by spreading the last few kilobytes across both tracks simultaneously. For large files it's negligible.

### Payload compression (optional — `--compress`)

zlib-compress (level 6) before encoding. If the compressed size is larger (e.g. JPEG, ZIP), it is skipped silently. Decoding is automatic.

Compression happens before ECC so the ECC overhead is proportional to the compressed size. For a text file that compresses 3:1, this cuts ECC overhead by 67% and reduces the number of video frames by the same ratio.

### Reed-Solomon error correction (optional — `--ecc`)

Payload is split into 255-byte blocks; each block gains `nsym` ECC bytes that allow correcting up to `nsym/2` byte errors within that block. This is GF(2^8) Reed-Solomon, the same construction used in QR codes and CDs.

| `--ecc NSYM` | Overhead | Errors correctable per block |
|---|---|---|
| `--ecc 8`  | ~3.2%  | up to 4  |
| `--ecc 16` | ~6.7%  | up to 8  **(default)** |
| `--ecc 32` | ~14.4% | up to 16 |
| `--ecc 64` | ~33.5% | up to 32 |
| `--ecc 0`  | 0%     | disabled |

The `nsym` value is stored in the header — decoding is automatic.

255-byte blocks are the maximum for GF(2^8) RS codes (GF(2^8) has 256 elements; one is reserved for zero, leaving 255 non-zero elements for codeword positions). Shorter blocks would waste ECC capacity; longer blocks would require a larger field. PixelVault's vectorised encoder processes all N blocks in a single loop of `(255 - nsym)` iterations — constant in the number of blocks, which is what makes it fast regardless of file size.

### Byte interleaving (optional — `--interleave`, requires `--ecc`)

YouTube's re-encoder doesn't distribute corruption uniformly — it produces **burst artefacts**: contiguous regions of frames where the quantiser is starved of bits and many pixels in a row are wrong. A burst of 30 wrong bytes in a row, without interleaving, can land entirely in one 255-byte RS block and exhaust its correction capacity (nsym=32 corrects 16 errors; 30 > 16).

Interleaving permutes the ECC byte stream using a stride of `depth` (set to the estimated number of data frames). The interleave is a simple matrix transpose: arrange the bytes into a matrix of `depth` rows, then read column-by-column. Each video frame then carries bytes from positions `depth` apart in the original ECC stream. Frame 0 carries bytes 0, depth, 2×depth, …; frame 1 carries bytes 1, 1+depth, 1+2×depth, …

After interleaving, a burst of K corrupt frames contributes at most `ceil(255/depth)` errors per RS block — roughly 1 error per block when depth ≈ number of frames. RS can correct that with nsym=8, compared to potentially exhausting nsym=64 without interleaving.

The tradeoff: interleaving adds padding to round the stream to a multiple of `depth`, and the depth value must be stored in the header. Decode deinterleaves before ECC correction.

```
without --interleave: burst of 30 corrupt bytes → 30 errors in one RS block → uncorrectable at nsym=32
with    --interleave: same burst → ~1 error per RS block → trivially corrected at nsym=8
```

---

## Calibration

Measure the actual error rates your account/region produces before uploading important files:

```bash
python calibrate.py --mode binary
python calibrate.py --mode nibble
python calibrate.py --mode palette
```

This generates a 512 KB pseudo-random test pattern, uploads it, downloads it, and prints personalised `--ecc` recommendations.

| Option | Default | Description |
|--------|---------|-------------|
| `--mode {binary,rgb_bin,palette,nibble}` | `binary` | Mode to calibrate |
| `--block-size N` | mode default | Block size in pixels |
| `--size BYTES` | 524288 (512 KB) | Test payload size |
| `--wait N` | 300 | Seconds to wait for YouTube processing |
| `--retries N` | 6 | Download retry attempts |
| `--no-hw` | off | Force libx264 |
| `--4k` | off | Calibrate at 4K (3840×2160) |

---

## Installation

### 1. System dependency — ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

> Python 3.10+ required.

---

## YouTube API Setup

To use `--upload` or the `upload` subcommand, you need Google OAuth 2.0 credentials.

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. Enable **YouTube Data API v3**: APIs & Services → Library → search → Enable.
4. Create credentials: APIs & Services → Credentials → **Create Credentials → OAuth client ID**.
   - Application type: **Desktop app**
5. Click **Download JSON** → save as `client_secret.json` in the project directory.

On first upload, a browser window opens for Google sign-in. After approval, `.youtube_token.json` is saved and reused on future runs.

**Both files are gitignored and must never be committed:**
```
client_secret.json      ← your Google OAuth client credentials
.youtube_token.json     ← auto-generated access token
```

PixelVault searches for `client_secret.json` in the current directory first, then `~/.pixelvault/`.

---

## Full CLI Reference

### `encode`

```
python main.py encode <file> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode {binary,rgb,rgb_bin,palette,nibble}` | `binary` | Encoding mode |
| `--block-size N` | auto | Pixels per logical-pixel edge |
| `--output PATH` / `-o` | `vault/encoded/<stem>.mp4` | Output MP4 path |
| `--compress` | off | zlib-compress the payload |
| `--ecc [NSYM]` | **16** | Reed-Solomon ECC symbols per block. Use `--ecc 0` to disable. Required for `nibble` mode. |
| `--interleave` | off | Spread ECC bytes across frames to convert burst errors (requires `--ecc`) |
| `--audio` | off | Encode payload tail in the audio track (local use only) |
| `--workers N` | auto | Frame-generation threads |
| `--no-hw` | off | Force libx264; disable NVENC/AMF/QSV |
| `--4k` | off | Encode at 3840×2160 (4× data/frame). YouTube re-encodes as VP9. Use `--4k` on decode too. |
| `--quiet` / `-q` | off | Suppress progress output |
| `--password PASS` | off | Encrypt the payload with AES-256-GCM (PBKDF2-SHA256 key derivation). Must supply the same password on `decode`. |
| `--chunk-mb MB` | off | Split into MB-sized chunks; writes a `.pvault` manifest. Combine with `--upload` for multi-video YouTube storage. |
| `--upload` | off | Upload to YouTube after encoding |
| `--title TEXT` | filename | YouTube video title |
| `--description TEXT` | `"Encoded with PixelVault"` | YouTube video description |
| `--privacy {public,unlisted,private}` | `unlisted` | YouTube privacy setting |

**Examples:**
```bash
# Binary 1080p (YouTube-safe), ECC nsym=16 on by default
python main.py encode secret.zip --upload

# 4K binary — 4× data density
python main.py encode secret.zip --4k --upload

# Maximum robustness for large files (YouTube)
python main.py encode archive.tar.gz --ecc 32 --interleave --upload

# Maximum robustness at 4K
python main.py encode archive.tar.gz --4k --ecc 32 --interleave --upload

# Split a 10 GB file across multiple 500 MB YouTube videos
python main.py encode huge_backup.tar.gz --chunk-mb 500 --upload
# → writes huge_backup.pvault (manifest with all video IDs)
# → python main.py decode vault/encoded/huge_backup.pvault

# Local-only, maximum density
python main.py encode archive.tar.gz --mode rgb_bin --audio --compress

# Local-only, nibble or palette
python main.py encode archive.tar.gz --mode nibble --ecc 32
```

---

### `decode`

```
python main.py decode <source> [options]
```

`<source>` accepts a local `.mp4` path, a YouTube URL, or `yt:VIDEO_ID`.

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir PATH` / `-o` | `vault/decoded/` | Directory to write the recovered file |
| `--4k` | off | Download 4K from YouTube. Required when the video was encoded with `--4k`. |
| `--workers N` | auto | Frame-decoding threads (0 = all CPU cores). |
| `--quiet` / `-q` | off | Suppress progress output |
| `--password PASS` | off | Decryption password (required if the video was encoded with `--password`). |

All encoding parameters are **auto-detected** from the video header — no flags needed on decode.

```bash
python main.py decode vault/encoded/secret.zip.mp4
python main.py decode yt:dQw4w9WgXcQ
python main.py decode yt:dQw4w9WgXcQ --4k        # for 4K-encoded videos
python main.py decode yt:dQw4w9WgXcQ -o ./recovered
```

---

### `upload`

Upload an already-encoded video without re-encoding.

```
python main.py upload <video.mp4> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--title TEXT` | filename stem | YouTube title |
| `--privacy {public,unlisted,private}` | `unlisted` | Privacy setting |

---

### `download`

Download a YouTube video without decoding.

```
python main.py download <url> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` / `-o` | `vault/encoded/downloaded.mp4` | Output file path |
| `--4k` | off | Download 4K resolution (required for videos encoded with `--4k`). |

---

## Self-Describing Videos

PixelVault videos are **self-describing**: all encoding parameters are stored in the first 128 bytes of the first video frame, so `decode` never needs extra flags.

The header is placed at the very start of the pixel stream — before any payload data — because the decoder needs it to know how many frames to read. Encoding parameters (mode, block_size) are not derivable from the video container, so the decoder tries each plausible `(mode, block_size)` combination in priority order, decoding 128 bytes from the first frame and checking for the magic bytes `BVI\x01` or `BVI\x02`. The first match is used. This auto-detection takes one frame and imposes no extra decode cost for correctly-formed videos.

### Header layout (128 bytes)

```
Offset  Size  Field
0       4     Magic bytes: BVI\x01 (video-only) or BVI\x02 (audio-extended)
4       1     Mode (0=binary, 1=rgb, 2=palette, 3=rgb_bin, 4=nibble, 5=gray4)
5       1     Block size
6       2     Filename length
8       64    Original filename (UTF-8)
72      8     Original file size (bytes)
80      4     Video payload padding bytes
84      4     Audio payload bytes (0 if no audio)
88      1     Audio n_levels (0 if no audio)
89      1     Audio block size (0 if no audio)
90      1     Flags: bit 0 = zlib-compressed, bit 1 = AES-256-GCM encrypted
91      8     Compressed/encrypted payload size (0 if neither)
99      1     ECC nsym (0 = no ECC)
100     4     Interleave depth (0 = no interleaving)
104     1     Encryption type (0=none, 1=AES-256-GCM)
105     12    AES-GCM nonce (zeros if not encrypted)
117     11    Reserved (zeros)
```

---

## Limitations & Notes

| Concern | Details |
|---------|---------|
| **YouTube audio channel** | `--audio` uses ALAC (lossless) locally. YouTube re-encodes audio to AAC, which corrupts the amplitude-level data — omit `--audio` for YouTube uploads. |
| **Nibble mode** | ECC is required and on by default (nsym=16). For extra safety use `--ecc 32`. YouTube's chroma compression introduces ~5–10% pixel misclassifications which ECC corrects. |
| **rgb_bin block size** | Must be ≥ 2. With block=1, YUV420p averages adjacent 2×2 chroma groups, causing bit errors. |
| **rgb mode** | Does not survive YouTube (YUV420p chroma rounding corrupts the header). |
| **Resolution** | Default 1920×1080. Add `--4k` for 3840×2160 (4× data density). The decoder must use the same `--4k` flag when downloading — resolution is not stored in the header but is visible from the video dimensions. YouTube may serve lower resolutions if processing is incomplete — the downloader retries automatically. 4K processing takes ~10–20 min. |
| **Minimum duration** | Videos are padded to at least 5 seconds so YouTube's pipeline accepts them. |
| **Silent audio track** | A silent AAC track is always muxed in. YouTube requires an audio stream. |
| **Hardware encoding** | NVENC/AMF/QSV is used automatically for `binary` and `rgb_bin` modes. `nibble`, `palette`, and `rgb` use libx264 CRF=0 (lossless local encode) so only YouTube's single re-encode introduces chroma error. Use `--no-hw` to force libx264 for binary/rgb_bin too. |
| **File size** | No hard limit. Use `--chunk-mb N` to split large files across multiple YouTube videos automatically — each chunk is encoded as an independent MP4 and a `.pvault` manifest is saved for one-command reassembly on decode. |
| **Terms of Service** | Automated uploads at scale may violate YouTube's ToS. Use for personal archival, not bulk/commercial uploading. |

---

## Architecture

```
PixelVault-Infinite--The-Eternal-Encoder/
├── pixelvault/
│   ├── __init__.py       exports
│   ├── palette.py        256-colour palette + 16-colour YCbCr nibble palette
│   ├── audio.py          amplitude-level PCM encoder/decoder (ALAC, local use)
│   ├── ecc.py            Reed-Solomon ECC encode/decode + byte interleaving
│   ├── encoder.py        file → [compress] → [ECC] → [interleave] → BGR frames → ffmpeg → MP4
│   ├── decoder.py        MP4 → ffmpeg → BGR frames → [deinterleave] → [ECC] → [decompress] → file
│   └── youtube.py        YouTube Data API v3 upload + yt-dlp download
├── main.py               CLI: encode / decode / upload / download
├── calibrate.py          measure YouTube error rates, recommend ECC settings
├── client_secret.json    OAuth client credentials (gitignored — add your own)
├── .youtube_token.json   OAuth token cache (gitignored — auto-generated)
├── requirements.txt
└── README.md
```

---

## Benchmarks & Comparison

The table below surveys ten known file-to-video encoders. All density figures are calculated from the project source or documentation; "YouTube-safe" means the authors verified lossless roundtrip through YouTube's re-encoder.

PixelVault's position in this landscape:

- **Only actively maintained Python tool that completes a roundtrip.** YouBit — the most-cited prior Python tool — crashes on any modern Python 3.10+ due to a `signal`-in-thread constraint. PixelVault is the only Python tool confirmed working end-to-end today.
- **38× faster encode than the only other working Python tool** (file2video): 7.3 s vs 285.6 s on a 15 MB file, measured with libx264 on an i7-1065G7 — see [benchmark methodology](#benchmark-methodology) below.
- **Highest DCT-safe data density at 4K:** 259,200 bytes/frame using 2×2 blocks, matching 1×1-block tools at 1080p but with 4× VP9 bitrate headroom.
- **Only tool with burst-error interleaving**, which converts YouTube's clustered corruption into near-uniform errors that Reed-Solomon handles efficiently.
- **Broadest feature set:** the only tool with all of configurable RS ECC, burst-error interleaving, 4K, hardware encoding, multi-threaded encode/decode, compression, AES-256-GCM encryption, audio-track encoding, self-describing header, YouTube API upload, and a calibration tool. No competitor has more than six of these twelve.

### Quick comparison

| Project | Lang | Stars | YouTube-safe | ECC | Best YouTube-safe density (1080p) | Active |
|---|---|---|---|---|---|---|
| **PixelVault** (this) | Python | — | ✅ binary, gray4¹ | RS nsym 8–64 + interleave | **64,800 B/fr** (bs=2) · **259,200 B/fr** (bs=1) | ✅ 2025–26 |
| [ISG](https://github.com/DvorakDwarf/Infinite-Storage-Glitch) | Rust | ~7,300 | ✅ binary only | None | 64,800 B/fr (2×2 blocks) | ❌ archived Mar 2023 |
| [YouBit](https://github.com/flonle/youbit) | Python/Cython | ~680 | ✅ BPP=1 (BPP=2 marginal) | RS (creedsolo) | **259,200 B/fr** (BPP=1) | ⚠️ last Oct 2022 |
| [YouTubeDrive](https://github.com/dzhang314/YouTubeDrive) | Wolfram | ~1,900 | ⚠️ unverified | None | ~864 B/fr (64×36 grid) | ❌ inactive |
| [yt-media-storage](https://github.com/PulseBeat02/yt-media-storage) | C++ | ~875 | ✅ DCT domain | Wirehair fountain (5× overhead) | ~13,000 B/fr net² (4K only) | ✅ 2026 |
| [bin2video](https://github.com/pixelomer/bin2video) | C | ~62 | ✅ compat mode | None | 259,200 B/fr (bs=1, 1080p) | ✅ 2025 |
| [file2video](https://github.com/karaketir16/file2video) | Python | ~67 | ⚠️ partial | RS(255,245) | unknown | ⚠️ unclear |
| [Data2Video](https://github.com/bfaure/Data2Video) | Python 2.7 | ~61 | ❌ GIF only | None | N/A (GIF, no YouTube) | ❌ abandoned |
| [DataToVideoEncoderDecoder](https://github.com/Incipiens/DataToVideoEncoderDecoder) | Python | ~154 | ⚠️ proof-of-concept | None | 10,368 B/fr (5×5 blocks)³ | ❌ ~2022 |
| [youtube-data-storage-challenge](https://github.com/code-mc/youtube-data-storage-challenge) | Python | ~0 | ✅ QR codes | QR built-in (~15%) | 3,391 B/fr (QR v40) | ❌ one-time |
| [qStore](https://github.com/therealOri/qStore) | Python | ~9 | ✅ QR codes | QR built-in | ~3,000 B/fr (est.) | ✅ 2025 |

> ¹ `gray4` is YouTube-safe at block_size ≥ 4 (not the default bs=2); see encoding modes table.  
> ² Raw density 64,800 B/fr at 4K before Wirehair's 5× repair overhead; effective net throughput ~12,960 B/fr.  
> ³ Critical bug: encodes 100 MB file consuming ~100 GB RAM — not practically usable.

---

### Benchmark methodology

**Test machine:** Intel Core i7-1065G7 (4 cores / 8 threads @ 1.30 GHz), Intel Iris Plus Graphics (no discrete GPU — libx264 software encoder used for all runs), 16 GB RAM, Windows 11 Pro, Python 3.14, ffmpeg May 2026 build.

**Methodology:** Python 3 programs only, using real GitHub code with minimal non-functional changes (CLI wiring, import compatibility). Each (tool, file) pair runs 3 times; encode and decode are timed separately. Results are checkpointed to `bench/jellyfish_results.json` after every run and can be reproduced with `python benchmark_jellyfish.py`.

Excluded: DataToVideoEncoderDecoder (pixel_size enc/dec mismatch, OOM on large files) · qStore (QR encoding too slow to finish within timeout).

### Benchmark results

> **Scope:** Python 3 programs only, using real GitHub code with minimal non-functional changes.
> Excluded: DataToVideoEncoderDecoder (pixel_size enc/dec mismatch, OOM on large files) ·
> qStore (requires system ffmpeg, QR encoding extremely slow).

#### ~15 MB HD (11 MB raw)

| Tool | Resolution | Density | ECC | Codec | Enc avg | Dec avg | MP4 size | Enc MB/s | Result |
|---|---|---|---|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1920×1080 | ~64,800 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 20.7s | 14.3s | 120 MB | 0.54 | ✅ PASS |
| ByteVault bs=1 ecc=16 1080p | 1920×1080 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 7.3s | 9.0s | 96 MB | 1.53 | ✅ PASS |
| ByteVault bs=2 ecc=16 4K | 3840×2160 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 25.1s | 27.9s | 121 MB | 0.45 | ✅ PASS |
| file2video RS-10 | 1080×1080 | ~9,112 B/fr (270×270 grid, RS-10) | RS(255,245) nsym=10 | H.264 CRF=40 | 4.8m | 1.9m | 35 MB | 0.04 | ✅ PASS |
| YouBit bpp=1 default | 1920×1080 | ~259,200 B/fr (1 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1920×1080 | ~518,400 B/fr (2 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

#### ~52 MB HD (37 MB raw)

| Tool | Resolution | Density | ECC | Codec | Enc avg | Dec avg | MP4 size | Enc MB/s | Result |
|---|---|---|---|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1920×1080 | ~64,800 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 53.9s | 49.7s | 399 MB | 0.69 | ✅ PASS |
| ByteVault bs=1 ecc=16 1080p | 1920×1080 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 27.4s | 32.1s | 321 MB | 1.36 | ✅ PASS |
| ByteVault bs=2 ecc=16 4K | 3840×2160 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 42.5s | 42.1s | 399 MB | 0.88 | ✅ PASS |
| file2video RS-10 | 1080×1080 | ~9,112 B/fr (270×270 grid, RS-10) | RS(255,245) nsym=10 | H.264 CRF=40 | 14.2m | 8.0m | 117 MB | 0.04 | ✅ PASS |
| YouBit bpp=1 default | 1920×1080 | ~259,200 B/fr (1 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1920×1080 | ~518,400 B/fr (2 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

#### ~150 MB HD (150 MB raw)

| Tool | Resolution | Density | ECC | Codec | Enc avg | Dec avg | MP4 size | Enc MB/s | Result |
|---|---|---|---|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1920×1080 | ~64,800 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 4.6m | 4.8m | 1596 MB | 0.55 | ✅ PASS |
| ByteVault bs=1 ecc=16 1080p | 1920×1080 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 3.4m | 3.3m | 1285 MB | 0.74 | ✅ PASS |
| ByteVault bs=2 ecc=16 4K | 3840×2160 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 16.7m | 4.2m | 1593 MB | 0.15 | ❌ ERROR: Command '['C:\\Users\\sahil\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe', 'C:\\Users\\sahil\\OneDrive\\Documents\\GitHub\\ByteVault-Infinite--The-Eternal-Encoder\\main.py', 'encode', 'C:\\ |
| file2video RS-10 | 1080×1080 | ~9,112 B/fr (270×270 grid, RS-10) | RS(255,245) nsym=10 | H.264 CRF=40 | — | — | — | — | ❌ TIMEOUT (>30 min) |
| YouBit bpp=1 default | 1920×1080 | ~259,200 B/fr (1 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1920×1080 | ~518,400 B/fr (2 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

#### ~375 MB HD (374 MB raw)

| Tool | Resolution | Density | ECC | Codec | Enc avg | Dec avg | MP4 size | Enc MB/s | Result |
|---|---|---|---|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1920×1080 | ~64,800 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 7.6m | 6.1m | 3990 MB | 0.82 | ✅ PASS |
| ByteVault bs=1 ecc=16 1080p | 1920×1080 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 2.2m | 3.1m | 3214 MB | 2.88 | ✅ PASS |
| ByteVault bs=2 ecc=16 4K | 3840×2160 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 5.7m | 5.1m | 3984 MB | 1.09 | ✅ PASS |
| file2video RS-10 | 1080×1080 | ~9,112 B/fr (270×270 grid, RS-10) | RS(255,245) nsym=10 | H.264 CRF=40 | — | — | — | — | ❌ TIMEOUT (>30 min) |
| YouBit bpp=1 default | 1920×1080 | ~259,200 B/fr (1 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1920×1080 | ~518,400 B/fr (2 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

#### ~450 MB 4K (452 MB raw)

| Tool | Resolution | Density | ECC | Codec | Enc avg | Dec avg | MP4 size | Enc MB/s | Result |
|---|---|---|---|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1920×1080 | ~64,800 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 7.8m | 9.2m | 4826 MB | 0.97 | ✅ PASS |
| ByteVault bs=1 ecc=16 1080p | 1920×1080 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 3.3m | 5.4m | 3888 MB | 2.28 | ✅ PASS |
| ByteVault bs=2 ecc=16 4K | 3840×2160 | ~259,200 B/fr | RS nsym=16 | H.264 NVENC/libx264 | 9.3m | 9.0m | 4817 MB | 0.81 | ✅ PASS |
| file2video RS-10 | 1080×1080 | ~9,112 B/fr (270×270 grid, RS-10) | RS(255,245) nsym=10 | H.264 CRF=40 | — | — | — | — | ❌ TIMEOUT (>30 min) |
| YouBit bpp=1 default | 1920×1080 | ~259,200 B/fr (1 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1920×1080 | ~518,400 B/fr (2 bpp) | RS (youbit default) | H.264 libx264 CRF=18 grain | — | — | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

<details>
<summary>Per-run timing detail</summary>

##### ~15 MB HD

| Tool | Run | Enc (s) | Dec (s) | MP4 size | Status |
|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1 | 26.8 | 12.7 | 120 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 2 | 19.0 | 16.2 | 120 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 3 | 16.2 | 14.2 | 120 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 1 | 7.2 | 8.9 | 96 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 2 | 7.3 | 9.2 | 96 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 3 | 7.5 | 9.1 | 96 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 1 | 13.1 | 27.1 | 121 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 2 | 33.2 | 30.8 | 121 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 3 | 29.2 | 25.7 | 121 MB | ✅ |
| file2video RS-10 | 1 | 362.5 | 117.9 | 35 MB | ✅ |
| file2video RS-10 | 2 | 257.3 | 129.1 | 35 MB | ✅ |
| file2video RS-10 | 3 | 237.0 | 94.5 | 35 MB | ✅ |
| YouBit bpp=1 default | 1 | 0.1 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 2 | 0.1 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 3 | 0.1 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1 | 0.1 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 2 | 0.1 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 3 | 0.1 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

##### ~52 MB HD

| Tool | Run | Enc (s) | Dec (s) | MP4 size | Status |
|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1 | 57.1 | 48.7 | 399 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 2 | 53.0 | 57.7 | 399 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 3 | 51.4 | 42.8 | 399 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 1 | 29.5 | 30.9 | 321 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 2 | 27.1 | 30.6 | 321 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 3 | 25.6 | 34.9 | 321 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 1 | 46.7 | 43.1 | 399 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 2 | 46.7 | 45.0 | 399 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 3 | 34.1 | 38.3 | 399 MB | ✅ |
| file2video RS-10 | 1 | 826.0 | 247.4 | 117 MB | ✅ |
| file2video RS-10 | 2 | 1036.8 | 689.7 | 117 MB | ✅ |
| file2video RS-10 | 3 | 684.6 | 497.6 | 117 MB | ✅ |
| YouBit bpp=1 default | 1 | 0.9 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 2 | 0.3 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 3 | 0.3 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1 | 0.4 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 2 | 0.3 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 3 | 0.3 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

##### ~150 MB HD

| Tool | Run | Enc (s) | Dec (s) | MP4 size | Status |
|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1 | 231.9 | 180.6 | 1596 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 2 | 191.0 | 239.7 | 1596 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 3 | 398.3 | 447.4 | 1596 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 1 | 362.3 | 252.1 | 1285 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 2 | 148.8 | 195.2 | 1285 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 3 | 99.0 | 152.5 | 1285 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 1 | 1001.2 | 251.6 | 1593 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 2 | 280.8 | — | — | ❌ ERROR: Command '['C:\\Users\\sahil\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe', 'C:\\Users\\sahil\\OneDrive\\Documents\\GitHub\\ByteVault-Infinite--The-Eternal-Encoder\\main.py', 'encode', 'C:\\ |
| ByteVault bs=2 ecc=16 4K | 3 | 387.9 | — | — | ❌ ERROR: Command '['C:\\Users\\sahil\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe', 'C:\\Users\\sahil\\OneDrive\\Documents\\GitHub\\ByteVault-Infinite--The-Eternal-Encoder\\main.py', 'encode', 'C:\\ |
| file2video RS-10 | 1 | 1800.1 | — | — | ❌ TIMEOUT (>30 min) |
| file2video RS-10 | 2 | 1800.1 | — | — | ❌ TIMEOUT (>30 min) |
| file2video RS-10 | 3 | 1800.3 | — | — | ❌ TIMEOUT (>30 min) |
| YouBit bpp=1 default | 1 | 1.0 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 2 | 0.8 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 3 | 0.9 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1 | 0.9 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 2 | 1.0 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 3 | 1.0 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

##### ~375 MB HD

| Tool | Run | Enc (s) | Dec (s) | MP4 size | Status |
|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1 | 796.6 | 441.4 | 3990 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 2 | 295.3 | 296.1 | 3990 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 3 | 272.4 | 369.1 | 3990 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 1 | 144.5 | 194.6 | 3214 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 2 | 112.4 | 170.2 | 3214 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 3 | 131.9 | 186.8 | 3214 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 1 | 262.6 | 298.7 | 3984 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 2 | 286.6 | 352.3 | 3984 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 3 | 476.3 | 266.8 | 3984 MB | ✅ |
| file2video RS-10 | 1 | 1800.0 | — | — | ❌ TIMEOUT (>30 min) |
| file2video RS-10 | 2 | 1800.0 | — | — | ❌ TIMEOUT (>30 min) |
| file2video RS-10 | 3 | 1800.0 | — | — | ❌ TIMEOUT (>30 min) |
| YouBit bpp=1 default | 1 | 1.5 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 2 | 1.0 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 3 | 0.8 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1 | 0.9 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 2 | 0.8 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 3 | 1.1 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

##### ~450 MB 4K

| Tool | Run | Enc (s) | Dec (s) | MP4 size | Status |
|---|---|---|---|---|---|
| ByteVault bs=2 ecc=16 1080p | 1 | 539.0 | 645.4 | 4826 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 2 | 416.3 | 520.8 | 4826 MB | ✅ |
| ByteVault bs=2 ecc=16 1080p | 3 | 445.9 | 495.1 | 4826 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 1 | 156.1 | 262.3 | 3888 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 2 | 165.0 | 330.7 | 3888 MB | ✅ |
| ByteVault bs=1 ecc=16 1080p | 3 | 272.8 | 372.8 | 3888 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 1 | 512.5 | 613.7 | 4817 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 2 | 564.6 | 513.8 | 4817 MB | ✅ |
| ByteVault bs=2 ecc=16 4K | 3 | 596.2 | 487.4 | 4817 MB | ✅ |
| file2video RS-10 | 1 | 1800.0 | — | — | ❌ TIMEOUT (>30 min) |
| file2video RS-10 | 2 | 1800.0 | — | — | ❌ TIMEOUT (>30 min) |
| file2video RS-10 | 3 | 1800.1 | — | — | ❌ TIMEOUT (>30 min) |
| YouBit bpp=1 default | 1 | 3.3 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 2 | 2.7 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=1 default | 3 | 2.5 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 1 | 2.3 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 2 | 2.5 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |
| YouBit bpp=2 default | 3 | 2.2 | — | — | ❌ ERROR: signal only works in main thread of the main interpreter |

</details>
---

### Detailed analysis

#### [Infinite-Storage-Glitch](https://github.com/DvorakDwarf/Infinite-Storage-Glitch) (ISG) — DvorakDwarf · Rust · archived 2023

The project that started the category going viral in February 2023. ISG introduced the two-mode approach that PixelVault builds on:

- **Binary mode:** 2×2 pixel blocks, black=0 / white=1. At 1080p: 960×540 logical pixels = **64,800 bytes/frame** — identical to PixelVault's `binary` default.
- **RGB mode:** one byte per channel per 2×2 block = **1,555,200 bytes/frame** — but YouTube's chroma re-encoding makes this local-only.

PixelVault's binary mode is density-for-density equivalent to ISG but adds: Reed-Solomon ECC, optional byte interleaving, hardware encoder support (NVENC/AMF/QSV), multi-threaded frame generation, 4K support, and a self-describing header. ISG has no ECC and was archived immediately after its viral moment.

---

#### [YouBit](https://github.com/flonle/youbit) — flonle · Python/Cython · last updated Oct 2022

The most technically mature YouTube-specific tool before ISG. YouBit's design choices are instructive:

- **Grayscale only**, at 1:1 pixel ratio (block_size=1). At BPP=1: 1920×1080 = **259,200 bytes/frame** — 4× PixelVault's `binary bs=2` default. PixelVault achieves the same density with `--block-size 1` but defaults to bs=2 for YouTube compression safety.
- **BPP=2:** 518,400 bytes/frame — two luma levels between 0/255, margins at ±43. Marginal on YouTube; ECC required.
- **BPP=3:** 777,600 bytes/frame — experimental, unreliable post-YouTube.
- **Encodes at 1 FPS** and exploits YouTube's minimum-6-FPS re-encode: decoder reads only keyframes, gaining determinism.
- **Reed-Solomon ECC** via `creedsolo` (Cython, same GF(2^8) construction as PixelVault).
- **Upload automation via Selenium** (browser cookie extraction) rather than the YouTube Data API — no OAuth setup, but brittle.
- **Gzip compression** of payload before encoding (PixelVault uses zlib).

PixelVault now outperforms YouBit on both encode speed (1.9 s vs 3.0 s for 1 MB) and decode speed (1.9 s vs 5.2 s) while providing identical density at `--block-size 1` and stronger burst protection via interleaving. PixelVault additionally adds: 4K support (4× more data per frame), hardware encoder acceleration (NVENC/QSV/AMF), and a proper YouTube Data API upload path.

---

#### [YouTubeDrive](https://github.com/dzhang314/YouTubeDrive) — dzhang314 · Wolfram Language · inactive

A proof-of-concept requiring proprietary Mathematica. Encodes into a 64×36 logical pixel grid (each block blown up 20× to fill 1280×720), giving just **864 bytes/frame**. The extreme upscaling (20×) wastes 99.75% of video bandwidth but does help DCT compression survival. No ECC, no Python, not practically usable without a Mathematica license. Historically notable as one of the earliest demonstrations of the concept.

---

#### [yt-media-storage](https://github.com/PulseBeat02/yt-media-storage) — PulseBeat02 · C++ · active 2026

The most sophisticated non-PixelVault entry. Uses a fundamentally different technique:

- **DCT-domain steganography** in 8×8 pixel blocks: encodes 4 bits per block (in low-frequency DCT coefficients {0,1}, {1,0}, {1,1}, {0,2}) by modulating their sign with ±500 units. DCT signs survive VP9/AV1 re-encoding better than raw pixel values.
- **Fixed 4K only.** At 3840×2160 with 8×8 blocks: (3840/8)×(2160/8) = 480×270 blocks × 4 bits = **64,800 bytes/frame raw** — same as PixelVault binary at 1080p.
- **Wirehair fountain codes** (rateless erasure codes) with **5× repair overhead** (REPAIR_OVERHEAD=5.0). After overhead: ~**12,960 effective bytes/frame**. Compare to PixelVault's RS at nsym=16: ~6.7% overhead, retaining 93.3% of raw density.
- **XChaCha20-Poly1305 encryption** (libsodium) — the only other tool with built-in encryption besides qStore.
- **Qt6 GUI, C++23, OpenMP** — powerful but complex build (CMake 3.22+, C++23 compiler, Qt6, libsodium).
- Also supports **live RTMP streaming** to YouTube/Twitch as a secondary mode.

The DCT approach is more theoretically elegant but the 5× fountain overhead makes its effective throughput lower than PixelVault binary at 1080p. PixelVault's RS at nsym=16 corrects up to 8 errors per 255-byte block at only 6.7% overhead.

---

#### [bin2video](https://github.com/pixelomer/bin2video) — pixelomer · C · active 2025

A flexible C tool with the widest configuration range in the survey:

- **1–24 bits per pixel**, configurable block size, configurable resolution.
- Default: 1 bpp, 5×5 blocks, 1280×720 → 4,608 bytes/frame (conservative).
- At 1 bpp, bs=1, 1080p: **259,200 bytes/frame** (same as YouBit BPP=1 / PixelVault binary bs=1).
- **ISG compatibility flag** (`-I`) to read/write ISG-format videos — unique interoperability feature.
- Larger metadata frame (10×10 blocks) for the first frame, smaller data blocks for subsequent frames.
- **No ECC.** Measured encode speed: 32 MB in 106–253 seconds on an M1 MacBook Air — slow compared to PixelVault's multi-threaded Python.
- YouTube compat mode demonstrated with yt-dlp roundtrip.

---

#### [Data2Video](https://github.com/bfaure/Data2Video) — bfaure · Python 2.7 · abandoned

Historical only. Outputs **GIF**, not MP4. Uses 1×1 pixels at 4K — giving 1,036,800 bytes/frame in theory, but 1×1 pixels have no compression resistance and the GIF format is not accepted by YouTube. The authors themselves noted the design flaw and suggested n×n blocks as a future fix.

---

#### [DataToVideoEncoderDecoder](https://github.com/Incipiens/DataToVideoEncoderDecoder) — Incipiens · Python · ~2022

Created for an XDA-Developers article. Uses 5×5 pixel clusters at 1080p: 384×216 logical pixels = **10,368 bytes/frame** — the lowest density of any video-based tool here. Critical flaw: encoding a 100 MB file consumes ~100 GB of RAM (the entire bit representation is held as a Python list in memory). No ECC. Impractical for any real use.

---

#### [youtube-data-storage-challenge](https://github.com/code-mc/youtube-data-storage-challenge) — code-mc · Python · one-time

A research challenge that demonstrated YouTube roundtrip via QR codes. Uses QR code version 40 (maximum) with medium ECC level: **3,391 bytes per QR frame** encoded at 1870×1870 pixels. Proved the concept works even at 360p. Density is extremely low (QR overhead consumes >99% of pixel budget), but robustness is extreme — works at any resolution YouTube serves. Not practical for bulk storage.

---

#### [qStore](https://github.com/therealOri/qStore) — therealOri · Python · active 2025

The only fully encrypted file-to-YouTube tool: **AES-GCM + ChaCha20-Poly1305** dual encryption (Chaeslib) applied before QR encoding. Each chunk of encrypted data becomes one QR frame; decoder uses zbar-tools to read QR codes from the downloaded video. Density is low (bounded by QR v40 at ~3,000 bytes/frame estimated) but the dual-encryption pipeline is the unique differentiator. No comparable to PixelVault for density; complements it for confidentiality.

---

### Data density head-to-head (1080p, YouTube-safe modes only)

```
Tool / mode                       Bytes/frame    bits/pixel   Note
─────────────────────────────────────────────────────────────────────────────
PixelVault  binary  bs=1           259,200        1.00         max density, more errors
PixelVault  binary  bs=2 (default)  64,800        0.25         default; DCT-safe
YouBit     BPP=1                  259,200        1.00         no interleaving
ISG        binary  2×2             64,800        0.25         no ECC
yt-media-storage  (4K, before FEC) 64,800        0.004 (4K)  DCT domain; only runs at 4K
yt-media-storage  (4K, net of 5×)  12,960        —           after Wirehair overhead
bin2video  bs=1, 1bpp             259,200        1.00         no ECC
youtube-data-storage-challenge      3,391        0.001        QR v40; very low density
qStore     QR                      ~3,000        —            est.; encrypted
YouTubeDrive                          864        0.0003       64×36 logical grid; Wolfram only
```

PixelVault at 4K binary bs=2: **259,200 bytes/frame** — same as YouBit/bin2video bs=1 at 1080p, but with the DCT-safety of a 2×2 block and 4K's higher VP9 bitrate providing more headroom.

---

### Feature matrix

| Feature | PixelVault | ISG | YouBit | yt-media-storage | bin2video |
|---|---|---|---|---|---|
| YouTube-safe mode | ✅ | ✅ | ✅ | ✅ | ✅ |
| Reed-Solomon ECC | ✅ configurable | ❌ | ✅ configurable | ❌ (fountain codes) | ❌ |
| Burst-error interleaving | ✅ | ❌ | ❌ | ❌ | ❌ |
| 4K support | ✅ | ❌ | ⚠️ slow | ✅ (only mode) | ❌ |
| Hardware encoder (NVENC/AMF/QSV) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multi-threaded encode | ✅ | ❌ | ❌ | ✅ (OpenMP) | ❌ |
| Multi-threaded decode | ✅ | ❌ | ❌ | ✅ (OpenMP) | ❌ |
| zlib compression | ✅ | ❌ | ✅ (gzip) | ❌ | ❌ |
| Encryption | ✅ AES-256-GCM (`--password`) | ❌ | ❌ | ✅ (XChaCha20) | ❌ |
| Audio track encoding | ✅ (local) | ❌ | ❌ | ❌ | ❌ |
| Self-describing header | ✅ | ✅ | ❌ (settings stored externally) | ❌ | ✅ (first-frame metadata) |
| YouTube API upload | ✅ | ❌ | ❌ (Selenium) | ❌ | ❌ |
| Calibration tool | ✅ | ❌ | ❌ | ❌ | ❌ |
| Configurable resolution | ✅ | ❌ | ⚠️ min 1080p | ❌ (4K only) | ✅ |
| Language | Python | Rust | Python/Cython | C++ | C |
| Active | ✅ | ❌ | ❌ | ✅ | ✅ |

---

## Contributing

Issues are welcome — bug reports, YouTube-compatibility findings, and benchmark results from other hardware are especially useful. PRs by discussion: open an issue first to align on scope before writing code. This is a personal research project; unsolicited feature PRs may not be merged.

---

## License

MIT — see [LICENSE](LICENSE).
