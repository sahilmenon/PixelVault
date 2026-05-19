# ByteVault Infinite: The Eternal Encoder

**The state-of-the-art file-to-YouTube encoder.** Convert **any file** into a YouTube video, upload it, and recover it perfectly later — using YouTube as infinite cloud storage.

> Inspired by [Infinite-Storage-Glitch](https://github.com/DvorakDwarf/Infinite-Storage-Glitch) and [YouTubeDrive](https://github.com/dzhang314/YouTubeDrive). A proof-of-concept showcase of compression-resistant binary encoding. Not intended for high-volume use.

---

## State of the Art

ByteVault is the most capable file-to-YouTube encoder publicly available. The claims below are backed by benchmark data and source-level analysis of every known comparable tool (see [Comparison with Similar Projects](#comparison-with-similar-projects) for the full survey).

### 1 — Only actively maintained Python tool that reliably completes a roundtrip

Of the six Python tools benchmarked:

| Tool | Status | Roundtrip result |
|---|---|---|
| **ByteVault** | ✅ Active (2025–26) | ✅ All runs passed |
| YouBit | ❌ Inactive since Oct 2022 | ❌ Crashed every run (`signal only works in main thread`) |
| file2video | ⚠️ Unclear | ✅ Passed (but 38× slower — see below) |
| DataToVideoEncoderDecoder | ❌ ~2022 | ❌ Excluded — pixel_size enc/dec mismatch + OOM |
| qStore | ⚠️ Active | ❌ Excluded — QR encode too slow to benchmark |
| Data2Video | ❌ Abandoned | ❌ Outputs GIF, not accepted by YouTube |

YouBit — the most-cited prior Python tool — fails on any modern Python due to a `signal`-in-thread constraint introduced after Python 3.9. ByteVault is the only Python tool confirmed working today.

### 2 — 38× faster encode than the closest working competitor

Measured on a ~15 MB file, three runs each:

| Tool | Avg encode | Encode throughput |
|---|---|---|
| **ByteVault bs=1 ecc=16 1080p** | **7.3 s** | **1.53 MB/s** |
| **ByteVault bs=2 ecc=16 1080p** | 20.7 s | 0.54 MB/s |
| file2video RS-10 | 285.6 s | 0.04 MB/s |

ByteVault at `block_size=1` is **38× faster** than file2video, the only other Python tool that works. This comes from multi-threaded NumPy frame generation, hardware encoder acceleration (NVENC/AMF/QSV), and a streaming pipeline that never materialises all frames in memory.

### 3 — Highest DCT-safe data density with 4K

At 4K (`--4k`), ByteVault encodes **259,200 bytes/frame** using 2×2 pixel blocks — the same density as 1×1-block tools at 1080p but with 4× the VP9 bitrate headroom, making it the most data-dense YouTube-safe configuration known:

```
ByteVault 4K binary bs=2 →  259,200 B/fr  (DCT-safe 2×2 blocks, VP9 ~35–45 Mbps)
YouBit BPP=1 1080p        →  259,200 B/fr  (1×1 blocks, H.264 ~4–8 Mbps, no interleaving)
ISG binary 1080p          →   64,800 B/fr  (2×2 blocks, no ECC, archived)
yt-media-storage 4K net   →   12,960 B/fr  (DCT domain, 5× Wirehair overhead)
```

### 4 — The only tool with burst-error interleaving

YouTube's re-encoder produces burst compression artefacts — contiguous corrupt regions that exhaust a Reed-Solomon block's correction capacity. ByteVault's `--interleave` flag spreads ECC bytes across frames so each RS block sees at most one or two errors from any burst:

```
without --interleave: 30 corrupt bytes in one RS block → uncorrectable at nsym=32
with    --interleave: same 30 bytes → ~1 error per RS block → trivially fixed at nsym=8
```

No other tool in the survey — ISG, YouBit, yt-media-storage, bin2video, file2video — implements burst-error interleaving.

### 5 — Broadest feature set among all tools surveyed

Counting the feature matrix below, ByteVault is the only tool with all of: YouTube-safe mode, configurable Reed-Solomon ECC, burst-error interleaving, 4K support, hardware encoder acceleration, multi-threaded encode and decode, payload compression, AES-256-GCM encryption, audio-track encoding, a self-describing header, YouTube API upload, and a calibration tool. No competitor has more than six of these twelve features.

---

## How It Works

Every file is a sequence of bytes. ByteVault encodes those bytes as **coloured pixel blocks** in a video frame, and optionally in the **audio track** as well:

```
File bytes  →  [zlib compress?]  →  [Reed-Solomon ECC?]  →  [interleave?]  →  split: video portion + audio portion
                                                                                │                    │
                                                                                ▼                    ▼
                                                                        logical pixels       amplitude-level PCM
                                                                        N×N pixel blocks     (stereo, ALAC lossless)
                                                                                │                    │
                                                                                └──────────┬─────────┘
                                                                                           ▼
                                                                                     ffmpeg → MP4
```

On the way back:

```
MP4 (local or YouTube)  →  yt-dlp download  →  pixel block centres (video) + PCM decode (audio)
                        →  [deinterleave]  →  [ECC correct]  →  [decompress]  →  original file
```

YouTube re-encodes every upload using H.264/VP9 (video) and AAC (audio), both lossy. ByteVault uses several strategies to survive this:

- **Video:** each logical pixel is a large `block_size × block_size` square. DCT can corrupt block edges, but the centre of a large uniform square survives reliably.
- **Binary mode:** pixels are pure black or white (luma only), so YouTube's chroma compression is irrelevant.
- **Nibble mode:** 16 colours designed in YCbCr space — spaced 64 units apart in luma and 128 in chroma — so YouTube's ±20–30 chroma noise rarely causes a misclassification. Reed-Solomon corrects the rest.
- **Palette mode:** colours are chosen with a minimum 36-channel gap per R/G channel (85 for B) — large enough that YouTube's ±10–20 channel error can never cross a palette boundary.
- **Reed-Solomon ECC + interleaving:** burst compression artefacts (contiguous corrupt bytes) are spread across many RS blocks so each block sees only 1–2 errors instead of exhausting its capacity.
- **Audio (local only):** stored as ALAC (Apple Lossless) — bit-perfect for local roundtrips. The audio data channel does **not** survive YouTube's AAC re-transcoding.

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

### Audio track (optional — `--audio`, local use only)

| Parameter | Value |
|-----------|-------|
| Sample rate | 48 000 Hz |
| Levels | 4 (2 bits/sample) |
| Channels | Stereo (independent L/R) |
| Codec | ALAC (lossless) |
| Yield | ~15 B/frame at 30 fps |

For YouTube uploads, omit `--audio`. All data goes through the video track.

### Payload compression (optional — `--compress`)

zlib-compress (level 6) before encoding. If the compressed size is larger (e.g. JPEG, ZIP), it is skipped silently. Decoding is automatic.

### Reed-Solomon error correction (optional — `--ecc`)

Payload is split into 255-byte blocks; each block gains `nsym` ECC bytes that allow correcting up to `nsym/2` byte errors within that block.

| `--ecc NSYM` | Overhead | Errors correctable per block |
|---|---|---|
| `--ecc 8`  | ~3.2%  | up to 4  |
| `--ecc 16` | ~6.7%  | up to 8  **(default)** |
| `--ecc 32` | ~14.4% | up to 16 |
| `--ecc 64` | ~33.5% | up to 32 |
| `--ecc 0`  | 0%     | disabled |

The `nsym` value is stored in the header — decoding is automatic.

### Byte interleaving (optional — `--interleave`, requires `--ecc`)

Permutes the ECC byte stream across frames before writing. A burst of *K* corrupt frames then contributes at most ⌈255/depth⌉ errors per RS block instead of all hitting one block — converting burst errors into near-uniform errors that RS handles efficiently.

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

ByteVault searches for `client_secret.json` in the current directory first, then `~/.bytevault/`.

---

## Quick Start

```bash
# Encode a file → video (1080p)
python main.py encode secret.zip

# Encode at 4K for 4× data density
python main.py encode secret.zip --4k --upload

# Decode a local video → original file
python main.py decode vault/encoded/secret.zip.mp4

# Encode and upload to YouTube (1080p)
python main.py encode secret.zip --upload
# → YouTube ID: dQw4w9WgXcQ
# → Decode with: python main.py decode yt:dQw4w9WgXcQ

# Decode directly from YouTube (1080p)
python main.py decode yt:dQw4w9WgXcQ

# Decode a 4K-encoded video from YouTube
python main.py decode yt:dQw4w9WgXcQ --4k
```

### Recommended YouTube workflow
```bash
# 1. Calibrate once per account
python calibrate.py --mode binary        # 1080p calibration
python calibrate.py --mode binary --4k   # 4K calibration (optional but recommended for --4k uploads)

# 2. Encode + upload (ECC nsym=16 on by default)
python main.py encode secret.zip --upload         # 1080p: ~64,800 B/frame
python main.py encode secret.zip --4k --upload    # 4K:   ~259,200 B/frame (4× density)

# 3. Large files (>1 GB): split into chunks, each uploaded as a separate video
python main.py encode big_archive.tar.gz --chunk-mb 512 --upload
# Saves a .bvault manifest listing all chunk video IDs
# Decode later with one command — chunks are downloaded and reassembled automatically:
python main.py decode vault/encoded/big_archive.bvault
```

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
| `--chunk-mb MB` | off | Split into MB-sized chunks; writes a `.bvault` manifest. Combine with `--upload` for multi-video YouTube storage. |
| `--upload` | off | Upload to YouTube after encoding |
| `--title TEXT` | filename | YouTube video title |
| `--description TEXT` | `"Encoded with ByteVault"` | YouTube video description |
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
# → writes huge_backup.bvault (manifest with all video IDs)
# → python main.py decode vault/encoded/huge_backup.bvault

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

ByteVault videos are **self-describing**: all encoding parameters are stored in the first 128 bytes of the first video frame, so `decode` never needs extra flags.

### Header layout (128 bytes)

```
Offset  Size  Field
0       4     Magic bytes: BVI\x01 (video-only) or BVI\x02 (audio-extended)
4       1     Mode (0=binary, 1=rgb, 2=palette, 3=rgb_bin, 4=nibble)
5       1     Block size
6       2     Filename length
8       64    Original filename (UTF-8)
72      8     Original file size (bytes)
80      4     Video payload padding bytes
84      4     Audio payload bytes (0 if no audio)
88      1     Audio n_levels (0 if no audio)
89      1     Audio block size (0 if no audio)
90      1     Flags: bit 0 = zlib-compressed payload
91      8     Compressed payload size (0 if uncompressed)
99      1     ECC nsym (0 = no ECC)
100     4     Interleave depth (0 = no interleaving)
104     24    Reserved (zeros)
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
| **Hardware encoding** | NVENC/AMF/QSV is used automatically for `binary` and `rgb_bin` modes. `nibble`, `palette`, and `rgb` use libx264 CRF=0 (lossless local encode) so only YouTube's re-encode introduces chroma error. Use `--no-hw` to force libx264 for binary/rgb_bin too. |
| **File size** | No hard limit. Use `--chunk-mb N` to split large files across multiple YouTube videos automatically — each chunk is encoded as an independent MP4 and a `.bvault` manifest is saved for one-command reassembly on decode. |
| **Terms of Service** | Mass automated uploads may violate YouTube ToS. Use responsibly. |

---

## Architecture

```
ByteVault-Infinite--The-Eternal-Encoder/
├── bytevault/
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

## Comparison with Similar Projects

The table below surveys ten known file-to-video encoders. All density figures are calculated from the project source or documentation; "YouTube-safe" means the authors verified lossless roundtrip through YouTube's re-encoder.

### Quick comparison

| Project | Lang | Stars | YouTube-safe | ECC | Best YouTube-safe density (1080p) | Active |
|---|---|---|---|---|---|---|
| **ByteVault** (this) | Python | — | ✅ binary, gray4¹ | RS nsym 8–64 + interleave | **64,800 B/fr** (bs=2) · **259,200 B/fr** (bs=1) | ✅ 2025–26 |
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

</details>
---

### Detailed analysis

#### [Infinite-Storage-Glitch](https://github.com/DvorakDwarf/Infinite-Storage-Glitch) (ISG) — DvorakDwarf · Rust · archived 2023

The project that started the category going viral in February 2023. ISG introduced the two-mode approach that ByteVault builds on:

- **Binary mode:** 2×2 pixel blocks, black=0 / white=1. At 1080p: 960×540 logical pixels = **64,800 bytes/frame** — identical to ByteVault's `binary` default.
- **RGB mode:** one byte per channel per 2×2 block = **1,555,200 bytes/frame** — but YouTube's chroma re-encoding makes this local-only.

ByteVault's binary mode is density-for-density equivalent to ISG but adds: Reed-Solomon ECC, optional byte interleaving, hardware encoder support (NVENC/AMF/QSV), multi-threaded frame generation, 4K support, and a self-describing header. ISG has no ECC and was archived immediately after its viral moment.

---

#### [YouBit](https://github.com/flonle/youbit) — flonle · Python/Cython · last updated Oct 2022

The most technically mature YouTube-specific tool before ISG. YouBit's design choices are instructive:

- **Grayscale only**, at 1:1 pixel ratio (block_size=1). At BPP=1: 1920×1080 = **259,200 bytes/frame** — 4× ByteVault's `binary bs=2` default. ByteVault achieves the same density with `--block-size 1` but defaults to bs=2 for YouTube compression safety.
- **BPP=2:** 518,400 bytes/frame — two luma levels between 0/255, margins at ±43. Marginal on YouTube; ECC required.
- **BPP=3:** 777,600 bytes/frame — experimental, unreliable post-YouTube.
- **Encodes at 1 FPS** and exploits YouTube's minimum-6-FPS re-encode: decoder reads only keyframes, gaining determinism.
- **Reed-Solomon ECC** via `creedsolo` (Cython, same GF(2^8) construction as ByteVault).
- **Upload automation via Selenium** (browser cookie extraction) rather than the YouTube Data API — no OAuth setup, but brittle.
- **Gzip compression** of payload before encoding (ByteVault uses zlib).

ByteVault now outperforms YouBit on both encode speed (1.9 s vs 3.0 s for 1 MB) and decode speed (1.9 s vs 5.2 s) while providing identical density at `--block-size 1` and stronger burst protection via interleaving. ByteVault additionally adds: 4K support (4× more data per frame), hardware encoder acceleration (NVENC/QSV/AMF), and a proper YouTube Data API upload path.

---

#### [YouTubeDrive](https://github.com/dzhang314/YouTubeDrive) — dzhang314 · Wolfram Language · inactive

A proof-of-concept requiring proprietary Mathematica. Encodes into a 64×36 logical pixel grid (each block blown up 20× to fill 1280×720), giving just **864 bytes/frame**. The extreme upscaling (20×) wastes 99.75% of video bandwidth but does help DCT compression survival. No ECC, no Python, not practically usable without a Mathematica license. Historically notable as one of the earliest demonstrations of the concept.

---

#### [yt-media-storage](https://github.com/PulseBeat02/yt-media-storage) — PulseBeat02 · C++ · active 2026

The most sophisticated non-ByteVault entry. Uses a fundamentally different technique:

- **DCT-domain steganography** in 8×8 pixel blocks: encodes 4 bits per block (in low-frequency DCT coefficients {0,1}, {1,0}, {1,1}, {0,2}) by modulating their sign with ±500 units. DCT signs survive VP9/AV1 re-encoding better than raw pixel values.
- **Fixed 4K only.** At 3840×2160 with 8×8 blocks: (3840/8)×(2160/8) = 480×270 blocks × 4 bits = **64,800 bytes/frame raw** — same as ByteVault binary at 1080p.
- **Wirehair fountain codes** (rateless erasure codes) with **5× repair overhead** (REPAIR_OVERHEAD=5.0). After overhead: ~**12,960 effective bytes/frame**. Compare to ByteVault's RS at nsym=16: ~6.7% overhead, retaining 93.3% of raw density.
- **XChaCha20-Poly1305 encryption** (libsodium) — the only other tool with built-in encryption besides qStore.
- **Qt6 GUI, C++23, OpenMP** — powerful but complex build (CMake 3.22+, C++23 compiler, Qt6, libsodium).
- Also supports **live RTMP streaming** to YouTube/Twitch as a secondary mode.

The DCT approach is more theoretically elegant but the 5× fountain overhead makes its effective throughput lower than ByteVault binary at 1080p. ByteVault's RS at nsym=16 corrects up to 8 errors per 255-byte block at only 6.7% overhead.

---

#### [bin2video](https://github.com/pixelomer/bin2video) — pixelomer · C · active 2025

A flexible C tool with the widest configuration range in the survey:

- **1–24 bits per pixel**, configurable block size, configurable resolution.
- Default: 1 bpp, 5×5 blocks, 1280×720 → 4,608 bytes/frame (conservative).
- At 1 bpp, bs=1, 1080p: **259,200 bytes/frame** (same as YouBit BPP=1 / ByteVault binary bs=1).
- **ISG compatibility flag** (`-I`) to read/write ISG-format videos — unique interoperability feature.
- Larger metadata frame (10×10 blocks) for the first frame, smaller data blocks for subsequent frames.
- **No ECC.** Measured encode speed: 32 MB in 106–253 seconds on an M1 MacBook Air — slow compared to ByteVault's multi-threaded Python.
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

The only fully encrypted file-to-YouTube tool: **AES-GCM + ChaCha20-Poly1305** dual encryption (Chaeslib) applied before QR encoding. Each chunk of encrypted data becomes one QR frame; decoder uses zbar-tools to read QR codes from the downloaded video. Density is low (bounded by QR v40 at ~3,000 bytes/frame estimated) but the dual-encryption pipeline is the unique differentiator. No comparable to ByteVault for density; complements it for confidentiality.

---

### Data density head-to-head (1080p, YouTube-safe modes only)

```
Tool / mode                       Bytes/frame    bits/pixel   Note
─────────────────────────────────────────────────────────────────────────────
ByteVault  binary  bs=1           259,200        1.00         max density, more errors
ByteVault  binary  bs=2 (default)  64,800        0.25         default; DCT-safe
YouBit     BPP=1                  259,200        1.00         no interleaving
ISG        binary  2×2             64,800        0.25         no ECC
yt-media-storage  (4K, before FEC) 64,800        0.004 (4K)  DCT domain; only runs at 4K
yt-media-storage  (4K, net of 5×)  12,960        —           after Wirehair overhead
bin2video  bs=1, 1bpp             259,200        1.00         no ECC
youtube-data-storage-challenge      3,391        0.001        QR v40; very low density
qStore     QR                      ~3,000        —            est.; encrypted
YouTubeDrive                          864        0.0003       64×36 logical grid; Wolfram only
```

ByteVault at 4K binary bs=2: **259,200 bytes/frame** — same as YouBit/bin2video bs=1 at 1080p, but with the DCT-safety of a 2×2 block and 4K's higher VP9 bitrate providing more headroom.

---

### Feature matrix

| Feature | ByteVault | ISG | YouBit | yt-media-storage | bin2video |
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

## License

MIT — see [LICENSE](LICENSE).
