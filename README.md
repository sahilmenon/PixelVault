# ByteVault Infinite: The Eternal Encoder

Convert **any file** into a YouTube video, upload it, and recover it perfectly later — using YouTube as infinite cloud storage.

> Inspired by [Infinite-Storage-Glitch](https://github.com/DvorakDwarf/Infinite-Storage-Glitch) and [YouTubeDrive](https://github.com/dzhang314/YouTubeDrive). A proof-of-concept showcase of compression-resistant binary encoding. Not intended for high-volume use.

---

## How It Works

Every file is a sequence of bytes. ByteVault encodes those bytes as **coloured pixel blocks** in a video frame:

```
File bytes  →  logical pixels  →  N×N pixel blocks  →  video frames  →  YouTube
```

On the way back:

```
YouTube  →  download (yt-dlp)  →  sample block centres  →  bytes  →  original file
```

YouTube re-encodes every upload using H.264 / VP9, which is **lossy**. The key insight that makes recovery possible is **block scaling**: each logical pixel is rendered as a large square of identical real pixels. The codec's DCT compression can corrupt the edges of a block, but the centre pixel of a large uniform square survives reliably.

---

## Encoding Modes

| Mode | Bits/logical px | Default block | Bytes/frame (1920×1080) | Overhead | YouTube resistance |
|------|----------------|---------------|------------------------|----------|--------------------|
| `binary` | 1 | 4×4 | 16,200 B | ~32× | ★★★ Best |
| `rgb` | 24 | 4×4 | 388,800 B | ~1.3× | ★ Fragile |
| `palette` | 8 | 8×8 | 32,400 B | ~4× | ★★ Good |

### `binary` (default — recommended)
Each bit maps to one logical pixel: **white = 1, black = 0**.  
Each logical pixel is a `block_size × block_size` square. After YouTube's VP9 encode, the centre of a 4×4 white square stays white; thresholding at grey=127 recovers the original bit with >99.9% accuracy.

### `rgb`
Each logical pixel stores 3 raw bytes — one per colour channel (R, G, B).  
Space-efficient but **not reliable after YouTube re-encoding** because VP9 chroma subsampling (4:2:0) and DCT rounding shift channel values unpredictably. Use only for videos you won't upload, or with very large block sizes.

### `palette`
Each byte (0–255) maps to one of 256 maximally-separated colours. The palette uses a 3-bit R, 3-bit G, 2-bit B layout (step ≥ 36 per R/G channel, 85 per B channel). YouTube compression introduces ±10–20 per channel at worst — still well inside the 18-unit tolerance. At `--block-size 16`, this mode survives even aggressive re-encoding.

### Choosing block size
Larger blocks = more YouTube-resistant, fewer bytes per frame (longer video).

```
--block-size 2   very dense,   low tolerance  (not recommended for YouTube)
--block-size 4   default binary/rgb
--block-size 8   default palette; safe for most uploads
--block-size 16  maximum tolerance; use for palette mode on compressed uploads
```

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

## Quick Start

### Encode a file → video
```bash
python main.py encode secret.zip
# → secret.zip.mp4  (binary mode, 1920×1080, 24 fps)
```

### Decode a local video → original file
```bash
python main.py decode secret.zip.mp4
# → secret.zip  (recovered in current directory)
```

### Encode + auto-upload to YouTube
```bash
python main.py encode secret.zip \
    --upload \
    --credentials client_secrets.json \
    --title "My Backup" \
    --privacy unlisted
# → https://www.youtube.com/watch?v=XXXXXXXXXXX
```

### Decode directly from a YouTube URL
```bash
python main.py decode "https://www.youtube.com/watch?v=XXXXXXXXXXX"
# Downloads + decodes → secret.zip
```

---

## Full CLI Reference

### `encode`

```
python main.py encode <file> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode {binary,rgb,palette}` | `binary` | Encoding mode (see table above) |
| `--block-size N` | auto | Pixels per logical-pixel edge |
| `--fps N` | `24` | Frames per second |
| `--width N` | `1920` | Frame width |
| `--height N` | `1080` | Frame height |
| `--output PATH` / `-o` | `<stem>.mp4` | Output MP4 path |
| `--quiet` / `-q` | off | Suppress progress output |
| `--upload` | off | Auto-upload to YouTube after encoding |
| `--credentials PATH` | — | `client_secrets.json` (required with `--upload`) |
| `--token PATH` | `~/.bytevault/token.json` | OAuth token cache |
| `--title TEXT` | filename | YouTube video title |
| `--description TEXT` | auto | YouTube description |
| `--privacy {public,unlisted,private}` | `unlisted` | YouTube privacy setting |

**Examples:**
```bash
# Palette mode, larger blocks for better compression resistance
python main.py encode photo.png --mode palette --block-size 16

# RGB mode for local use only (no YouTube upload)
python main.py encode data.db --mode rgb --output data_rgb.mp4

# Custom resolution
python main.py encode file.txt --width 1280 --height 720 --block-size 4

# Encode and upload in one step
python main.py encode archive.tar.gz \
    --mode binary --block-size 8 \
    --upload --credentials client_secrets.json \
    --privacy private
```

---

### `decode`

```
python main.py decode <source> [options]
```

`<source>` is either:
- A local `.mp4` file path, or
- A YouTube watch URL (`https://www.youtube.com/watch?v=...`)

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir PATH` / `-o` | `.` | Directory to write the recovered file |
| `--cookies PATH` | — | Netscape `cookies.txt` for private/age-gated videos |
| `--quiet` / `-q` | off | Suppress progress output |

The encoding parameters (mode, block size) are **auto-detected** from the video header — no flags needed.

**Examples:**
```bash
# Decode a local file
python main.py decode encoded.mp4 --output-dir ./recovered

# Decode from YouTube URL
python main.py decode "https://www.youtube.com/watch?v=XXXXXXXXXXX" -o ./out

# Private video with cookies
python main.py decode "https://www.youtube.com/watch?v=XXXXXXXXXXX" \
    --cookies cookies.txt --output-dir ./out
```

---

### `upload`

Upload an already-encoded video without re-encoding.

```
python main.py upload <video> --credentials client_secrets.json [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--credentials PATH` | **required** | Path to `client_secrets.json` |
| `--token PATH` | `~/.bytevault/token.json` | OAuth token cache |
| `--title TEXT` | filename stem | YouTube title |
| `--description TEXT` | auto | Description |
| `--tags TEXT` | — | Comma-separated tags |
| `--privacy {public,unlisted,private}` | `unlisted` | Privacy setting |

**Example:**
```bash
python main.py upload encoded.mp4 \
    --credentials client_secrets.json \
    --title "ByteVault: archive.tar.gz" \
    --privacy private
```

---

### `download`

Download a YouTube video without decoding it.

```
python main.py download <url> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir PATH` / `-o` | `.` | Destination directory |
| `--cookies PATH` | — | Cookies file for private videos |
| `--quiet` / `-q` | off | Suppress output |

---

## YouTube API Setup

To use `--upload` or the `upload` subcommand, you need Google OAuth 2.0 credentials.

### Step-by-step

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Enable the **YouTube Data API v3**: APIs & Services → Library → search "YouTube Data API v3" → Enable.
4. Create OAuth credentials: APIs & Services → Credentials → **Create Credentials** → **OAuth client ID**.
   - Application type: **Desktop app**
   - Name: anything (e.g. "ByteVault")
5. Download the JSON file → save it as `client_secrets.json` in the project directory.
6. First run will open your browser for consent. The token is cached at `~/.bytevault/token.json` for future runs.

> **Important:** YouTube requires the video category to be valid and the account to not be restricted. Uploaded videos default to **unlisted** so they won't appear in search results.

---

## Self-Describing Videos

ByteVault videos are **self-describing**: the encoding parameters are stored in the first 128 bytes of the encoded payload (the header), so `decode` never needs extra flags. The header structure:

```
Offset  Size  Field
0       4     Magic bytes: BVI\x01
4       1     Mode (0=binary, 1=rgb, 2=palette)
5       1     Block size
6       2     Filename length
8       64    Original filename (UTF-8)
72      8     Original file size (bytes)
80      4     Padding byte count
84      44    Reserved
```

If the video was downloaded from YouTube, `decode` auto-tries all reasonable (mode, block_size) combinations against the first frame until one produces valid magic bytes.

---

## Limitations & Notes

| Concern | Details |
|---------|---------|
| **File size** | No hard limit, but RAM usage scales with payload size. Keep files under ~500 MB for comfortable single-pass encoding. |
| **YouTube re-encoding** | YouTube always re-encodes. Binary mode + block ≥ 4 survives reliably. RGB mode may corrupt. |
| **Video duration** | Binary mode: ~1 MB ≈ 5 s at 1920×1080, 24 fps, block=4. A 100 MB file ≈ ~8 min. |
| **Terms of Service** | Mass automated uploads may violate YouTube ToS. Use responsibly. |
| **Error correction** | None built-in. Block redundancy acts as implicit ECC. For critical data, compress + encrypt before encoding. |
| **Private videos** | Use `--privacy private`. Download with `--cookies` if the account requires authentication. |
| **No audio** | Encoded videos are silent; YouTube requires audio on some upload paths — ffmpeg will add a silent audio track automatically if needed. |

---

## Architecture

```
ByteVault-Infinite--The-Eternal-Encoder/
├── bytevault/
│   ├── __init__.py       exports
│   ├── palette.py        256-colour palette + nearest-colour lookup
│   ├── encoder.py        file → raw BGR frames → ffmpeg pipe → MP4
│   ├── decoder.py        MP4 → ffmpeg pipe → raw BGR frames → file
│   ├── uploader.py       YouTube Data API v3 OAuth2 upload
│   └── downloader.py     yt-dlp download wrapper
├── main.py               argparse CLI (encode / decode / upload / download)
├── requirements.txt
└── README.md
```

---

## Similar Projects

- [DvorakDwarf/Infinite-Storage-Glitch](https://github.com/DvorakDwarf/Infinite-Storage-Glitch) — the original, written in Rust
- [dzhang314/YouTubeDrive](https://github.com/dzhang314/YouTubeDrive) — Mathematica proof-of-concept
- [ianling/steg-experiments](https://github.com/ianling/steg-experiments) — colour-palette tile approach
- [gasman's gist](https://gist.github.com/gasman/1253b764049cfab3e29739d3f217c9c6) — monochrome + FFV1 + upscale

---

## License

MIT — see [LICENSE](LICENSE).
