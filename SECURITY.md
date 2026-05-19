# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Email **sahilmenon01@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- Affected version(s)

You will receive an acknowledgement within **72 hours** and a resolution timeline within **7 days**. If the vulnerability is confirmed, we will credit you in the release notes unless you prefer otherwise.

## Scope

Areas of particular interest:

- **AES-256-GCM encryption** (`pixelvault/encoder.py`, `pixelvault/decoder.py`) — nonce reuse, key derivation weaknesses
- **Reed-Solomon ECC** (`pixelvault/ecc.py`) — bypass leading to silent data corruption
- **YouTube OAuth credentials** — anything that could expose or mishandle `client_secret.json` or `.youtube_token.json`
- **Arbitrary file write** during decode — path traversal in output filename reconstruction

## Out of Scope

- Issues requiring physical access to the machine
- Theoretical attacks with no practical exploit path
- YouTube Terms of Service violations (these are user responsibility)
