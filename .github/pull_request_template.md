## Summary

<!-- What does this PR do? Link the issue it closes if applicable. -->
Closes #

## Changes

<!-- Bullet-point list of what changed and why. -->
-

## Test plan

- [ ] Ran `pytest tests/test_smoke.py` — all pass
- [ ] Ran a local encode→decode roundtrip (`python main.py encode <file> && python main.py decode vault/encoded/<stem>.mp4`)
- [ ] If adding a new encoding mode: tested bit-exact recovery
- [ ] If touching ECC: verified error correction still works at nsym/2 errors per block
- [ ] If touching YouTube upload/download: tested with a real video (or noted why not)
- [ ] Updated `CHANGELOG.md` under `[Unreleased]`
- [ ] No secrets, credentials, or large binary files committed
