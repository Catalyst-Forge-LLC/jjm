# TODO

## Build (current phase)

- [x] Contract parsing, validation, digests, and criterion compiler
- [x] Synthetic adapter, gate engine, envelope self-check, and CLI (`evaluate`, `capabilities`, `doctor`)
- [x] AnyJev L0 adapter with every setting pinned
- [x] All 25 spec-pack fixtures pass the invariant port
- [x] Real Transformers route on the owner's GPU (Qwen3-1.7B, revision pinned)
- [x] Fit-rank trial harness, with criterion wording frozen as version 1
- [x] Exec Foundry read-only exporter (`pnpm export:fit-trial`)
- [x] Download Qwen3-4B (revision 1cfa9a72) and add a `shared_prefix` profile option so it fits 12 GB
- [x] Export production jobs with Sonnet ranks (`--prod-db`) and run the trial with 4B and 1.7B
- [ ] Decide the next step for fit: a larger or quantized model, a calibration for this scope (not in v1), or a different criterion shape
- [ ] Owner blind-labels a subset, and a Sonnet re-run supplies the ceiling (brief section 2a)
- [ ] Decide whether to add `accelerate` to the `hf` extra (two upstream AnyJev tests need it)

## Later

- Publishing, the JJM.dev site, and a push to GitHub all wait for the owner's go-ahead.
