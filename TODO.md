# TODO

## Build (current phase)

- [x] Contract parsing, validation, digests, and criterion compiler
- [x] Synthetic adapter, gate engine, envelope self-check, and CLI (`evaluate`, `capabilities`, `doctor`)
- [x] AnyJev L0 adapter with every setting pinned
- [x] All 25 spec-pack fixtures pass the invariant port
- [x] Real Transformers route on the owner's GPU (Qwen3-1.7B, revision pinned)
- [x] Fit-rank trial harness, with criterion wording frozen as version 1
- [x] Exec Foundry read-only exporter (`pnpm export:fit-trial`)
- [ ] Pick and download a model large enough for fit (owner approval needed; Qwen3-4B in bfloat16 fits 12 GB)
- [ ] Export production jobs with Sonnet ranks (`--prod-db`) and run the trial
- [ ] Owner blind-labels a subset, and a Sonnet re-run supplies the ceiling (brief section 2a)
- [ ] Decide whether to add `accelerate` to the `hf` extra (two upstream AnyJev tests need it)

## Later

- Publishing, the JJM.dev site, and a push to GitHub all wait for the owner's go-ahead.
