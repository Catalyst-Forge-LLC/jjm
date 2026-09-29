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
- [x] Fitted-head route: Hugging Face and Ollama encoders, feature cache, head artifacts with reuse and promotion, teachers (Ollama, LiteLLM), `evaluate_batch`, and `jjm models / fit / heads` (decision-fitted-head-engine)
- [x] Banking77 routing benchmark (`trials/routing/README.md`)
- [ ] Cut envelope overhead in batches (7 ms per request with every feature cached)
- [ ] Faster head fitting (now 2 to 5 minutes on 10,000 examples)
- [ ] Held-out validation report for a head, so a scope can move past `fitted_unvalidated`
- [ ] Re-run the fit trial on the fitted-head route, now that it is part of the engine
- [ ] Decide the next step for fit: a larger or quantized model, a calibration for this scope, or a different criterion shape
- [ ] Owner blind-labels a subset, and a Sonnet re-run supplies the ceiling (brief section 2a)
- [ ] Decide whether to add `accelerate` to the `hf` extra (two upstream AnyJev tests need it)

## Later

- Publishing, the JJM.dev site, and a push to GitHub all wait for the owner's go-ahead.
