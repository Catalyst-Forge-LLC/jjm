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
- [x] Cut envelope schema-check cost (a no-model request is about 0.9 ms, down from 3.0 ms)
- [x] Faster head fitting (about 8 seconds on 10,000 examples; the penalty is chosen on 2,048 stratified rows)
- [x] Held-out validation report for a head. `jjm accept --scope` records it; a matching scope can require `validated_in_scope`
- [x] Re-run the fit trial on the fitted-head route (`trials/fit_rank/README.md`). On the rendered state the head does not beat always answering Strong; the earlier 0.52 correlation used a different prompt
- [ ] Decide the next step for fit: a larger or quantized model, a calibration for this scope, or a different criterion shape
- [ ] Owner blind-labels a subset, and a Sonnet re-run supplies the ceiling (brief section 2a)
- [ ] Decide whether to add `accelerate` to the `hf` extra (two upstream AnyJev tests need it)

## Later

- The private GitHub repo is https://github.com/Catalyst-Forge-LLC/jjm. The JJM.dev site and a package upload still wait for a clean-machine run of the fitted-head try-it.
