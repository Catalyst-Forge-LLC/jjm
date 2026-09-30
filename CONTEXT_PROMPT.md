# JJM context prompt

Read this first in a new session. The locked plan is `docs/PHASE_1_BRIEF.md`, the design is `specs/jjm-spec-pack/`, and the current phase is in `appledger/profiles/forgetrail.yaml`.

## What JJM is

JJM (Just Judgment Model) is an open-source, local judgment runtime. An application sends a state snapshot and explicit criteria, and gets back weights over allowed outcomes, provenance, and a `pass` / `hold` / `reject` against its own declared requirements. JJM never writes prose on the decision path and never acts. It is a local stand-in for TypeSafe's Jev, with no wire compatibility and no affiliation.

## Stack

- Python 3.10+ with uv. The package is `jjm-engine` and the CLI is `jjm`, under `src/jjm`.
- It depends on `anyjev==0.2.0`, pinned, with no fork. The `hf` extra adds torch (from PyTorch's CUDA 13.0 index on Windows and Linux) and transformers.
- Contract 0.1: the schemas in `src/jjm/schemas` are byte copies of the spec pack's.

## Rules that hold

- Every AnyJev Decider setting is pinned (`DECIDER_SETTINGS`), and a test fails if upstream adds or changes one.
- Each criterion gets a fresh Decider, so no state carries across requests.
- Every envelope passes schema validation and the invariant checks before it is printed.
- Downloads are off unless a profile sets `allow_download: true`.
- Synthetic results are labeled `execution_mode: synthetic`.
- Any change to rendering bumps `COMPILER_VERSION` (now `jjm-compile-2`), which changes the rendering digest.
- Changing the brief requires a decision record in `appledger/records/decision/`.
- Commit after substantive work. Do not push unless the owner asks.

## Where things stand (2026-09-28)

- The runtime, the synthetic route, and the fixture pack all pass (98 tests).
- The real route runs with Qwen3-1.7B on the owner's RTX 4000 Ada laptop GPU (12 GB). The model is at `C:/Users/acmegeek/.cache/huggingface/...`, and its profile is in the gitignored `jjm-profiles.local.json`.
- The fit-rank trial harness is in `trials/fit_rank`, with criterion wording frozen as version 1. The exporter is `pnpm export:fit-trial` in Exec Foundry.
- The production trial (253 jobs, `trials/fit_rank/README.md`) found that neither model's picked rank beats always answering "Strong". Qwen3-4B's weights do track Sonnet's ordering (rank correlation 0.43), on a compressed scale. It takes 5.0 s per job with `shared_prefix: "off"`, against Sonnet's 5.1 s median. Qwen3-1.7B takes 0.72 s.

## Fitted-head route (2026-09-29)

- `decision-fitted-head-engine` adds a `fitted-head` profile: a frozen encoder plus a small fitted head. Code is in `src/jjm/heads/`, `src/jjm/adapters/head_adapter.py`, and `src/jjm/discover.py`.
- No model is hard-coded. Encoders are any Hugging Face checkpoint or Ollama embedding model. Teachers are Ollama models or LiteLLM providers (the optional `providers` extra), and run only outside `evaluate`.
- Heads bind to criterion digest, encoder identity, and method. They are reused on an identical training digest, served only after promotion, and refused with `ARTIFACT_MISMATCH` otherwise. A fit holds out about 20% of each outcome and leaves the head `fitted_unvalidated`. `jjm accept --scope` records that report for one scope. A request that requires `validated_in_scope` then gets `calibrated_estimate` only when its scope matches, and that envelope carries the artifact manifest.
- Banking77 routing (`trials/routing/README.md`): Qwen3-Embedding-0.6B head 97.1% against human labels, about 5 ms per message in bulk. qwen3.5:9b zero-shot 88%, and the AnyJev 1.7B readout 75%. Profiles are in `jjm-profiles.local.json` (gitignored). The data sits in `data/banking77` (gitignored).
- 122 tests pass.

## Useful commands

```text
uv run pytest
uv run jjm --profiles jjm-profiles.local.json doctor
uv run python -m trials.fit_rank.run --input <export.jsonl> --profiles jjm-profiles.local.json \
    --profile local-advisory --hardware "RTX 4000 Ada Laptop 12 GB, CUDA 13.0, bfloat16"
uv run jjm models
uv run python -m trials.routing.bench --help
```
