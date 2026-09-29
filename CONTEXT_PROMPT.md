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
- Qwen3-1.7B cannot judge fit: it answered "Strong Fit" for all 70 local jobs. The trial needs a larger model and a production export with Sonnet ranks.

## Useful commands

```text
uv run pytest
uv run jjm --profiles jjm-profiles.local.json doctor
uv run python -m trials.fit_rank.run --input <export.jsonl> --profiles jjm-profiles.local.json \
    --profile local-advisory --hardware "RTX 4000 Ada Laptop 12 GB, CUDA 13.0, bfloat16"
```
