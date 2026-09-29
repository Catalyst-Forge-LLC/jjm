# Fit-rank shadow trial

This harness runs the trial protocol from `docs/PHASE_1_BRIEF.md` §2a. JJM reads the same inputs Exec Foundry gives Sonnet and picks one of five fit levels. The harness then compares that pick with Sonnet's rank and with the owner's labels. It never writes anything back to Exec Foundry.

## Status

- **Criterion wording is a draft** (`criterion.py`, `CRITERION_STATUS = "draft"`). Results only count once the owner freezes the five level descriptions. Every report prints the wording status and the criterion digest, so runs under different wording can't be mixed up.
- **The export does not exist yet.** Exec Foundry needs a small read-only script that writes the JSONL format described below. That script belongs in the Exec Foundry repo and hasn't been written.
- **Real runs need a model.** Only the synthetic profile runs today. Its numbers come from hashed weights and say nothing about fit; the report says so at the top.

## Deviation from the brief

§2a says jobs with no candidate material should come out as `not_observable`. The harness instead declares the candidate evidence required (`required_evidence_ids`). A job with no resume, profile, supplementary context, or work preferences then abstains with `MISSING_REQUIRED_EVIDENCE` before any model runs. This check is deterministic and costs nothing, so the model is never asked to guess a middle rank from the listing alone. The owner still needs to approve this change.

## Export format

The export is one JSON object per line. Unknown fields are refused, which keeps Sonnet's justification from reaching the harness by accident.

| Field | Type | Source in Exec Foundry |
| --- | --- | --- |
| `job_id` | string | job id |
| `exported_at` | RFC 3339 time | export time |
| `url` | string or null | listing URL |
| `listing_text` | string | scraped listing text (truncated here to 120,000 characters, as for Sonnet) |
| `listing_hints` | object or null | `title`, `company`, `workType` from the scrape |
| `location_instructions` | string or null | location scoring instructions |
| `base_resume`, `candidate_profile`, `supplementary_context`, `work_preferences` | string or null | the candidate material given to Sonnet |
| `sonnet_rank` | 1–5 or null | `fitJustificationRank` (the baseline) |
| `board_rank` | 1–5 or null | `fitRank` on the board. It counts as an owner override only where it differs from `sonnet_rank`. |
| `owner_label` | 1–5 or null | blind owner label, if collected |
| `sonnet_rerun_rank` | 1–5 or null | a second Sonnet run on the same inputs (the agreement ceiling) |

`tests/data/fit_export.sample.jsonl` is a made-up example.

## Running

```text
uv run python -m trials.fit_rank.run --input export.jsonl --profile synthetic
uv run python -m trials.fit_rank.run --input export.jsonl --profiles my-profiles.json \
    --profile local-advisory --hardware "RTX 4000 Ada 12 GB, CUDA 12"
```

Output goes to `runs/fit-rank-<time>/`, which git ignores:

- `envelopes.jsonl`: the full JJM response for each job.
- `rows.json`: one line per job with JJM's rank, weights, labels, latency, and context tokens.
- `summary.json` and `summary.md`, which contain:
  - Status counts.
  - Exact and within-one agreement, quadratic weighted kappa, and mean signed offset, with seeded bootstrap 95% intervals. These are reported for JJM against Sonnet, JJM against the owner, Sonnet against the owner, JJM against the owner's overrides, and Sonnet's re-run against Sonnet.
  - Model load time, first request, and warm p50 and p95 latency.
  - A determinism check that re-runs up to `--repeat` jobs.

A real profile must name its hardware, so the latency figures always say what they were measured on.
