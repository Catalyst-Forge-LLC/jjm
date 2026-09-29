# Fit-rank shadow trial

This harness runs the trial protocol from `docs/PHASE_1_BRIEF.md` §2a. JJM reads the same inputs Exec Foundry gives Sonnet and picks one of five fit levels. The harness then compares that pick with Sonnet's rank and with the owner's labels. It never writes anything back to Exec Foundry.

## Status

- **Criterion wording is frozen** as version 1 (`criterion.py`). Any change needs a new version. Every report prints the wording status and the criterion digest, so runs under different wording can't be mixed up.
- **The exporter lives in Exec Foundry:** `pnpm export:fit-trial --email <you> [--prod-db]` writes `temp/jjm-fit-export.jsonl` there. It is read-only and makes no model calls. It rebuilds each job's fit inputs with the same helpers `reassessJobFit` uses. The candidate material is today's, which may differ from what Sonnet saw when it ranked an older job.
- **Local data has no Sonnet ranks.** The 70 jobs in local PocketBase predate `fit_justification_rank`, so only the board-rank comparison has data. The real baseline needs a production export.
- **First real run:** Qwen3-1.7B answered "Strong Fit" for all 70 local jobs (weighted kappa 0 against the board). The harness and timing work, but that model is too small for this judgment.

## Missing candidate evidence

The criterion declares the candidate evidence required (`required_evidence_ids`). A job with no resume, profile, supplementary context, or work preferences abstains with `MISSING_REQUIRED_EVIDENCE` before any model runs. The check is deterministic and free, so the model is never asked to guess a middle rank from the listing alone. This replaces the `not_observable` wording in brief §2a (`decision-fit-evidence-abstention`).

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
  - Exact and within-one agreement, quadratic weighted kappa, and mean signed offset, with seeded bootstrap 95% intervals. These are reported for JJM against Sonnet, the owner, the board rank, and the owner's overrides, plus Sonnet against the owner and Sonnet's re-run against Sonnet.
  - Model load time, first request, and warm p50 and p95 latency.
  - A determinism check that re-runs up to `--repeat` jobs.

A real profile must name its hardware, so the latency figures always say what they were measured on.
