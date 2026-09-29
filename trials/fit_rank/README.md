# Fit-rank shadow trial

This harness runs the trial protocol from `docs/PHASE_1_BRIEF.md` §2a. JJM reads the same inputs Exec Foundry gives Sonnet and picks one of five fit levels. The harness then compares that pick with Sonnet's rank and with the owner's labels. It never writes anything back to Exec Foundry.

## Status

- **Criterion wording is frozen** as version 1 (`criterion.py`). Any change needs a new version. Every report prints the wording status and the criterion digest, so runs under different wording can't be mixed up.
- **The exporter lives in Exec Foundry:** `pnpm export:fit-trial --email <you> [--prod-db]` writes `temp/jjm-fit-export.jsonl` there. It is read-only and makes no model calls. It rebuilds each job's fit inputs with the same helpers `reassessJobFit` uses. The candidate material is today's, which may differ from what Sonnet saw when it ranked an older job.
- **Local data has no Sonnet ranks.** The 70 jobs in local PocketBase predate `fit_justification_rank`, so only the board-rank comparison has data. The real baseline needs a production export.
- **First real run:** Qwen3-1.7B answered "Strong Fit" for all 70 local jobs (weighted kappa 0 against the board). The harness and timing work, but that model is too small for this judgment.

## Results: production export, 2026-09-29

These cover the owner's 253 production jobs, 241 of them with a Sonnet rank. They ran on an RTX 4000 Ada laptop GPU (12 GB, CUDA 13.0, bfloat16) with criterion version 1. All 253 jobs were evaluated with no abstentions, and 10 of 10 repeats were identical. There are no owner labels yet, so Sonnet is the reference here, not ground truth.

| | Qwen3-4B | Qwen3-1.7B | Always answer "Strong" |
| --- | --- | --- | --- |
| Exact match with Sonnet | 0.27 [0.22, 0.33] | 0.47 [0.41, 0.54] | 0.56 |
| Within one rank | 0.89 | 0.89 | 0.98 |
| Weighted kappa | 0.19 | 0.24 | 0 |
| Mean offset (positive = JJM less favorable) | -0.79 | +0.41 | |
| Rank correlation, expected rank vs Sonnet | 0.43 | 0.30 | |
| AUC, Sonnet 1–2 vs 3–5, from expected rank | 0.76 | 0.67 | 0.5 |
| Warm latency p50 / p95 | 5.0 s / 5.7 s | 0.72 s / 0.89 s | |

- **Sonnet's ranks are concentrated:** 33 Perfect, 136 Strong, 67 Good, 3 Stretch, 2 Long Shot. Always answering "Strong" matches 56% exactly, so exact agreement on its own says little.
- **The labels JJM picks are not usable yet.** The 4B model only ever answered Perfect (145) or Strong (96). The 1.7B mostly answered Strong or Good, and gave 21 Long Shots that Sonnet did not.
- **The 4B weights carry ranking signal.** Its expected rank rises with Sonnet's rank at every step (1.19, 1.32, 1.59, 1.79, 2.00), but the scale is compressed toward Perfect. Mapping that onto five levels would need a fitted calibration for this scope, which v1 does not include.
- **Speed:**
  - Sonnet 4.6 fit re-assessments: p50 5.1 s and p95 12.2 s, about $0.012 per call (9 calls in Exec Foundry's `llm_usage`). Sonnet also writes the justification, so the task is not identical.
  - Qwen3-4B matches Sonnet's median with a much tighter tail. It runs with `shared_prefix: "off"`, because five copies of a shared prefix cache spill past 12 GB.
  - Qwen3-1.7B, with the shared prefix, is about 7 times faster than Sonnet.
  - Either local model costs nothing per call once downloaded.
### Same model, two methods

`generate_baseline.py` gives Qwen3-4B the exact `reassessJobFit` prompt Sonnet gets and parses the `fitRank` it writes. Decoding is greedy with thinking off; Sonnet runs at its default temperature. This holds the model fixed and changes only the method. The model and the 241 Sonnet-ranked jobs are the same as above.

| Qwen3-4B | JJM readout | Writes the rank (Sonnet's prompt) |
| --- | --- | --- |
| Exact match with Sonnet | 0.27 [0.22, 0.33] | 0.18 [0.13, 0.23] |
| Within one rank | 0.89 | 0.69 |
| Weighted kappa | 0.19 [0.14, 0.25] | 0.10 [0.01, 0.20] |
| Rank correlation with Sonnet | 0.38 (0.43 from expected rank) | 0.14 |
| Ranks used | 1: 145, 2: 96 | 1: 107, 2: 9, 3: 57, 4: 68 |
| Warm latency p50 / p95 | 5.0 s / 5.7 s (5 prompts, shared prefix off) | 1.4 s / 1.7 s to the rank; 17.3 s / 24.1 s for the full JSON with justification (30 jobs) |
| Repeats identical | 10 of 10 | 10 of 10 |

- **With the model held fixed, JJM's method beats generation on every agreement measure.** The written ranks are erratic: mostly 1 or 4, almost never Sonnet's most common 2. The readout is compressed toward Perfect, but it stays ordered.
- **Rank only versus rank plus justification:** stopping generation right after `fitRank` gave the same rank on 30 of 30 jobs as letting it write the full JSON. That check shows the rank-only run changes nothing.
- **Speed:** JJM scores five full prompts because the shared prefix does not fit 12 GB with this model. Generation stopped at the rank needs one prompt pass plus six tokens. Writing the justification the way Sonnet does takes 17 s locally.

- **Caveat:** candidate material in the export is current. Sonnet ranked many jobs when they were imported, possibly from an older resume or career direction.

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
