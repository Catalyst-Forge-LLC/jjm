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

### One pass, then a small head

`features.py` saves one forward pass per job. `probe_eval.py` scores those features against Sonnet on the same 241 jobs. Zero-shot rows never see Sonnet's ranks. Learned rows are ridge heads inside nested cross-validation (10 outer folds, an inner 5-fold choice of ridge strength, 3 shuffles), so no job is scored by a head that trained on it. Forward times are medians on the same GPU.

| Method | Trained on Sonnet? | Rank correlation | AUC, 1–2 vs rest | Exact | Within one | Weighted kappa | ms/job |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Always answer "Strong" | no | 0 | 0.50 | 0.564 | 0.979 | 0 | |
| JJM readout, Qwen3-4B (expected rank) | no | 0.43 | 0.76 | 0.27 | 0.89 | 0.19 | 5000 |
| Listing–candidate cosine, Embedding-0.6B | no | 0.457 | 0.741 | | | | 275 |
| Reranker-0.6B, P(yes) | no | 0.429 | 0.753 | | | | 186 |
| Requirement coverage, mean | no | 0.350 | 0.691 | | | | 275 |
| 1.7B mean hidden state, layer 7 | yes, cross-validated | 0.479 | 0.773 | 0.581 | 0.982 | 0.273 | 366 |
| 4B mean hidden state, layer 18 | yes, cross-validated | 0.522 | 0.808 | 0.622 | 0.978 | 0.362 | 918 |
| Reranker hidden-state head | yes, cross-validated | 0.515 | 0.804 | 0.640 | 0.989 | 0.450 | 186 |

- **Zero-shot, the reranker and the cosine score order jobs about as well as the 4B readout, much faster.** Rank correlation is 0.43 and 0.46 against the readout's 0.43, at 186 ms and 275 ms instead of 5 s. They do not emit a 1–5 label, so exact agreement and kappa are not defined for them.
- **The best trained head is the reranker's hidden state.** Weighted kappa is 0.45 against the readout's 0.19, and exact agreement is 0.64 against always answering "Strong" at 0.56, at 186 ms. That head was fit to predict Sonnet, so this is distillation. It is not an independent judge, and it cannot be more right than Sonnet.
- **Mean-pooled hidden states beat the last token.** The 4B last token at layer 18 had rank correlation −0.02. Adding requirement coverage to the embedding head barely moved the score (0.433 against 0.426 for the listing embedding alone).
- **These heads learn one profile.** The candidate text is the same on every job, so a head trained here will not transfer to another person without their labels. Blind owner labels are still the check that matters.

- **Caveat:** candidate material in the export is current. Sonnet ranked many jobs when they were imported, possibly from an older resume or career direction.

### The same jobs through the fitted-head engine

`head_run.py` renders each job as a JJM state, encodes it with the reranker's last token (8,192 token cap; nothing was truncated), and fits the engine's softmax head. Five folds. A job is scored only by a head that did not train on it. The two Long Shot jobs stay in every fit, because the fitter keeps at least two examples of an outcome, so the scored set is 239. Sonnet's ranks are the teacher's labels.

| Method | Rank correlation | AUC, 1–2 vs rest | Exact | Within one | Weighted kappa | ms/job |
| --- | --- | --- | --- | --- | --- | --- |
| Always answer "Strong", these 239 jobs | 0 | 0.50 | 0.569 | 0.987 | 0 | |
| Engine softmax head, expected rank | 0.259 | 0.660 | 0.498 | 0.941 | 0.133 | 365 |
| Same vectors, ridge, as in the table above | 0.141 | 0.583 | 0.553 | 0.976 | −0.009 | 365 |

Exact-match 95% interval for the softmax head is 0.43 to 0.56. The kappa interval is 0.01 to 0.26. The head's picks were Perfect 26, Strong 144, Good 69, and neither Stretch nor Long Shot.

- **This rendering does not carry the signal the earlier reranker prompt did.** That prompt asked whether the listing fit the candidate, with the candidate as the query and the listing as the document, and a ridge on its hidden state reached correlation 0.52. The same ridge on these JJM-rendered vectors reaches 0.14. The softmax head's probabilities order jobs a little (correlation 0.26) and its picked rank does not beat always answering "Strong" (0.50 against 0.57).
- **Speed:** the first encode of 191 jobs took 70 seconds, about 365 ms each, on the RTX 4000 Ada. Once a feature is cached, an envelope is about 16 ms.

### Same jobs, same words, 2026-09-30

Nimble 9B and Tev1 4B answered the five fit options through Ollama `/v1/systemone`, with no labels. The fitted head is the reranker softmax from the section above. Two texts: the JJM rendered state, and the dedicated prompt (candidate as the query, listing as the document). Sonnet is the teacher. Every decision-model reply parsed. Stock Tev1 refuses prompts over 2,050 tokens, so Tev1 ran as a local alias with the same weights and an 8,192-token window, matching Nimble. Nothing was truncated. The head on the dedicated prompt scored 239 jobs; the decision models scored all 241.

| Method | Text | Exact | Within one | Weighted kappa | Expected-rank correlation | ms/job |
| --- | --- | --- | --- | --- | --- | --- |
| Always answer "Strong" | | 0.564 of 241; 0.569 of 239 | | 0 | 0 | |
| Nimble 9B | rendered state | 0.241 | 0.884 | 0.388 | 0.621 | 2200 |
| Tev1 4B | rendered state | 0.162 | 0.788 | 0.161 | 0.630 | 1570 |
| Fitted softmax head | rendered state | 0.498 of 239 | 0.941 | 0.133 | 0.259 | 365 encode |
| Nimble 9B | dedicated prompt | 0.452 | 0.888 | 0.435 | 0.602 | 1100 |
| Tev1 4B | dedicated prompt | 0.295 | 0.900 | 0.315 | 0.605 | 690 |
| Fitted softmax head | dedicated prompt | 0.460 of 239 | 0.950 | 0.104 | 0.201 | 390 encode |

The head's criterion prefix sits in front of the dedicated prompt. The earlier ridge correlation of 0.52 used the reranker's chat wrapper and read the last token at the assistant position, so it is not this row.

- **The dedicated prompt helps the picked letter.** Nimble's exact agreement rises from 0.24 to 0.45, and Tev1's from 0.16 to 0.29. Neither picked letter beats always answering "Strong."
- **The decision models' probabilities track Sonnet on either text,** at about 0.60. The fitted head's probabilities do not: 0.26 on the rendered state and 0.20 on the dedicated prompt. On that prompt the head still mostly says Strong (160 of 239).
- **Speed:** after the encode, the head is the cheap one to serve. Nimble and Tev1 pay about a second on every job.

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
uv run python -m trials.fit_rank.head_run --input export.jsonl --profiles jjm-profiles.local.json
uv run python -m trials.fit_rank.head_run --input export.jsonl --profiles jjm-profiles.local.json --text probe
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
