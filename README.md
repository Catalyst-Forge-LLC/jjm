# JJM — Just Judgment Model

An open-source judgment runtime. Just judgment. Your application makes the decision.

**Status: pre-release.** Nothing is published. The synthetic route works. The local-model route has run on a real model (Qwen3-1.7B on an RTX 4000 Ada laptop GPU); see [Real-model results so far](#real-model-results-so-far).

## What it does

Your application sends a JSON request with a state snapshot and explicit criteria. JJM returns a JSON envelope. For each criterion, the envelope reports:

- A weight over every allowed outcome.
- What produced it: method, model, runtime, and digests of the inputs.
- Whether it meets your declared requirements (`pass`, `hold`, or `reject`), with a reason for each failed check.

JJM does not write prose on the decision path and never takes an action. A `pass` means "meets the requirements you declared." It does not mean the answer is true, and it does not authorize anything.

Criteria in v1:

- `choice`: two to 26 caller-defined options.
- `assertion`: three fixed outcomes. `supported` and `contradicted` mean the supplied evidence settles the question one way or the other. `not_observable` means it does not settle it.

Weights are labeled `normalized_readout`: normalized model output, not calibrated probabilities. A request that requires validated calibration abstains.

## Quickstart (synthetic, no model download)

Needs Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jjm evaluate --request examples/route-ticket.json
uv run jjm evaluate --request examples/export-completed.json
uv run pytest
```

The `synthetic` profile is a contract demonstration. Its weights are a hash of the input, so they carry no meaning. In the first example it picks `technical` for a billing ticket, and the result is held because its top weight is below the requested 0.5. In the second example the required `export-receipt` evidence is missing, so the criterion abstains before any evaluation runs.

Requests can also come on stdin: `uv run jjm evaluate < request.json`.

## Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | An envelope was printed. Read each result's `acceptance`. Exit 0 is not approval. |
| 2 | Invalid request or configuration. stdout has `{"error": {...}}` naming the field. |
| 3 | Execution failure, such as a model that cannot load. |

## Local model route

Real profiles come from an operator file, not from the request. See `examples/jjm-profiles.example.json`. Pass it with `--profiles` or `JJM_PROFILES`. The route needs the `hf` extra (torch and transformers). On Windows and Linux, uv installs torch from PyTorch's CUDA 13.0 index (`pyproject.toml`), because PyPI's Windows wheels are CPU-only. Downloads are off unless the profile sets `allow_download: true`.

A profile may set `"shared_prefix": "off"`. That scores each rotation as a full prompt instead of sharing one prefix cache across all K rotations. The weights are the same, but it needs far less GPU memory on long prompts. The setting is part of the profile's identity and the effective config digest.

A request names its `profile_id`, and the CLI refuses a `--profile` that differs from it rather than switching profiles quietly.

```bash
uv sync --extra hf
uv run jjm --profiles my-profiles.json doctor
uv run jjm --profiles my-profiles.json evaluate --request request.json
```

This route uses AnyJev's L0 readout with permutation only: the options are shown in every rotation, and the letter-token probabilities are averaged in log space. There is no prior correction. See `UPSTREAM.md` for the pinned AnyJev release and every setting JJM fixes.

## Fitted-head route

When you have labelled examples for a criterion, a `fitted-head` profile is much faster and, on the tasks tried so far, more accurate. A frozen encoder reads the state once, and a small head fitted on your examples turns that into weights. The encoder can be any Hugging Face checkpoint or Ollama embedding model; nothing is hard-coded.

```bash
uv run jjm models                       # what is installed locally, with suggested roles
uv run jjm --profiles p.json fit --profile route-embed --criterion c.json --examples ex.jsonl --promote
uv run jjm --profiles p.json accept --profile route-embed --criterion c.json --scope my-app
uv run jjm --profiles p.json heads --profile route-embed
uv run jjm --profiles p.json evaluate-batch --requests requests.jsonl
```

- `--examples` is JSONL of `{"state": {...}, "label": "<outcome id>"}`. Or pass `--teacher <id>` to have a teacher model label the states instead, and the artifact records that the labels came from it.
- A head is bound to the exact criterion digest, encoder identity, and head method. Fitting the same examples again returns the stored head. A head serves only after `--promote`; otherwise the criterion abstains with `ARTIFACT_MISMATCH`.
- Results report `calibration.status: fitted_unvalidated` until you accept the held-out report. Fitting holds out about 20% of each outcome, scores it, and leaves those rows out of the penalty choice and the fit. `jjm accept --scope <scope_id>` records that the report applies to that scope. A request that sets `calibration` to `validated_in_scope` then gets `calibrated_estimate` weights when its `scope_id` matches, and abstains otherwise. The numbers are the same head output; the status says an accepted report exists for this scope.
- Teachers are listed under `teachers` in the profiles file: a local Ollama model, or any provider [LiteLLM](https://docs.litellm.ai/) supports (`uv sync --extra providers`). Teachers label examples. They never run inside `evaluate`.
- Speed settings live in the profile and its identity: `batch_size`, `max_tokens`, `layer` (early exit, as a layer count or depth fraction), `prefix_reuse`, and a feature `cache` directory.

See `examples/jjm-profiles.routing.example.json`.

## Real-model results so far

These were run with Qwen3-1.7B at revision `70d244cc`, in bfloat16 on a 12 GB RTX 4000 Ada laptop GPU with CUDA 13.0.

- **Route example:** the billing ticket selects `billing` at 0.944 and passes. The model's raw pick changed with option order (`order_flip_raw` 1.0), which is the bias the rotations average out. The first run picked `other` with near-certainty. That traced to an explicit "(no evidence items supplied)" line, and the compiler (`jjm-compile-2`) now leaves the evidence section out when there is none.
- **Fit rank on 70 local Exec Foundry jobs:** every job was evaluated. Model load took 16 s, the warm median was 1.8 s per job (p95 4.3 s), prompts ran about 3,200 tokens, and repeats were identical. The model answered "Strong Fit" for all 70, and its expected rank did not track the board rank (weighted kappa 0). A 1.7B model is too small for this judgment.
- **Fit rank on 253 production jobs, against Sonnet 4.6:** the ranks JJM picks do not beat always answering "Strong" (56% exact).
  - Qwen3-4B's weights still order jobs the way Sonnet does (rank correlation 0.43, AUC 0.76 for top-two versus the rest), on a scale compressed toward Perfect.
  - Latency was 5.0 s per job for Qwen3-4B and 0.72 s for Qwen3-1.7B. Sonnet's measured median is 5.1 s, at about $0.012 per call.
  - With the model held fixed, JJM's readout beat letting Qwen3-4B write the rank from Sonnet's own prompt. Within one rank was 89% against 69%, weighted kappa 0.19 against 0.10, and rank correlation 0.38 against 0.14.
  - One forward pass is much faster and, zero-shot, orders jobs about as well. A 0.6B reranker's yes/no probability has rank correlation 0.43 with Sonnet at 186 ms, against the 4B readout's 0.43 at 5 s. A small head trained on Sonnet's ranks (cross-validated) reaches weighted kappa 0.45 at that same 186 ms. That head is distillation, not an independent judge.
  - Details are in `trials/fit_rank/README.md`.
- **Banking77 routing, eight teams, against human labels** (`trials/routing/README.md`): a Qwen3-Embedding-0.6B head reached 97.1% on 3,080 test messages, at about 5 ms per message in bulk and 11 ms per request through `evaluate_batch`. On a 400-message subset, qwen3.5:9b asked directly got 88% at about 400 ms, and the AnyJev readout on Qwen3-1.7B got 75%. With 16 labelled messages per team, the head already beat the 9B model.

## Layout

- `src/jjm/`: contract, compiler, adapters, engine, CLI.
- `specs/jjm-spec-pack/`: the design spec and the canonical schemas.
- `docs/PHASE_1_BRIEF.md`: what v1 is and is not.
- `UPSTREAM.md`: the AnyJev pin, tests run, and reconciliation.
- `trials/fit_rank/`: the Exec Foundry fit-rank shadow trial.
- `trials/routing/`: the Banking77 routing benchmark (data not included; CC-BY-4.0, PolyAI).

## Credits

JJM depends on [AnyJev](https://github.com/nokia-applied-research/AnyJev), Apache-2.0, by Jiamu Zhang, Tianze Yang, Yucheng Shi, and Liang Wu. JJM is not affiliated with TypeSafe AI or Jev. See `NOTICE`.

License: Apache-2.0.
