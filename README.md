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

A request names its `profile_id`, and the CLI refuses a `--profile` that differs from it rather than switching profiles quietly.

```bash
uv sync --extra hf
uv run jjm --profiles my-profiles.json doctor
uv run jjm --profiles my-profiles.json evaluate --request request.json
```

This route uses AnyJev's L0 readout with permutation only: the options are shown in every rotation, and the letter-token probabilities are averaged in log space. There is no prior correction. See `UPSTREAM.md` for the pinned AnyJev release and every setting JJM fixes.

## Real-model results so far

These were run with Qwen3-1.7B at revision `70d244cc`, in bfloat16 on a 12 GB RTX 4000 Ada laptop GPU with CUDA 13.0.

- **Route example:** the billing ticket selects `billing` at 0.944 and passes. The model's raw pick changed with option order (`order_flip_raw` 1.0), which is the bias the rotations average out. The first run picked `other` with near-certainty. That traced to an explicit "(no evidence items supplied)" line, and the compiler (`jjm-compile-2`) now leaves the evidence section out when there is none.
- **Fit rank on 70 local Exec Foundry jobs:** every job was evaluated. Model load took 16 s, the warm median was 1.8 s per job (p95 4.3 s), prompts ran about 3,200 tokens, and repeats were identical. The model answered "Strong Fit" for all 70, and its expected rank did not track the board rank (weighted kappa 0). A 1.7B model is too small for this judgment.

## Layout

- `src/jjm/`: contract, compiler, adapters, engine, CLI.
- `specs/jjm-spec-pack/`: the design spec and the canonical schemas.
- `docs/PHASE_1_BRIEF.md`: what v1 is and is not.
- `UPSTREAM.md`: the AnyJev pin, tests run, and reconciliation.
- `trials/fit_rank/`: the Exec Foundry fit-rank shadow trial.

## Credits

JJM depends on [AnyJev](https://github.com/nokia-applied-research/AnyJev), Apache-2.0, by Jiamu Zhang, Tianze Yang, Yucheng Shi, and Liang Wu. JJM is not affiliated with TypeSafe AI or Jev. See `NOTICE`.

License: Apache-2.0.
