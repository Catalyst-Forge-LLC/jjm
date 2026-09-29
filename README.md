# JJM — Just Judgment Model

An open-source judgment runtime. Just judgment. Your application makes the decision.

**Status: pre-release.** Nothing is published. The synthetic route works. The local-model route is built and tested against AnyJev's fake backend, but it has not been run on a real model yet.

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

Real profiles come from an operator file, not from the request. See `examples/jjm-profiles.example.json`. Pass it with `--profiles` or `JJM_PROFILES`. The route needs the `hf` extra (torch and transformers). Downloads are off unless the profile sets `allow_download: true`.

```bash
uv sync --extra hf
uv run jjm --profiles my-profiles.json doctor
uv run jjm --profiles my-profiles.json evaluate --request request.json
```

This route uses AnyJev's L0 readout with permutation only: the options are shown in every rotation, and the letter-token probabilities are averaged in log space. There is no prior correction. See `UPSTREAM.md` for the pinned AnyJev release and every setting JJM fixes.

## Layout

- `src/jjm/`: contract, compiler, adapters, engine, CLI.
- `specs/jjm-spec-pack/`: the design spec and the canonical schemas.
- `docs/PHASE_1_BRIEF.md`: what v1 is and is not.
- `UPSTREAM.md`: the AnyJev pin, tests run, and reconciliation.

## Credits

JJM depends on [AnyJev](https://github.com/nokia-applied-research/AnyJev), Apache-2.0, by Jiamu Zhang, Tianze Yang, Yucheng Shi, and Liang Wu. JJM is not affiliated with TypeSafe AI or Jev. See `NOTICE`.

License: Apache-2.0.
