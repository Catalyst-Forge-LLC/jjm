<img width="254" alt="JJM" src="brand/jjm-wordmark.png#gh-light-mode-only">
<img width="254" alt="JJM" src="brand/jjm-wordmark-on-dark.png#gh-dark-mode-only">

# JJM — Just Judgment Machine

An open-source judgment runtime. Just judgment. Your application makes the decision.

**Status: alpha.** This repository is public so people can try it and argue with the numbers. The synthetic quickstart needs no model. The eight-ticket fitted-head walkthrough downloads one small encoder and has been run on the author's machine. Measured runs are in [Real-model results so far](#real-model-results-so-far). Read [Limits of the numbers](#limits-of-the-numbers) for what those runs cover.

## PE:TL;DR

The payoff is a fast, solid judgment an application can use. Fit it on cases you have already labeled, in the buckets you use. JJM does not try to talk. It makes the judgment: a weight for each option you allowed, the method and the model that produced it, and a pass, hold, or reject against checks you declared. Your application decides what happens next.

## What you might hand it

- A support note, to sort into the teams you already use.
- A receipt, a log, or a short passage, to see whether it settles the question.
- An application or a review, to place into the ratings you defined, once you have labelled a few.
- A tool result or a document snippet, to return as one of the answers you allowed.

## What it does

Your application sends a JSON request with a state snapshot and explicit criteria. JJM returns a JSON envelope. For each criterion, the envelope reports:

- A weight over every allowed outcome.
- What produced it: method, model, runtime, and digests of the inputs.
- Whether it meets your declared requirements (`pass`, `hold`, or `reject`), with a reason for each failed check.

JJM does not write prose on the decision path and never takes an action. The request names the allowed options, so the reply is this envelope rather than text you then parse. A `pass` means "meets the requirements you declared." It does not mean the answer is true, and it does not authorize anything.

Criteria in v1:

- `choice`: two to 26 caller-defined options.
- `assertion`: three fixed outcomes. `supported` and `contradicted` mean the supplied evidence settles the question one way or the other. `not_observable` means it does not settle it.

Weights are labeled `normalized_readout`: normalized model output, not calibrated probabilities. Accepting a held-out report does not change that label. A request that requires assessed probability calibration abstains, because JJM does not measure that yet.

## Quickstart (synthetic, no model download)

Needs Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jjm evaluate --request examples/route-ticket.json
uv run jjm evaluate --request examples/export-completed.json
uv run pytest
```

The `synthetic` profile is a contract demonstration. Its weights are a hash of the input, so they carry no meaning. On the current build the first example selects `billing` at about 0.892 and the check passes the requested 0.5. That pass is the hash clearing a threshold, not a judgment that the ticket is billing. In the second example the required `export-receipt` evidence is missing, so the criterion abstains before any evaluation runs.

The request, shortened:

```json
{
  "state": {
    "content": {
      "ticket": "I was charged twice for the same subscription."
    }
  },
  "criteria": [{
    "id": "route-ticket",
    "kind": "choice",
    "question": "Which team should own this ticket?",
    "options": [
      { "id": "billing" },
      { "id": "technical" },
      { "id": "other" }
    ]
  }],
  "requirements": { "min_top_mass": 0.5 }
}
```

The return, shortened and rounded. The command prints the full envelope, including digests and checks.

```json
{
  "results": [{
    "criterion_id": "route-ticket",
    "prediction": {
      "weights": { "billing": 0.892, "technical": 0.005, "other": 0.103 },
      "semantics": "normalized_readout",
      "selected": "billing"
    },
    "acceptance": { "status": "pass" },
    "provenance": {
      "method": "mock:normalized",
      "model": { "id": "jjm/synthetic-no-model" }
    }
  }]
}
```

Requests can also come on stdin: `uv run jjm evaluate < request.json`.

## Fitted head on eight labelled tickets

This exercises `jjm fit` without the Banking77 download or any private export. Eight invented tickets, two teams. The first run downloads `Qwen/Qwen3-Embedding-0.6B` (the profile sets `allow_download: true`) and fits on CPU. Heads and the feature cache go under `runs/try/`, which git ignores. Eight rows show the command. They are not a benchmark.

```bash
uv sync --extra hf
uv run jjm --profiles examples/try/profile.json fit \
    --profile try-embed --criterion examples/try/team.criterion.json \
    --examples examples/try/examples.jsonl --promote
uv run jjm --profiles examples/try/profile.json evaluate --request examples/try/request.json
```

The envelope stays `fitted_unvalidated` until `jjm accept --scope`. A request that requires `accepted_in_scope` abstains until that scope is accepted. Acceptance records the held-out report. The weights stay `normalized_readout`.

## Limits of the numbers

- The eight-ticket walkthrough has been run on the author's machine. It downloads one small encoder and fits those eight rows.
- Routing was measured on Banking77, eight teams, human labels, on one laptop GPU. The fitted head used the training labels. About 64 labels per team is where it lines up with Nimble on that set.
- Fit rank was measured on one candidate. Agreement with the teacher is not correctness, and it moved with how the job was written.

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
- Results report `validation.status: fitted_unvalidated` until you accept the held-out report. Fitting holds out about 20% of each outcome, scores it, and leaves those rows out of the penalty choice and the fit. `jjm accept --scope <scope_id>` records that the report applies to that scope. A request that sets `validation` to `accepted_in_scope` passes when its `scope_id` matches, and abstains otherwise. The weights stay `normalized_readout`. `probability_calibration` stays `not_assessed`. A weight of 0.95 is not a measurement that the outcome occurs about 95% of the time. Asking for `assessed_in_scope` abstains until a calibrator and its reliability report exist.
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
- **Banking77 routing, eight teams, against human labels** (`trials/routing/README.md`), on one laptop GPU:

  | | Right | Time per note |
  | --- | --- | --- |
  | Fitted head | 97.1% of 3,080 | about 5 ms in bulk |
  | Nimble, a 9B decision model | 91.6% of 3,080 | 694 ms |
  | qwen3.5:9b, asked to pick | 88% of 400 | 390 ms |

  Banking77 here is 77 intents mapped to eight teams. The head had seen the human training labels. Nimble and the chat model had seen none. About 64 labelled messages per team, 512 in all, is where the embedding head matches Nimble's 92.5% on the 400-message subset. The 97.1% is the earlier fit on the full test set. A refit scored 96.8%. In bulk that run was a few milliseconds per note, about 5 ms. Through the full contract, a batch of 400 took about 11 ms each, and one request at a time with nothing cached took 97 ms median on 100 requests. The other times in the table are medians. The AnyJev readout on Qwen3-1.7B got 75% of that 400. Fit rank, on one candidate, is a different task: agreement with the teacher is not correctness. The comparison is in `docs/EXPERIMENT_LOG.md`.

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
