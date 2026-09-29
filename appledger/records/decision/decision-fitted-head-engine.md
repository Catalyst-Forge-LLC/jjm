---
format_version: 0.1.0
id: decision-fitted-head-engine
kind: decision
title: Fitted heads over frozen encoders become a JJM route
record_status: active
created_at: 2026-09-29T19:27:42Z
updated_at: 2026-09-29T19:27:42Z
recorded_by:
  id: cursor-agent
  type: agent
visibility: internal
relations:
  - type: affects
    target: app-36f50a0b-90a2-4e99-a024-197c6cba62c8
claims: []
data:
  status: accepted
  choice: >-
    Add a `fitted-head` profile. A frozen encoder (any Hugging Face checkpoint or Ollama embedding model)
    reads the state once, and a small head fitted on labelled examples maps the features to weights.
    Heads are content-addressed artifacts bound to the exact criterion digest, encoder identity, and head
    method. They are reused when the training digest matches, served only after explicit promotion, and
    refused with ARTIFACT_MISMATCH otherwise. Results report `fitted_unvalidated` calibration and
    `numeric_path: hidden_state_head`. Teachers (a local Ollama model, or any provider through the optional
    LiteLLM extra) label examples or take escalated cases outside `evaluate`, never as a hidden fallback.
    No model is hard-coded; `jjm models` lists what is installed.
  rationale: >-
    On the Exec Foundry fit trial, a 0.6B reranker with a head matched or beat the 4B readout at 186 ms
    instead of 5 s. On Banking77 routing (`trials/routing/README.md`), a Qwen3-Embedding-0.6B head reached
    97.1% against human labels at a few milliseconds per message, against 88% for a 9B model asked zero-shot
    and 75% for the AnyJev readout. Contract 0.1 already had every field this needs, so there is no schema
    change.
  alternatives:
    - Keep v1 to the AnyJev readout and defer fitted heads
    - Fine-tune a model per task instead of a head over frozen features
    - Call a hosted model inside `evaluate` when the head is unsure
  authority: "Owner, 2026-09-29: \"all of your speedup loops are excellent! Let's get on that! Agree on the adjustments. Let's implement these\""
---

# Fitted heads

The locked brief (`docs/PHASE_1_BRIEF.md`, section 10) put fitted calibration artifacts, artifact
promotion, and hosted provider adapters out of v1. The owner approved this route on 2026-09-29.

What stays the same:

- A fitted head's weights are a readout, not a calibrated probability. A request that requires
  validated calibration still abstains.
- Cross-validation chose the head's setting, so its report is a selection estimate, not a held-out
  validation. Validated calibration needs a separate held-out report, which does not exist yet.
- A candidate fit never promotes itself. Promotion is an explicit `--promote`.
- Teacher labels are recorded as that teacher's labels, never as ground truth.
- Hosted providers are teachers only. They do not run inside `evaluate`.
