# 04. Proposed judgment contract

Status: proposed JJM contract family `0.1`, not an externally adopted standard. Schemas in this pack define shape. This document and the semantic validator define cross-field invariants.

## Request

An evaluation contains `schema_version`, a client request ID, a state snapshot, one or more criteria, an allowlisted profile ID, a deployment scope, and execution requirements.

`state.id` and `state.revision` identify the caller's snapshot. `observed_at` is a caller-supplied observation time, not independently verified freshness. `content` is bounded JSON-compatible data. `evidence` contains inline items with unique IDs, kind, content, and optional observation time. JJM must not dereference paths or URLs embedded in content.

Criterion IDs must be unique in a request. IDs are stable application labels, not sufficient artifact keys. Definitions have versions and content digests. Exact question text, option IDs, option descriptions, order, and rubric semantics matter.

## Criterion kinds

### Choice

Requires two or more explicit options. Option IDs are output identifiers. Descriptions define their meaning. The compiler deterministically maps descriptions to upstream labels and maps results back to stable IDs. Distinct IDs with identical descriptions are rejected as ambiguous in the initial profile.

The option list is mutually exclusive for this question. Multiple true conditions should be separate criteria, not a normalized choice list. An `other` or `not_observable` option is caller-defined when appropriate. JJM does not invent either silently.

### Assertion

Evaluates the relationship between supplied evidence and a proposition. It uses these three fixed outcomes:

- `supported`: the supplied evidence supports the proposition under the stated criterion.
- `contradicted`: the supplied evidence supports the contrary under the stated criterion.
- `not_observable`: supplied evidence is insufficient, missing, ambiguous, or conflicting enough that the criterion does not settle the proposition.

These are evidence-status definitions, not guaranteed truth values. The compiler includes them explicitly. For the AnyJev adapter, assertion compiles to a three-option choice. It is not manufactured from a binary probability or equated automatically to another provider's third label.

### Score

Requires two to ten ordered levels with descriptions. Version 0.1 restricts values to consecutive integer indices starting at zero. This preserves a simple mapping to the upstream explicit-level score without a hidden nonuniform scale conversion. The output value is the expected index, calculated from the distribution. It is not a physical measurement.

Missing evidence is not automatically a zero score. A caller that needs observability must provide a separate assertion criterion and combine the results in application code.

## Execution status versus semantic outcome

`evaluated`: a complete, validated prediction was produced. `not_observable` is an evaluated semantic outcome, not a transport failure.

`abstained`: JJM intentionally did not emit a prediction, for example because required evidence or an admissible method was unavailable.

`error`: evaluation failed, for example because a backend response was malformed or the execution timed out.

A rejected or held evaluated prediction may be retained for diagnostics. It must never be silently presented as accepted.

## Prediction

`weights` covers every expected option ID exactly once, has finite values in [0,1], and sums to one within 1e-6. A caller must not interpret weights as calibrated event probabilities unless the semantics and validation status support that interpretation.

`semantics` is one of:

- `normalized_readout`: normalized model output, not claimed empirically calibrated.
- `calibrated_estimate`: a fitted estimator with an approved, in-scope held-out validation report and an exact matching deployment profile.
- `provider_reported`: a provider supplies probability-like values, but JJM has not established the preceding condition.

Do not put embedding similarities or arbitrary scores in `weights`. Reject an adapter that cannot populate this field honestly. A future score-only adapter needs a separately versioned prediction variant.

`selected` is the highest-weight option, using the declared option order as the deterministic tie-breaker. `top_mass`, `margin`, and normalized entropy are deterministic summaries. They are not independent confidence assessments. `value` is non-null only for score and equals the distribution-weighted level index.

## Acceptance

Acceptance is a deterministic record with `pass`, `hold`, or `reject`, reason codes, and individual checks. `pass` means the result met the caller's declared consumption requirements. It does not mean the proposition is true or an action is authorized.

`hold` covers potentially usable but insufficient results. `reject` covers structurally invalid, incompatible, or forbidden results. For absent predictions, `pass` is prohibited.

`accept_outcomes` is an optional per-criterion consumption filter. Its default is empty, meaning no outcome filter, not automatic approval. This filter changes result admission, not the model question or its artifact fingerprint.

## Requirements

Global requirements apply independently to every criterion. They include calibrated-in-scope estimates when requested, allowed method IDs, optional top-mass and margin thresholds, optional model-revision verification, and optional state-age limits. Criterion-specific evidence IDs and outcome filters also apply.

All mandatory unknown checks fail closed. A requested method is not the achieved method. Early releases must return `UNSUPPORTED_REQUIREMENT` rather than ignoring a field they cannot enforce.

## Batched evaluation

All criteria use the same supplied snapshot. They do not consume one another's answers. The implementation may batch or loop internally. Shared request shape does not guarantee a single forward pass. Always return results in request order and preserve unique criterion IDs.

One criterion can fail while another succeeds. An invalid envelope is rejected before any inference. A well-formed but unsupported criterion becomes an item-level abstention or error according to its code.

## Errors

Stable codes include `INVALID_REQUEST`, `DUPLICATE_CRITERION_ID`, `UNSUPPORTED_KIND`, `TOO_MANY_OPTIONS`, `CONTEXT_LIMIT_EXCEEDED`, `MISSING_REQUIRED_EVIDENCE`, `STATE_TOO_OLD`, `ARTIFACT_MISMATCH`, `CALIBRATION_NOT_VALIDATED`, `METHOD_NOT_ALLOWED`, `MODEL_IDENTITY_UNKNOWN`, `INCOMPLETE_LABEL_DISTRIBUTION`, `NONFINITE_RESULT`, `BACKEND_TIMEOUT`, and `INTERNAL_ERROR`.

Error records never contain provider keys, raw private state, or unrestricted tracebacks. `retryable` is explicit. No error is converted into zero probability.
