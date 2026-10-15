# 05. Uncertainty, abstention, and acceptance

## Four separate questions

What did the model select? How concentrated is its output? What validation supports interpreting its numbers? Does this result meet this application's consumption requirements?

Keep those answers in separate fields. A single `confidence` number must not stand in for all four.

## Arithmetic summaries

For weights p over K outcomes, top mass is max(p). Margin is largest minus second largest. Normalized entropy is -sum(p log p)/log K, with 0 log 0 treated as zero. These are deterministic functions of one distribution. They do not establish factual correctness, evidence authenticity, or calibration.

A 0.98 weight on an outcome in an uncalibrated readout is not a demonstrated 98% chance of correctness. A threshold on such a number is allowed for advisory experimentation only when the caller explicitly permits uncalibrated results.

## Acceptance and calibration are separate, and both are scoped

An accepted validation report refers to one estimator, one held-out evaluation, one use domain, one criterion definition, and a pinned effective system. It is not a global quality stamp, and it does not make the weights calibrated probabilities. Distribution shift, changed criteria, model revision, changed prompt rendering, and modified artifact state may invalidate that acceptance.

`probability_calibration` records whether those weights were assessed as frequencies. `calibrated_probability` is allowed only when that status is `assessed_in_scope`.

Do not convert upstream level ordering into cross-provider assurance. There is no rule that all L2 results are safer than all L1 results. Method choice may depend on measured quality, latency, and cost for a task.

## Evidence absence versus contradictory evidence

A missing upload receipt cannot establish that an upload failed. An agent's claim that it succeeded cannot independently establish success. For an assertion criterion, both may result in `not_observable`, depending on its explicit definitions and supplied evidence.

The deterministic absence of a caller-required evidence ID is different again: preflight abstains without inference, using `MISSING_REQUIRED_EVIDENCE`. This saves computation and avoids asking a model to satisfy an impossible requirement.

## Gate precedence

1. Invalid schema, unsupported version, or duplicate identifiers: reject the request before execution.
2. Forbidden destination, incompatible artifact, unsupported method requirement, missing required evidence, or stale state: abstain for the affected criterion before execution when detectable.
3. Backend failure or malformed output: error, no prediction.
4. Complete result with insufficient output margin, disallowed semantic outcome, or unknown mandatory validation: evaluated but held.
5. Complete result meeting all declared checks: evaluated and passed for consumption.

The final status is the most restrictive applicable result. Unknown checks cannot be treated as passes. Default requirements permit uncalibrated advisory judgments, but require nonempty provenance and complete distributions.

## Threshold policy

No product-wide magic threshold is shipped as a safety guarantee. Examples use explicitly synthetic values for contract illustration. Production thresholds are selected from held-out evaluation for the actual use, loss asymmetry, and review budget.

Threshold changes have a policy version and hash. They do not retrain the model, but they change acceptance and must be auditable.

## Safe downstream example

An application receives a `supported` finding for an export-completion criterion. Its acceptance record passes. The application still verifies that a receipt references the correct task and expected object version, then checks permission and current state before changing anything.

JJM's response does not replace that verification, and its gate cannot authorize the state transition.

## No blended probabilities by default

Do not average Jev, Span-01, and AnyJev scores into a seemingly stronger probability. Their numbers may have different meanings and correlated errors. An ensemble would need a separately specified, fitted, and validated method with its own provenance.
