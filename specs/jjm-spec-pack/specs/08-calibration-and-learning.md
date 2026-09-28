# 08. Calibration, fitted heads, and learning lifecycle

## Separate inference from fitting

A model may produce useful rankings before there are labelled examples. JJM must expose that experimental mode honestly. Calibrated probability semantics require a stronger workflow, not a different label in the response.

AnyJev documents zero-label debiasing, label-fitted temperature scaling, and label-fitted heads. Its methods documentation also bounds transfer and calibration assumptions. Treat these as available mechanisms, not assurance guarantees. [U12]

## Artifact lifecycle

1. Collect examples only with explicit retention consent and a stated purpose.
2. Assign labels with provenance and preserve disagreement rather than silently overwriting it.
3. Split by source, time, or group as appropriate to prevent related-state leakage.
4. Fit a candidate using only permitted training and calibration partitions.
5. Evaluate on a separately held-out set not used for method selection, threshold tuning, or promotion decisions that would invalidate independence.
6. Review the evaluation against predeclared release criteria.
7. Promote an immutable artifact to a deployment profile generation.
8. Monitor on new independently labelled observations and roll back when necessary.

A candidate fit never promotes itself. Cross-validation used to choose a head or temperature is not a substitute for the final held-out evaluation.

## Metadata

The artifact manifest includes ID/version, SHA-256 of payload, method, exact criterion and effective-system bindings, creation time, training/calibration/validation data identifiers, split method, label provenance, label counts per outcome, and intended domain.

The validation report includes task metrics, probabilistic metrics, acceptance thresholds, selective risk and coverage, sample counts, uncertainty intervals where supported, and reviewer/promotion status. Keep raw data separate and access-controlled.

`label_source` distinguishes human adjudication, deterministic outcomes, provider labels, synthetic labels, and mixtures. A model-generated label is not silently renamed ground truth. Agreement with a teacher is reported as agreement.

## Status values

`not_fitted`: no fitted artifact.

`fitted_unvalidated`: a fitted artifact exists, but a suitable held-out report has not been accepted.

`validated_in_scope`: an approved report exists and all pinned scope and identity checks pass for the selected use.

`out_of_scope`: a report exists but its scope, model, criterion, rendering, or deployment assumptions do not apply.

`unknown`: the provider or artifact does not supply enough information.

Only `validated_in_scope` can support `calibrated_estimate` in the proposed public contract. This is an evidence-based release convention, not a mathematical promise that every individual output is accurate.

## Scope checks

Treat domain membership as an explicit deployment declaration plus verifiable metadata, not a model guessing that its own calibration applies. Record whether scope was operator-attested or mechanically checked. A validated scope does not prove absence of future distribution shift.

Exact matching includes model revision, tokenizer and template, readout layer, label ordering, criterion semantics, method parameters, and the frozen prior or feature normalization. A change requires revalidation unless a reviewed compatibility record demonstrates equivalence.

## Default estimator state

For the initial L0 advisory profile use full permitted cyclic rotations with `prior=none`, `adaptive_shifts=false`, and adaptation disabled. This is a deliberate permutation-only variant, not a claim to reproduce upstream's default L0 measurements. Report `prior_method=none` in provenance.

When frozen-prior support is added, prior injection must be a narrow tested engine seam. Do not manually edit several private accumulators. L1 uses the artifact's frozen prior and exact compatible settings. Head profiles load only the exact selected head, with automatic cross-question routing disabled.

## Learning limits

The first release has no `observe` endpoint. Later data collection is opt-in, bounded, application-scoped, and excluded from default logs. Fit jobs run separately from serving workers. Reject arbitrary executable payload formats and unsafe object deserialization.

Private examples are not exported with an artifact by default. Exporting weights, metadata, labels, and original observations are distinct choices.

## Drift and feedback

Do not use the model's own accepted selections as unlabelled “successes” or verified labels. Feedback comes from independently established outcomes or accountable review. Track disagreement and failure cases, not just examples admitted by the existing gate.

An updated criterion is a new evaluation target. Do not silently reuse a head simply because answer labels still look similar.
