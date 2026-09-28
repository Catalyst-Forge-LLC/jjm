# 06. Provenance and reproducibility

## Effective-system identity

Record the JJM version, upstream commit when known, adapter identity/version, deployment profile ID/generation, base-model ID and immutable revision when available, tokenizer revision, runtime version, numerical precision, compiler/template version, and method settings.

For a fitted artifact also record artifact hash, head/readout layer, fit method, label-source metadata, and validation report ID. For prior correction record the estimator, strength, prior snapshot ID, and whether it was applied. For a fallback record requested behavior, achieved behavior, and reason.

A friendly served-model alias is not a verified weights revision. Use `verified`, `operator_attested`, or `unknown` to distinguish what the service can actually establish. The strict `require_verified_model_revision` gate accepts only a locally verified immutable revision or a deployment attestation verification mechanism explicitly configured by the operator. Merely receiving a model name from a provider does not satisfy it.

## Digests

Use SHA-256 of RFC 8785 canonical JSON for structured contract content. Preserve array order. Reject duplicate object keys, non-finite numbers, and incompatible numeric values at ingress. RFC 8785 is the selected canonicalization reference, not an invented JJM standard. [U11]

Define separate digests:

- `state_digest`: the complete effective state supplied to the adapter, including the selected inline evidence after any declared redaction.
- `criterion_digest`: kind, question, ordered option IDs/descriptions or score levels, and assertion compiler definitions. Exclude client IDs, outcome-admission filters, and runtime thresholds.
- `rendering_digest`: compiler version, templates, and all prompt-affecting settings.
- `effective_config_digest`: model and tokenizer revisions, adapter/runtime versions, method settings, artifact identity, and rendering identity.
- `policy_digest`: all requirements that affect acceptance.

Artifact identity in the effective configuration uses the immutable payload digest and fixed artifact ID, not a hash of a manifest that contains the configuration digest. This avoids a circular hash dependency. The artifact manifest binds that configuration separately.

Opaque digest examples in this pack are synthetic identifiers, not hashes of actual model artifacts. The fixtures are not full replay bundles.

## Model input correspondence

A valid digest of a stored request does not prove that the adapter sent the same content. The adapter records a digest of its effective input and, in a locally retained debug mode, the exact rendered input. Tests establish deterministic correspondence between structured state and rendering.

Do not claim that hashing verifies the truth of evidence or the correctness of a judgment. Hashes identify content. Signatures, when added, authenticate a signer and content integrity, not truth.

## Artifact binding

A head is bound to exact effective criterion, model revision, tokenizer/rendering, layer definition, dtype assumptions, and artifact parameters. The deployment loader rejects mismatches before inference.

A semantic alias is not inferred from shared labels. Future aliases are explicit records with reviewer, reason, source/target digests, evaluation evidence, and version. They are disabled in the first release.

## Frozen versus adaptive operation

Production default: immutable artifacts and fixed estimator state. Experimental learning is a separate offline or shadow process. Every updated head, prior, or feature normalization becomes a new candidate identity. Promotion pins a new deployment generation, with rollback to the prior one.

No unreported traffic-dependent state is allowed in reproducibility claims. Any future online mode must name the exact pre-request and post-request state snapshots and serialize updates within its scope. It is intentionally deferred.

## Replay levels

`metadata_only`: enough to identify what was claimed, not rerun it.

`input_retained`: exact sanitized input and configuration are retained, but model assets or hardware may differ.

`replayable`: input, assets or immutable obtainable revisions, artifacts, runtime, and numerical settings are available.

Even `replayable` does not promise bit-identical GPU execution. Define numerical tolerances and compare both weights and selections. Report actual determinism rather than assuming it.

## Privacy

Input hashes can leak equality or be guessed for small domains. Keep them private by default, partition by application, and use keyed external identifiers when publishing cross-boundary records. Do not place raw state, prompts, artifact observations, or personal evidence in default logs.
