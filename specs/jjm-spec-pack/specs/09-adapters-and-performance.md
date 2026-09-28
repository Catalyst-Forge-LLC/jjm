# 09. Adapters, capabilities, and performance

## Capability contract

Each configured profile advertises implemented kinds, maximum options and score levels, accepted input forms, available method IDs, number semantics, evidence selection support, revision verification, and execution limits.

Capability discovery is descriptive. Preflight still checks a specific request and loaded artifacts. A schema-valid score request can be unsupported by an early choice-only release.

Capabilities are not inferred from an OpenAI-compatible URL. They are established through adapter-specific tests and the configured deployment.

## Initial adapter

Retain the Transformers inference route. The fake backend is a separately named synthetic adapter that must never masquerade as real inference.

Compile choice options deterministically. Compile assertion to its three explicit options. Preserve ordered score behavior when added. Do not turn separately evaluated binary scores into a normalized multiclass distribution without a separately specified estimator.

Require finite outputs, correct vector dimension, exact option coverage, expected model identity, and a known numeric path. A failed check yields an error, not a plausible-looking repair.

## vLLM route

When added, distinguish the logits-serving deployment from the pooling deployment. Test requested-label coverage and the exact token-ID mapping. Reject absent logprobs rather than using a fallback numeric value.

For a head, validate model revision, representation layer, hidden dimension, pooling position, and absence of unintended normalization or softmax. The fact that a server has an embeddings endpoint does not make its ordinary semantic embeddings compatible with a fitted hidden-state head.

A truncated checkpoint is a separate identified asset with a mapping from source layers to deployed layers. Do not treat the layer number of a truncated model as the same representation of a full model without that mapping.

## Hosted judgment adapters

Reserve the architecture, not the implementation. Jev and Span-01 are candidates for later discovery and testing. Before implementing either, read its then-current official API, obtain actual access, and create provider-specific contract tests. This pack does not invent endpoint URLs or claim SDK compatibility.

Preserve provider-native semantics and method metadata. A `present/absent/not_observable` response is not automatically the same criterion as JJM's evidence-status assertion. Verify definitions or reject that mapping.

Unknown calibration remains `provider_reported` with calibration status `unknown` or a supported attestation. Do not automatically translate a vendor confidence score to AnyJev's top mass.

## Backend eligibility

A text-generation-only API is not a supported logits backend. Ordinary embeddings are not automatically head-compatible hidden states. Limited top-k logprobs that omit required labels are not complete distributions.

Ollama, MLX, llama.cpp, and other engines are not promised by this pack. Add them only when a concrete adapter demonstrates the necessary readout and passes conformance. Keep Python inference separate from the Node client rather than rewriting tensor execution for familiarity.

## Execution accounting

Record actual adapter calls, prompt evaluations, token-readout calls, elapsed wall time, queue time when measurable, cache-hit status when known, and head early-exit behavior. Unknown counters are null, not zero.

A shared state in the API may still require multiple prompts. L0 variants may evaluate rotations. Additional evidence-selection tasks cost additional inference. Cross-question prefix reuse and cross-request reuse are separate features.

Do not sum overlapping per-criterion wall times to claim total request latency. Report request wall time separately. Avoid deriving a dollar amount from undocumented provider usage. Unknown cost is null.

## Bench protocol

Measure cold and warm calls, state length, question count, option count, dtype, model revision, concurrency, hardware, and cache configuration. Report p50/p95/p99 with sample counts rather than an isolated best observation. Separate queue time from model execution where possible.

Compare methods using the same task and effective question meanings. Preserve unsuccessful runs and missing measurements. Do not compare a fake backend to a real model or compare a one-question run to a full multi-criterion request without explicit normalization.

No “encode once forever,” “one forward for everything,” or “faster than Jev” claim without a directly supporting measurement and scoped explanation.
