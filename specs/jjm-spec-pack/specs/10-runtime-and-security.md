# 10. Runtime, failure behavior, and security

## Minimal deployment

CLI first, one isolated local process or environment, no mandatory account, no telemetry by default, and no hosted control plane. The later HTTP service binds loopback by default.

A private-network deployment requires explicit bind configuration, authentication, request limits, and a trusted reverse proxy or equivalent protection where appropriate. The request `scope_id` partitions state and caches but is not an authorization credential.

## Input controls

Set explicit limits for UTF-8 request bytes, nesting depth, item counts, evidence bytes, criterion count, model context length, and per-call timeout. Default initial service targets: 1 MiB request body, depth 32, 32 criteria, 64 evidence items, and a configurable 30-second request timeout. These are engineering defaults, not claims that any model can process them at a given speed.

Calculate token bounds using the actual rendered prompt and tokenizer. Reject oversize input. Do not silently trim old messages, options, or evidence to make a request fit. A future explicit truncation policy must report exactly what was removed and change the state digest.

Reject duplicate JSON object keys, NaN, infinities, unsupported numeric ranges, and malformed Unicode. Network and CLI parsers use the same strict ingress path.

## Data as data

Treat state and evidence as untrusted content, including strings that resemble system instructions. Keep compiler instructions separate. The model may still be influenced by prompt injection, so enforce filesystem, network, artifact, and action boundaries outside it.

Criteria contain declarative questions and descriptions only. No embedded Python, JavaScript, SQL, shell, templates with evaluation, or user-controlled import paths.

## No arbitrary destinations

Requests select allowlisted profile IDs. They do not contain provider base URLs, model download URLs, filesystem paths, or artifact import commands. Server configuration sets those values. Evidence URLs are not fetched.

A local failure never causes automatic cloud fallback. A future fallback chain must be an explicit operator configuration, permitted by caller requirements and data classification, and reported per result.

## Mutable state isolation

Keep artifacts and priors immutable during a request. Namespace caches and persisted metadata by application/deployment scope. Never combine unrelated tenants' calibration observations or live priors.

If the inherited inference object maintains counters or mutable statistics, use a per-profile lock, immutable adapter snapshot, or request-scoped object as needed. Do not run mutable state updates concurrently merely because HTTP handlers are asynchronous. Weight sharing is allowed independently of estimator-state sharing.

## Runtime errors

Cancellation and timeouts stop result production and signal cancellation downstream when supported. No late result is accepted after the request has been cancelled. Bound queues and return backpressure errors rather than exhausting memory.

OOM, missing weights, tokenizer failures, invalid distributions, and incompatible artifacts produce structured errors. Retry only documented transient failures. Do not retry an invalid head by changing the question or choosing a new provider.

## Artifacts and model loading

Default `trust_remote_code` to false. Any exception is an explicit deployment decision, not a request field. Pin model revisions where feasible and record the actual resolved identity.

Use safe, size-bounded metadata parsing and non-executable array formats. Validate dimensions, dtypes, finite parameters, hashes, and allowed paths. Do not load arbitrary pickles from users. Use atomic writes and replacement for artifact promotion and retain a rollback record.

## Logs and retention

Public diagnostics use an allowlist of non-content metadata. Do not forward arbitrary upstream diagnostic dictionaries or provider bodies into a response without sanitization.

Default logs contain request IDs, criterion IDs, timing, method IDs, rejection codes, and high-level metrics. Do not log raw content, keys, prompts, or label observations. Debug retention is opt-in, application-scoped, access-controlled, time-bounded, and clearly announced.

Model inference does not consent the user to training-data storage. Data collection and exports require separate explicit choices. Do not imply that a locally computed digest anonymizes sensitive material.

## Application action boundary

A stale but accepted judgment must not authorize acting on a changed application state. The caller compares state revision at the moment of action and reruns or rechecks when needed. Authorization, idempotency, and deterministic receipt verification remain the caller's responsibility.
