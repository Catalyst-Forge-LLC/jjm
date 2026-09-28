# 03. Architecture and ownership

## Component boundary

```text
Application or agent
  owns state, goals, permissions, and actions
             |
             v
JJM request validator and criterion compiler
             |
             v
Pinned deployment profile + capability preflight
             |
             v
Judgment adapter
  AnyJev local inference now
  direct judgment-provider adapters later
             |
             v
Result validator + provenance + acceptance checks
             |
             v
Typed judgment envelope
             |
             v
Application combines deterministic checks and makes its own decision
```

## Modules

`jjm.contract`: schema validation, finite-number validation, typed records, version negotiation, and stable error codes. No GPU imports.

`jjm.compile`: deterministic rendering of state, evidence, criterion, answer descriptions, and output mapping. Compiler version and template hashes are part of the effective-system identity.

`jjm.profiles`: immutable configured deployment profiles. Each profile fixes adapter, model revisions, method settings, artifact, permitted data destinations, and limits. Requests choose an allowlisted profile ID, not arbitrary network URLs or filesystem paths.

`jjm.adapters.anyjev`: conversion between JJM contracts and upstream objects. It owns strict output coverage, exact artifact binding, and method metadata. It does not own application workflow.

`jjm.acceptance`: deterministic, non-executable checks such as required evidence presence, calibration scope, model identity, state age, minimum margin, and accepted outcomes. It reports why a result can or cannot be consumed. It never executes an action.

`jjm.cli`: local JSON input/output. `jjm.service`: optional loopback HTTP wrapper with the same behavior. Neither implements independent judgment logic.

`clients/typescript`: a thin client and generated or schema-derived types. Node code calls a service, not Python internals. Browser bundles must not contain model paths, provider keys, or local-only dependencies.

`jjm.artifacts`: immutable metadata, staged artifacts, validation reports, and explicit promotion. Use local files first. A database is not necessary for the initial runtime.

## Data ownership

The application supplies state and its revision. JJM does not mutate or reconstruct the application's source of truth. The application supplies criterion definitions and records their versions. JJM owns only execution records and explicitly retained artifact metadata.

A judgment record is an observation about a state snapshot under a criterion and model configuration. It is not the state itself, a permanent fact, or proof that a later state still has the same properties.

## Direct provider versus inference backend

An AnyJev backend supplies restricted logits or hidden states. A hosted judgment provider supplies completed judgments with its own semantics. These are different seams.

Do not implement a provider that already returns judgments by converting them to fake logits. Do not implement a chat-only provider by asking it to invent confidence percentages. A later generative evaluator could be a separate explicit adapter category, but it is outside the initial product promise and must never be a silent fallback.

## Deployment

Local machine is the default. A private LAN service is an optional deployment of the same runtime, not a separate product. Cloud deployments and hosted adapters must be explicit, configured, and observable. The architecture is not local-only, but it is local-first and network-neutral at the contract level.

Keep per-deployment profiles immutable and each active request pinned. A promotion changes a profile generation for subsequent requests. It cannot change a method halfway through a request.

## No persistent encoded-world promise

The public contract is stateless by default. It accepts complete supplied snapshots. Prefix reuse may optimize a request, and an exact-result cache may be added later. Neither is a general persistent world representation. Persistent state handles require a separate design for model-specific cache lifetime, invalidation, access control, and accounting.
