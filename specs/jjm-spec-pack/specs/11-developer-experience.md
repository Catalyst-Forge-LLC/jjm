# 11. Developer experience and transport

## One understandable route

The first documentation page explains what runs, where it runs, what the user supplies, and what returns. Separate install, configure, evaluate, and interpret. Avoid asking a newcomer to choose among many engines before seeing a working result.

Deliver a synthetic quickstart with no model download, then one actually tested real-model quickstart. Clearly label the synthetic route as a contract demonstration. It does not demonstrate semantic intelligence or calibration.

The planned public quickstart is at `/docs/quickstart` on **JJM.dev**. `specs/17-website-and-launch.md` defines the documentation routes and publication gates. Before a tested install exists, use an explicit pre-release notice and the runnable spec-pack checks, not a guessed package-manager command. The site is documentation, not an implicit hosted runtime.

## Proposed CLI

The following commands are design targets, not commands implemented by this specification pack:

```bash
python -m jjm doctor --profile local-advisory
python -m jjm evaluate --profile local-advisory --request request.json
python -m jjm capabilities --profile local-advisory
python -m jjm serve --profile local-advisory --host 127.0.0.1 --port 8765
```

Read input from a file or stdin, never from shell-evaluated content. JSON goes to stdout. Diagnostics go to stderr. `doctor` reports precise missing capability or dependency and does not install packages, download models, change configuration, or send private state without permission.

Use separate exit meanings: 0 for a completed transport operation with at least an envelope, 2 for invalid request/configuration, and 3 for process-level execution failure. Item-level acceptance remains in the JSON. Document this clearly so exit 0 is not confused with an accepted or correct judgment.

## Proposed HTTP transport

`GET /v1/capabilities` returns capability metadata for authorized configured profiles.

`POST /v1/judgments` accepts the request schema and returns the response schema. `GET /health/live` reports process liveness without testing the GPU. `GET /health/ready` reports whether the configured runtime can accept work, without leaking secrets.

Use HTTP 400 for malformed envelopes, 413 for size limits, 401/403 for authentication/authorization, 429 for backpressure, and 503 for unavailable configured execution. Well-formed partial evaluation returns HTTP 200 with item statuses. The service must not return a successful judgment-shaped body when it has only an error message.

These are paths on a separately configured runtime base URL, initially loopback. They are not public endpoints on `https://jjm.dev`, and the client must not default to that domain or send state there.

No streaming text, conversational session, or mutable state handle in v1. Do not add an idempotency cache until cache keys include effective system, policy, input, scope, and artifact identity.

## TypeScript client

Keep a tiny client that validates requests and responses, supports cancellation and timeouts, preserves null/unknown provenance, and exposes the complete envelope. The client must not collapse every result to `true/false` or discard acceptance checks.

Use a supported Node LTS version selected and tested at implementation time. Publish through an owner-controlled package scope after name checks. Generate types from the canonical schema or test handwritten types against it. No Python process orchestration hidden inside the client package.

Browser use is not a first-release target. Browser clients must never receive provider keys or privileged local service credentials through bundled configuration.

## Documentation example

A complete example includes input, exact configuration, result, and interpretation. It explains why the supplied numbers are uncalibrated or what validation applies. Include a second example where the model output is held, and a third where missing evidence prevents inference.

Document the meaning of `not_observable`, `abstained`, and `error` together. Avoid the phrase “cannot hallucinate.” Explain that bounded output prevents unexpected labels but not mistaken selections.

## Agent usability

Machine-readable capabilities, JSON Schema, stable codes, and explicit provenance are sufficient first-release agent affordances. A skill or MCP wrapper is optional later and should call the same read-only contract. It must not create a second inference path or acquire action permissions by default.
