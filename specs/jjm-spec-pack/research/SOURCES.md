# Primary sources and inspection scope

Inspected September 25, 2026. Public `main` branch pages were retrieved through web tools. These references are mutable unless pinned by the implementing agent. Exact commit was not independently resolved. A local clone failed because the runtime could not resolve GitHub, so no source-tree test or inference run was performed.

The observations in `specs/01-upstream-findings.md` are based on the following source files. Proposed design requirements throughout the pack are original recommendations, not claims that upstream already implements them.

| ID | Primary source | Used for |
|---|---|---|
| U1 | [AnyJev README](https://github.com/nokia-applied-research/AnyJev) | Repository identity and project scope |
| U2 | [Package metadata](https://github.com/nokia-applied-research/AnyJev/blob/main/pyproject.toml) | Declared version, language, and packaging |
| U3 | [Backend protocol](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/backends/base.py) | Low-level backend boundary |
| U4 | [Question definitions](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/question.py) | Primitive limits and mappings |
| U5 | [Decider source](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/decider.py) | Routing, mutable estimator state, and artifact workflow |
| U6 | [vLLM backend](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/backends/vllm.py) | Label coverage and one-token readout behavior |
| U7 | [Result definitions](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/result.py) | Serialization and method-level requirements |
| U8 | [Transformers backend](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/backends/hf.py) | Local inference and checked prefix reuse |
| U9 | [Repository license](https://github.com/nokia-applied-research/AnyJev/blob/main/LICENSE) | Apache-2.0 fork and attribution requirements |
| U10 | [Third-party materials](https://github.com/nokia-applied-research/AnyJev/blob/main/THIRD_PARTY.md) | Asset-specific licensing and label-source caveats |
| U11 | [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785) | Selected JSON canonicalization reference |
| U12 | [Level documentation](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/levels.md) | Mechanism and calibration limitations |
| U13 | [Readout source](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/readout.py) | Prompt/label mapping to recheck during implementation |
| U14 | [Agent working agreement](https://github.com/nokia-applied-research/AnyJev/blob/main/AGENTS.md) | Upstream contribution and verification conventions |
| U15 | [Methods documentation](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/method_v3.md) | Background for deeper implementation review |

## Evidence limitations

No benchmarks were reproduced. No provider access was tested. No package or domain availability was independently established in the original inspection. No complete security audit, license audit of all weights, or trademark clearance was performed.

For spec-pack 0.2, the user reported acquiring **JJM.dev**. The selected name and domain are updated on that basis. This update did not repeat the upstream web inspection, query registries, test DNS/HTTPS, inspect a deployed website, or verify trademark clearance. The original upstream-source findings and unresolved commit requirement are retained unchanged.

The source-inspection concerns are specific behaviors to test and harden, not claims of observed exploitation. Upstream may change before implementation. Reconcile against the pinned fork commit.

## Naming search

Exact-phrase searches for the proposed expansion and its alternate spelling did not establish an authoritative matching project. This is not an exhaustive search and provides no exclusivity or clearance. The specification does not depend on a claim that the name is original.
