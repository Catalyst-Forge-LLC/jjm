# JJM: Just Judgment Machine
## AnyJev fork specification, version 0.2

**An open-source judgment runtime.**

A programmable judgment layer for applications and agents.

**Just judgment. Your application makes the decision.**

Give it state, questions, and defined outcomes. Get bounded judgments with explicit method, provenance, uncertainty, and acceptance information. Your application remains responsible for authorization and actions.

Updated September 25, 2026. **JJM, Just Judgment Machine**, is the selected name. **JJM.dev** is the canonical project domain, acquired by the user. The canonical web origin is `https://jjm.dev`. Registration is user-reported, not an independent DNS, deployment, or ownership check. Distribution names, repository location, handles, and trademark clearance remain unresolved.

This is **spec-pack version 0.2**, not a runtime release. The proposed contract remains **0.1**, and the original schemas, fixtures, and runtime release sequence are unchanged. See `CHANGELOG.md` for the update and `BRAND.json` for the selected identity.

## What this package is

An implementation specification for an independently maintained fork of `nokia-applied-research/AnyJev`, not an implemented fork, a pretrained model, or a claim of performance. It includes proposed JSON contracts, synthetic examples, a fixture validator, an implementation sequence, and release criteria.

The upstream public code and documentation were inspected through web retrieval. A local clone could not be obtained in this environment, so upstream tests and model inference were not run. The inspected branch was `main`. Its exact commit was not independently resolved. Phase 0 requires an immutable commit before implementation.

**Read `JJM-OVERVIEW.md` for the summary. Start implementation with `KICKOFF.md`, then `specs/00-product-and-scope.md`.**

## Delivery shape

Keep the existing Python inference machinery and its tests. Add a small JJM contract and service layer around it. Add a thin TypeScript client for Node applications without porting model inference to JavaScript. The first useful release is one local backend, one end-to-end path, explicit limitations, and synthetic contract tests. Provider adapters and optional integrations follow only after that path works.

## Reading map

| File | Purpose |
|---|---|
| `KICKOFF.md` | Agent handoff, boundaries, and first actions |
| `BRAND.json` | Selected identity, canonical domain, and unresolved distribution names |
| `CHANGELOG.md` | Spec-pack changes, distinct from runtime and contract versions |
| `specs/00-product-and-scope.md` | Product definition, MVP, and non-goals |
| `specs/01-upstream-findings.md` | Existing capabilities versus changes to make |
| `specs/02-fork-and-license.md` | Attribution, packaging, and upstream maintenance |
| `specs/03-architecture.md` | Python core, contract, adapter, and client boundaries |
| `specs/04-contract.md` | Normative request and response behavior |
| `specs/05-uncertainty-and-acceptance.md` | Uncertainty, abstention, and admission checks |
| `specs/06-provenance-and-reproducibility.md` | Effective-system identity and replay |
| `specs/07-evidence.md` | Evidence boundaries and non-generative references |
| `specs/08-calibration-and-learning.md` | Artifact scope, validation, promotion, and drift |
| `specs/09-adapters-and-performance.md` | Backend capabilities and honest execution accounting |
| `specs/10-runtime-and-security.md` | Isolation, data handling, failures, and deployment |
| `specs/11-developer-experience.md` | CLI, API, TypeScript client, and quickstart requirements |
| `specs/12-testing-and-evaluation.md` | Tests, real-model evaluation, and release gates |
| `specs/13-implementation-plan.md` | Ordered work items with acceptance criteria |
| `specs/14-optional-composition.md` | Interfaces for other tools, with no hard integrations |
| `specs/15-naming-and-positioning.md` | JJM name, spelling, message, and visual brief |
| `specs/16-decisions-and-open-questions.md` | Defaults, tradeoffs, and genuinely unresolved choices |
| `specs/17-website-and-launch.md` | JJM.dev content, routes, status, and publication gates |
| `website/HOMEPAGE-COPY.md` | Homepage copy and truthful pre-release variants |
| `research/SOURCES.md` | Primary-source evidence and inspection limitations |
| `schemas/` | Proposed contract schemas, not an existing public standard |
| `examples/` | Synthetic positive fixtures and deliberately invalid cases |
| `reference/` | Integration examples and expected behaviors |
| `scripts/validate_pack.py` | Schema, semantic, and adversarial fixture checks |
| `scripts/validate_identity.py` | Offline identity, homepage, route, and version checks |
| `VALIDATION.md` | What was and was not tested in preparing this package |

## Validate this package

The contract validator requires Python 3.10+ and the `jsonschema` package with Draft 2020-12 support. The application-boundary check also requires Node. The identity validator uses only the Python standard library. These checks validate the proposed contract, fixtures, and documentation consistency, not an implemented runtime.

```bash
python scripts/validate_pack.py
node reference/consume-judgment.test.mjs
python scripts/validate_identity.py
```

It does not install JJM, download weights, call providers, or validate model quality.

## Central invariants

- An answer schema limits output shape, not the possibility of a wrong answer.
- A normalized weight, a calibrated probability estimate, and a similarity score are different things.
- A model method is not a universal assurance grade.
- Missing evidence, an adverse finding, and runtime abstention are different outcomes.
- Model, criterion, input rendering, and artifact identities are explicit.
- No silent backend downgrade, fabricated probability, or cloud fallback.
- State ingestion is not automatic training, and fitting a candidate is not promoting it.
- A passing acceptance check is permission to consume a result under a stated policy, never authorization to take an action.

## Default release sequence

**0.1:** fork hygiene, deterministic contracts, synthetic backend, local Transformers L0, bounded choice and assertion, evidence presence checks, strict rejection behavior, CLI, and real-model smoke tests.

**0.2:** fitted L1/L2 artifacts with explicit validation status, ordered score, loopback HTTP, and a TypeScript client.

**0.3 or later:** vLLM hardening, provider adapters, opt-in learning workflows, and composition adapters.

The schemas describe the intended 0.1 contract family. An early implementation must advertise only the subset it actually supports.

The JJM.dev publication track is separate from runtime delivery. Domain ownership does not create a hosted inference service. A website is not a prerequisite for the first working local route, and publishing one does not make unimplemented runtime features available.
