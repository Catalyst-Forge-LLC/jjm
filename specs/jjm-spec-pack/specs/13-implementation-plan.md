# 13. Implementation sequence

Work in small changes with tests and a brief changelog entry. Keep the scope narrow enough that the maintainer can review each behavior change independently. Runtime release numbers below are not spec-pack versions. This document is updated in spec-pack 0.2 while the first runtime release target remains 0.1.

## Phase 0: establish the fork

**JJM-001, upstream baseline.** Resolve and record immutable commit, inspect relevant modules and agent instructions, run inherited CPU tests, and reconcile this pack with current source. Acceptance: baseline report distinguishes observed results, skipped tests, and unverified claims.

**JJM-002, identity and attribution.** Apply the selected JJM name, Just Judgment Machine expansion, JJM.dev home, and tagline from `BRAND.json`. Add fork documentation, retain license/credits, define isolated packaging, and resolve distribution names before publication. Acceptance: no removed copyright notices, no implied sponsorship, no silent import-file collision, and no claim that owning the domain reserves package namespaces.

## Phase 1: a complete synthetic path

**JJM-010, contract.** Implement schema and semantic validation, error codes, and strict JSON ingress. Acceptance: T01, T03, T06, T15, T16, and T17.

**JJM-011, criterion compiler.** Implement deterministic choice and assertion rendering with stable option-ID mappings. Acceptance: snapshot tests, changed definitions produce changed digests, and no unknown labels appear.

**JJM-012, fake adapter.** Provide an explicitly synthetic deterministic adapter. Acceptance: `execution_mode=synthetic`, documented fixed outputs, no network or model downloads, and a complete envelope.

**JJM-013, acceptance checks.** Enforce required evidence, method requirements, model identity requirements, and outcome filters. Acceptance: missing evidence is a pre-inference abstention, not a made-up model outcome.

## Phase 2: one real local path, release 0.1

**JJM-020, AnyJev adapter.** Wrap objects directly, preserve complete diagnostics, disable permissive head routing, and expose exact method. Use permutation-only L0 as the initial frozen advisory profile. Acceptance: real local smoke test, no hidden calibration claim, and documented divergence from upstream defaults.

**JJM-021, provenance.** Record effective model/runtime/compiler/input identity and unknown metadata honestly. Acceptance: T05, T09, and replay metadata test.

**JJM-022, frozen state and limits.** Add scoped estimator isolation, token/context checks, timeout behavior, and controlled configuration. Acceptance: T10, T11, T12, and T14.

**JJM-023, CLI and quickstart.** Deliver one synthetic and one real route. Acceptance: clean-environment commands are recorded, stdout is JSON, and no success is inferred from process exit alone.

**Release 0.1 gate:** no fitted-calibration promise required. Choice/assertion work locally with complete metadata and clear acceptance. Stop here for a useful personal tool rather than implementing the entire roadmap at once.

## Phase 3: calibrated use and integration, release 0.2

**JJM-030, artifact manifest and exact loading.** Add payload identity and effective-system binding with strict refusal. Acceptance: wrong-question heads fail even when labels match.

**JJM-031, fitting and validation.** Add separate fit/evaluate/promote workflow, immutable deployment generations, and held-out validation metadata. Acceptance: fitting cannot auto-promote, and unvalidated artifacts cannot claim validated estimates.

**JJM-032, ordered score.** Implement the indexed-level score contract. Acceptance: T07 and tested source-ID mappings.

**JJM-033, HTTP and TypeScript.** Add loopback service and thin client. Acceptance: transport contract matches CLI, cancellation works, and the client preserves uncertainty and errors.

## Phase 4: independently useful additions

**JJM-040, vLLM hardening.** Validate every expected logprob and head-vector property. Acceptance: T06 plus a real-engine test for each advertised serving mode.

**JJM-041, direct-provider adapters.** Add only after current official API inspection and real access. Acceptance: no invented endpoints, no numeric-semantics laundering, and explicit privacy routing.

**JJM-042, evidence candidate selection.** Separate optional inference with truthful provenance and subset checks. Acceptance: no implied deterministic verification.

**JJM-043, feedback and shadow fitting.** Opt-in labelled observation collection and staged candidates. Acceptance: no online auto-promotion, no cross-scope data, and explicit retention.

**JJM-044, optional composition adapters.** Emit or consume references for existing systems without owning their state. Acceptance: core works with all composition adapters absent.

## Separate publication track: JJM.dev

This track does not block phases 0 through 2 and does not authorize a deployment. A pre-release information page may ship only with truthful status and owner approval. Runtime functionality must never depend on the site.

**JJM-050, identity propagation.** Use `BRAND.json` consistently across README, site metadata, docs, and eventual distribution metadata. Acceptance: `python scripts/validate_identity.py` passes for the spec pack, published metadata is checked against the same identity, and unresolved registry names remain explicit.

**JJM-051, static homepage and docs.** Implement `specs/17-website-and-launch.md` and the release-appropriate copy in `website/HOMEPAGE-COPY.md`. Acceptance: primary use and input/output are apparent without scrolling through architecture, the three evidence-status meanings are clear, and there is no upload/chat form or hidden inference call.

**JJM-052, documentation and schema publishing.** Publish only implemented quickstarts and valid documentation routes. Make contract schemas available as static downloads while retaining their URN identities and offline resolution. Acceptance: schema downloads are byte-for-byte the release's canonical files, no runtime network fetch is required, and no proposed command is presented as tested.

**JJM-053, launch verification.** Verify DNS, HTTPS, canonical URLs, link destinations, displayed release status, accessible layout, and the actual install path before an authorized public release. Acceptance: the checklist in spec 17 has evidence for each completed item, with no fabricated repository URL, support address, model-performance claim, or live-service claim.

## Stop conditions

Do not ship a permissive workaround because an adapter lacks logits, labels, a model revision, or artifact metadata. Report the missing capability. Do not create a universal model-training pipeline, new standard registry, graph service, or workflow engine to complete an optional phase.
