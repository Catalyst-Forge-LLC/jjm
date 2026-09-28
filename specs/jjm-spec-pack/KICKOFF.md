# Coding-agent kickoff

Build **JJM, Just Judgment Model**, as a small, maintainable AnyJev fork with an explicit judgment contract. Preserve the inference research rather than rebuilding it.

The selected product descriptor is **an open-source judgment runtime**. The user has acquired **JJM.dev**, and `https://jjm.dev` is the canonical web origin. Use `BRAND.json` for the identity and **Just judgment. Your application makes the decision.** as the tagline. Do not restart naming or domain-selection work. Registry names and repository placement still need resolution.

This is spec-pack 0.2. Do not change `schema_version: "0.1"`, schema URNs, or fixture engine identifiers merely to match the pack version.

## Read first

Read this file, `BRAND.json`, `CHANGELOG.md`, `specs/00-product-and-scope.md`, `specs/01-upstream-findings.md`, and `specs/13-implementation-plan.md`. Read `specs/15-naming-and-positioning.md` before writing public copy and `specs/17-website-and-launch.md` for website work. Read the upstream repository's current agent instructions and license before editing it. This package does not authorize commits, pushes, package publication, cloud deployment, or model downloads that incur cost.

## First implementation task

Resolve the upstream commit, record it in `UPSTREAM.md`, inspect the actual tree, and run its CPU-only tests. Record failures without concealing or relabelling them. Compare the current sources to the findings in this package. Fix stale paths or assumptions in a short reconciliation note before implementation.

Then implement the smallest end-to-end route:

```text
synthetic state + choice criterion
  -> strict request validation
  -> fake backend with known values
  -> JJM judgment envelope
  -> deterministic acceptance checks
  -> JSON on stdout
```

Make the same route work with one tested local Transformers checkpoint. Do not add a website, provider marketplace, database service, agent runtime, or plugin registry first. The JJM.dev publication track is separately scoped and must not block or quietly expand the local runtime. Domain registration is not deployment authorization.

## Required boundaries

Preserve upstream copyright and attribution. Keep an upstream remote. Add narrow extension seams when needed, not a directory-wide rename of the inference core. Use Python for the retained inference engine and a thin TypeScript client for Node consumers. The public JJM interface is provider-independent, but v1 does not promise every provider is supported.

Every result must say what actually produced it. Unknown metadata remains unknown. The result must not imply that calibrated estimates are available merely because the upstream level is `L1` or `L2`. Exact criterion matching is the default for loaded heads.

No action execution. No arbitrary code in criteria. No evidence URL fetches. No generation of explanations or citations on the decision path. No silent truncation. No silent cloud fallback. No training or artifact promotion as an inference side effect.

## Completion report

Report the immutable upstream base, implemented task IDs, changed files, exact commands and test outcomes, unresolved failures, supported runtime/model combinations, intentional deviations, and the one tested quickstart. Separate mocked behavior, real inference, and benchmark measurements.

Do not claim the fork is ready because the contract fixtures pass. Do not turn synthetic example values into benchmark claims.
