# Spec-pack changelog

These are specification-package revisions, not shipped runtime releases. The runtime roadmap and the JSON contract have separate version numbers.

## Name, October 8, 2026

The owner selected the expansion **Just Judgment Machine**. The September 25 record below used Model. Current identity files use Machine.

## 0.2, September 25, 2026

### Selected identity

The user has acquired **JJM.dev**. The pack now treats **JJM, Just Judgment Model**, as the selected project name and `https://jjm.dev` as its canonical web origin. Registration is recorded from the user's report. No independent registration, DNS, HTTPS, or deployed-site verification was performed.

The primary descriptor is **an open-source judgment runtime**. The positioning is **a programmable judgment layer for applications and agents**, and the tagline is **Just judgment. Your application makes the decision.** Public spelling is American **judgment**. A segmented J has the same shape as a mirrored L, so the optional wordmark joke is **JJM** beside **LLM**. The name still has to work as plain text.

Added `BRAND.json` as one machine-readable identity record. Updated the overview, kickoff, scope, packaging, developer experience, implementation plan, naming specification, decisions, and source-scope notes so they no longer send the implementing agent back into name or domain selection.

### Website and publication

Added `specs/17-website-and-launch.md` and `website/HOMEPAGE-COPY.md`. They cover the homepage, documentation routes, synthetic examples, status-aware calls to action, canonical metadata, schema downloads, and a release checklist.

Added JJM-050 through JJM-053 as a separate publication track. A website does not block the first working local route and does not create a hosted inference service, default API endpoint, account requirement, telemetry destination, or deployment authorization.

Registry names, owner-controlled package scopes, repository location, social handles, and trademark review remain unresolved. Domain acquisition does not establish any of those.

### Compatibility and validation

The proposed contract remains **0.1**. All six schema files, all example files, and the original contract validator are preserved byte-for-byte from spec-pack 0.1. Their synthetic version and digest labels were not rewritten merely to match the pack number. The proposed runtime releases remain 0.1, 0.2, and later as previously scoped. The application-boundary example was later renamed to `reference/consume-judgment.mjs`, and its reason code is `JUDGMENT_NOT_ADMITTED`, so that example is no longer byte-identical with the 0.1 pack.

Added an offline identity validator and its test output. Re-ran the original schema/fixture and application-boundary checks. Rebuilt the manifest and archive. See `VALIDATION.md` for exact scope and recorded results.

This update does not implement the fork or website, deploy anything, publish a package, re-inspect upstream, run a model, or reproduce a benchmark.

## 0.1, September 25, 2026

Initial specification pack: 17 focused specifications, six contract schemas, synthetic positive and negative examples, a contract validator, an application-boundary example, and upstream inspection notes.
