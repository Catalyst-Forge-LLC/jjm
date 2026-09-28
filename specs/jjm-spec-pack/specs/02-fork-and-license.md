# 02. Fork, license, attribution, and packaging

## Fork strategy

Use a normal fork with upstream history intact. Maintain an `upstream` remote and a dedicated record of the last incorporated commit. Keep the inference package internally recognizable for straightforward merges. The public project, documentation, CLI, and new API use JJM.

Do not bulk rename upstream files or identifiers in the first patch. A clean public API does not require obscuring source lineage.

## License obligations

The inspected repository is Apache-2.0. Its license permits derivative works subject to redistribution conditions, including a license copy, retained notices, and prominent notices in modified files. Preserve any applicable NOTICE content present at the actual fork commit. The license does not grant general rights to the upstream's trademarks. [U9]

Use Apache-2.0 for the fork's new code as the default maintainer decision, subject to final owner approval. Preserve original author metadata in credits and source notices. Add the fork maintainer separately. This specification is an implementation checklist, not a trademark or patent clearance opinion.

The repository's third-party file distinguishes dataset and external-code licenses. Do not assume the repository license also covers separately downloaded weights, tokenizer assets, datasets, or provider terms. Review each item before redistribution. [U10]

## Files to add or maintain

`UPSTREAM.md`: upstream URL, exact commit, fork date, incorporated updates, tests run, and deliberate behavior differences.

`CREDITS.md`: inherited credits plus JJM contributions, with no implication of Nokia or TypeSafe sponsorship.

`FORK_CHANGES.md`: narrow list of semantic departures, especially exact head binding, immutable runtime profiles, strict logprob coverage, and expanded provenance.

`LICENSE` and applicable notice files: keep legally relevant upstream material. For each modified inherited source file, add a short prominent modification notice without deleting existing headers.

`THIRD_PARTY.md`: continue asset-specific license tracking and distinguish redistribution from runtime download.

## Distribution collision

Do not publish a new wheel named `anyjev`. Do not install both upstream AnyJev and a fork-owned distribution containing the same `anyjev` import tree into one environment.

For initial deployment, use an isolated virtual environment or container containing the fork. Proposed public Python distribution name: `jjm-engine`, pending registry checks. Proposed import and CLI: `jjm`. Retained `anyjev` internals remain private implementation detail and are documented as conflicting with a separately installed upstream distribution.

Before public package release, either keep a documented isolated distribution with an install-time collision check or complete a dedicated vendoring namespace migration with import and artifact compatibility tests. Do not quietly create two distributions owning the same installed files.

The TypeScript client should use an owner-controlled scope selected at release time. Do not assume the unscoped `jjm` package or a particular organization is available. **JJM.dev** has been acquired by the user, but domain registration does not reserve a PyPI name, an npm scope, a GitHub repository, or a trademark.

Use `https://jjm.dev` as the project homepage in release metadata after the site is deployed. Set documentation URLs only for routes actually published. Resolve the repository URL and distribution names before publication, and keep them explicitly unset in `BRAND.json` until then. Do not invent `@jjm/*` install commands, support email addresses, or an `api.jjm.dev` service.

## Maintenance

Run an upstream reconciliation before merging updates. Apply unchanged numerical regression tests first, then JJM contract tests. Record intentional result changes as behavior changes even when the JSON schema is unchanged.

Retain upstream's benchmark attribution. Do not relabel inherited measurements as measurements of JJM. A new performance claim requires a JJM run with a traceable configuration and output artifact.
