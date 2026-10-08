# 17. JJM.dev website and launch

## Purpose and scope

JJM.dev is the canonical project home for **JJM, Just Judgment Machine**. Its job is to explain the use, show the contract, and give developers one reliable path to trying the runtime. It is not a hosted inference product, model marketplace, account system, or replacement for the local CLI.

The user reports acquiring the domain. DNS, HTTPS, deployment status, repository location, and package-registry availability were not verified for this update. Every route below is a publication target, not a statement that a page already exists.

The website is a separate workstream. Do not build it before the first end-to-end local path merely because the domain is available. A simple pre-release information page is acceptable with owner approval and a clear status notice. Deployment, DNS changes, package publication, and external account changes require separate authorization.

## Identity and implementation target

Use `BRAND.json` for the selected name, expansion, spelling, descriptor, positioning, tagline, and canonical origin. Use `website/HOMEPAGE-COPY.md` for the copy hierarchy and release-stage variants.

Target a static-first SvelteKit site on Cloudflare Pages, consistent with the owner's web-stack preference. This is a design target, not a claim that a project or deployment exists. Keep the first site static. Add no database, authentication, server-side inference, or browser connection to a local runtime. Verify current framework and deployment instructions during implementation rather than relying on untested setup commands in this pack.

Keep the docs near the code and derive published examples and schemas from canonical release files. Prefer a straightforward build over a new documentation platform. Do not make Python inference depend on the JavaScript site toolchain.

## Homepage information order

The first screen should answer what JJM is, what goes in, what comes out, and who controls the next step. Lead with readable product text, not the segmented-J wordmark, an unexplained diagram, or benchmark claims.

1. **Name and concrete description.** JJM, Just Judgment Machine. An open-source judgment runtime. A programmable judgment layer for applications and agents.
2. **Input and output.** Supplied state and explicit criteria become structured findings with evaluation context. The application retains responsibility for actions.
3. **One small example.** Show the report-export scenario, then the missing-required-receipt variant. Label both as synthetic. Allow access to full JSON without forcing the complete envelope into the hero.
4. **Why the envelope matters.** Explain the actual model/method, number semantics, evidence boundaries, and acceptance outcome in short, concrete terms.
5. **Boundaries and status.** No chatbot or action execution, no claim that bounded outputs are always correct, and no hidden hosted requirement. Distinguish current functionality from planned functionality.
6. **One next step.** Use the release-appropriate documentation action. Keep source code as a secondary destination only after its repository is known and accessible.

The visual mark is optional. A segmented J is a mirrored L, so the mark can rhyme **JJM** with **LLM**, but the letters must stay readable. Use real heading text, accessible names, a plain-text fallback, and a monochrome option. This spec does not include or authorize publishing a generated logo.

## Planned routes

| Route | Purpose | Publication gate |
|---|---|---|
| `/` | Product explanation, synthetic example, and visible status | Copy matches actual release stage |
| `/docs/quickstart` | Install, configure, evaluate, and interpret | Before runtime release, clearly limited to spec-pack validation. Afterwards, exact clean-environment commands are tested |
| `/docs/concepts` | State, criteria, output meanings, evidence, and acceptance | Wording matches contract and status distinctions |
| `/docs/contract` | Request/response shape, versions, and semantic constraints | Examples are validated and unsupported features are labelled |
| `/docs/methods` | Actual readout methods and numeric semantics | Implemented, experimental, and planned methods are separated |
| `/docs/limitations` | Semantic errors, calibration scope, evidence, data handling, and hardware limits | No safety or accuracy guarantees beyond evidence |
| `/about` | Purpose, lineage, attribution, and project status | Repository and contributor details are confirmed before publication |
| `/changelog` | Runtime and documentation releases | Spec-pack updates are not presented as shipped runtime features |
| `/schemas/0.1` | Index and static downloads of the six schema files | Files match the canonical released contract exactly |

Only expose navigation to routes that are actually published. A static schema file would use `/schemas/0.1/request.schema.json`, for example. Retain `urn:jjm:request:0.1` as its schema identity. The URL is a distribution location, not a reason to rewrite the schema or fetch it during inference.

Use `https://jjm.dev` for canonical page URLs and metadata after deployment. Choose one canonical host. Redirect other controlled hosts only if configured and verified. Do not advertise a `www` alias, API subdomain, support mailbox, or download host that has not been provisioned.

## Status-aware copy and calls to action

While the runtime is not implemented, display **In development. This is a specification, not an installable runtime.** Keep that notice adjacent to feature explanations and examples, not buried in a footer.

In that stage, the primary action is **Read the specification**, targeting `/docs/contract` only once a real specification page is published. The secondary action is **See the example**, targeting an actual section on the page. Provide runnable spec-pack checks separately and say what they do not test. Do not display `pip install jjm`, `npm install jjm`, or any other unverified distribution command.

When the first local runtime release passes its release gate, the primary action becomes **Get started**, targeting `/docs/quickstart`. Publish the exact supported Python/runtime/model route and show what is required for the real-model example. Display the supported version and preserve the synthetic label for fixture-based demonstrations.

Show **View source** only when the real fork URL is verified. Do not substitute the upstream AnyJev repository as though it were the JJM fork. Credit upstream separately. Omit unavailable actions rather than creating dead links or collecting an email address as a substitute for a useful explanation.

## Example requirements

Use `examples/02-supported.request.json` and `examples/02-supported.response.json` for the main synthetic example. The secondary example uses `examples/03-missing-evidence.request.json` and `examples/03-missing-evidence.response.json`. A concepts page may also show `examples/04-not-observable.response.json`.

Render summaries directly from the fixture data and link to the full request/response pair. Do not create independently maintained abbreviated JSON that appears to satisfy the API contract when required fields are missing. Any shortened view must be labelled as a summary, not executable request or response JSON.

Keep `supported`, `contradicted`, and `not_observable` distinct from `evaluated`, `abstained`, and `error`. Missing a deterministically required receipt produces preflight abstention in the included example. It is not the same thing as a model selecting `not_observable` after evaluation.

All example weights, durations, and outcomes in this pack are synthetic. Do not describe a weight as verified confidence, a duration as a speed result, or a supplied receipt as independently authenticated. Explain that the application must still verify required receipts, current state, and authorization.

An interactive example may switch among bundled fixtures. It must not accept private user input, invoke inference, contact a provider, or probe localhost. A public live playground is not part of this release.

## Usability, privacy, and honest claims

Support keyboard navigation, visible focus, readable contrast, mobile layouts, and code blocks that can be read without making the whole page overflow. Do not rely on color alone to distinguish evidence outcomes. Put the main explanation in HTML, with no animation required to understand it. Use a compact navigation hierarchy rather than a wall of feature cards.

Keep the initial site free of analytics SDKs, advertising, external chat widgets, and input collection by default. This is a design choice, not a claim of no hosting-provider access logs. Any later analytics addition needs explicit owner approval and appropriate disclosure. Never put input states, evidence contents, credentials, or local service tokens into page URLs, telemetry, or browser bundles.

The public description must not imply every adapter has the same performance, one forward pass, calibrated probabilities, permanent encoded state, or universal compatibility. Do not republish upstream benchmark charts as measurements of JJM. A future hosted adapter does not turn the documentation domain into an inference destination.

## Release checklist

Mark items complete only after recording the relevant evidence. No boxes are pre-completed by this pack.

- [ ] The identity, spelling, tagline, and canonical origin match `BRAND.json`.
- [ ] Site status matches what is implemented and available, with planned features visibly labelled.
- [ ] The actual fork URL, distribution names, and owner-controlled namespaces are resolved wherever public links or install commands use them.
- [ ] The quickstart is tested in a clean environment, or is explicitly limited to spec-pack validation before release.
- [ ] Fixture summaries are derived from source JSON, and synthetic data is clearly identified.
- [ ] The missing-required-evidence example and the `not_observable` explanation preserve the contract distinction.
- [ ] Schema downloads match the published contract bytes and retain offline URN resolution.
- [ ] Keyboard, mobile, code-block, and readable-text checks pass.
- [ ] Internal links, canonical metadata, real HTTPS delivery, and any configured redirects are checked after authorized deployment.
- [ ] No page makes an unexpected inference, localhost, analytics, or private-state request.
- [ ] Attribution is present, and no support address, performance result, or package availability is invented.
- [ ] The owner has separately authorized the publication being performed.

See JJM-050 through JJM-053 in `specs/13-implementation-plan.md`. Passing the spec-pack validators does not complete this website checklist.
