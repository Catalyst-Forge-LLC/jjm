# JJM — Phase 1 architecture brief

_Structured capture of planning and architecture before code scaffolding. Goal: the build can start from this file plus `appledger/` without re-reading the whole planning chat._

**Status:** `draft`  
**Last updated:** 2026-09-28  
**Phase 1 exit:** Do not mark the `plan` phase complete until this brief is **locked** and the major commitments are accepted decision records.

Source material: `specs/jjm-spec-pack/` (spec-pack 0.2, contract 0.1) and the decisions already made in this project. This brief does not replace the spec pack. It says what to build first.

---

## 1. Problem and outcome

**What we are building (2–4 sentences):**

JJM (Just Judgment Model) is an open-source local judgment runtime. An application supplies a state snapshot, explicit criteria, and an allowlisted profile. JJM returns typed findings with method, provenance, uncertainty, and acceptance checks. The application still decides and acts. It is a local stand-in for the job TypeSafe Jev does as a hosted call: classification and judgment without writing prose. The implementation is a fork of Nokia's AnyJev, not a new set of foundation-model weights.

**Project archetype:** `product`

**What “done” looks like for v1 (measurable where possible):**

One local command takes a JSON request on stdin and prints a JSON judgment envelope on stdout. A synthetic backend proves the contract with no model download. The same route then runs on one tested local Transformers checkpoint for choice and three-way assertion. Weights are labeled `normalized_readout`. A request that demands validated calibration is rejected. Upstream commit, commands, and test outcomes are written down. Passing the spec-pack fixtures alone is not enough.

---

## 2. Users and hero flow

**Primary user(s):**

A developer embedding bounded judgments in their own application. The first real audience is the owner, using Exec Foundry as the proof project.

**The single most important workflow (hero flow) end-to-end:**

```text
synthetic state + choice criterion
  -> strict request validation
  -> fake backend with known values
  -> JJM judgment envelope
  -> deterministic acceptance checks
  -> JSON on stdout
```

Then the same route with one local Transformers checkpoint. Stdout is JSON. Process exit 0 means the envelope was produced, not that the judgment was accepted or correct.

**Secondary workflows (if any) for v1:**

- Shadow-score Exec Foundry fit rank (Perfect Fit through Long Shot) on jobs already on the board. Compare the selected level and the latency with the current Sonnet rank. Do not change the board, and do not replace the written fit justification.
- JJM.dev explains the product. It does not run inference.

---

## 3. Constraints

- **Technical:** Python for the retained AnyJev inference engine. Contract version stays 0.1. American spelling `judgment` in public prose and new names. Exact criterion-to-head binding. No fabricated probabilities, silent backend downgrade, silent truncation, evidence URL fetches, generated explanations on the decision path, or training as a side effect of inference. Unknown metadata stays unknown. A passing acceptance check is not permission to act.
- **Business / timeline:** No hosted service, accounts, or billing in v1. Domain `https://jjm.dev` is user-reported and not independently verified. Package names and trademark clearance are still open. Do not publish a package or deploy the site until those are resolved and the owner asks.
- **Explicit non-goals for v1:** See section 10.

---

## 4. Stack and tooling

| Area            | Choice   | Status | Notes / WHY |
| --------------- | -------- | ------ | ----------- |
| Framework       | Python CLI around AnyJev; FilePress for the site | proposed | The runtime is not a SvelteKit app. The spec pack's SvelteKit site target is superseded by the FilePress landing guide. |
| Language        | Python for the engine; TypeScript only for a later thin client | proposed | Spec release 0.2, not v1. |
| DB / backend    | None | proposed | A judgment is stateless. The caller supplies the snapshot. |
| Auth / storage  | None | proposed | No accounts. Local files only if a profile or artifact must be pinned. |
| Styling         | FilePress landing layout; JJM mark, colors, and type | confirmed | `decision-filepress-landing`. Do not copy Finetuna's teal and orange. |
| Deploy / CI     | None in v1 | proposed | Local CLI first. Site publish is a separate track. |
| Git host        | GitHub `Catalyst-Forge-LLC/jjm` | proposed | Owner named this remote. `BRAND.json` still has `repository_url` unset until it is verified. |
| DNS             | Cloudflare, when the site is published | proposed | Not authorized by this brief. |
| Registrar       | Already acquired: jjm.dev | confirmed | User report, 2026-09-25. Not an independent DNS check. |
| Package manager | pnpm for any Node package; the Python engine uses its own environment | proposed | Isolated from a separately installed upstream `anyjev`. |
| State persistence | Stateless request. No PocketBase. | proposed | Nothing has to outlive the process except files the operator pins. |

---

## 5. Data model (sketch)

**Core entities:**

- **Request:** schema 0.1, state snapshot, unique criteria, allowlisted profile, requirements.
- **Criterion:** `choice` or `assertion` in v1. Assertion outcomes are `supported`, `contradicted`, and `not_observable`. `score` stays in the schema for later.
- **Envelope:** per-criterion status `evaluated`, `abstained`, or `error`; weights; semantics; acceptance `pass`, `hold`, or `reject`; provenance.
- **Profile:** immutable local deployment settings. v1 freezes permutation-only L0, which differs from upstream's default prior correction and must be labeled as such.

**Relationships:**

One request, one snapshot, many criteria, one profile. Criteria do not read each other's answers. The application owns state, permissions, and actions.

**Existing data / migration:**

None inside JJM. Exec Foundry already stores jobs, resumes, and Sonnet fit ranks. The shadow trial reads those; it does not migrate them.

---

## 6. Integrations and external systems

| Integration | Purpose | Auth / secrets | Risk notes |
| ------------- | ------- | -------------- | ---------- |
| AnyJev upstream | Inference machinery to preserve | None for the fork itself | Pin an immutable commit in Phase 0. Reconcile findings F01–F07. Do not patch a behavior upstream has already fixed. |
| Local Transformers checkpoint | The one real v1 route | Local files only. No required cloud key. | Choose the checkpoint from the machine that will run it. Do not assume dtype or GPU. |
| Exec Foundry | First external proof, after the local route works | Uses Exec Foundry's existing data. JJM does not take its keys. | Shadow only. The written justification stays on Sonnet until the score is good enough to stand alone. |
| JJM.dev / FilePress | Explanation and, later, distribution | Publish credentials only when a deploy is authorized | The site must not become an inference API. |

---

## 6a. Content-generation pattern

Skip for the runtime. JJM does not generate prose. The site copy is hand-authored from `website/HOMEPAGE-COPY.md`, adjusted to the FilePress landing shape. Exec Foundry's fit justification remains that app's existing Sonnet call during the shadow trial.

| Field | Value |
| ----- | ----- |
| **Pattern** | Not used for JJM output |
| **Provider / model** | One local Transformers checkpoint, chosen at build time |
| **Env vars** | None required for the synthetic path. Model path is local configuration, not a cloud key. |

---

## 7. Hardest problems and risks

1. **Upstream drift.** The spec pack inspected public `main` without a commit hash and without running tests. The first build task is to pin the commit, run CPU tests, and record what no longer matches.
2. **Honest numbers.** A normalized readout can look like a calibrated probability. v1 must label it and reject requests that ask for validated calibration.
3. **Fresh git vs a real fork.** This repo was initialized with the spec pack. The spec wants upstream history and an `upstream` remote. Resolve that at the start of the build without discarding the spec pack, wordmark, or ledger.
4. **Fit-rank comparison will not equal Jev.** The trial measures agreement with Exec Foundry's current Sonnet ranks and how often the margin is wide. It is not a claim about TypeSafe's weights.

---

## 8. Architectural decisions (numbered)

**D1.** Keep AnyJev's Python inference and wrap it. Do not rewrite it in JavaScript and do not train a new foundation model. Rejected: a from-scratch model, a JS port of inference.

**D2.** v1 is one local CLI path: synthetic backend, then one Transformers route, choice and assertion only. Rejected: starting with the website, vLLM, hosted providers, or a TypeScript client.

**D3.** The public site is a FilePress landing page using the FilePress landing design spec. JJM keeps its segmented-J wordmark, colors, and type. Rejected: the spec pack's SvelteKit site as the publication target.

**D4.** No accounts, database, or hosted inference in v1. The caller sends a complete snapshot. Rejected: PocketBase, login, and `api.jjm.dev`.

**D5.** The first external trial is an Exec Foundry fit-rank shadow score, after D2 works. JJM returns the distribution over the five fit levels. Exec Foundry keeps the star override and the written justification. Rejected: replacing tailoring, the dossier, or outreach drafts.

**D6.** Contract stays 0.1. Spec-pack version 0.2 does not bump schema URNs or fixture engine ids. Rejected: rewriting fixtures to match the pack number.

**D7.** Public spelling is American `judgment`. British `judgement` is search metadata only. Rejected: keeping the spec pack's original British spelling as the brand.

---

## 9. Open questions (before or during Phase 2)

| # | Question | Owner / resolve by |
| - | -------- | ------------------ |
| 1 | Which immutable AnyJev commit is the base? | Build start, before editing inference code |
| 2 | How does this fresh repo gain upstream history without losing the spec pack? | Build start |
| 3 | Which checkpoint and runtime is the one tested local route? | After the synthetic path, on the machine that will run it |
| 4 | Are `jjm-engine`, the npm scope, and `Catalyst-Forge-LLC/jjm` actually available? | Before any publish |
| 5 | When should JJM.dev be deployed, and which install command may it show? | Separate publication track, after a local route exists |

---

## 10. Explicitly out of scope (v1)

- Chat, action execution, RAG, a workflow engine, accounts, billing, and a public model registry
- Score criteria, fitted calibration artifacts, loopback HTTP, and the TypeScript client
- vLLM, hosted provider adapters, and learning or artifact promotion
- Changing Exec Foundry's board or replacing its written fit justification
- Treating L0/L1/L2 as an assurance grade
- Publishing packages, deploying the site, or downloading a model during the synthetic path

---

## 11. First feature batch (post-scaffold)

1. Pin the upstream commit, run its CPU tests, and write the reconciliation note.
2. Attribution, license notices, and an isolated Python environment. Do not publish a wheel named `anyjev`.
3. Strict request validation and the synthetic choice path through acceptance checks to JSON stdout.
4. Assertion, including missing-evidence abstention before inference.
5. One local Transformers route with honest provenance and the frozen permutation-only L0 profile.
6. Record the exact quickstart commands and separate mocked runs from real inference.
7. Only then, shadow-score Exec Foundry fit rank.
8. FilePress home page in the landing shape, with truthful pre-release status. It does not block steps 1–6.

---

## 12. Handoff checklist (before leaving Phase 1)

- [ ] User has confirmed stack, folder shape, data sketch, hero flow, and v1 boundaries
- [ ] This brief is **locked** (no `[draft]` ambiguity) or remaining items are only in §9 Open questions
- [ ] `appledger/` updated: a decision record for each major D#, and the session records the sign-off
- [ ] The build starts by reading this file, `appledger/profiles/forgetrail.yaml`, and the latest session record

---

## Folded-in cautions

- Code owns the envelope. The model fills values. Do not ask a generative model to invent the JSON shape.
- Report the real error when validation fails. Do not turn a missing field into a zero probability.
- Do not mark this phase complete, and do not start the fork, until this draft is locked.
