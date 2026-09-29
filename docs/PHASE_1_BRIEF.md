# JJM — Phase 1 architecture brief

_Structured capture of planning and architecture before code scaffolding. Goal: the build can start from this file plus `appledger/` without re-reading the whole planning chat._

**Status:** `locked`  
**Last updated:** 2026-09-28  
**Locked:** 2026-09-28, on the owner's instruction "proceed as far as you can without my intervention", after the second review. Changes after this point go through a new decision record.  
**Phase 1 exit:** Do not mark the `plan` phase complete until this brief is **locked** and the major commitments are accepted decision records.

Source material: `specs/jjm-spec-pack/` (spec-pack 0.2, contract 0.1) and the decisions already made in this project. This brief does not replace the spec pack. It says what to build first.

---

## 1. Problem and outcome

**What we are building (2–4 sentences):**

JJM (Just Judgment Model) is an open-source local judgment runtime. An application supplies a state snapshot, explicit criteria, and an allowlisted profile. JJM returns typed findings with method, provenance, uncertainty, and acceptance checks. The application still decides and acts. It is a local stand-in for the job TypeSafe Jev does as a hosted call: classification and judgment without writing prose. Local Jev-style readout already exists in `nokia-applied-research/AnyJev` and several other open projects. JJM's contribution is the contract: exact criterion binding, honest weight semantics, deterministic acceptance, and provenance. The engine is a pinned `anyjev` release used as a dependency, not a new set of foundation-model weights.

**Project archetype:** `product`

**What “done” looks like for v1 (measurable where possible):**

One local command takes a JSON request on stdin and prints a JSON judgment envelope on stdout. A synthetic backend proves the contract with no model download. The same route then runs on one tested local Transformers checkpoint for choice and three-way assertion. Weights are labeled `normalized_readout`. A request that demands validated calibration is rejected. Upstream version, commands, and test outcomes are written down. Passing the spec-pack fixtures alone is not enough. The Exec Foundry shadow trial comes after v1 and is not part of its exit.

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

- After v1: shadow-score Exec Foundry fit rank (Perfect Fit through Long Shot) on jobs already on the board, using the protocol in section 2a. Do not change the board, and do not replace the written fit justification.
- JJM.dev explains the product. It does not run inference.

---

## 2a. Fit-rank shadow protocol

The trial asks one `choice` criterion with five options and reports it as a choice. Score criteria stay out of scope. The levels are ordered, and the envelope does not claim to know that. Under permutation-only L0 the five options are rotated like any other choice. The trial report says so.

- **Level descriptions:** Write and freeze a one-line description for each of the five levels before the first run. Exec Foundry today has labels only. Sonnet's rule "Good Fit (3) when there is no candidate evidence" becomes a separate `assertion` about whether candidate evidence exists. If the evidence is missing, the fit criterion abstains and does not return a middle rank.
- **Baseline:** Sonnet's own rank is `fitJustificationRank`, not `fitRank`, because `fitRank` may hold the owner's star override. Owner overrides are reported separately as a human signal.
- **Ceiling:** Re-run Sonnet on the same jobs to measure how often Sonnet agrees with itself. JJM is not expected to beat that.
- **Ground truth:** The owner labels about 50 jobs blind, without seeing either model's rank. Agreement with Sonnet is agreement with the model being replaced, not accuracy.
- **Sample:** About 150 or more jobs.
- **Metrics:** Confusion matrix, exact and within-one agreement, quadratic weighted kappa, mean signed offset, and bootstrap confidence intervals. Repeat a subset to show that local runs are deterministic.
- **No leakage:** A test asserts that no `fitJustification*` or `fitRank*` field reaches the JJM state.
- **Same inputs:** JJM gets the same listing text, resume, and fit context that Sonnet gets, including the same `LLM_PROMPT_LARGE_TEXT_MAX` truncation (120,000 characters). Token counts are logged for every job.
- **Latency:** p50 and p95, cold and warm, on named hardware. Sonnet's latency includes writing the justification. The report says so and does not present it as a like-for-like comparison.

---

## 3. Constraints

- **Technical:** Python for the retained AnyJev inference engine. Contract version stays 0.1. American spelling `judgment` in public prose and new names. Exact criterion-to-head binding. No fabricated probabilities, silent backend downgrade, silent truncation, evidence URL fetches, generated explanations on the decision path, or training as a side effect of inference. Unknown metadata stays unknown. A passing acceptance check is not permission to act.
- **Business / timeline:** No hosted service, accounts, or billing in v1. Domain `https://jjm.dev` is user-reported and not independently verified. Package names and trademark clearance are still open. On 2026-09-28, `jjm` was already taken on both PyPI and npm, and `jjm-engine` was free on PyPI. jjm.dev uses Cloudflare nameservers. v1 does not offer Jev wire compatibility or use Jev's name in its API. Do not publish a package or deploy the site until those are resolved and the owner asks.
- **Explicit non-goals for v1:** See section 10.

---

## 4. Stack and tooling

| Area            | Choice   | Status | Notes / WHY |
| --------------- | -------- | ------ | ----------- |
| Framework       | Python CLI with a pinned `anyjev` dependency; FilePress for the site | proposed | The runtime is not a SvelteKit app. The spec pack's SvelteKit site target is superseded by the FilePress landing guide. |
| Language        | Python for the engine; TypeScript only for a later thin client | proposed | Spec release 0.2, not v1. |
| DB / backend    | None | proposed | A judgment is stateless. The caller supplies the snapshot. |
| Auth / storage  | None | proposed | No accounts. Local files only if a profile or artifact must be pinned. |
| Styling         | FilePress landing layout; JJM mark, colors, and type | confirmed | `decision-filepress-landing`. Do not copy Finetuna's teal and orange. |
| Deploy / CI     | None in v1 | proposed | Local CLI first. Site publish is a separate track. |
| Git host        | GitHub `Catalyst-Forge-LLC/jjm` | proposed | Owner named this remote. `BRAND.json` still has `repository_url` unset until it is verified. |
| DNS             | Cloudflare, when the site is published | proposed | Not authorized by this brief. |
| Registrar       | Already acquired: jjm.dev | confirmed | User report, 2026-09-25. Not an independent DNS check. |
| Package manager | pnpm for any Node package; the Python engine uses its own environment | proposed | `anyjev` is pinned to an exact version and hash in that environment. |
| State persistence | Stateless request. No PocketBase. | proposed | Nothing has to outlive the process except files the operator pins. |

---

## 5. Data model (sketch)

**Core entities:**

- **Request:** schema 0.1, state snapshot, unique criteria, allowlisted profile, requirements.
- **Criterion:** `choice` or `assertion` in v1. Assertion outcomes are `supported`, `contradicted`, and `not_observable`. `score` stays in the schema for later.
- **Envelope:** per-criterion status `evaluated`, `abstained`, or `error`; weights; semantics; acceptance `pass`, `hold`, or `reject`; provenance.
- **Profile:** immutable local deployment settings. v1 freezes permutation-only L0, which differs from upstream's default prior correction and must be labeled as such. The profile pins every AnyJev `Decider` setting explicitly: prior, shifts, `adaptive_shifts`, `canonical_order`, and the combine rule. It never relies on upstream defaults, because upstream has said its defaults will change.

**Relationships:**

One request, one snapshot, many criteria, one profile. Criteria do not read each other's answers. The application owns state, permissions, and actions.

**Existing data / migration:**

None inside JJM. Exec Foundry already stores jobs, resumes, and Sonnet fit ranks. The shadow trial reads those; it does not migrate them.

---

## 6. Integrations and external systems

| Integration | Purpose | Auth / secrets | Risk notes |
| ------------- | ------- | -------------- | ---------- |
| AnyJev upstream | Inference machinery, used as a dependency | None | Apache-2.0. Pin an exact release. Fork only when a patch is unavoidable, and send it upstream. Reconcile findings F01–F07 against the pinned release. Do not patch a behavior upstream has already fixed. |
| Local Transformers checkpoint | The one real v1 route | Local files only. No required cloud key. | Choose the checkpoint from the machine that will run it. Start with the Qwen3 family, which upstream has published results for. Do not assume dtype or GPU. |
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

1. **Upstream drift.** The spec pack inspected public `main` at 0.1.0 without a commit hash and without running tests. AnyJev 0.2.0 was published on PyPI on 2026-09-28. It adds opt-in `adaptive_shifts` and `canonical_order` settings and says they will become defaults in a later release. The first build task is to pin a release, run its CPU tests, and re-check findings F01–F07, especially F03 and F04.
2. **Honest numbers.** A normalized readout can look like a calibrated probability. v1 must label it and reject requests that ask for validated calibration.
3. **Dependency, not a history fork.** Using a pinned `anyjev` release removes the need to graft upstream history into this repo. If a patch becomes unavoidable, a fork happens then, and the spec pack, wordmark, and ledger stay where they are.
4. **Fit-rank comparison will not equal Jev.** The trial measures agreement with Sonnet, the owner, and Sonnet's own repeatability. It is not a claim about TypeSafe's weights.
5. **Context size and CPU latency.** A resume plus a listing is likely 5–15k tokens. Five rotated passes on a CPU could take minutes per job. This is an estimate, not a measurement. Choose and name the hardware before committing to the trial.
6. **Ordered levels as a choice.** Rotating ordered levels can hurt accuracy, and upstream never rotates score levels. v1 accepts this and reports it. It does not claim ordinal calibration.
7. **Crowded field.** Many open Jev alternatives appeared in the two weeks after launch. JJM stands out only if the contract, honesty rules, and determinism hold in practice.

---

## 8. Architectural decisions (numbered)

**D1.** Keep AnyJev's Python inference and wrap it as a pinned `anyjev` dependency. Fork only when a patch is unavoidable. Do not rewrite it in JavaScript and do not train a new foundation model. Rejected: a from-scratch model, a JS port of inference, and a full history fork at the start.

**D2.** v1 is one local CLI path: synthetic backend, then one Transformers route, choice and assertion only. Rejected: starting with the website, vLLM, hosted providers, or a TypeScript client.

**D3.** The public site is a FilePress landing page using the FilePress landing design spec. JJM keeps its segmented-J wordmark, colors, and type. Rejected: the spec pack's SvelteKit site as the publication target.

**D4.** No accounts, database, or hosted inference in v1. The caller sends a complete snapshot. Rejected: PocketBase, login, and `api.jjm.dev`.

**D5.** The first external trial is an Exec Foundry fit-rank shadow score, after v1 works. It runs as a five-option `choice` with frozen level descriptions, plus an assertion for missing candidate evidence, under the section 2a protocol. Exec Foundry keeps the star override and the written justification. Rejected: replacing tailoring, the dossier, or outreach drafts, and bringing `score` into v1 for this trial.

**D6.** Contract stays 0.1. Spec-pack version 0.2 does not bump schema URNs or fixture engine ids. Rejected: rewriting fixtures to match the pack number.

**D7.** Public spelling is American `judgment`. British `judgement` is search metadata only. Rejected: keeping the spec pack's original British spelling as the brand.

---

## 9. Open questions (before or during Phase 2)

| # | Question | Owner / resolve by |
| - | -------- | ------------------ |
| 1 | Which exact `anyjev` release is pinned (0.2.0 or later)? | Build start, before writing adapters |
| 2 | Which hardware runs the trial, and is a CPU-only latency figure acceptable? | Before the fit-rank trial |
| 3 | Which checkpoint and runtime is the one tested local route? | After the synthetic path, on the machine that will run it |
| 4 | Which package names: `jjm-engine` on PyPI, which npm scope, and is `Catalyst-Forge-LLC/jjm` created? (`jjm` is taken on both registries.) | Before any publish |
| 5 | When should JJM.dev be deployed, and which install command may it show? | Separate publication track, after a local route exists |

---

## 10. Explicitly out of scope (v1)

- Chat, action execution, RAG, a workflow engine, accounts, billing, and a public model registry
- Score criteria, fitted calibration artifacts, loopback HTTP, and the TypeScript client
- Jev wire compatibility. A mapping to Jev's Choice, Score, and Noul shapes may come later. Noul is not the same as a three-way assertion.
- vLLM, hosted provider adapters, and learning or artifact promotion
- Changing Exec Foundry's board or replacing its written fit justification
- Treating L0/L1/L2 as an assurance grade
- Publishing packages, deploying the site, or downloading a model during the synthetic path

---

## 11. First feature batch (post-scaffold)

1. Pin the `anyjev` release, run its CPU tests, and write the reconciliation note against findings F01–F07.
2. Attribution, license notices, and an isolated Python environment. Do not publish a wheel named `anyjev`.
3. Strict request validation and the synthetic choice path through acceptance checks to JSON stdout.
4. Assertion, including missing-evidence abstention before inference.
5. One local Transformers route with honest provenance and the frozen permutation-only L0 profile.
6. Record the exact quickstart commands and separate mocked runs from real inference.
7. Only then, write the fit level descriptions and run the section 2a shadow trial.
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
- Do not mark this phase complete, and do not start the build, until this draft is locked.
- Record which numbers are vendor claims, independent results, or our own measurements. Many public Jev figures are vendor claims.
