# JJM.dev homepage copy

This file supplies copy, not a rendered or deployed website. Use the pre-release version until an actual runtime release has passed its gates. `BRAND.json` holds the selected identity. `specs/17-website-and-launch.md` controls publication and example requirements.

## Identity and metadata

**Site name:** JJM

**Expanded name:** Just Judgment Model

**Canonical origin:** `https://jjm.dev`

**Page title:** JJM | Just Judgment Model

**Pre-release meta description:** JJM is a proposed open-source judgment runtime. Evaluate supplied state against explicit criteria, return structured findings, and keep your application in control.

**Runtime-release meta description:** JJM is an open-source judgment runtime. Evaluate supplied state against explicit criteria, return structured findings, and keep your application in control.

Do not advertise a current release, install command, or repository URL until it is verified. Do not silently switch to runtime-release copy because the domain has been deployed.

## First screen

# JJM

## Just Judgment Model

**An open-source judgment runtime.**

A programmable judgment layer for applications and agents.

Supply the state, define the criteria, and get structured judgments with uncertainty and provenance. Keep control of what happens next.

**Just judgment. Your application makes the decision.**

### Pre-release notice and actions

**In development. This is a specification, not an installable runtime.**

The examples below show the proposed contract using synthetic data, not a running model.

Primary action: **Read the specification**. Use `/docs/contract` after the page exists.

Secondary action: **See the example**. Use the actual example section anchor.

### Replacement actions after a tested runtime release

Primary action: **Get started**. Use `/docs/quickstart` with the exact tested installation route.

Secondary action: **View source**. Use the verified JJM fork URL.

Replace the pre-release notice with a precise release and support statement. Do not claim a wider backend or model matrix than was tested.

## What goes in, and what comes back

The proposed interaction is straightforward. Your application supplies a ticket, a tool result, a document passage, or an application snapshot. It defines the questions and what the allowed outcomes mean.

JJM returns structured findings with the evaluation context: the model and method, what the numbers mean, which supplied evidence was considered, and whether the result meets your stated consumption requirements.

Your application checks authorization, verifies required records, and decides what happens next. JJM does not take the action.

## Example: did the export complete?

**Synthetic contract example. No real model was run.**

The application supplies a report-export request, the agent's completion message, and a tool receipt. The criterion asks whether the supplied evidence establishes completion.

The included fixture returns **supported**, with an evaluated result and a passing consumption check. That does not authenticate the receipt or authorize the application to act. Those remain application checks.

Now remove the required receipt. The second fixture returns **abstained**, no prediction, and **MISSING_REQUIRED_EVIDENCE**. The absence is caught before inference, rather than being turned into a made-up probability.

Build these views from `examples/02-supported.request.json`, `examples/02-supported.response.json`, `examples/03-missing-evidence.request.json`, and `examples/03-missing-evidence.response.json`. Show them as human-readable summaries with access to the full JSON. Do not present the prose above as a complete API payload.

## A result you can inspect

**Know what produced it.** The contract records the actual method and model identity. Unavailable metadata stays unknown.

**Know what the numbers mean.** A normalized readout is not automatically a calibrated probability. Any calibration claim must carry its applicable validation context.

**Keep missing evidence separate from contrary evidence.** `supported`, `contradicted`, and `not_observable` describe evidence judgments. `abstained` and `error` describe whether evaluation could be completed.

**Keep control of the next step.** Acceptance checks describe whether a result meets stated consumption requirements. They do not grant permissions or guarantee correctness.

These are proposed contract requirements while the project remains pre-release. After implementation, label any unsupported portions rather than implying they already work.

## Just judgment

Not a chatbot. Not a retriever. Not an action executor.

JJM is being built as a narrow interface over existing inference machinery, not a new foundation model. It is intended to run locally first, without making this website part of the inference path.

A bounded answer can still be wrong. Supplied evidence can still be incomplete or misleading. Your application remains responsible for independent checks and accountable decisions.

After runtime release, replace “is being built” with wording that names the implemented route. Do not imply that all proposed methods or adapters have shipped.

## Lineage and footer

Pre-release: **A planned independent fork of AnyJev, with a stricter application-facing contract.**

After the fork exists: **An independent fork of AnyJev, with a stricter application-facing contract.**

**Not affiliated with or endorsed by the upstream authors or TypeSafe AI.**

Footer identity: **JJM, Just Judgment Model. JJM.dev.**

Link only to real documentation, source, release notes, and attribution pages. Add no guessed contact email, user count, testimonial, performance number, or availability badge.
