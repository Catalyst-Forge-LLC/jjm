# JJM: Just Judgment Machine

**An open-source judgment runtime.**

A programmable judgment layer for applications and agents.

**JJM.dev** is the canonical project domain, acquired by the user. The site and package registries were not checked during this update.

JJM would be an independent AnyJev fork that preserves its inference machinery and adds a stricter application-facing contract. The product is a runtime and interface, not a claim to have trained a new foundation model.

## The core interaction

```text
State + explicit criteria + bounded outcomes
                 |
                 v
      Configured judgment method
                 |
                 v
Typed findings + uncertainty + provenance + admission checks
                 |
                 v
   Application decides and controls actions
```

## What the fork adds

The contract identifies the actual model, method, criterion, input snapshot, and calibration context. It distinguishes normalized readouts from validated calibrated estimates. It preserves the difference between adverse evidence, insufficient evidence, runtime abstention, and execution failure.

Exact criterion binding prevents a head from being reused merely because its answer labels match another question. Complete-output checks reject missing probabilities instead of inventing values. Frozen deployment profiles keep live traffic and experimental learning from silently changing inference behavior.

Evidence IDs identify material supplied to the evaluator. They do not become model-selected citations or independently verified proof. Deterministic application checks still enforce receipts, permissions, and current state.

## What stays out

No chat assistant, action executor, RAG retriever, persistent world model, cloud control plane, or automatic model-training platform. No required companion products. No hidden provider fallback or generated confidence percentages.

## A useful first release

Keep Python inference. Add a local CLI, strict JSON contract, synthetic demo, and one tested Transformers route for choice and three-way assertion. Start with explicitly uncalibrated advisory use. Add fitted-artifact validation, score, loopback HTTP, and a thin TypeScript client next.

The whole roadmap need not be implemented before the tool is useful.

## Selected identity

**JJM, Just Judgment Machine**, is the selected name. **JJM.dev** is its canonical home, with `https://jjm.dev` as the canonical web origin. The optional wordmark uses a segmented J, which has the same shape as a mirrored L, so **JJM** rhymes visually with **LLM**. The name does not depend on noticing that joke. Registry package names, repository location, and trademark clearance remain unresolved.

**Just judgment. Your application makes the decision.**

“Just” is a scope boundary. JJM evaluates supplied state and returns findings. It does not acquire action permissions, own a workflow, become a chat assistant, or require a hosted service. The name remains useful when nobody notices the segmented-J wordmark.

The homepage brief, release-aware copy, and documentation routes are in `specs/17-website-and-launch.md` and `website/HOMEPAGE-COPY.md`. The website is a separate publication track, not a new dependency of inference.

## Package status

This is an implementation spec, not a completed fork. Spec-pack version 0.2 contains 18 focused specifications, selected identity metadata, homepage copy, six schema files, synthetic request/response scenarios, and runnable contract/application-boundary/identity checks. The contract remains version 0.1. The original inference requirements and synthetic fixtures are unchanged. Upstream tests, provider calls, and model inference were not run.

Begin with `KICKOFF.md`. Source-inspection findings and primary references are in `specs/01-upstream-findings.md` and `research/SOURCES.md`.
