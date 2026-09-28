# 15. Selected identity, message, and visual direction

## Selected name and home

**JJM, Just Judgment Model.**

**JJM.dev** is the canonical project domain. The user reports acquiring it on September 25, 2026. Use `https://jjm.dev` as the canonical web origin. This pack does not verify DNS, HTTPS, a live deployment, or registry namespaces.

`BRAND.json` is the machine-readable source of identity choices for this pack. It is build/documentation metadata, not a runtime API schema or a service-discovery mechanism. Do not add it to every inference request. Update public copy and identity checks together when the owner changes a choice.

Use American spelling `judgment` in public prose and new contract naming, including `POST /v1/judgments`. Keep inherited upstream identifiers unchanged where that reduces maintenance. Do not create two competing API spellings. Search metadata can acknowledge the British `judgement` spelling without making it a second brand.

JJM is preferred over LJM because it does not imply a model-size class or new large pretrained weights. “Just” describes the narrow job: interpretation for a bounded decision, not a whole chat or agent platform. It is not a claim of moral or judicial authority.

## Primary message

Name: **JJM**

Expansion: **Just Judgment Model**

Product descriptor: **An open-source judgment runtime.**

Positioning: **A programmable judgment layer for applications and agents.**

Explanation: **Supply the state, define the criteria, and get structured judgments with uncertainty and provenance. Keep control of what happens next.**

Tagline: **Just judgment. Your application makes the decision.**

This is a runtime and interface built on existing models, not an announcement of newly trained foundation-model weights. In pre-release contexts say “JJM is being built as…” or “The proposed runtime…” when describing functionality that has not shipped. See `website/HOMEPAGE-COPY.md` for explicit pre-release and release variants.

A supporting line, rather than a competing main headline, may be: **Give your software a judgment call, not a conversation.**

Lineage statement for an actual fork: **An independent fork of AnyJev, with a stricter application-facing contract. Not affiliated with or endorsed by the upstream authors or TypeSafe AI.** Until the fork exists, say **planned independent fork** rather than implying implementation is complete.

## Visual brief

The wordmark joke is **JJM** beside **LLM**. A segmented J has the same shape as a mirrored L, so a segmented `JJM` reads as a visual rhyme with `LLM`. A conventional curved J is not a mirrored L. The correspondence is a deliberate glyph construction, and the product must not require the reader to decode it.

Keep a plain-text fallback, legible small sizes, and a monochrome version. Treat `JJM` as text in page titles, navigation, and accessible names. Do not reverse letters in body copy or replace the semantic heading with an unlabeled symbol.

Avoid copying upstream marks or visual identity. This pack includes a visual brief only, not a generated logo or trademark clearance. The name and the product explanation must still work when the reader never notices the visual joke.

## What is settled, and what is not

The name, expansion, spelling, canonical domain, primary positioning, and tagline are selected for this project. Do not spend implementation time looking for another domain.

Python and npm distribution names, owner-controlled scopes, the repository URL, social handles, and trademark review remain unresolved. Keep the corresponding `BRAND.json` fields explicit until verified. Domain registration does not establish those rights or reservations. Use owner-scoped packages or a distinctive suffix when needed.

The domain is not a public inference endpoint. Do not invent an API subdomain, telemetry host, support mailbox, or model download location from the fact that it is owned. Website routes in spec 17 are publication targets, not claims of currently reachable resources.

## Claims to avoid

Do not call the initial fork a newly trained model. Do not say every supported model is calibrated out of the box. Do not claim zero possibility of hallucination, universally correct decisions, guaranteed action safety, arbitrary-model compatibility, or persistent encode-once behavior.

Do not use inherited AnyJev measurements as JJM measurements. Do not say “no training” without distinguishing frozen base weights from fitted calibration and heads. Do not say “no generation anywhere” for a backend that makes a one-token readout call.

Do not imply a site, package, checkpoint, or integration is live because this spec names it. Keep implemented, experimental, proposed, and unavailable capabilities visibly distinct.

## Product boundary

“Just” must survive feature requests. JJM evaluates supplied context and returns findings. It does not own the application state, select and execute actions, obtain new permissions, or silently turn into a workflow engine, chat assistant, or agent framework.

The tagline is a boundary statement, not a claim that a model should substitute for accountable human judgment.
