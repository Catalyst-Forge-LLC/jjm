# 00. Product and scope

## Product definition

JJM is a callable, bounded judgment layer. An application supplies the state it wants evaluated, the questions that matter, and the permissible meanings of the answers. JJM returns typed findings and their evaluation context. Ordinary application code decides whether and how to use them.

**Public name:** JJM, Just Judgment Machine. Selected, not a placeholder.

**Canonical home:** JJM.dev, with `https://jjm.dev` as the canonical web origin. The user reports acquiring the domain. This does not establish a live site, hosted API, or cleared package namespace.

**Tagline:** Just judgment. Your application makes the decision.

**Technical description:** an open-source judgment runtime and contract, initially implemented as an AnyJev fork. The name does not mean that JJM is a new set of foundation-model weights.

**Core promise:** make judgment outputs easier to integrate, inspect, test, reject, and replace without hiding how they were obtained.

## The user problem

An application may need to interpret a ticket, a tool result, a retrieved passage, or an application snapshot. A prose answer is often an unnecessary intermediate representation. However, simply putting a category and a number in JSON does not tell the application whether the number is calibrated, whether evidence was available, which exact model was used, or whether the runtime silently changed methods.

JJM's contribution is a stronger boundary around such judgments, not a promise of universally better model reasoning.

## Concrete first workflow

An agent attempts a document export. The application passes the requested outcome, the observed tool result, and any supplied receipt as structured state. It asks whether the supplied evidence supports successful completion. JJM returns `supported`, `contradicted`, or `not_observable`, plus normalized model weights and method metadata. A required-receipt check may hold the result even when the model selected `supported`. The application independently checks the receipt and permissions before marking its task complete.

The judgment does not create the receipt, validate a remote resource, or change the task record.

## First release

Implement choice and evidence-status assertion. Provide an offline fake backend for deterministic contract tests, then one real local Transformers route. Include input bounds, explicit method identity, truthful number semantics, evidence-presence checks, request-scoped operation, and machine-readable rejection codes. Provide a CLI and a single tested example.

Support advisory uncalibrated outputs so a user can experiment immediately. Reject requests that demand unavailable calibration. A useful uncalibrated mode with clear labelling is better than falsely implying assurance.

The larger schema anticipates score and additional methods so later releases do not need a second unrelated envelope. Capability discovery must distinguish schema validity from implemented support.

## Later releases

Add calibration and head-artifact lifecycle, HTTP transport, a TypeScript client, and vLLM only when individually tested. Add direct hosted judgment providers through their own adapters rather than pretending they are AnyJev logits backends.

## Explicit non-goals

JJM is not a chat assistant, planner, action executor, RAG retriever, persistent world model, knowledge graph, audit-log service, identity provider, or new model-training platform. It is not an ethical or legal decision authority. It does not replace deterministic validation, application permissions, or accountable human decisions.

Do not add a general policy language, orchestration workflow, automatic provider competition, vector database, accounts, billing, or a public registry in v1.

“Just” is an architectural constraint, not only a name. The site explains and distributes the project. It does not become the default inference endpoint, store user states, or introduce an account requirement. Website publication follows a separate track in `specs/17-website-and-launch.md`.

## Success criteria

A developer can evaluate one state, understand every returned numeric field, reject a result with insufficient evidence or method support, reproduce its effective configuration, and replace the backend without changing criterion IDs or accepting changed semantics silently.

A maintainer can update upstream with a small and inspectable diff. A reviewer can distinguish what was observed, what was inferred, and what remains unverified.
