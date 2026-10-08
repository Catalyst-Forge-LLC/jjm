# 16. Decisions, tradeoffs, and open questions

## Defaults chosen by this specification

D01: selected name **JJM, Just Judgment Machine**. **JJM.dev** was acquired by the user and is the canonical project home. Use `https://jjm.dev` as the canonical web origin. This is not independent registration verification, trademark clearance, a package reservation, or a live-deployment claim.

D02: actual fork with retained Python inference machinery, not a JavaScript rewrite and not a standalone pretrained model.

D03: a provider-independent application contract, with one local adapter first.

D04: local CLI first. Loopback HTTP and TypeScript client next. No mandatory hosted account or cloud control plane.

D05: exact criterion/model/artifact binding. Alias routing and online adaptation off by default.

D06: uncalibrated advisory mode is useful and allowed when explicit. Calibration requirements fail closed.

D07: evidence references begin as supplied/considered IDs, not generated explanations or invented proof.

D08: the assertion type has three evidence-status outcomes. Runtime abstention remains separate.

D09: first L0 profile uses permutation-only readout with no mutable live prior. This intentionally differs from upstream's default prior correction and must be labelled and evaluated as such.

D10: JJM returns deterministic consumption checks but never owns authorization or executes actions.

D11: JSON Schema and semantic rules are the common contract. Initial rollout is a project contract, not a claim of ecosystem-wide standard adoption.

D12: package install isolation is mandatory until a tested vendoring strategy prevents ownership collisions.

D13: public product descriptor **an open-source judgment runtime**, positioning **a programmable judgment layer for applications and agents**, and tagline **Just judgment. Your application makes the decision.** `BRAND.json` records these choices.

D14: JJM.dev is a documentation and distribution home, not a mandatory inference service. Site publication is separately scoped and must not delay the first working local path.

D15: spec-pack 0.2 retains contract version 0.1 and the existing `urn:jjm:*:0.1` schema identities. Domain acquisition does not require a protocol migration or runtime network dependency.

## Tradeoffs

Keeping upstream's internals recognizable lowers merge friction but needs isolated packaging. A later vendoring move can solve broader distribution compatibility, at the cost of a larger maintenance diff.

Requiring complete probability data rejects some otherwise usable APIs. That is preferable to manufacturing confidence. A future score-only interface may support such providers explicitly without weakening the current contract.

Exact artifact matching sacrifices some convenient head reuse. That is deliberate until equivalence is demonstrated for a specific criterion, not inferred from common answer words.

Freezing priors and adaptation may give up gains on a particular workload. It provides a reproducible starting point. Reintroduce adaptive methods as separately identified experiments with measured benefit and controlled state.

Three-way assertion increases output complexity compared with a yes/no flag. It preserves the difference between evidence against a proposition and insufficient evidence to decide.

## Questions requiring implementation-time resolution

What immutable upstream commit is the base? Resolve in phase 0.

Which owner-controlled distribution names are available? Resolve before publishing, not before implementing local functionality.

Which checkpoint/runtime combination is the first tested reference deployment? Select using available hardware, access rights, and a small real smoke test. Do not assume all local hosts support the same dtype, engine, or acceleration.

Which exact calibration acceptance criteria apply to the first production workload? Define from the workload's failure costs and labelled data before automatic use.

Does the maintainer prefer a long-lived isolated fork distribution or a dedicated private inference namespace before public packaging? Resolve by the packaging release gate.

Which hosted judgment provider, if any, is needed? None is required for MVP.

When should JJM.dev be deployed, and which repository URL and verified install path should it advertise? Resolve through the separate publication track. No deployment is authorized by this package update.

## Future work deliberately deferred

Permanent encoded-state handles, online head routing, automatic artifact promotion, ensemble probabilities, arbitrary score-only providers, evidence span attribution, MCP, public registry, and cloud billing are not implied commitments.
