# 14. Optional composition with other systems

These are proposed boundaries, not inspected integrations or claims about the current APIs of other projects. Implement none in the core MVP. Read the actual target contracts before writing adapters.

## Application state and AppLedger

An application-state system may provide a versioned projection for a judgment request. JJM consumes that projection and returns its result with the source revision. It does not own, rebuild, or keep a duplicate application ledger.

Any AppLedger integration should preserve the distinction between a proposed interpretation and an accepted application fact. JJM may evaluate an assertion about a feature or task, but it does not promote that assertion to truth automatically.

## xFacts suite

A typed-facts system may supply stable entity or criterion references. JJM may emit an explicitly attributed judgment record that another system can store or inspect. A calibrated estimate is not a proven fact, and a missing observation is not proof of absence.

Do not invent an xFacts label type or schema here. Use an adapter after inspecting the relevant existing specification. JJM's contract remains independently usable.

## ForgeTrail

A workflow tool may request readiness or completion judgments at a step boundary. It owns the workflow and deterministic completion requirements. JJM returns a bounded assessment and evidence metadata, not a new copy of progress tracking or process state.

Do not automatically mark a step completed because the model selected a favorable outcome. The calling tool enforces its own criteria and state transition rules.

## Efficacy and proof-oriented tooling

A tool-result system may supply receipts or measured outcomes as evidence, or retain a judgment as one explicitly probabilistic assessment. Artifact integrity, observed outcome measurements, and semantic evaluation remain different categories.

Do not rename a model's self-assessment as measured success. A signature does not prove that a reported outcome is true. JJM is one possible evaluator, not a proof issuer.

## Agents and skills

An agent calls JJM with a bounded request. It receives no additional action permissions from doing so. A skill should document suitable questions, the meaning of uncertain results, and application-controlled escalation.

## Integration acceptance criteria

The core CLI works with no companion projects installed. A target project remains its own source of truth. Records use immutable IDs and revisions. Each adapter documents whether it supplied facts, claims, measurements, or inferred judgments. No integration is required for the first release.
