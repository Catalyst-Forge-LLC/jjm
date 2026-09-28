# 01. Upstream inspection and required changes

Inspected September 25, 2026 through public web retrieval. Branch: `main`. Exact commit unresolved. Treat these as source-inspection findings, not demonstrated runtime exploits or independently reproduced benchmark outcomes. Resolve and recheck them in phase 0. Primary references are in `research/SOURCES.md`.

## Existing foundation to preserve

AnyJev already provides typed questions, result objects, logit readout, debiasing, temperature calibration, fitted heads, artifacts, and backend seams. Its declared Python package version is `0.1.0`. Preserve these mechanics and their tests rather than inventing a replacement inference stack. [U1, U2, U3]

The public constructors bound letter-readout choices to 26 options and ordered scores to 10 levels. Keep backend limits visible rather than promising unrestricted criteria. [U4]

## Findings with implementation consequences

### F01. Bind a head to a criterion, not just its answer vocabulary

In the inspected `Decider.route`, exact-key lookup is followed by matching kind and option texts, including alternate order for some heads. Thus differently worded questions sharing an option vocabulary can reach the same head. This is broader than a proof of semantic equivalence. [U5]

**JJM requirement:** exact effective criterion and model identity by default. Do not call automatic routing with a pool of unrelated heads. A future alias requires an explicit, reviewed equivalence record and evaluation. Add a regression with two unrelated yes/no questions and a three-outcome assertion pair. Both must reject the other question's head.

### F02. Never substitute a fabricated low probability

The inspected vLLM label mapping uses `-30.0` when an expected token is absent from the response. [U6]

**JJM requirement:** missing label data is `INCOMPLETE_LABEL_DISTRIBUTION`, not evidence against the label. Validate token IDs, exact expected coverage, finite values, and dimensions before normalization. No partial distribution may become a complete-looking result.

### F03. Preserve result diagnostics in the public envelope

Upstream result objects carry diagnostics, but the inspected `Decision.to_dict()` exports a narrower representation that omits that field. [U7]

**JJM requirement:** read result objects directly through a typed adapter. Export the actual method, calibration status, early-exit status, and any relevant fallback. Do not reconstruct missing provenance from a shortened serialized result.

### F04. Freeze deployment behavior independently of learning

Upstream exposes mutable prior statistics, label-free adaptation, and label observation/refitting. [U5]

**JJM requirement:** inference uses an immutable deployment snapshot. Learning produces a separate candidate artifact. Neither arriving traffic nor an automatic fit promotes a new deployment. No cross-application statistics.

### F05. Use method identity, not an assurance ladder

Upstream result levels are ordered for its `require` API. [U7]

**JJM requirement:** retain `raw`, `L0`, `L1`, and `L2` as namespaced method metadata, not a cross-provider safety score. A fitted head with no held-out validation does not pass a request for validated in-scope estimates.

### F06. Describe actual execution, not a slogan

The vLLM logits path issues a one-token completion request. [U6]

**JJM requirement:** distinguish no prose generation from zero tokens emitted anywhere. Report `logit_readout`, `single_token_readout`, or `hidden_state_head` as appropriate. Keep generated text and chain-of-thought out of the public result.

### F07. Preserve backend-specific prefix reuse

The Transformers backend has a shared-prefix path with a token-boundary correctness check and a full-prompt fallback. [U8]

**JJM requirement:** keep the check and report reuse as actual, absent, or unknown. Never advertise persistent encode-once behavior from a shared-state request alone.

## Proposed module map

The following existing paths were inspected: `anyjev/question.py`, `result.py`, `decider.py`, `state.py`, `readout.py`, and `backends/{base,hf,vllm}.py`. Inspect calibration and head implementations before editing them.

Add JJM-specific contract/compiler/adapter/service modules outside the retained inference core. A narrow change to expose exact routing or immutable prior injection is preferable to reaching into multiple private dictionaries from application code.

## Reconciliation rule

When upstream has changed, record which finding no longer applies. Do not implement a workaround for a fixed behavior simply to match this document. Preserve the contract outcome and test, not an obsolete patch.
