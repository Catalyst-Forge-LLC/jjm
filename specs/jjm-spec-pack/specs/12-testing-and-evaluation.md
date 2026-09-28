# 12. Tests, evaluation, and release gates

## Test layers

Contract tests validate JSON shapes and semantic cross-field constraints. Compiler tests validate deterministic state/criterion rendering and option-ID mapping. Adapter tests validate result coverage and actual method reporting. Engine regression tests preserve upstream numerical behavior except for explicit fork changes. Real-model tests demonstrate a specific deployment works. Task evaluation measures usefulness for a particular judgment workload.

Passing an earlier layer does not establish a later one.

## Mandatory regression cases

T01: duplicate criterion and evidence IDs are rejected before inference.

T02: a missing required evidence item abstains with no prediction and no model invocation.

T03: `not_observable` remains an evaluated outcome when selected by an assertion model.

T04: an unrelated question with identical option strings cannot borrow a fitted head.

T05: changed option descriptions, tokenizer revision, renderer, model revision, layer, or prior causes an artifact mismatch unless explicitly compatible under a reviewed record.

T06: absent requested logprobs, NaN, infinity, wrong dimensions, duplicated token IDs, and partial distributions become errors.

T07: ordinal score levels are not permuted, and value equals the weighted index.

T08: `L0` cannot satisfy validated-in-scope calibration merely because its weights sum to one. `L1` and `L2` need suitable artifact metadata and validation too.

T09: unavailable model revision stays unknown. A friendly model alias does not pass verified-revision requirements.

T10: no silent output-level upgrade, hidden fallback, truncation, or cloud egress.

T11: repeated requests in frozen mode do not change decisions because unrelated traffic arrived.

T12: estimator state, cached prefixes, and artifacts are isolated across application scopes.

T13: single-token API readout is reported distinctly from direct logit access. No free-form generation or generated evidence claims appear.

T14: timeout and cancellation never produce late accepted results. OOM reports a bounded error.

T15: response order and IDs match the request. Unknown fields and unsupported schema versions are rejected.

T16: acceptance `pass` is impossible without a complete evaluated prediction. It does not override an outcome filter or unknown mandatory check.

T17: evidence `selected_ids` are empty when no selection operation ran and always a subset of considered IDs when selection is supported.

T18: immutable profile promotion affects only subsequent requests. Rollback restores the prior binding.

T19: package installation cannot overwrite a separately installed AnyJev distribution silently.

T20: marketing examples and quickstart values distinguish fixture data from real inference and real benchmarks.

## Evaluation tasks

Start with a small set relevant to the user's workflows: explicit ticket routing, evidence-supported task completion, relevance of a passage to a stated criterion, and a bounded rubric score when supported. Include unavailable evidence, contradictory evidence, negation, ambiguous requests, irrelevant context, and adversarial instruction-shaped content.

Use real observed outcomes or accountable annotation for the task. Preserve disagreements and unknowns. Do not derive all gold labels from the same model being evaluated.

## Metrics

Report per-class precision/recall and confusion matrices, macro-F1 for classification, and Brier/NLL plus a defined calibration diagnostic for probabilistic outputs. For thresholded use, report coverage and conditional error among accepted items. Specify top-label versus classwise calibration and binning choices rather than calling all of them “ECE.”

For assertions, separately measure false support, false contradiction, and misuse of `not_observable`. A high aggregate score must not hide a weak minority outcome. For score, compare both the distribution and expected index against the chosen annotation method.

Provide sample counts, splits, model/runtime versions, and uncertainty estimates where justified. Do not treat an in-sample calibration fit as evidence of deployment accuracy. Do not claim guaranteed risk bounds from a small test set.

## Performance

Measure end-to-end and model-only latency separately where possible. Include warm/cold runs, concurrency, state length, criteria count, rotations, and actual prefix reuse. Report unavailable measurements as unavailable.

## Release gates

MVP: all contract and compiler tests pass, exact-head and incomplete-logprob regressions exist, the inherited CPU suite status is documented, one real local route passes a smoke test, and no unknown method/calibration fields are misrepresented.

Calibration release: held-out validation and promotion workflow pass, scope mismatch causes rejection, and replay metadata can identify the exact deployed artifact.

Provider release: actual account/API access tested, capability and semantics mappings are documented, privacy controls pass, and there is no fake probability conversion.

Public launch: names and licenses reviewed, upstream attribution retained, install collision behavior tested, quickstart run from a clean environment, and every performance claim traces to real result data.

## What this spec pack tests

`scripts/validate_pack.py` checks the included proposed schemas, positive synthetic fixtures, negative fixtures, and cross-field invariants. It does not exercise AnyJev, a GPU, provider APIs, actual confidence calibration, or a live service. See `VALIDATION.md` for the executed result.
