# 07. Evidence and source references

## Three distinct layers

**Input evidence** is material supplied by the application. **Model-selected evidence** is material a model judges relevant under an explicit selection task. **Verified evidence** is material that a separate deterministic or accountable process has checked for a specific claim.

Do not collapse these layers. In particular, an item classified by the caller as `tool_receipt` is not authenticated merely because it has that name.

## Version 0.1 evidence behavior

Accept inline evidence items with stable IDs, a kind, content, and optional observation time. Reject duplicate IDs. Include the effective selected input evidence in the state digest.

For every criterion, return which evidence IDs were actually included in its input as `considered_ids`. This means exposed to the evaluator, not proven attended to by the model.

Return `selection_method: none` and an empty `selected_ids` list in the first AnyJev adapter. Do not populate supporting citations from the input list simply because it was available. A bounded classifier result does not itself supply a reliable explanation or source attribution.

`required_evidence_ids` is a deterministic presence requirement. Missing required evidence produces an abstention before model execution. Presence checks do not validate authenticity, completeness, or meaning.

## Optional evidence selection

A later adapter may expose a separate bounded candidate-selection operation. The caller supplies candidate IDs and exact content. JJM asks narrowly scoped relevance or support questions about those candidates. Return only IDs from the supplied set, with the selector method and its own evaluation metadata.

Selection is additional inference with additional latency and error risk. It is not free provenance from the original judgment. Its result must not be promoted to independent verification. Candidate selection is deferred from the first release.

## No fabricated citations

Never generate URLs, quote spans, line ranges, or receipt IDs that were not supplied and validated. A future span feature must use deterministic UTF-8 byte offsets into the exact retained content, with boundary and substring tests. JavaScript UTF-16 indices and Python character indices must not be mixed.

No runtime network fetch of evidence URLs. No local filesystem dereference from a user-controlled string. Application code may fetch permitted material before the request and supply the resulting content explicitly.

## Claims versus observations

A supplied statement from an agent is an assertion. A tool's response is an observation whose reliability depends on the tool and context. A verified external receipt is a stronger artifact for specific checks, but JJM does not authenticate it by default.

The evidence schema preserves the caller's classification without treating it as an instruction to trust the content. A malicious payload saying that it is a system message remains data.

## Example boundary

Input contains `agent_note` with “The export succeeded.” There is no required receipt.

The model may evaluate the assertion criterion as `not_observable`. It may also make a mistake. The application can avoid depending on that judgment alone by declaring `export_receipt` required, which makes its absence a deterministic preflight abstention.

A matching receipt later arrives. A new state revision is submitted. JJM evaluates the new snapshot, and the application checks the receipt against the expected task and object independently.
