/**
 * Example application-side consumption boundary.
 * This is not a JJM client, engine, or a replacement for full schema validation.
 * receiptVerified and authorized must come from independent application checks.
 * The function has no side effects and never executes an action.
 */
export function assessExportCompletion({
  request,
  response,
  criterionId,
  currentStateRevision,
  receiptVerified = false,
  authorized = false,
}) {
  const hold = reason => ({ readyForApplicationAction: false, reason })
  if (!request || !response || response.request_id !== request.request_id) {
    return hold('REQUEST_MISMATCH')
  }
  if (currentStateRevision !== request.state?.revision) {
    return hold('STATE_CHANGED')
  }
  const criteria = request.criteria?.filter(c => c.id === criterionId) ?? []
  const results = response.results?.filter(r => r.criterion_id === criterionId) ?? []
  if (criteria.length !== 1 || results.length !== 1) {
    return hold('CRITERION_MISMATCH')
  }
  const result = results[0]
  if (criteria[0].kind !== 'assertion' || result.kind !== 'assertion') {
    return hold('ASSERTION_REQUIRED')
  }
  if (result.provenance?.state_id !== request.state.id ||
      result.provenance?.state_revision !== request.state.revision) {
    return hold('RESULT_STATE_MISMATCH')
  }
  if (result.status !== 'evaluated' || result.acceptance?.status !== 'pass') {
    return hold('JUDGMENT_NOT_ADMITTED')
  }
  if (result.prediction?.selected !== 'supported') {
    return hold('OUTCOME_NOT_SUPPORTED')
  }
  if (!receiptVerified) return hold('RECEIPT_NOT_VERIFIED')
  if (!authorized) return hold('ACTION_NOT_AUTHORIZED')
  return {
    readyForApplicationAction: true,
    reason: 'APPLICATION_CHECKS_PASSED',
    stateRevision: currentStateRevision,
  }
}
