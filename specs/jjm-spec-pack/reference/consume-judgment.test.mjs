import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { assessExportCompletion } from './consume-judgment.mjs'

const load = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'))
const request = load('../examples/02-supported.request.json')
const response = load('../examples/02-supported.response.json')
const base = {
  request,
  response,
  criterionId: 'export-completed',
  currentStateRevision: 'r1',
  receiptVerified: true,
  authorized: true,
}

assert.equal(assessExportCompletion(base).readyForApplicationAction, true)
assert.equal(assessExportCompletion({ ...base, receiptVerified: false }).reason, 'RECEIPT_NOT_VERIFIED')
assert.equal(assessExportCompletion({ ...base, authorized: false }).reason, 'ACTION_NOT_AUTHORIZED')
assert.equal(assessExportCompletion({ ...base, currentStateRevision: 'r2' }).reason, 'STATE_CHANGED')
assert.equal(assessExportCompletion({ ...base, criterionId: 'missing' }).reason, 'CRITERION_MISMATCH')
const badId = structuredClone(response)
badId.request_id = 'another-request'
assert.equal(assessExportCompletion({ ...base, response: badId }).reason, 'REQUEST_MISMATCH')
const held = structuredClone(response)
held.results[0].acceptance.status = 'hold'
assert.equal(assessExportCompletion({ ...base, response: held }).reason, 'JUDGMENT_NOT_ADMITTED')
const unobserved = structuredClone(response)
unobserved.results[0].prediction.selected = 'not_observable'
assert.equal(assessExportCompletion({ ...base, response: unobserved }).reason, 'OUTCOME_NOT_SUPPORTED')
console.log('PASS: 8 application-boundary checks using synthetic fixtures. No action executed.')
