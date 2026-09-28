#!/usr/bin/env python3
"""Validate proposed JJM schemas and synthetic contract fixtures, not model quality."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource
except ImportError as exc:
    raise SystemExit('Install jsonschema with Draft 2020-12 support before running this validator.') from exc

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6
ASSERTION_OUTCOMES = ['supported', 'contradicted', 'not_observable']


def fail(message: str) -> None:
    raise ValueError(message)


def reject_constant(value: str) -> Any:
    fail(f'Non-finite JSON literal is forbidden: {value}')


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            fail(f'Duplicate JSON object key: {key}')
        out[key] = value
    return out


def finite_json(value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        fail('Integer exceeds this contract fixture validator\'s safe JSON range')
    if isinstance(value, float) and not math.isfinite(value):
        fail('Non-finite number')
    if isinstance(value, str):
        try:
            value.encode('utf-8', errors='strict')
        except UnicodeEncodeError as exc:
            raise ValueError('Invalid Unicode scalar data') from exc
    if isinstance(value, list):
        for child in value:
            finite_json(child)
    if isinstance(value, dict):
        for key, child in value.items():
            finite_json(key)
            finite_json(child)


def load(path: Path) -> Any:
    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
    finite_json(value)
    return value


def schema_registry() -> tuple[dict[str, Any], Registry]:
    schemas: dict[str, Any] = {}
    registry = Registry()
    for path in sorted((ROOT / 'schemas').glob('*.schema.json')):
        schema = load(path)
        Draft202012Validator.check_schema(schema)
        schemas[path.name] = schema
        registry = registry.with_resource(schema['$id'], Resource.from_contents(schema))
    return schemas, registry


def validate_shape(value: Any, name: str, schemas: dict[str, Any], registry: Registry) -> None:
    Draft202012Validator(schemas[f'{name}.schema.json'], registry=registry,
                        format_checker=FormatChecker()).validate(value)


def unique(values: list[str], description: str) -> None:
    if len(values) != len(set(values)):
        fail(f'Duplicate {description}')


def outcome_ids(criterion: dict[str, Any]) -> list[str]:
    if criterion['kind'] == 'assertion':
        return ASSERTION_OUTCOMES
    source = criterion['levels'] if criterion['kind'] == 'score' else criterion['options']
    return [entry['id'] for entry in source]


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=EPS):
        fail(f'{label} does not match its defined arithmetic')


def iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def validate_pair(request: dict[str, Any], response: dict[str, Any], artifact: dict[str, Any] | None = None) -> None:
    criteria = request['criteria']
    unique([c['id'] for c in criteria], 'criterion ID')
    input_evidence = {e['id'] for e in request['state']['evidence']}
    unique([e['id'] for e in request['state']['evidence']], 'evidence ID')
    if response['request_id'] != request['request_id']:
        fail('Request/response IDs differ')
    if [r['criterion_id'] for r in response['results']] != [c['id'] for c in criteria]:
        fail('Result IDs or ordering differ from request')

    for criterion, result in zip(criteria, response['results']):
        ids = outcome_ids(criterion)
        unique(ids, 'option ID')
        if criterion['kind'] != 'assertion':
            source = criterion['levels'] if criterion['kind'] == 'score' else criterion['options']
            unique([x['description'] for x in source], 'option description')
        if not set(criterion['accept_outcomes']) <= set(ids):
            fail('Unknown accepted outcome')
        if criterion['kind'] == 'score':
            if [level['value'] for level in criterion['levels']] != list(range(len(ids))):
                fail('Score indices must be consecutive from zero')
        if result['kind'] != criterion['kind']:
            fail('Result kind differs from criterion')
        p = result['prediction']
        prov = result['provenance']
        gate = result['acceptance']
        evidence = result['evidence']
        if prov['state_id'] != request['state']['id'] or prov['state_revision'] != request['state']['revision']:
            fail('Response state identity differs from request')
        if prov['profile_id'] != request['profile_id']:
            fail('Response profile differs from requested configured profile')
        if not set(evidence['considered_ids']) <= input_evidence:
            fail('Considered evidence was not supplied')
        if not set(evidence['selected_ids']) <= set(evidence['considered_ids']):
            fail('Selected evidence was not considered')
        if evidence['selection_method'] == 'none' and evidence['selected_ids']:
            fail('Evidence selections reported without a selection operation')
        missing = set(criterion['required_evidence_ids']) - input_evidence
        if set(evidence['missing_required_ids']) != missing:
            fail('Missing required evidence metadata is inaccurate')
        if missing and result['status'] != 'abstained':
            fail('Missing required evidence must cause preflight abstention')
        if result['status'] != 'evaluated':
            if p is not None or gate['status'] == 'pass':
                fail('A non-evaluated result cannot have a prediction or pass')
        else:
            if p is None or prov['method'] is None or prov['runtime']['numeric_path'] == 'none':
                fail('An evaluated result needs prediction and actual method metadata')
        if result['status'] == 'error' and result['error'] is None:
            fail('Error status requires an error object')
        if result['status'] != 'error' and result['error'] is not None:
            fail('Only error status may carry an error object')
        if gate['status'] != 'pass' and not gate['reason_codes']:
            fail('Held or rejected results require reason codes')
        if gate['status'] == 'pass':
            if gate['reason_codes']:
                fail('Passed result cannot carry rejection reasons')
            if any(ch['mandatory'] and ch['status'] != 'pass' for ch in gate['checks']):
                fail('An unknown or failed mandatory check cannot pass')
        if prov['model']['revision_status'] == 'verified' and not prov['model']['revision']:
            fail('A verified model revision must be identified')
        cal = prov['calibration']
        if cal['status'] == 'validated_in_scope':
            if not cal['artifact_id'] or not cal['validation_report_id'] or not cal['scope_match']:
                fail('In-scope calibration needs a named artifact, report, and scope match')
        if p is None:
            continue
        weights = p['weights']
        if set(weights) != set(ids):
            fail('Distribution must contain exactly the expected option IDs')
        close(sum(weights.values()), 1.0, 'Weight sum')
        if p['selected'] != max(ids, key=lambda key: weights[key]):
            fail('Selected outcome is not the deterministic argmax')
        ranked = sorted(weights.values(), reverse=True)
        close(p['top_mass'], ranked[0], 'Top mass')
        close(p['margin'], ranked[0] - ranked[1], 'Margin')
        entropy = -sum(v * math.log(v) for v in weights.values() if v) / math.log(len(weights))
        close(p['normalized_entropy'], entropy, 'Normalized entropy')
        if criterion['kind'] == 'score':
            if p['value'] is None:
                fail('Score needs an expected-index value')
            close(p['value'], sum(weights[level['id']] * level['value'] for level in criterion['levels']), 'Score value')
        elif p['value'] is not None:
            fail('Non-score predictions cannot contain score values')
        if p['semantics'] == 'calibrated_estimate':
            if cal['status'] != 'validated_in_scope':
                fail('Calibrated semantics require validated in-scope metadata')
            if artifact is None:
                fail('This synthetic calibrated fixture needs its artifact manifest')
            if artifact['artifact_id'] != cal['artifact_id'] or artifact['validation_report_id'] != cal['validation_report_id']:
                fail('Artifact or report identity mismatch')
            for key in ['criterion_digest', 'effective_config_digest', 'rendering_digest', 'model', 'method']:
                if artifact[key] != prov[key]:
                    fail(f'Artifact binding mismatch: {key}')
            if artifact['scope_id'] != request['scope_id']:
                fail('Artifact scope differs from request')
            if not artifact['promoted'] or artifact['validation_status'] != 'validated' or artifact['heldout_count'] <= 0:
                fail('Calibrated fixture requires approved held-out validation metadata')
        if gate['status'] == 'pass':
            requirements = request['requirements']
            allowed = requirements['allowed_methods']
            if allowed and prov['method'] not in allowed:
                fail('Actual method is not allowed')
            if requirements['calibration'] == 'validated_in_scope':
                if cal['status'] != 'validated_in_scope' or p['semantics'] != 'calibrated_estimate':
                    fail('Required calibration is unavailable')
            if requirements['min_top_mass'] is not None and p['top_mass'] + EPS < requirements['min_top_mass']:
                fail('Top mass below requirement')
            if requirements['min_margin'] is not None and p['margin'] + EPS < requirements['min_margin']:
                fail('Margin below requirement')
            if requirements['require_verified_model_revision'] and prov['model']['revision_status'] != 'verified':
                fail('Verified model revision is required')
            accepted = criterion['accept_outcomes']
            if accepted and p['selected'] not in accepted:
                fail('Selected outcome fails consumption filter')
            max_age = requirements['max_state_age_seconds']
            if max_age is not None:
                age = (iso(response['evaluated_at']) - iso(request['state']['observed_at'])).total_seconds()
                if age > max_age or age < -5:
                    fail('State freshness requirement not satisfied')


def verify_manifest() -> int:
    manifest_path = ROOT / 'MANIFEST.json'
    if not manifest_path.exists():
        fail('MANIFEST.json does not exist')
    manifest = load(manifest_path)
    expected = set(manifest['files'])
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p != manifest_path}
    if expected != actual:
        fail('Manifest file inventory differs from package contents')
    for relative, digest in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            fail('Manifest path escapes package root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            fail(f'Manifest digest mismatch: {relative}')
    return len(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-manifest', action='store_true')
    args = parser.parse_args()
    schemas, registry = schema_registry()
    index = load(ROOT / 'examples/index.json')
    valid = invalid = 0
    failures: list[str] = []
    for case in index['cases']:
        try:
            request = load(ROOT / case['request'])
            response = load(ROOT / case['response'])
            artifact = load(ROOT / case['artifact']) if case.get('artifact') else None
            validate_shape(request, 'request', schemas, registry)
            validate_shape(response, 'response', schemas, registry)
            if artifact is not None:
                validate_shape(artifact, 'artifact', schemas, registry)
            validate_pair(request, response, artifact)
        except Exception as exc:
            if case['expected'] == 'invalid':
                invalid += 1
                print(f'PASS rejected {case["id"]}: {str(exc).splitlines()[0]}')
            else:
                failures.append(f'{case["id"]}: {exc}')
        else:
            if case['expected'] == 'valid':
                valid += 1
                print(f'PASS accepted {case["id"]}')
            else:
                failures.append(f'{case["id"]}: invalid fixture was accepted')
    validate_shape(load(ROOT / 'examples/capabilities.synthetic.json'), 'capabilities', schemas, registry)
    # Check strict parser behavior independently of JSON Schema.
    parser_rejections = 0
    for payload in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}']:
        try:
            json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except ValueError:
            parser_rejections += 1
        else:
            failures.append(f'Strict parser accepted {payload}')
    if args.check_manifest:
        count = verify_manifest()
        print(f'PASS manifest: {count} files verified')
    if failures:
        print('\nFAILED\n' + '\n'.join(failures), file=sys.stderr)
        return 1
    print(f'\nSUMMARY: {len(schemas)} schemas valid, {valid} positive pairs accepted, '
          f'{invalid} negative pairs rejected, 1 capability fixture valid, '
          f'{parser_rejections} strict parser cases rejected.')
    print('Scope: synthetic contract checks only. No model, GPU, provider, or calibration evaluation.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
