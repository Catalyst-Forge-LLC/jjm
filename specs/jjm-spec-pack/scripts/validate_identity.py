#!/usr/bin/env python3
"""Check this spec revision's identity and publication boundaries without network access.

These are documentation checks. They do not verify a domain, registry, website,
trademark, runtime implementation, or release readiness.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), f'{path.name} must contain a JSON object')
    return value


def validate_identity(identity: dict[str, Any]) -> None:
    expected = {
        'spec_pack_version': '0.2',
        'contract_version': '0.1',
        'name': 'JJM',
        'expanded_name': 'Just Judgment Model',
        'name_status': 'selected',
        'descriptor': 'An open-source judgment runtime.',
        'positioning': 'A programmable judgment layer for applications and agents.',
        'tagline': 'Just judgment. Your application makes the decision.',
        'spelling': 'judgment',
    }
    for key, value in expected.items():
        require(identity.get(key) == value, f'Identity mismatch: {key}')
    domain = identity['domain']
    require(domain['display'] == 'JJM.dev', 'Unexpected display domain')
    require(domain['canonical_host'] == 'jjm.dev', 'Unexpected canonical host')
    origin = urlsplit(domain['canonical_origin'])
    require(origin.scheme == 'https' and origin.netloc == domain['canonical_host']
            and not origin.path and not origin.query and not origin.fragment,
            'Canonical origin must be the HTTPS origin without a path')
    require(domain['registration_status'] == 'acquired'
            and domain['registration_basis'] == 'user_report'
            and domain['recorded_on'] == '2026-09-25',
            'Acquisition must retain its user-reported basis')
    require(domain['deployment_status'] == 'not_verified',
            'This spec update did not verify a deployment')
    website = identity['website']
    require(website['stage'] == 'spec_only' and website['routes_status'] == 'planned',
            'Website routes must remain labelled as planned in this spec revision')
    for flag in ['public_inference_api', 'online_inference_demo', 'runtime_requires_site']:
        require(website[flag] is False, f'Unexpected website capability: {flag}')
    routes = list(website['routes'].values())
    require(len(routes) == 9 and len(set(routes)) == len(routes),
            'Expected nine distinct planned routes')
    for route in routes:
        parsed = urlsplit(route)
        require(route.startswith('/') and not route.startswith('//')
                and not parsed.scheme and not parsed.netloc
                and not parsed.query and not parsed.fragment
                and '..' not in route.split('/'), 'Invalid relative site route')
    dist = identity['distribution']
    require(dist['python_candidate'] == 'jjm-engine' and dist['python_status'] == 'unverified',
            'Python distribution is a candidate, not a verified reservation')
    require(dist['npm_package'] is None and dist['npm_status'] == 'unverified'
            and dist['repository_url'] is None and dist['repository_status'] == 'unverified',
            'Unresolved namespaces must not be invented')
    require(dist['trademark_clearance'] == 'not_performed',
            'This revision did not perform trademark clearance')
    require(dist['cli'] == 'jjm' and dist['cli_status'] == 'proposed',
            'The CLI is proposed, not implemented by this pack')


def main() -> int:
    identity = read_json(ROOT / 'BRAND.json')
    validate_identity(identity)
    print('PASS selected identity, domain provenance, versions, and publication boundaries')

    core_docs = [
        'README.md', 'JJM-OVERVIEW.md', 'KICKOFF.md',
        'specs/00-product-and-scope.md', 'specs/15-naming-and-positioning.md',
    ]
    for relative in core_docs:
        text = (ROOT / relative).read_text(encoding='utf-8')
        for required in [identity['expanded_name'], identity['domain']['display'], identity['tagline']]:
            require(required in text, f'{relative} omits selected identity text: {required}')
        require('open-source judgment runtime' in text.lower(),
                f'{relative} omits the runtime descriptor')
    print(f'PASS identity and runtime positioning in {len(core_docs)} core documents')

    site = (ROOT / 'specs/17-website-and-launch.md').read_text(encoding='utf-8')
    for route in identity['website']['routes'].values():
        require(f'`{route}`' in site, f'Planned route lacks a spec entry: {route}')
    print('PASS nine planned routes have publication requirements')

    copy_text = (ROOT / 'website/HOMEPAGE-COPY.md').read_text(encoding='utf-8')
    for text in [identity['expanded_name'], identity['descriptor'], identity['positioning'],
                 identity['tagline'], identity['domain']['canonical_origin'],
                 'This is a specification, not an installable runtime.',
                 'Synthetic contract example. No real model was run.',
                 'MISSING_REQUIRED_EVIDENCE']:
        require(text in copy_text, f'Homepage copy omits required text: {text}')
    for relative in ['examples/02-supported.request.json', 'examples/02-supported.response.json',
                     'examples/03-missing-evidence.request.json', 'examples/03-missing-evidence.response.json']:
        require((ROOT / relative).is_file() and relative in copy_text and relative in site,
                f'Homepage fixture reference is missing: {relative}')
    print('PASS homepage identity, pre-release notice, synthetic labels, and fixture references')

    schemas = list((ROOT / 'schemas').glob('*.schema.json'))
    require(len(schemas) == 6, 'Expected the original six schema entry files')
    for path in schemas:
        schema = read_json(path)
        kind = path.name.removesuffix('.schema.json')
        require(schema['$id'] == f'urn:jjm:{kind}:0.1', f'Schema identity changed: {path.name}')
    pairs = list((ROOT / 'examples').glob('*.request.json')) + list((ROOT / 'examples').glob('*.response.json'))
    require(len(pairs) == 50, 'Expected 25 request/response fixture pairs')
    for path in pairs:
        require(read_json(path)['schema_version'] == '0.1', f'Fixture contract version changed: {path.name}')
    print('PASS six schema URNs and all 50 request/response fixture versions remain at contract 0.1')

    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    require('specification, version 0.2' in readme and '18 focused specifications' in
            (ROOT / 'JJM-OVERVIEW.md').read_text(encoding='utf-8'), 'Incorrect pack version or spec count')
    require(len(list((ROOT / 'specs').glob('*.md'))) == 18, 'Expected 18 focused specifications')
    for relative in ['BRAND.json', 'CHANGELOG.md', 'specs/17-website-and-launch.md',
                     'website/HOMEPAGE-COPY.md', 'scripts/validate_identity.py']:
        require((ROOT / relative).is_file() and relative in readme, f'New file missing from reading map: {relative}')
    print('PASS pack version, 18-spec inventory, and updated reading map')

    mutations: list[tuple[str, Any]] = [
        ('contract-version', lambda x: x.update(contract_version='0.2')),
        ('domain', lambda x: x['domain'].update(canonical_origin='https://example.org')),
        ('deployment', lambda x: x['domain'].update(deployment_status='verified')),
        ('runtime-dependency', lambda x: x['website'].update(runtime_requires_site=True)),
        ('invented-package', lambda x: x['distribution'].update(npm_package='jjm', npm_status='reserved')),
    ]
    for name, mutate in mutations:
        changed = copy.deepcopy(identity)
        mutate(changed)
        try:
            validate_identity(changed)
        except ValueError:
            pass
        else:
            raise ValueError(f'Invalid identity mutation accepted: {name}')
    print(f'PASS rejected {len(mutations)} deliberate identity/version/publication mutations')
    print('\nSUMMARY: 7 identity/documentation check groups passed, including 5 negative mutation cases.')
    print('Scope: offline spec consistency only. No domain, registry, website, runtime, or model verification.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f'FAIL identity validation: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
