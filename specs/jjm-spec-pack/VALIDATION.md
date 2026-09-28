# Validation performed

Spec-pack 0.2, updated September 25, 2026. This report describes checks actually run during this revision.

## Executed checks

`python scripts/validate_pack.py` completed successfully. It validated six Draft 2020-12 schema files, accepted all 10 positive synthetic request/response pairs, rejected all 15 deliberately invalid pairs, validated the capability fixture, and rejected four invalid strict-parser inputs.

`node reference/consume-judgment.test.mjs` completed successfully. All eight synthetic application-boundary checks passed. No action was executed.

`python scripts/validate_identity.py` completed successfully. Seven identity/documentation check groups passed, including five deliberate invalid-metadata mutations. The checks cover the selected name and domain, the user-reported basis for acquisition, runtime-versus-website boundaries, unresolved namespaces, five core documents, nine planned site routes, homepage fixture references, and the separation between pack version 0.2 and contract version 0.1.

The first identity-check run caught two fixture response paths that the site spec referred to indirectly rather than by exact filename. Both references were made explicit, and the full check then passed. This was a documentation consistency correction, not a model or contract change.

The preparation environment used Python 3.13.5, jsonschema 4.26.0, and Node v22.16.0. These are observed test-environment versions, not a claim about the latest available versions.

Captured outputs are in `research/contract-validation-output.txt`, `research/application-validation-output.txt`, and `research/identity-validation-output.txt`.

## Compatibility with spec-pack 0.1

The original archive was checked before editing. Its manifest and contract/application-boundary checks passed.

A byte comparison against the original ZIP confirmed that the six JSON schema files, every file under `examples/`, and `scripts/validate_pack.py` were unchanged by the identity update. The spelling correction renamed the application-boundary example to `reference/consume-judgment.mjs` and changed its prose and the `JUDGMENT_NOT_ADMITTED` reason code. The identity script is additional tooling, not a replacement for the original validator. Contract semantics, fixture versions, and synthetic digest values were not rewritten for the naming or spelling update.

## Scope and limitations

These checks validate the proposed contract, synthetic fixtures, an illustrative application-side boundary, and specification consistency. They do not test a JJM runtime implementation, actual model probabilities, real calibration, an inference backend, GPU performance, a hosted provider, or the correctness of semantic judgments.

Upstream public code and documentation were inspected for the original pack. This update did not repeat that inspection. No upstream tests, provider calls, or model inference ran. The exact upstream commit remains a phase-0 requirement.

JJM.dev acquisition is recorded from the user's report. This update did not verify domain ownership, DNS, HTTPS, a deployed website, registry namespaces, trademark clearance, or the website release checklist. No fork implementation, website implementation, deployment, package publication, or external account change was performed.

Fixture digests are synthetic identifiers. No real model artifact payload was hashed or verified. The validator checks artifact metadata correspondence, not asset integrity or truthfulness of a calibration-validation report. Production checks remain implementation work.

## Archive integrity

`MANIFEST.json` records SHA-256 digests of every other file in this package. Run the following from the unmodified extracted directory:

```bash
python scripts/validate_pack.py --check-manifest
node reference/consume-judgment.test.mjs
python scripts/validate_identity.py
```

The final ZIP is checked for archive corruption, extracted into a clean directory, and validated using these commands before delivery. The manifest establishes package integrity against itself, not authorship, a live deployment, or validity of scientific claims.
