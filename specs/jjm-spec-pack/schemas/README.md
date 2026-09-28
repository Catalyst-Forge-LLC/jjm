# Proposed contract schemas

Draft 2020-12. `definitions.schema.json` is the local canonical definition bundle. The other files are entry points referencing its URN. Load all schemas into a local resolver registry. Do not fetch the URNs from the network.

Schemas validate structure, not every cross-field relationship. `scripts/validate_pack.py` also checks identifier uniqueness, complete distributions, selected outcomes, summaries, score arithmetic, provenance correspondence, calibration claims, and consumption requirements in the included fixtures.

Production implementations must additionally enforce strict byte/token limits, profile authorization, artifact payload verification, actual effective-config digest calculation, and backend execution behavior. The fixture validator is not a production judgment runtime or a full security validator.

`0.1` is this proposal's contract version. It is not an external standard or published package version. Unsupported kinds must be rejected even when they are representable in the schema.

Spec-pack 0.2 changes identity and publication requirements, not these schema bytes. Keep the existing `urn:jjm:*:0.1` identifiers. The proposed JJM.dev distribution route is `/schemas/0.1/<filename>`, but those URLs are not claimed live. When publishing, serve exact release files without rewriting their `$id` or `$ref` values. Runtime validators continue to resolve the bundled schemas offline.
