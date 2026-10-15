"""Shared promotion and validation lifecycle for fitted adapters."""
from __future__ import annotations

from typing import Any, Optional

from jjm import errors
from jjm.compile import CompiledCriterion
from jjm.heads.artifact import ArtifactStore
from jjm.heads.fit import contract_manifest


class FittedArtifactAdapter:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self._served: dict[str, Any] = {}
        self._refused: dict[str, str] = {}

    def _load(self, compiled: CompiledCriterion):
        raise NotImplementedError

    def _serve(self, compiled: CompiledCriterion):
        key = compiled.criterion_digest
        if key not in self._served:
            try:
                self._served[key] = self._load(compiled)
            except errors.AdapterError as exc:
                self._served[key] = None
                self._refused[key] = exc.code
        return self._served[key]

    def artifact(self, compiled: CompiledCriterion) -> Optional[dict[str, Any]]:
        served = self._serve(compiled)
        return None if served is None else served[0]

    def preflight_codes(self, compiled: CompiledCriterion) -> list[str]:
        if self._serve(compiled) is not None:
            return []
        return [self._refused.get(compiled.criterion_digest, errors.ARTIFACT_MISMATCH)]

    def validation(self, compiled: CompiledCriterion, scope_id: Optional[str] = None) -> dict[str, Any]:
        served = self._serve(compiled)
        if served is None:
            return {"status": "not_fitted", "artifact_id": None, "validation_report_id": None, "scope_match": False}
        art = served[0]
        acceptance = self.store.acceptance(art["binding_digest"])
        if acceptance is None or acceptance.get("artifact_id") != art["artifact_id"]:
            return {"status": "fitted_unvalidated", "artifact_id": art["artifact_id"],
                    "validation_report_id": None, "scope_match": False}
        match = acceptance.get("scope_id") == scope_id
        return {"status": "accepted_in_scope" if match else "out_of_scope",
                "artifact_id": art["artifact_id"],
                "validation_report_id": acceptance["validation_report_id"],
                "scope_match": match}

    def validation_manifest(self, compiled: CompiledCriterion, scope_id: str,
                            provenance: dict[str, Any]) -> Optional[dict[str, Any]]:
        served = self._serve(compiled)
        if served is None:
            return None
        art = served[0]
        acceptance = self.store.acceptance(art["binding_digest"])
        if (acceptance is None or acceptance.get("artifact_id") != art["artifact_id"]
                or acceptance.get("scope_id") != scope_id):
            return None
        return contract_manifest(art, provenance, scope_id)

