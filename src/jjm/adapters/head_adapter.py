"""Fitted-head adapter: a frozen encoder reads the state once, a promoted head maps it to weights.

A criterion runs only when a promoted head's binding matches its criterion digest and this
encoder's identity exactly. Otherwise it abstains with ARTIFACT_MISMATCH before any model call.
There is no fallback to another method. Weights stay `normalized_readout`. Validation status stays
`fitted_unvalidated` until a held-out report is explicitly accepted for the request's scope.
Acceptance does not assess probability calibration.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from jjm import errors
from jjm.adapters.base import AdapterOutput
from jjm.adapters.fitted import FittedArtifactAdapter
from jjm.compile import CompiledCriterion
from jjm.heads.artifact import ArtifactStore
from jjm.heads.encoders import encoder_input
from jjm.heads.fit import head_from_artifact, served_artifact
from jjm.heads.head import Head
from jjm.heads.runtime import Featurizer

METHOD = "jjm:hidden-state-head"


class FittedHeadAdapter(FittedArtifactAdapter):
    adapter_id = "jjm-head"
    adapter_version = "jjm-head-1"
    execution_mode = "real"
    method = METHOD
    upstream_level = None
    numeric_path = "hidden_state_head"
    semantics = "normalized_readout"
    supported_kinds = ("choice", "assertion")
    max_choice_options = 26

    def __init__(self, featurizer: Featurizer, store: ArtifactStore, *, max_context_tokens: Optional[int]):
        self.featurizer = featurizer
        self.encoder = featurizer.encoder
        super().__init__(store)
        self.runtime_id = self.encoder.runtime_id
        self.runtime_version = self.encoder.runtime_version
        self.dtype = self.encoder.dtype
        self.max_context_tokens = max_context_tokens

    def settings_identity(self) -> dict[str, Any]:
        return {"encoder": self.encoder.identity(), "encoder_digest": self.featurizer.digest,
                "compute": self.encoder.compute_settings(), "feature_cache": self.featurizer.cache is not None}

    def model_identity(self) -> dict[str, Any]:
        return self.encoder.model_identity()

    def _load(self, compiled: CompiledCriterion) -> Optional[tuple[dict[str, Any], Head]]:
        art = served_artifact(self.store, self.featurizer, compiled)
        return None if art is None else (art, head_from_artifact(art))

    def context_tokens(self, state_text: str, compiled: CompiledCriterion) -> Optional[int]:
        prefix, body = encoder_input(state_text, compiled, self.encoder.identity()["prefix_template"])
        return self.encoder.count_tokens(prefix, body)

    def prefetch(self, pairs: Sequence[tuple[str, CompiledCriterion]]) -> None:
        """Encode many states in batches ahead of per-request evaluation. Results land in the feature caches."""
        wanted = [(t, c) for t, c in pairs if self._serve(c) is not None]
        if wanted:
            template = self.encoder.identity()["prefix_template"]
            self.featurizer.features([encoder_input(t, c, template) for t, c in wanted])

    def evaluate(self, state_text: str, compiled: CompiledCriterion) -> AdapterOutput:
        served = self._serve(compiled)
        if served is None:
            raise errors.AdapterError(errors.ARTIFACT_MISMATCH, "No promoted head matches this criterion and encoder.")
        art, head = served
        try:
            run = self.featurizer.features(self.featurizer.inputs([state_text], compiled))
        except errors.JJMError:
            raise
        except MemoryError:
            raise errors.AdapterError(errors.INTERNAL_ERROR, "The encoder ran out of memory.", True) from None
        except Exception as exc:  # encoder failures must become bounded item errors
            name = exc.__class__.__name__
            raise errors.AdapterError(errors.INTERNAL_ERROR, f"The encoder failed with {name}.",
                                      "OutOfMemory" in name) from None
        weights = [float(w) for w in head.weights(run.features)[0]]
        ran = run.encoded > 0
        stats = run.encode
        return AdapterOutput(
            weights=weights,
            prompt_evaluations=1 if ran else 0,
            adapter_calls=stats.forward_passes if ran and stats else 0,
            prefix_reuse=stats.prefix_reuse if ran and stats else "not_used",
            early_exit=self.encoder.early_exit if ran else "not_used",
            diagnostics={"artifact_id": art["artifact_id"], "feature_cache": "miss" if ran else "hit",
                         "head_method": art["binding"]["method"],
                         "truncated": bool(stats.truncated) if ran and stats else False},
        )
