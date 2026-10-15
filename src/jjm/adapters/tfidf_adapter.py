"""CPU text classification with the same judgment and validation contract."""
from __future__ import annotations

import json

from jjm import errors
from jjm.adapters.base import AdapterOutput
from jjm.adapters.fitted import FittedArtifactAdapter
from jjm.digest import digest
from jjm.heads.artifact import binding
from jjm.heads.tfidf import METHOD, VERSION, TextModel, fit_text, identity, sklearn


class TfidfAdapter(FittedArtifactAdapter):
    adapter_id = "jjm-tfidf"
    adapter_version = VERSION
    execution_mode = "real"
    method = METHOD
    upstream_level = None
    numeric_path = "provider_native"
    semantics = "normalized_readout"
    supported_kinds = ("choice",)
    max_choice_options = 26
    max_context_tokens = None
    runtime_id = "scikit-learn"
    runtime_version = sklearn.__version__
    dtype = "float64"

    def __init__(self, store, settings):
        super().__init__(store)
        self.settings = dict(settings)

    def settings_identity(self):
        return identity(self.settings)

    def model_identity(self):
        return {"id": "jjm/tfidf-logistic", "revision": VERSION,
                "revision_status": "operator_attested", "tokenizer_revision": VERSION}

    def context_tokens(self, state_text, compiled):
        return None  # No language-model context window.

    def _load(self, compiled):
        try:
            art = self.store.promoted(digest(binding(compiled.criterion_digest, identity(self.settings), METHOD)))
            if art is None:
                return None
            if art["outcome_ids"] != list(compiled.outcome_ids):
                raise ValueError("outcome order changed")
            return art, TextModel.from_payload(art["payload"], self.settings, len(compiled.outcome_ids))
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            raise errors.AdapterError(errors.ARTIFACT_MISMATCH, "Invalid or unreadable TF-IDF artifact.") from None

    def fit(self, compiled, texts, labels, **kwargs):
        result = fit_text(self.store, self.settings, compiled, texts, labels, **kwargs)
        self._served.clear()
        self._refused.clear()
        return result

    def evaluate(self, state_text, compiled):
        served = self._serve(compiled)
        if served is None:
            raise errors.AdapterError(errors.ARTIFACT_MISMATCH, "No promoted TF-IDF artifact matches this criterion.")
        art, model = served
        try:
            features = model.vector.transform([state_text])
            weights = model.model.predict_proba(features)[0].tolist()
        except MemoryError:
            raise errors.AdapterError(errors.INTERNAL_ERROR, "TF-IDF ran out of memory.", True) from None
        except Exception as exc:
            raise errors.AdapterError(errors.INTERNAL_ERROR, f"TF-IDF failed with {type(exc).__name__}.") from None
        return AdapterOutput(weights=weights, prompt_evaluations=0, adapter_calls=1, prefix_reuse="not_used",
                             diagnostics={"artifact_id": art["artifact_id"], "feature_type": "tfidf",
                                          "active_features": features.nnz, "all_features_unknown": features.nnz == 0})
