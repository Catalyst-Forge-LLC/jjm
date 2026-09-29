"""The synthetic adapter: a contract demonstration with no model.

Weights are a softmax over bytes of SHA-256(criterion digest, rendered state). They carry no meaning. The same request always returns the
same weights, on any machine, with no network access or download.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

from jjm.adapters.base import AdapterOutput
from jjm.compile import CompiledCriterion

NOTICE = "Synthetic contract demonstration. Weights are a hash of the input, not a model judgment."


class SyntheticAdapter:
    adapter_id = "mock"
    adapter_version = "jjm-synthetic-1"
    execution_mode = "synthetic"
    method = "mock:normalized"
    upstream_level = None
    numeric_path = "synthetic"
    semantics = "normalized_readout"
    runtime_id = "jjm-synthetic"
    runtime_version = "1"
    dtype = None
    supported_kinds = ("choice", "assertion")
    max_choice_options = 26
    max_context_tokens = None

    def model_identity(self) -> dict[str, Any]:
        return {"id": "jjm/synthetic-no-model", "revision": None, "revision_status": "unknown",
                "tokenizer_revision": None}

    def context_tokens(self, state_text: str, compiled: CompiledCriterion) -> Optional[int]:
        return None

    def evaluate(self, state_text: str, compiled: CompiledCriterion) -> AdapterOutput:
        seed = hashlib.sha256(f"{compiled.criterion_digest}\n{state_text}".encode("utf-8")).digest()
        raw = [math.exp((seed[i % len(seed)] - 128) / 32.0) for i in range(compiled.k)]
        total = sum(raw)
        weights = [r / total for r in raw]
        weights[-1] = 1.0 - sum(weights[:-1])
        return AdapterOutput(weights=weights, prompt_evaluations=0, adapter_calls=0, prefix_reuse="not_used",
                             early_exit="not_used", diagnostics={"synthetic_notice": NOTICE})
