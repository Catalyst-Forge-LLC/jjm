"""The seam between JJM and an inference backend."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from jjm import errors
from jjm.compile import CompiledCriterion

SUM_TOLERANCE = 1e-6


@dataclass
class AdapterOutput:
    """Weights in the criterion's declared option order, plus honest execution accounting.
    Counters the adapter cannot measure are None, not zero."""

    weights: list[float]
    prompt_evaluations: Optional[int]
    adapter_calls: Optional[int]
    prefix_reuse: str = "unknown"
    early_exit: str = "not_used"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Adapter(Protocol):
    adapter_id: str
    adapter_version: str
    execution_mode: str          # "synthetic" | "real"
    method: str
    upstream_level: Optional[str]
    numeric_path: str
    semantics: str
    runtime_id: str
    runtime_version: str
    dtype: Optional[str]
    supported_kinds: tuple[str, ...]
    max_choice_options: int
    max_context_tokens: Optional[int]

    def model_identity(self) -> dict[str, Any]: ...

    def context_tokens(self, state_text: str, compiled: CompiledCriterion) -> Optional[int]:
        """Tokens in the longest prompt this criterion will send, or None when not measurable."""
        ...

    def evaluate(self, state_text: str, compiled: CompiledCriterion) -> AdapterOutput: ...


def check_weights(weights: Sequence[Any], k: int) -> list[float]:
    """Reject anything that is not a complete, finite distribution over exactly k outcomes (T06).
    Nothing is repaired or renormalized."""
    if len(weights) != k:
        raise errors.AdapterError(errors.INCOMPLETE_LABEL_DISTRIBUTION,
                                  f"The backend returned {len(weights)} weights for {k} outcomes. "
                                  "No prediction was emitted.")
    out = []
    for w in weights:
        try:
            value = float(w)
        except (TypeError, ValueError):
            raise errors.AdapterError(errors.NONFINITE_RESULT, "The backend returned a non-numeric weight.") from None
        if not math.isfinite(value):
            raise errors.AdapterError(errors.NONFINITE_RESULT, "The backend returned a NaN or infinite weight.")
        if value < 0.0 or value > 1.0:
            raise errors.AdapterError(errors.NONFINITE_RESULT, f"The backend returned a weight outside [0, 1]: {value}")
        out.append(value)
    total = sum(out)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise errors.AdapterError(errors.INCOMPLETE_LABEL_DISTRIBUTION,
                                  f"The backend's weights sum to {total!r}, not 1. No prediction was emitted.")
    return out
