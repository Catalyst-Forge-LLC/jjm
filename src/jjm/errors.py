"""Stable error codes (spec 04) and the exceptions that carry them."""
from __future__ import annotations

from dataclasses import dataclass

INVALID_REQUEST = "INVALID_REQUEST"
DUPLICATE_CRITERION_ID = "DUPLICATE_CRITERION_ID"
UNSUPPORTED_KIND = "UNSUPPORTED_KIND"
TOO_MANY_OPTIONS = "TOO_MANY_OPTIONS"
CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"
MISSING_REQUIRED_EVIDENCE = "MISSING_REQUIRED_EVIDENCE"
STATE_TOO_OLD = "STATE_TOO_OLD"
ARTIFACT_MISMATCH = "ARTIFACT_MISMATCH"
CALIBRATION_NOT_VALIDATED = "CALIBRATION_NOT_VALIDATED"
METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
MODEL_IDENTITY_UNKNOWN = "MODEL_IDENTITY_UNKNOWN"
INCOMPLETE_LABEL_DISTRIBUTION = "INCOMPLETE_LABEL_DISTRIBUTION"
NONFINITE_RESULT = "NONFINITE_RESULT"
BACKEND_TIMEOUT = "BACKEND_TIMEOUT"
INTERNAL_ERROR = "INTERNAL_ERROR"

# Acceptance-only reason codes used by the pack's fixtures.
TOP_MASS_BELOW_MINIMUM = "TOP_MASS_BELOW_MINIMUM"
MARGIN_BELOW_MINIMUM = "MARGIN_BELOW_MINIMUM"
OUTCOME_NOT_ACCEPTED = "OUTCOME_NOT_ACCEPTED"
UNSUPPORTED_REQUIREMENT = "UNSUPPORTED_REQUIREMENT"
BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"


@dataclass
class JJMError(Exception):
    """A whole-request failure: no envelope is produced."""

    code: str
    message: str
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


class RequestError(JJMError):
    """The request or configuration is invalid. CLI exit code 2."""


class ExecutionError(JJMError):
    """The configured runtime cannot run, or the engine broke its own contract. CLI exit code 3."""


class AdapterError(JJMError):
    """One criterion's evaluation failed. Becomes an item-level `error` result."""
