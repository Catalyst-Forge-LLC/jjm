"""Request -> envelope. Gate precedence follows spec 05.

1. Invalid request: rejected before this module runs (contract.load_request).
2. Detectable preflight failures: the criterion abstains, no model call.
3. Backend failure or malformed output: error, no prediction.
4. Complete result that misses a declared requirement: evaluated, held.
5. Complete result meeting every declared check: evaluated, passed for consumption.

`pass` never authorizes an action. The caller re-checks state before acting.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from jjm import ENGINE_VERSION, errors
from jjm.adapters.base import Adapter, AdapterOutput, check_weights
from jjm.compile import RENDERING_DIGEST, CompiledCriterion, compile_criterion, policy_digest, render_state, state_digest
from jjm.contract import validate_shape
from jjm.digest import digest, text_digest
from jjm.profiles import Profile
from jjm.verify import InvariantError, parse_time, validate_pair

FUTURE_SKEW_SECONDS = 5


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _check(check_id: str, ok: bool, code: str) -> dict[str, Any]:
    return {"id": check_id, "mandatory": True, "status": "pass" if ok else "fail", "code": "OK" if ok else code}


def summarize(ids: tuple[str, ...], weights: list[float], semantics: str) -> dict[str, Any]:
    top = max(range(len(weights)), key=lambda i: (weights[i], -i))
    ranked = sorted(weights, reverse=True)
    entropy = -sum(w * math.log(w) for w in weights if w > 0) / math.log(len(weights))
    return {
        "weights": {i: w for i, w in zip(ids, weights)},
        "semantics": semantics,
        "selected": ids[top],
        "value": None,
        "top_mass": ranked[0],
        "margin": ranked[0] - ranked[1],
        "normalized_entropy": min(1.0, max(0.0, entropy)),
    }


def effective_config(adapter: Adapter, profile: Profile) -> dict[str, Any]:
    settings = None
    if adapter.adapter_id == "anyjev":
        settings = adapter.settings_identity()
    return {
        "adapter": {"id": adapter.adapter_id, "version": adapter.adapter_version},
        "profile": {"id": profile.id, "generation": profile.generation},
        "model": adapter.model_identity(),
        "runtime": {"id": adapter.runtime_id, "version": adapter.runtime_version, "dtype": adapter.dtype},
        "method": adapter.method,
        "method_settings": settings,
        "rendering_digest": RENDERING_DIGEST,
        "artifact": None,
    }


class Evaluation:
    """One request against one profile."""

    def __init__(self, request: dict[str, Any], profile: Profile, *, clock: Callable[[], datetime] = utc_now):
        self.request = request
        self.profile = profile
        self.clock = clock
        self.adapter = profile.adapter()
        self.state = request["state"]
        self.state_text = render_state(self.state)
        self.state_digest = state_digest(self.state)
        self.evidence_ids = [e["id"] for e in self.state["evidence"]]
        self.config_digest = digest(effective_config(self.adapter, profile))
        self.started = time.monotonic()
        self.deadline = self.started + request["timeout_ms"] / 1000.0

    # ---- provenance ---------------------------------------------------
    def _provenance(self, compiled: CompiledCriterion, *, ran: bool, output: Optional[AdapterOutput],
                    elapsed_ms: Optional[float], context_tokens: Optional[int]) -> dict[str, Any]:
        a = self.adapter
        diagnostics: dict[str, Any] = {}
        if a.execution_mode == "synthetic":
            diagnostics["synthetic_notice"] = "Synthetic contract demonstration, not a model judgment."
        if output is not None:
            diagnostics.update(output.diagnostics)
        if context_tokens is not None:
            diagnostics["context_tokens"] = context_tokens
        diagnostics["rendered_state_digest"] = text_digest(self.state_text)
        return {
            "execution_mode": a.execution_mode,
            "engine_version": ENGINE_VERSION,
            "upstream_commit": None,
            "adapter_id": a.adapter_id,
            "adapter_version": a.adapter_version,
            "profile_id": self.profile.id,
            "profile_generation": self.profile.generation,
            "state_id": self.state["id"],
            "state_revision": self.state["revision"],
            "state_digest": self.state_digest,
            "criterion_digest": compiled.criterion_digest,
            "rendering_digest": RENDERING_DIGEST,
            "effective_config_digest": self.config_digest,
            "policy_digest": policy_digest(self.request, compiled.criterion),
            "model": a.model_identity(),
            "method": a.method if output is not None else None,
            "upstream_level": a.upstream_level if output is not None else None,
            "calibration": {"status": "not_fitted", "artifact_id": None, "validation_report_id": None,
                            "scope_match": False},
            "runtime": {
                "id": a.runtime_id,
                "version": a.runtime_version,
                "dtype": a.dtype,
                "numeric_path": a.numeric_path if output is not None else "none",
                "prompt_evaluations": output.prompt_evaluations if output is not None else (None if ran else 0),
                "adapter_calls": output.adapter_calls if output is not None else (None if ran else 0),
                "prefix_reuse": output.prefix_reuse if output is not None else ("unknown" if ran else "not_used"),
                "early_exit": output.early_exit if output is not None else ("unknown" if ran else "not_used"),
                "elapsed_ms": elapsed_ms,
            },
            "diagnostics": diagnostics,
        }

    def _evidence(self, compiled: CompiledCriterion, exposed: bool) -> dict[str, Any]:
        required = compiled.criterion["required_evidence_ids"]
        return {
            "considered_ids": list(self.evidence_ids) if exposed else [],
            "selected_ids": [],
            "missing_required_ids": [x for x in required if x not in self.evidence_ids],
            "selection_method": "none",
            "verification": "not_performed",
        }

    # ---- gates --------------------------------------------------------
    def _state_age(self, at: datetime) -> float:
        return (at - parse_time(self.state["observed_at"])).total_seconds()

    def _preflight(self, compiled: CompiledCriterion) -> list[dict[str, Any]]:
        req = self.request["requirements"]
        a = self.adapter
        checks = []
        if compiled.kind not in a.supported_kinds:
            checks.append(_check("kind-supported", False, errors.UNSUPPORTED_KIND))
        elif compiled.k > a.max_choice_options:
            checks.append(_check("option-count", False, errors.TOO_MANY_OPTIONS))
        missing = [x for x in compiled.criterion["required_evidence_ids"] if x not in self.evidence_ids]
        if missing:
            checks.append(_check("required-evidence", False, errors.MISSING_REQUIRED_EVIDENCE))
        if req["calibration"] == "validated_in_scope":
            checks.append(_check("calibration", False, errors.CALIBRATION_NOT_VALIDATED))
        if req["allowed_methods"] and a.method not in req["allowed_methods"]:
            checks.append(_check("method-allowed", False, errors.METHOD_NOT_ALLOWED))
        if req["require_verified_model_revision"] and a.model_identity()["revision_status"] != "verified":
            checks.append(_check("model-revision", False, errors.MODEL_IDENTITY_UNKNOWN))
        max_age = req["max_state_age_seconds"]
        if max_age is not None:
            age = self._state_age(self.clock())
            if age > max_age or age < -FUTURE_SKEW_SECONDS:
                checks.append(_check("state-age", False, errors.STATE_TOO_OLD))
        return checks

    def _postchecks(self, compiled: CompiledCriterion, prediction: dict[str, Any], evaluated_at: datetime
                    ) -> list[dict[str, Any]]:
        req = self.request["requirements"]
        checks = [_check("prediction-complete", True, "OK")]
        if req["allowed_methods"]:
            checks.append(_check("method-allowed", True, "OK"))
        if req["require_verified_model_revision"]:
            checks.append(_check("model-revision", True, "OK"))
        if req["min_top_mass"] is not None:
            ok = prediction["top_mass"] + 1e-9 >= req["min_top_mass"]
            checks.append(_check("min-top-mass", ok, errors.TOP_MASS_BELOW_MINIMUM))
        if req["min_margin"] is not None:
            ok = prediction["margin"] + 1e-9 >= req["min_margin"]
            checks.append(_check("min-margin", ok, errors.MARGIN_BELOW_MINIMUM))
        accepted = compiled.criterion["accept_outcomes"]
        if accepted:
            checks.append(_check("accept-outcomes", prediction["selected"] in accepted, errors.OUTCOME_NOT_ACCEPTED))
        if req["max_state_age_seconds"] is not None:
            age = self._state_age(evaluated_at)
            ok = -FUTURE_SKEW_SECONDS <= age <= req["max_state_age_seconds"]
            checks.append(_check("state-age", ok, errors.STATE_TOO_OLD))
        return checks

    # ---- per criterion -----------------------------------------------
    def _result(self, compiled: CompiledCriterion, status: str, prediction: Optional[dict[str, Any]],
                checks: list[dict[str, Any]], provenance: dict[str, Any], exposed: bool,
                error: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        reasons = []
        for ch in checks:
            if ch["status"] != "pass" and ch["code"] not in reasons:
                reasons.append(ch["code"])
        if status == "error":
            gate = "reject"
        elif status == "abstained" or reasons:
            gate = "hold"
        else:
            gate = "pass"
        return {
            "criterion_id": compiled.id,
            "kind": compiled.kind,
            "status": status,
            "prediction": prediction,
            "acceptance": {"status": gate, "reason_codes": reasons, "checks": checks},
            "evidence": self._evidence(compiled, exposed),
            "provenance": provenance,
            "error": error,
        }

    def _error_result(self, compiled: CompiledCriterion, exc: errors.JJMError, ran: bool,
                      elapsed_ms: Optional[float], context_tokens: Optional[int]) -> dict[str, Any]:
        prov = self._provenance(compiled, ran=ran, output=None, elapsed_ms=elapsed_ms, context_tokens=context_tokens)
        return self._result(compiled, "error", None, [_check("evaluation", False, exc.code)], prov, ran,
                            exc.to_dict())

    def _criterion(self, criterion: dict[str, Any]) -> tuple[CompiledCriterion, dict[str, Any], Optional[AdapterOutput]]:
        compiled = compile_criterion(criterion)
        if time.monotonic() > self.deadline:
            exc = errors.AdapterError(errors.BACKEND_TIMEOUT, "The request timed out before this criterion ran.", True)
            return compiled, self._error_result(compiled, exc, False, 0.0, None), None
        checks = self._preflight(compiled)
        context_tokens = None
        if not checks and self.adapter.max_context_tokens is not None:
            try:
                context_tokens = self.adapter.context_tokens(self.state_text, compiled)
            except errors.AdapterError as exc:
                return compiled, self._error_result(compiled, exc, False, 0.0, None), None
            if context_tokens is not None and context_tokens > self.adapter.max_context_tokens:
                checks.append(_check("context-limit", False, errors.CONTEXT_LIMIT_EXCEEDED))
        if checks:
            prov = self._provenance(compiled, ran=False, output=None, elapsed_ms=0.0, context_tokens=context_tokens)
            return compiled, self._result(compiled, "abstained", None, checks, prov, False), None

        t0 = time.monotonic()
        try:
            output = self.adapter.evaluate(self.state_text, compiled)
            weights = check_weights(output.weights, compiled.k)
        except errors.AdapterError as exc:
            elapsed = round((time.monotonic() - t0) * 1000.0, 3)
            return compiled, self._error_result(compiled, exc, True, elapsed, context_tokens), None
        elapsed = round((time.monotonic() - t0) * 1000.0, 3)
        if time.monotonic() > self.deadline:
            exc = errors.AdapterError(errors.BACKEND_TIMEOUT,
                                      f"Evaluation finished after the {self.request['timeout_ms']} ms timeout. "
                                      "The late result was discarded.", True)
            return compiled, self._error_result(compiled, exc, True, elapsed, context_tokens), None
        output.weights = weights
        prov = self._provenance(compiled, ran=True, output=output, elapsed_ms=elapsed, context_tokens=context_tokens)
        prediction = summarize(compiled.outcome_ids, weights, self.adapter.semantics)
        # Acceptance for evaluated results is finalized once evaluated_at is known.
        return compiled, self._result(compiled, "evaluated", prediction, [], prov, True), output

    def run(self) -> dict[str, Any]:
        pending = [self._criterion(c) for c in self.request["criteria"]]
        evaluated_at = self.clock()
        results = []
        for compiled, result, output in pending:
            if result["status"] == "evaluated":
                checks = self._postchecks(compiled, result["prediction"], evaluated_at)
                result = self._result(compiled, "evaluated", result["prediction"], checks, result["provenance"], True)
            results.append(result)
        response = {
            "schema_version": "0.1",
            "request_id": self.request["request_id"],
            "evaluated_at": format_time(evaluated_at),
            "elapsed_ms": round((time.monotonic() - self.started) * 1000.0, 3),
            "results": results,
        }
        self_check(self.request, response)
        return response


def self_check(request: dict[str, Any], response: dict[str, Any]) -> None:
    """The engine never returns an envelope that breaks its own contract."""
    try:
        validate_shape(response, "response", code=errors.INTERNAL_ERROR)
        validate_pair(request, response)
    except (errors.RequestError, InvariantError) as exc:
        detail = exc.message if isinstance(exc, errors.JJMError) else str(exc)
        raise errors.ExecutionError(errors.INTERNAL_ERROR, f"Engine produced an invalid envelope: {detail}") from None


def evaluate(request: dict[str, Any], profiles: dict[str, Profile], *,
             clock: Callable[[], datetime] = utc_now) -> dict[str, Any]:
    profile = profiles.get(request["profile_id"])
    if profile is None:
        known = ", ".join(sorted(profiles))
        raise errors.RequestError(errors.INVALID_REQUEST,
                                  f"profile_id {request['profile_id']!r} is not configured. Configured: {known}")
    return Evaluation(request, profile, clock=clock).run()
