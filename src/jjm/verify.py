"""Cross-field response invariants.

This is the rule set of `specs/jjm-spec-pack/scripts/validate_pack.py::validate_pair`.
The engine runs it on every envelope before returning it, and the test suite runs
it on the pack's positive and negative fixtures to keep the two in step.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Optional

from jjm.contract import outcome_ids

EPS = 1e-6


class InvariantError(ValueError):
    pass


def fail(message: str) -> None:
    raise InvariantError(message)


def _unique(values: list[str], description: str) -> None:
    if len(values) != len(set(values)):
        fail(f"Duplicate {description}")


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=EPS):
        fail(f"{label} does not match its defined arithmetic")


_RFC3339 = re.compile(r"^(\d{4}-\d{2}-\d{2})[Tt ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?([Zz]|[+-]\d{2}:\d{2})$")


def parse_time(value: str) -> datetime:
    """RFC 3339 date-time to an aware datetime. Fractions beyond microseconds are truncated."""
    m = _RFC3339.match(value.strip())
    if not m:
        raise InvariantError(f"Not an RFC 3339 date-time: {value!r}")
    date, clock, fraction, offset = m.groups()
    offset = "+00:00" if offset in ("Z", "z") else offset
    fraction = f".{fraction[:6].ljust(6, '0')}" if fraction else ""
    return datetime.fromisoformat(f"{date}T{clock}{fraction}{offset}")


def validate_pair(request: dict[str, Any], response: dict[str, Any], artifact: Optional[dict[str, Any]] = None) -> None:
    criteria = request["criteria"]
    _unique([c["id"] for c in criteria], "criterion ID")
    input_evidence = {e["id"] for e in request["state"]["evidence"]}
    _unique([e["id"] for e in request["state"]["evidence"]], "evidence ID")
    if response["request_id"] != request["request_id"]:
        fail("Request/response IDs differ")
    if [r["criterion_id"] for r in response["results"]] != [c["id"] for c in criteria]:
        fail("Result IDs or ordering differ from request")

    for criterion, result in zip(criteria, response["results"]):
        ids = outcome_ids(criterion)
        _unique(ids, "option ID")
        if criterion["kind"] != "assertion":
            source = criterion["levels"] if criterion["kind"] == "score" else criterion["options"]
            _unique([x["description"] for x in source], "option description")
        if not set(criterion["accept_outcomes"]) <= set(ids):
            fail("Unknown accepted outcome")
        if criterion["kind"] == "score":
            if [level["value"] for level in criterion["levels"]] != list(range(len(ids))):
                fail("Score indices must be consecutive from zero")
        if result["kind"] != criterion["kind"]:
            fail("Result kind differs from criterion")
        p = result["prediction"]
        prov = result["provenance"]
        gate = result["acceptance"]
        evidence = result["evidence"]
        if prov["state_id"] != request["state"]["id"] or prov["state_revision"] != request["state"]["revision"]:
            fail("Response state identity differs from request")
        if prov["profile_id"] != request["profile_id"]:
            fail("Response profile differs from requested configured profile")
        if not set(evidence["considered_ids"]) <= input_evidence:
            fail("Considered evidence was not supplied")
        if not set(evidence["selected_ids"]) <= set(evidence["considered_ids"]):
            fail("Selected evidence was not considered")
        if evidence["selection_method"] == "none" and evidence["selected_ids"]:
            fail("Evidence selections reported without a selection operation")
        missing = set(criterion["required_evidence_ids"]) - input_evidence
        if set(evidence["missing_required_ids"]) != missing:
            fail("Missing required evidence metadata is inaccurate")
        if missing and result["status"] != "abstained":
            fail("Missing required evidence must cause preflight abstention")
        if result["status"] != "evaluated":
            if p is not None or gate["status"] == "pass":
                fail("A non-evaluated result cannot have a prediction or pass")
        else:
            if p is None or prov["method"] is None or prov["runtime"]["numeric_path"] == "none":
                fail("An evaluated result needs prediction and actual method metadata")
        if result["status"] == "error" and result["error"] is None:
            fail("Error status requires an error object")
        if result["status"] != "error" and result["error"] is not None:
            fail("Only error status may carry an error object")
        if gate["status"] != "pass" and not gate["reason_codes"]:
            fail("Held or rejected results require reason codes")
        if gate["status"] == "pass":
            if gate["reason_codes"]:
                fail("Passed result cannot carry rejection reasons")
            if any(ch["mandatory"] and ch["status"] != "pass" for ch in gate["checks"]):
                fail("An unknown or failed mandatory check cannot pass")
        if prov["model"]["revision_status"] == "verified" and not prov["model"]["revision"]:
            fail("A verified model revision must be identified")
        cal = prov["calibration"]
        if cal["status"] == "validated_in_scope":
            if not cal["artifact_id"] or not cal["validation_report_id"] or not cal["scope_match"]:
                fail("In-scope calibration needs a named artifact, report, and scope match")
        if p is None:
            continue
        weights = p["weights"]
        if set(weights) != set(ids):
            fail("Distribution must contain exactly the expected option IDs")
        _close(sum(weights.values()), 1.0, "Weight sum")
        if p["selected"] != max(ids, key=lambda key: weights[key]):
            fail("Selected outcome is not the deterministic argmax")
        ranked = sorted(weights.values(), reverse=True)
        _close(p["top_mass"], ranked[0], "Top mass")
        _close(p["margin"], ranked[0] - ranked[1], "Margin")
        entropy = -sum(v * math.log(v) for v in weights.values() if v) / math.log(len(weights))
        _close(p["normalized_entropy"], entropy, "Normalized entropy")
        if criterion["kind"] == "score":
            if p["value"] is None:
                fail("Score needs an expected-index value")
            _close(p["value"], sum(weights[level["id"]] * level["value"] for level in criterion["levels"]),
                   "Score value")
        elif p["value"] is not None:
            fail("Non-score predictions cannot contain score values")
        if p["semantics"] == "calibrated_estimate":
            if cal["status"] != "validated_in_scope":
                fail("Calibrated semantics require validated in-scope metadata")
            if artifact is None:
                fail("A calibrated result needs its artifact manifest")
            if (artifact["artifact_id"] != cal["artifact_id"]
                    or artifact["validation_report_id"] != cal["validation_report_id"]):
                fail("Artifact or report identity mismatch")
            for key in ["criterion_digest", "effective_config_digest", "rendering_digest", "model", "method"]:
                if artifact[key] != prov[key]:
                    fail(f"Artifact binding mismatch: {key}")
            if artifact["scope_id"] != request["scope_id"]:
                fail("Artifact scope differs from request")
            if not artifact["promoted"] or artifact["validation_status"] != "validated" or artifact["heldout_count"] <= 0:
                fail("Calibrated result requires approved held-out validation metadata")
        if gate["status"] == "pass":
            requirements = request["requirements"]
            allowed = requirements["allowed_methods"]
            if allowed and prov["method"] not in allowed:
                fail("Actual method is not allowed")
            if requirements["calibration"] == "validated_in_scope":
                if cal["status"] != "validated_in_scope" or p["semantics"] != "calibrated_estimate":
                    fail("Required calibration is unavailable")
            if requirements["min_top_mass"] is not None and p["top_mass"] + EPS < requirements["min_top_mass"]:
                fail("Top mass below requirement")
            if requirements["min_margin"] is not None and p["margin"] + EPS < requirements["min_margin"]:
                fail("Margin below requirement")
            if requirements["require_verified_model_revision"] and prov["model"]["revision_status"] != "verified":
                fail("Verified model revision is required")
            accepted = criterion["accept_outcomes"]
            if accepted and p["selected"] not in accepted:
                fail("Selected outcome fails consumption filter")
            max_age = requirements["max_state_age_seconds"]
            if max_age is not None:
                age = (parse_time(response["evaluated_at"]) - parse_time(request["state"]["observed_at"])).total_seconds()
                if age > max_age or age < -5:
                    fail("State freshness requirement not satisfied")
