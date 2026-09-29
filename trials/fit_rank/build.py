"""Export record -> JJM request.

Only whitelisted fields are read. Sonnet's rank and justification never enter the
request, and `assert_no_leak` checks that on every built request.
"""
from __future__ import annotations

from typing import Any, Optional

from trials.fit_rank.criterion import CANDIDATE_EVIDENCE_ID, fit_criterion

LISTING_MAX_CHARS = 120_000  # Exec Foundry LLM_PROMPT_LARGE_TEXT_MAX
CANDIDATE_FIELDS = ("base_resume", "candidate_profile", "supplementary_context", "work_preferences")
LABEL_FIELDS = ("sonnet_rank", "board_rank", "owner_label", "sonnet_rerun_rank")
RECORD_FIELDS = {"job_id", "exported_at", "url", "listing_text", "listing_hints", "location_instructions",
                 *CANDIDATE_FIELDS, *LABEL_FIELDS}
LEAK_MARKERS = ("fitjustification", "fitrank", "fit_justification", "fit_rank", "sonnet", "owner_label",
                "board_rank")


class RecordError(ValueError):
    pass


def _text(record: dict[str, Any], key: str) -> Optional[str]:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RecordError(f"job {record.get('job_id')!r}: {key} must be a string or null")
    value = value.strip()
    return value or None


def _rank(record: dict[str, Any], key: str) -> Optional[int]:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
        raise RecordError(f"job {record.get('job_id')!r}: {key} must be an integer 1-5 or null")
    return value


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    extra = sorted(set(record) - RECORD_FIELDS)
    if extra:
        raise RecordError(f"job {record.get('job_id')!r}: unknown fields {', '.join(extra)}; "
                          "the export must not carry anything else")
    if not isinstance(record.get("job_id"), str) or not record["job_id"]:
        raise RecordError("every record needs a job_id string")
    if not _text(record, "listing_text"):
        raise RecordError(f"job {record['job_id']!r}: listing_text is empty")
    if not isinstance(record.get("exported_at"), str):
        raise RecordError(f"job {record['job_id']!r}: exported_at must be an RFC 3339 time")
    for key in LABEL_FIELDS:
        _rank(record, key)
    return record


def labels(record: dict[str, Any]) -> dict[str, Optional[int]]:
    return {key: _rank(record, key) for key in LABEL_FIELDS}


def _safe_id(job_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_.:/-" else "-" for ch in job_id)
    return ("job-" + cleaned)[:128]


def build_request(record: dict[str, Any], *, profile_id: str, request_suffix: str = "") -> dict[str, Any]:
    validate_record(record)
    hints = record.get("listing_hints") or {}
    if not isinstance(hints, dict):
        raise RecordError(f"job {record['job_id']!r}: listing_hints must be an object")
    listing = {
        "url": _text(record, "url"),
        "scraped_title": hints.get("title"),
        "scraped_company": hints.get("company"),
        "scraped_work_type": hints.get("workType"),
        "text": _text(record, "listing_text")[:LISTING_MAX_CHARS],
    }
    content: dict[str, Any] = {"job_listing": listing}
    instructions = _text(record, "location_instructions")
    if instructions:
        content["location_scoring_instructions"] = instructions
    candidate = {key: _text(record, key) for key in CANDIDATE_FIELDS}
    candidate = {k: v for k, v in candidate.items() if v}
    evidence = []
    if candidate:
        evidence.append({"id": CANDIDATE_EVIDENCE_ID, "kind": "document", "content": candidate,
                         "observed_at": record["exported_at"]})
    job = _safe_id(record["job_id"])
    request = {
        "schema_version": "0.1",
        "request_id": (f"fit-{job}{request_suffix}")[:128],
        "scope_id": "exec-foundry.fit-rank-trial",
        "profile_id": profile_id,
        "state": {"id": job, "revision": "export", "observed_at": record["exported_at"], "content": content,
                  "evidence": evidence},
        "criteria": [fit_criterion()],
        "requirements": {"calibration": "not_required", "allowed_methods": [], "min_top_mass": None,
                         "min_margin": None, "require_verified_model_revision": False,
                         "max_state_age_seconds": None},
        "timeout_ms": 300000,
    }
    assert_no_leak(request["state"])
    return request


def assert_no_leak(value: Any, path: str = "state") -> None:
    """No key anywhere in the state may name a rank, a justification, or a label source."""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(marker in lowered for marker in LEAK_MARKERS):
                raise RecordError(f"label leak: {path}.{key} must not reach the model")
            assert_no_leak(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            assert_no_leak(child, f"{path}[{i}]")
