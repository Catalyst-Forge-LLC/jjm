"""Strict JSON ingress, schema validation, and request-level semantic checks.

No GPU or model imports. Every failure names the field and the reason.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources
from typing import Any

import fastjsonschema
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import best_match
from referencing import Registry, Resource

from jjm import errors

MAX_REQUEST_BYTES = 1024 * 1024
MAX_DEPTH = 32
MAX_SAFE_INTEGER = 2**53 - 1
ASSERTION_OUTCOMES = ("supported", "contradicted", "not_observable")
SCHEMA_NAMES = ("definitions", "request", "response", "criterion", "capabilities", "artifact")


def _reject_constant(value: str) -> Any:
    raise errors.RequestError(errors.INVALID_REQUEST, f"Non-finite JSON literal is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise errors.RequestError(errors.INVALID_REQUEST, f"Duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _check_values(value: Any, depth: int = 1, path: str = "$") -> None:
    if depth > MAX_DEPTH:
        raise errors.RequestError(errors.INVALID_REQUEST, f"JSON nesting deeper than {MAX_DEPTH} at {path}")
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise errors.RequestError(errors.INVALID_REQUEST, f"Integer outside the safe JSON range at {path}")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise errors.RequestError(errors.INVALID_REQUEST, f"Non-finite number at {path}")
        return
    if isinstance(value, list):
        for i, child in enumerate(value):
            _check_values(child, depth + 1, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _check_values(child, depth + 1, f"{path}.{key}")


def parse_json_bytes(data: bytes, *, max_bytes: int = MAX_REQUEST_BYTES) -> Any:
    """Parse untrusted JSON: size-bounded, strict UTF-8, no duplicate keys, no NaN or infinity."""
    if len(data) > max_bytes:
        raise errors.RequestError(errors.INVALID_REQUEST,
                                  f"Request is {len(data)} bytes; the limit is {max_bytes} bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise errors.RequestError(errors.INVALID_REQUEST, f"Request is not valid UTF-8: {exc}") from None
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except errors.RequestError:
        raise
    except json.JSONDecodeError as exc:
        raise errors.RequestError(errors.INVALID_REQUEST,
                                  f"Request is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
                                  ) from None
    _check_values(value)
    return value


@lru_cache(maxsize=1)
def _schemas() -> tuple[dict[str, Any], Registry]:
    schemas: dict[str, Any] = {}
    registry: Registry = Registry()
    root = resources.files("jjm") / "schemas"
    for name in SCHEMA_NAMES:
        schema = json.loads((root / f"{name}.schema.json").read_text(encoding="utf-8"))
        schemas[name] = schema
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return schemas, registry


@lru_cache(maxsize=None)
def schema_validator(name: str) -> Draft202012Validator:
    schemas, registry = _schemas()
    return Draft202012Validator(schemas[name], registry=registry, format_checker=FormatChecker())


def _urn_schema(uri: str) -> dict[str, Any]:
    schemas, _ = _schemas()
    base = uri.split("#", 1)[0]
    found = next((schema for schema in schemas.values() if schema["$id"] == base), None)
    if found is None:
        raise ValueError(f"unknown schema {uri}")
    return found


def _refuse_remote(uri: str) -> dict[str, Any]:
    raise ValueError(f"schema ref is not local: {uri}")


@lru_cache(maxsize=None)
def _compiled(name: str):
    """A generated checker for one schema. It resolves urn: refs from the local files and never fetches."""
    schemas, _ = _schemas()
    return fastjsonschema.compile(schemas[name], handlers={"urn": _urn_schema, "http": _refuse_remote,
                                                           "https": _refuse_remote}, use_default=False)


def _format_path(path: Any) -> str:
    out = "$"
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def schema_errors(value: Any, name: str) -> list[str]:
    """All schema violations, most specific first, as 'path: message' strings."""
    validator = schema_validator(name)
    found = []
    for err in sorted(validator.iter_errors(value), key=lambda e: str(list(e.absolute_path))):
        # oneOf over criterion kinds only says "not valid under any"; best_match descends to the closest branch.
        leaf = best_match([err]) or err
        found.append(f"{_format_path(leaf.absolute_path)}: {leaf.message}")
    return found


def validate_shape(value: Any, name: str, code: str = errors.INVALID_REQUEST) -> None:
    """Reject a document that breaks schema 0.1.

    The success path uses a checker generated from the same schema files. A rejection falls back to
    the interpreter so the message still names the field. The fixture suite checks that the generated
    checker accepts and rejects the same documents the interpreter does.
    """
    try:
        _compiled(name)(value)
        return
    except fastjsonschema.JsonSchemaValueException:
        pass
    problems = schema_errors(value, name)
    if problems:
        shown = "; ".join(problems[:5])
        more = f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""
        raise errors.RequestError(code, f"{name} does not match schema 0.1: {shown}{more}")


def outcome_ids(criterion: dict[str, Any]) -> list[str]:
    if criterion["kind"] == "assertion":
        return list(ASSERTION_OUTCOMES)
    source = criterion["levels"] if criterion["kind"] == "score" else criterion["options"]
    return [entry["id"] for entry in source]


def validate_request_semantics(request: dict[str, Any]) -> None:
    """Cross-field rules the schema cannot express. Violations reject the request before inference."""
    seen: set[str] = set()
    for criterion in request["criteria"]:
        if criterion["id"] in seen:
            raise errors.RequestError(errors.DUPLICATE_CRITERION_ID,
                                      f"Criterion ID {criterion['id']!r} appears more than once")
        seen.add(criterion["id"])

    evidence_ids = [e["id"] for e in request["state"]["evidence"]]
    duplicates = sorted({x for x in evidence_ids if evidence_ids.count(x) > 1})
    if duplicates:
        raise errors.RequestError(errors.INVALID_REQUEST, f"Duplicate evidence IDs: {', '.join(duplicates)}")

    for criterion in request["criteria"]:
        cid = criterion["id"]
        ids = outcome_ids(criterion)
        if len(set(ids)) != len(ids):
            raise errors.RequestError(errors.INVALID_REQUEST, f"Criterion {cid!r} repeats an option ID")
        if criterion["kind"] != "assertion":
            source = criterion["levels"] if criterion["kind"] == "score" else criterion["options"]
            descriptions = [x["description"] for x in source]
            if len(set(descriptions)) != len(descriptions):
                raise errors.RequestError(errors.INVALID_REQUEST,
                                          f"Criterion {cid!r} gives two options the same description")
        unknown = sorted(set(criterion["accept_outcomes"]) - set(ids))
        if unknown:
            raise errors.RequestError(errors.INVALID_REQUEST,
                                      f"Criterion {cid!r} accepts unknown outcomes: {', '.join(unknown)}")
        if criterion["kind"] == "score":
            values = [level["value"] for level in criterion["levels"]]
            if values != list(range(len(values))):
                raise errors.RequestError(errors.INVALID_REQUEST,
                                          f"Criterion {cid!r} score levels must be indexed 0, 1, 2, ... in order")


def load_request(data: bytes) -> dict[str, Any]:
    """Parse, shape-check, and semantically check a request. Raises RequestError."""
    request = parse_json_bytes(data)
    if not isinstance(request, dict):
        raise errors.RequestError(errors.INVALID_REQUEST, "Request must be a JSON object")
    version = request.get("schema_version")
    if version != "0.1":
        raise errors.RequestError(errors.INVALID_REQUEST,
                                  f"Unsupported schema_version {version!r}; this engine accepts '0.1'")
    validate_shape(request, "request")
    validate_request_semantics(request)
    return request
