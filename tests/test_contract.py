import json

import pytest

from conftest import PACK, fixture_cases, load_json, route_request
from jjm import errors
from jjm.contract import MAX_DEPTH, MAX_REQUEST_BYTES, load_request, parse_json_bytes, schema_errors, validate_shape
from jjm.profiles import capabilities, load_profiles


def body(value) -> bytes:
    return json.dumps(value).encode("utf-8")


def test_packaged_schemas_are_the_pack_schemas_byte_for_byte():
    from importlib import resources

    packaged = resources.files("jjm") / "schemas"
    for path in sorted((PACK / "schemas").glob("*.schema.json")):
        assert (packaged / path.name).read_bytes() == path.read_bytes(), path.name


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'])
def test_strict_parser_rejects_duplicate_keys_and_non_finite(payload):
    with pytest.raises(errors.RequestError) as exc:
        parse_json_bytes(payload.encode())
    assert exc.value.code == errors.INVALID_REQUEST


def test_parser_rejects_invalid_utf8_oversize_and_deep_nesting():
    with pytest.raises(errors.RequestError, match="UTF-8"):
        parse_json_bytes(b'{"x":"\xff"}')
    with pytest.raises(errors.RequestError, match="limit"):
        parse_json_bytes(b" " * (MAX_REQUEST_BYTES + 1))
    deep = "[" * (MAX_DEPTH + 1) + "]" * (MAX_DEPTH + 1)
    with pytest.raises(errors.RequestError, match="nesting"):
        parse_json_bytes(deep.encode())


def test_every_positive_fixture_request_loads():
    for case in fixture_cases():
        if case["expected"] == "valid":
            load_request((PACK / case["request"]).read_bytes())


def test_t01_duplicate_criterion_rejected_before_inference():
    request = route_request()
    request["criteria"].append(dict(request["criteria"][0]))
    with pytest.raises(errors.RequestError) as exc:
        load_request(body(request))
    assert exc.value.code == errors.DUPLICATE_CRITERION_ID


def test_t01_duplicate_evidence_rejected():
    request = route_request()
    item = {"id": "e1", "kind": "observation", "content": "x", "observed_at": None}
    request["state"]["evidence"] = [item, dict(item)]
    with pytest.raises(errors.RequestError, match="Duplicate evidence IDs: e1"):
        load_request(body(request))


def test_t15_unknown_fields_and_versions_rejected_with_the_field_named():
    request = route_request()
    request["surprise"] = True
    with pytest.raises(errors.RequestError, match="surprise"):
        load_request(body(request))
    request = route_request()
    request["schema_version"] = "0.2"
    with pytest.raises(errors.RequestError, match="schema_version"):
        load_request(body(request))


def test_schema_error_names_the_nested_field():
    request = route_request()
    del request["criteria"][0]["options"][1]["description"]
    with pytest.raises(errors.RequestError) as exc:
        load_request(body(request))
    assert "description" in exc.value.message


def test_duplicate_option_descriptions_and_unknown_accept_outcomes_rejected():
    request = route_request()
    request["criteria"][0]["options"][1]["description"] = request["criteria"][0]["options"][0]["description"]
    with pytest.raises(errors.RequestError, match="same description"):
        load_request(body(request))
    request = route_request()
    request["criteria"][0]["accept_outcomes"] = ["refunds"]
    with pytest.raises(errors.RequestError, match="refunds"):
        load_request(body(request))


def test_bad_timestamp_is_rejected():
    request = route_request()
    request["state"]["observed_at"] = "yesterday"
    with pytest.raises(errors.RequestError, match="observed_at"):
        load_request(body(request))


@pytest.mark.parametrize("text,iso", [
    ("2026-09-25T18:00:01Z", "2026-09-25T18:00:01+00:00"),
    ("2026-09-25T18:00:01.5Z", "2026-09-25T18:00:01.500000+00:00"),
    ("2026-09-25t18:00:01.123456789z", "2026-09-25T18:00:01.123456+00:00"),
    ("2026-09-25T14:00:01-04:00", "2026-09-25T14:00:01-04:00"),
])
def test_parse_time_keeps_offsets(text, iso):
    from jjm.verify import parse_time

    parsed = parse_time(text)
    assert parsed.utcoffset() is not None
    assert parsed.isoformat() == iso


def test_capabilities_match_schema():
    record = capabilities(load_profiles(None)["synthetic"])
    assert schema_errors(record, "capabilities") == []
    assert record["execution_mode"] == "synthetic"
    assert record["number_semantics"] == ["normalized_readout"]


def test_pack_capability_fixture_still_valid():
    assert schema_errors(load_json(PACK / "examples" / "capabilities.synthetic.json"), "capabilities") == []


def _agrees(document: dict, name: str) -> None:
    from fastjsonschema import JsonSchemaValueException

    from jjm.contract import _compiled

    problems = schema_errors(document, name)
    try:
        _compiled(name)(document)
        generated_ok = True
    except JsonSchemaValueException:
        generated_ok = False
    assert generated_ok == (problems == []), (name, generated_ok, problems[:3])
    if problems:
        with pytest.raises(errors.RequestError, match="schema 0.1"):
            validate_shape(document, name)
    else:
        validate_shape(document, name)


def test_generated_checker_matches_the_interpreter_on_the_pack():
    for case in fixture_cases():
        request = load_json(PACK / case["request"])
        response = load_json(PACK / case["response"])
        _agrees(request, "request")
        _agrees(response, "response")
        if case.get("artifact"):
            _agrees(load_json(PACK / case["artifact"]), "artifact")
    capabilities_doc = load_json(PACK / "examples" / "capabilities.synthetic.json")
    _agrees(capabilities_doc, "capabilities")
    broken = dict(response)
    del broken["results"]
    _agrees(broken, "response")
    broken = dict(response)
    broken["evaluated_at"] = "yesterday"
    _agrees(broken, "response")
