"""jjm.verify must accept and reject exactly what the pack validator does."""
import pytest

from conftest import PACK, fixture_cases, load_json
from jjm.contract import schema_errors
from jjm.verify import validate_pair


@pytest.mark.parametrize("case", fixture_cases(), ids=lambda c: c["id"])
def test_fixture_parity(case):
    request = load_json(PACK / case["request"])
    response = load_json(PACK / case["response"])
    artifact = load_json(PACK / case["artifact"]) if case.get("artifact") else None
    problems = schema_errors(request, "request") + schema_errors(response, "response")
    if artifact is not None:
        problems += schema_errors(artifact, "artifact")
    if not problems:
        try:
            validate_pair(request, response, artifact)
        except Exception as exc:
            problems.append(str(exc))
    if case["expected"] == "valid":
        assert problems == []
    else:
        assert problems, f"{case['id']} should be rejected"
