import json
import subprocess
import sys

from conftest import route_request
from jjm.contract import schema_errors


def jjm(*args, stdin: bytes = b""):
    proc = subprocess.run([sys.executable, "-m", "jjm", *args], input=stdin, capture_output=True, timeout=60)
    return proc.returncode, proc.stdout.decode("utf-8"), proc.stderr.decode("utf-8")


def test_evaluate_from_stdin_prints_a_valid_envelope():
    code, out, err = jjm("evaluate", stdin=json.dumps(route_request()).encode())
    assert code == 0, err
    response = json.loads(out)
    assert schema_errors(response, "response") == []
    assert "not passed for consumption" in err


def test_shipped_examples_run_on_the_synthetic_profile():
    from conftest import ROOT

    for path in sorted((ROOT / "examples").glob("*.json")):
        if path.name.startswith("jjm-profiles"):
            continue
        code, out, err = jjm("evaluate", "--request", str(path))
        assert code == 0, f"{path.name}: {err}"
        assert json.loads(out)["results"][0]["provenance"]["execution_mode"] == "synthetic"


def test_invalid_json_exits_2_with_an_error_object():
    code, out, err = jjm("evaluate", stdin=b'{"schema_version": "0.1",')
    assert code == 2
    payload = json.loads(out)
    assert payload["error"]["code"] == "INVALID_REQUEST"
    assert "results" not in payload
    assert "INVALID_REQUEST" in err


def test_profile_flag_must_match_request():
    code, out, _ = jjm("evaluate", "--profile", "local-advisory", stdin=json.dumps(route_request()).encode())
    assert code == 2
    assert "differs" in json.loads(out)["error"]["message"]


def test_missing_profiles_file_exits_2():
    code, out, _ = jjm("--profiles", "does-not-exist.json", "capabilities", "--profile", "synthetic")
    assert code == 2
    assert "not found" in json.loads(out)["error"]["message"]


def test_capabilities_and_doctor():
    code, out, _ = jjm("capabilities", "--profile", "synthetic")
    assert code == 0
    assert schema_errors(json.loads(out), "capabilities") == []
    code, out, _ = jjm("doctor")
    report = json.loads(out)
    assert report["packages"]["anyjev"] == "0.2.0"
    assert "synthetic" in report["profiles"]
    assert code == (0 if report["ok"] else 2)


def test_hf_profile_without_torch_or_model_is_an_execution_failure(tmp_path):
    config = tmp_path / "profiles.json"
    config.write_text(json.dumps({"profiles": [{
        "id": "local-advisory", "adapter": "anyjev-hf",
        "model": {"id": "example/not-downloaded", "path": str(tmp_path / "missing-model"), "revision": None},
        "device": "cpu", "dtype": "float32", "max_context_tokens": 4096}]}))
    request = route_request()
    request["profile_id"] = "local-advisory"
    code, out, _ = jjm("--profiles", str(config), "evaluate", stdin=json.dumps(request).encode())
    assert code == 3
    assert json.loads(out)["error"]["code"] == "BACKEND_UNAVAILABLE"
