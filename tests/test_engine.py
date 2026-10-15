import copy
import time

import pytest

from conftest import fixed_clock, fixture_request, profile_with, route_request
from jjm import errors
from jjm.adapters.base import AdapterOutput
from jjm.adapters.synthetic import SyntheticAdapter
from jjm.engine import evaluate


class SpyAdapter(SyntheticAdapter):
    """Synthetic adapter that records calls and can return scripted weights."""

    def __init__(self, weights=None, delay=0.0):
        self.calls = 0
        self.weights = weights
        self.delay = delay

    def evaluate(self, state_text, compiled):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.weights is None:
            return super().evaluate(state_text, compiled)
        return AdapterOutput(weights=list(self.weights), prompt_evaluations=1, adapter_calls=1)


def run(request, adapter=None):
    if adapter is None:
        from jjm.profiles import load_profiles

        return evaluate(request, load_profiles(None), clock=fixed_clock)
    request = copy.deepcopy(request)
    request["profile_id"] = "test-profile"
    return evaluate(request, profile_with(adapter), clock=fixed_clock)


@pytest.mark.parametrize("case_id,status,reason", [
    ("01-routing", "evaluated", None),
    ("02-supported", "evaluated", None),
    ("03-missing-evidence", "abstained", "MISSING_REQUIRED_EVIDENCE"),
    ("04-not-observable", "evaluated", None),
    ("05-calibration-required", "abstained", "VALIDATION_NOT_ACCEPTED"),
    ("06-low-margin", "evaluated", None),
    ("08-score", "abstained", "UNSUPPORTED_KIND"),
    ("09-calibrated-shape", "abstained", "VALIDATION_NOT_ACCEPTED"),
    ("11-probability-calibration-required", "abstained", "PROBABILITY_CALIBRATION_NOT_ASSESSED"),
    ("10-unknown-model", "abstained", "MODEL_IDENTITY_UNKNOWN"),
])
def test_positive_fixture_requests_run_on_the_synthetic_profile(case_id, status, reason):
    response = run(fixture_request(case_id))
    result = response["results"][0]
    assert result["status"] == status
    assert result["provenance"]["execution_mode"] == "synthetic"
    if reason:
        assert reason in result["acceptance"]["reason_codes"]
        assert result["acceptance"]["status"] == "hold"
        assert result["prediction"] is None


def test_t02_missing_required_evidence_never_calls_the_adapter():
    spy = SpyAdapter()
    result = run(fixture_request("03-missing-evidence"), spy)["results"][0]
    assert spy.calls == 0
    assert result["status"] == "abstained"
    assert result["evidence"]["missing_required_ids"] == ["export-receipt"]
    assert result["provenance"]["runtime"]["adapter_calls"] == 0
    assert result["provenance"]["method"] is None


def test_t03_not_observable_is_an_evaluated_outcome():
    spy = SpyAdapter(weights=[0.1, 0.1, 0.8])
    request = fixture_request("04-not-observable")
    result = run(request, spy)["results"][0]
    assert result["status"] == "evaluated"
    assert result["prediction"]["selected"] == "not_observable"
    assert result["acceptance"]["reason_codes"] == ["OUTCOME_NOT_ACCEPTED"]


@pytest.mark.parametrize("weights,code", [
    ([0.5, 0.5], errors.INCOMPLETE_LABEL_DISTRIBUTION),
    ([0.5, 0.3, float("nan")], errors.NONFINITE_RESULT),
    ([0.5, 0.3, float("inf")], errors.NONFINITE_RESULT),
    ([0.5, 0.3, 0.1], errors.INCOMPLETE_LABEL_DISTRIBUTION),
    ([1.2, -0.1, -0.1], errors.NONFINITE_RESULT),
])
def test_t06_bad_backend_output_becomes_an_error_not_a_repair(weights, code):
    result = run(route_request(), SpyAdapter(weights=weights))["results"][0]
    assert result["status"] == "error"
    assert result["prediction"] is None
    assert result["error"]["code"] == code
    assert result["acceptance"]["status"] == "reject"


def test_t11_same_request_same_answer():
    first = run(route_request())["results"][0]["prediction"]
    run(fixture_request("02-supported"))
    second = run(route_request())["results"][0]["prediction"]
    assert first == second


def test_t14_late_result_is_discarded():
    request = route_request(timeout_ms=20)
    result = run(request, SpyAdapter(weights=[0.9, 0.05, 0.05], delay=0.05))["results"][0]
    assert result["status"] == "error"
    assert result["error"]["code"] == errors.BACKEND_TIMEOUT
    assert result["error"]["retryable"] is True
    assert result["prediction"] is None


def test_t16_pass_requires_complete_evaluated_prediction_and_respects_filters():
    request = route_request()
    result = run(request, SpyAdapter(weights=[0.8, 0.15, 0.05]))["results"][0]
    assert result["acceptance"]["status"] == "pass"
    request["criteria"][0]["accept_outcomes"] = ["technical"]
    result = run(request, SpyAdapter(weights=[0.8, 0.15, 0.05]))["results"][0]
    assert result["acceptance"]["status"] == "hold"
    assert result["acceptance"]["reason_codes"] == ["OUTCOME_NOT_ACCEPTED"]


def test_thresholds_hold_low_margin_and_low_top_mass():
    request = route_request()
    request["requirements"]["min_margin"] = 0.2
    request["requirements"]["min_top_mass"] = 0.6
    result = run(request, SpyAdapter(weights=[0.45, 0.40, 0.15]))["results"][0]
    assert result["status"] == "evaluated"
    assert result["acceptance"]["status"] == "hold"
    assert set(result["acceptance"]["reason_codes"]) == {"MARGIN_BELOW_MINIMUM", "TOP_MASS_BELOW_MINIMUM"}


def test_t17_no_selected_evidence_without_selection():
    result = run(fixture_request("02-supported"))["results"][0]
    assert result["evidence"]["considered_ids"] == ["export-receipt"]
    assert result["evidence"]["selected_ids"] == []
    assert result["evidence"]["selection_method"] == "none"


def test_method_not_allowed_abstains():
    request = route_request()
    request["requirements"]["allowed_methods"] = ["anyjev:L0-permutation"]
    result = run(request)["results"][0]
    assert result["status"] == "abstained"
    assert result["acceptance"]["reason_codes"] == ["METHOD_NOT_ALLOWED"]


def test_stale_state_abstains():
    request = route_request()
    request["state"]["observed_at"] = "2026-09-25T17:00:00Z"
    request["requirements"]["max_state_age_seconds"] = 60
    result = run(request)["results"][0]
    assert result["status"] == "abstained"
    assert result["acceptance"]["reason_codes"] == ["STATE_TOO_OLD"]


def test_fresh_state_passes_age_check():
    request = route_request()
    request["requirements"]["max_state_age_seconds"] = 60
    result = run(request, SpyAdapter(weights=[0.8, 0.15, 0.05]))["results"][0]
    assert result["acceptance"]["status"] == "pass"
    assert any(c["id"] == "state-age" and c["status"] == "pass" for c in result["acceptance"]["checks"])


def test_results_follow_request_order_and_one_failure_does_not_sink_the_rest():
    request = fixture_request("02-supported")
    extra = copy.deepcopy(route_request()["criteria"][0])
    extra["id"] = "route-ticket"
    missing = copy.deepcopy(request["criteria"][0])
    missing["id"] = "needs-other-receipt"
    missing["required_evidence_ids"] = ["other-receipt"]
    request["criteria"] = [extra, missing, request["criteria"][0]]
    response = run(request)
    assert [r["criterion_id"] for r in response["results"]] == ["route-ticket", "needs-other-receipt",
                                                                 "export-completed"]
    assert [r["status"] for r in response["results"]] == ["evaluated", "abstained", "evaluated"]


def test_unknown_profile_is_a_request_error():
    request = route_request()
    request["profile_id"] = "no-such-profile"
    from jjm.profiles import load_profiles

    with pytest.raises(errors.RequestError, match="no-such-profile"):
        evaluate(request, load_profiles(None))


def test_criterion_digest_ignores_filters_but_tracks_definitions():
    base = run(route_request())["results"][0]["provenance"]
    request = route_request()
    request["criteria"][0]["accept_outcomes"] = ["billing"]
    request["criteria"][0]["id"] = "renamed"
    filtered = run(request)["results"][0]["provenance"]
    assert filtered["criterion_digest"] == base["criterion_digest"]
    assert filtered["policy_digest"] != base["policy_digest"]
    request = route_request()
    request["criteria"][0]["options"][0]["description"] = "Billing questions"
    changed = run(request)["results"][0]["provenance"]
    assert changed["criterion_digest"] != base["criterion_digest"]


def test_empty_evidence_renders_no_evidence_section():
    from jjm.compile import EVIDENCE_HEADER, render_state

    state = route_request()["state"]
    assert state["evidence"] == []
    assert EVIDENCE_HEADER not in render_state(state)
    state["evidence"] = [{"id": "e1", "kind": "note", "content": "x", "observed_at": None}]
    assert EVIDENCE_HEADER in render_state(state)
