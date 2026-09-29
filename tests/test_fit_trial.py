import json

import pytest

from conftest import ROOT
from jjm.contract import load_request
from jjm.engine import evaluate
from jjm.profiles import load_profiles
from trials.fit_rank import metrics
from trials.fit_rank.build import RecordError, assert_no_leak, build_request
from trials.fit_rank.run import main, read_export

SAMPLE = ROOT / "tests" / "data" / "fit_export.sample.jsonl"


def records():
    return read_export(SAMPLE)


def test_built_requests_are_valid_contract_requests():
    for record in records():
        request = build_request(record, profile_id="synthetic")
        load_request(json.dumps(request).encode())


def test_labels_never_reach_the_state():
    for record in records():
        text = json.dumps(build_request(record, profile_id="synthetic")["state"])
        assert "sonnet" not in text and "owner_label" not in text and "board_rank" not in text


def test_export_with_extra_fields_is_refused():
    record = dict(records()[0], fitJustification="**Strongest alignment.** ...")
    with pytest.raises(RecordError, match="unknown fields fitJustification"):
        build_request(record, profile_id="synthetic")


def test_leak_guard_catches_nested_keys():
    with pytest.raises(RecordError, match="label leak"):
        assert_no_leak({"content": {"job": {"fit_rank": 2}}})


def test_listing_is_truncated_like_sonnet():
    record = dict(records()[0], listing_text="x" * 130_000)
    request = build_request(record, profile_id="synthetic")
    assert len(request["state"]["content"]["job_listing"]["text"]) == 120_000


def test_no_candidate_material_abstains_instead_of_a_middle_rank():
    record = records()[3]
    response = evaluate(build_request(record, profile_id="synthetic"), load_profiles(None))
    result = response["results"][0]
    assert result["status"] == "abstained"
    assert result["acceptance"]["reason_codes"] == ["MISSING_REQUIRED_EVIDENCE"]


def test_metrics_known_values():
    assert metrics.quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert metrics.exact([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    assert metrics.within_one([1, 2, 5], [1, 3, 3]) == pytest.approx(2 / 3)
    assert metrics.mean_signed_offset([2, 3], [1, 1]) == 1.5
    assert metrics.quadratic_weighted_kappa([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]) == pytest.approx(-1.0)
    assert metrics.quadratic_weighted_kappa([3, 3, 3], [3, 3, 3]) is None
    ci = metrics.bootstrap_ci(metrics.exact, [1, 2, 3, 4] * 5, [1, 2, 3, 5] * 5)
    assert ci[0] <= 0.75 <= ci[1]
    assert metrics.percentile([10, 20, 30, 40], 0.5) == 25


def test_end_to_end_synthetic_run(tmp_path):
    assert main(["--input", str(SAMPLE), "--profile", "synthetic", "--out", str(tmp_path), "--repeat", "2"]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["counts"] == {"jobs": 4, "evaluated": 3, "abstained": 1, "error": 0,
                                 "abstain_reasons": ["MISSING_REQUIRED_EVIDENCE"], "error_codes": []}
    assert summary["determinism"]["same_selection"] == 2
    assert summary["determinism"]["max_abs_weight_diff"] == 0.0
    assert summary["comparisons"]["sonnet_rerun_vs_sonnet"]["n"] == 3
    assert summary["comparisons"]["jjm_vs_owner_overrides"]["n"] == 1
    report = (tmp_path / "summary.md").read_text()
    assert "Synthetic profile" in report and "Criterion wording" not in report
    assert summary["criterion_status"] == "frozen"
    assert len((tmp_path / "envelopes.jsonl").read_text().splitlines()) == 4


def test_real_profile_requires_named_hardware(tmp_path):
    config = tmp_path / "profiles.json"
    config.write_text(json.dumps({"profiles": [{
        "id": "local-advisory", "adapter": "anyjev-hf",
        "model": {"id": "example/not-downloaded", "path": None, "revision": None},
        "max_context_tokens": 4096}]}))
    code = main(["--input", str(SAMPLE), "--profiles", str(config), "--profile", "local-advisory",
                 "--out", str(tmp_path / "out")])
    assert code == 2
