"""Tests for consequential comparison and adapter failure modes."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from laya_adapter import readout
from bench import threshold_from_validation, selection, request, profile, latency_sample
from jjm.engine import evaluate
from laya_adapter import LayaAdapter
import task_data as data

def raw():
    ids = data.TEAM_IDS
    values = [0.3333, 0.3333, 0.3333, 0, 0, 0, 0, 0]
    return {"answers":{"route":{"probabilities":dict(zip(ids,values)), "choice":ids[0], "confidence":0.2}},
            "usage":{"truncated":False,"input_tokens":30}}

def test_rounding_is_explicit_and_raw_values_survive():
    weights, audit = readout(raw(),data.TEAM_IDS)
    assert sum(weights) == pytest.approx(1)
    assert audit["raw_sum"] == pytest.approx(.9999)
    assert audit["raw_probabilities"][data.TEAM_IDS[0]] == .3333

@pytest.mark.parametrize("mutation",["missing","nonfinite","bad_sum","truncation","collapsed","wrong_choice"])
def test_refuses_invalid_or_lossy_outputs(mutation):
    result = raw()
    route = result["answers"]["route"]
    if mutation == "missing": route["probabilities"].pop(data.TEAM_IDS[0])
    elif mutation == "nonfinite": route["probabilities"][data.TEAM_IDS[0]] = float("nan")
    elif mutation == "bad_sum": route["probabilities"][data.TEAM_IDS[0]] = .1
    elif mutation == "truncation": result["usage"]["truncated"] = True
    elif mutation == "collapsed": result["usage"]["options"] = {"route":{"distinct":2}}
    else: route["choice"] = data.TEAM_IDS[-1]
    with pytest.raises(ValueError): readout(result,data.TEAM_IDS)

def test_no_qualifying_validation_policy_is_explicit():
    weights = np.array([[.9,.1],[.6,.4]])
    labels = np.array([0,1])
    assert threshold_from_validation(weights, labels, .02, 2) is None
    t = threshold_from_validation(weights, labels, .02, 1)
    assert t == .9
    # Apply unchanged to independent test data, even when it makes a mistake.
    assert selection(np.array([[.95,.05]]), np.array([1]), t)["errors"] == 1

def test_adapter_passes_jjm_contract_and_does_not_claim_calibration():
    class Agent:
        dtype = "float32"
        def predict(self,*args,**kwargs): return copy.deepcopy(raw())
    adapter = LayaAdapter(Agent(),"test/laya","test-revision")
    req = request({"row_id":"test:0","text":"I need a new card"},adapter)
    result = evaluate(req,profile(adapter))["results"][0]
    assert result["prediction"]["semantics"] == "normalized_readout"
    assert result["provenance"]["probability_calibration"]["status"] == "not_assessed"
    req["requirements"]["probability_calibration"] = "assessed_in_scope"
    assert evaluate(req,profile(adapter))["results"][0]["prediction"] is None


def test_latency_sample_is_reproducible_proportional_and_not_a_file_prefix():
    rows = [{"row_id": f"{team}:{i}", "team": team} for team, count in
            zip(data.TEAM_IDS, [80, 24, 40, 36, 16, 52, 24, 36]) for i in range(count)]
    original = list(rows)
    chosen = latency_sample(rows, 100, 45)
    assert chosen == latency_sample(rows, 100, 45)
    assert chosen != latency_sample(rows, 100, 46)
    assert len(chosen) == len({r["row_id"] for r in chosen}) == 100
    assert set(r["team"] for r in chosen) == set(data.TEAM_IDS)
    for team in data.TEAM_IDS:
        expected = sum(r["team"] == team for r in rows) * 100 / len(rows)
        assert abs(sum(r["team"] == team for r in chosen) - expected) < 1
    assert chosen != rows[:100]
    assert rows == original
    assert {r["row_id"] for r in latency_sample(rows, len(rows) + 10, 45)} == {r["row_id"] for r in rows}
