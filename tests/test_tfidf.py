"""Contract, split isolation, persistence, and CPU-only CLI tests."""
import builtins
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from conftest import fixed_clock, route_request
from jjm import errors
from jjm.adapters.tfidf_adapter import TfidfAdapter
from jjm.cli import main
from jjm.compile import compile_criterion, render_state
from jjm.engine import evaluate, evaluate_batch
from jjm.heads.artifact import ArtifactStore
from jjm.heads.fit import accept_validation, holdout_positions
from jjm.heads.tfidf import METHOD, TextModel, vectorizer
from jjm.profiles import Profile, capabilities, load_profiles

SETTINGS = {"ngram_range": [1, 2], "min_df": 1, "max_features": None, "C": 4.0, "max_iter": 1000}


def fixture(tmp_path, binary=False, settings=None):
    req = route_request()
    req["profile_id"] = "text"
    req["requirements"]["allowed_methods"] = [METHOD]
    if binary:
        req["criteria"][0]["options"] = req["criteria"][0]["options"][:2]
    compiled = compile_criterion(req["criteria"][0])
    texts, labels = [], []
    for label, word in zip(compiled.outcome_ids, ["refund invoice charged", "crash login broken", "hello weather"]):
        for i in range(8):
            texts.append(render_state({"content": {"ticket": f"{word} uniqueterm{label}{chr(97+i)}"}, "evidence": []}))
            labels.append(label)
    adapter = TfidfAdapter(ArtifactStore(tmp_path / "heads"), settings or SETTINGS)
    profile = Profile("text", "test-1", "real", "tfidf", {}, lambda: adapter)
    return req, compiled, texts, labels, adapter, {"text": profile}


def fit(compiled, texts, labels, adapter, promote=True):
    return adapter.fit(compiled, texts, labels, label_source={"kind": "human"}, promote=promote)


@pytest.mark.parametrize("binary", [False, True])
def test_training_isolation_and_reload(tmp_path, binary):
    req, comp, texts, labels, adapter, profiles = fixture(tmp_path, binary)
    train, hold = holdout_positions(texts, labels)
    art = fit(comp, texts, labels, adapter).artifact
    expected = vectorizer(SETTINGS).fit([texts[i] for i in train])
    assert art["payload"]["vocabulary"] == expected.vocabulary_
    assert art["payload"]["idf"] == pytest.approx(expected.idf_)
    for i in hold:
        assert f"uniqueterm{labels[i]}{chr(97+i%8)}" not in art["payload"]["vocabulary"]
    model = TextModel.from_payload(art["payload"], SETTINGS, len(comp.outcome_ids))
    req["state"]["content"] = {"ticket": "refund invoice charged"}
    result = evaluate(req, profiles, clock=fixed_clock)["results"][0]
    assert result["prediction"]["selected"] == "billing"
    assert result["prediction"]["semantics"] == "normalized_readout"
    direct = model.weights([render_state(req["state"])])[0]
    assert list(result["prediction"]["weights"]) == list(comp.outcome_ids)
    assert list(result["prediction"]["weights"].values()) == pytest.approx(direct)
    assert result["provenance"]["validation"]["status"] == "fitted_unvalidated"
    assert result["provenance"]["runtime"]["prompt_evaluations"] == 0
    fresh = fixture(tmp_path, binary)[-1]
    assert evaluate(req, fresh, clock=fixed_clock)["results"][0]["prediction"] == result["prediction"]
    batch = evaluate_batch([req], profiles, clock=fixed_clock)
    assert batch[0]["results"][0]["prediction"] == result["prediction"]


def test_reuse_promotion_and_invalidation(tmp_path):
    req, comp, texts, labels, adapter, profiles = fixture(tmp_path)
    assert evaluate(req, profiles, clock=fixed_clock)["results"][0]["acceptance"]["reason_codes"] == ["ARTIFACT_MISMATCH"]
    first = fit(comp, texts, labels, adapter, False)
    assert evaluate(req, profiles, clock=fixed_clock)["results"][0]["status"] == "abstained"
    second = fit(comp, texts[::-1], labels[::-1], adapter)
    assert second.reused and first.artifact["artifact_id"] == second.artifact["artifact_id"]
    assert evaluate(req, profiles, clock=fixed_clock)["results"][0]["status"] == "evaluated"
    changed = copy.deepcopy(req)
    changed["criteria"][0]["question"] += " Changed rubric."
    assert evaluate(changed, profiles, clock=fixed_clock)["results"][0]["acceptance"]["reason_codes"] == ["ARTIFACT_MISMATCH"]
    other = fixture(tmp_path, settings={**SETTINGS, "C": 1.0})[-1]
    assert evaluate(req, other, clock=fixed_clock)["results"][0]["status"] == "abstained"


def test_acceptance_does_not_claim_calibration(tmp_path):
    req, comp, texts, labels, adapter, profiles = fixture(tmp_path)
    art = fit(comp, texts, labels, adapter).artifact
    req["requirements"]["validation"] = "accepted_in_scope"
    assert evaluate(req, profiles, clock=fixed_clock)["results"][0]["status"] == "abstained"
    accept_validation(adapter.store, art, "wrong-scope")
    assert evaluate(req, profiles, clock=fixed_clock)["results"][0]["provenance"]["validation"]["status"] == "out_of_scope"
    accept_validation(adapter.store, art, req["scope_id"])
    result = evaluate(req, profiles, clock=fixed_clock)["results"][0]
    assert result["status"] == "evaluated"
    assert result["provenance"]["probability_calibration"]["status"] == "not_assessed"
    req["requirements"]["probability_calibration"] = "assessed_in_scope"
    assert evaluate(req, profiles, clock=fixed_clock)["results"][0]["acceptance"]["reason_codes"] == ["PROBABILITY_CALIBRATION_NOT_ASSESSED"]


@pytest.mark.parametrize("damage", ["coef", "malformed", "shape", "classes"])
def test_bad_artifacts_abstain(tmp_path, damage):
    req, comp, texts, labels, adapter, _ = fixture(tmp_path)
    art = fit(comp, texts, labels, adapter).artifact
    path = next((tmp_path / "heads").rglob("head-*.json"))
    if damage == "coef":
        art["payload"]["coef"][0][0] += 10
    elif damage in ("shape", "classes"):
        # Verify defensive shape checks, even for a payload with a matching digest.
        from jjm.digest import digest
        art["payload"]["coef" if damage == "shape" else "classes"] = []
        art["payload_digest"] = digest(art["payload"])
    path.write_text("{" if damage == "malformed" else json.dumps(art), encoding="utf-8")
    profiles = fixture(tmp_path)[-1]
    result = evaluate(req, profiles, clock=fixed_clock)["results"][0]
    assert result["acceptance"]["reason_codes"] == ["ARTIFACT_MISMATCH"]


def test_bad_training_data_and_no_holdout(tmp_path):
    _, comp, texts, labels, adapter, _ = fixture(tmp_path)
    for ts, ys in [(texts[:-1], labels), (texts, ["unknown"]+labels[1:]),
                   (texts[:8], labels[:8]), (texts+texts[:1], labels+labels[:1])]:
        with pytest.raises(errors.RequestError):
            fit(comp, ts, ys, adapter)
    with pytest.raises(errors.RequestError):
        adapter.fit(comp, texts, labels, method="ordinal", label_source={"kind": "human"})
    indices = [0, 1, 8, 9, 16, 17]
    art = fit(comp, [texts[i] for i in indices], [labels[i] for i in indices], adapter).artifact
    with pytest.raises(errors.RequestError, match="no held-out report"):
        accept_validation(adapter.store, art, "scope")


def test_unknown_features_are_reported_without_claiming_abstention(tmp_path):
    _, comp, texts, labels, adapter, _ = fixture(tmp_path)
    fit(comp, texts, labels, adapter)
    out = adapter.evaluate("zzzzunknownword", comp)
    assert out.diagnostics["all_features_unknown"]
    assert sum(out.weights) == pytest.approx(1)


def profile_file(tmp_path, **settings):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": [{"id": "text", "adapter": "tfidf", "artifacts": "heads", **settings}]}))
    return path


@pytest.mark.parametrize("settings", [{"C": 0}, {"C": True}, {"C": float("inf")}, {"ngram_range": [2, 1]},
                                      {"min_df": 0}, {"max_iter": True}, {"max_features": -1}, {"device": "cuda"}])
def test_profile_rejects_invalid_settings(tmp_path, settings):
    with pytest.raises(errors.RequestError):
        load_profiles(str(profile_file(tmp_path, **settings)))


def test_missing_dependency_is_bounded_and_capabilities_still_work(tmp_path, monkeypatch):
    path = profile_file(tmp_path)
    profile = load_profiles(str(path))["text"]
    assert capabilities(profile)["supported_kinds"] == ["choice"]
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "jjm.adapters.tfidf_adapter":
            raise ImportError("simulated missing extra")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(errors.ExecutionError, match="text.*extra"):
        profile.adapter()


def test_cli_lifecycle_and_no_neural_imports(tmp_path, capsys):
    req, comp, texts, labels, _, _ = fixture(tmp_path)
    cfg = profile_file(tmp_path)
    criterion = tmp_path / "criterion.json"
    criterion.write_text(json.dumps(req["criteria"][0]))
    examples = tmp_path / "examples.jsonl"
    examples.write_text("\n".join(json.dumps({"state": {"content": t, "evidence": []}, "label": y}) for t,y in zip(texts,labels)))
    args = ["--profiles", str(cfg)]
    assert main(args+["fit","--profile","text","--criterion",str(criterion),"--examples",str(examples),"--promote"]) == 0
    assert json.loads(capsys.readouterr().out)["heldout"]["n"] > 0
    assert main(args+["heads","--profile","text"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    assert main(args+["accept","--profile","text","--criterion",str(criterion),"--scope",req["scope_id"]]) == 0
    capsys.readouterr()
    code = ("import sys; from jjm.profiles import load_profiles; "
            f"p=load_profiles({str(cfg)!r})['text']; p.adapter(); "
            "assert not any(m in sys.modules for m in ('torch','transformers','laya'))")
    subprocess.run([sys.executable, "-c", code], check=True)
