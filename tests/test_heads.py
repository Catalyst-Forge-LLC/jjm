import copy
import json
import zlib

import numpy as np
import pytest

from conftest import fixed_clock, route_request
from jjm import errors
from jjm.adapters.head_adapter import METHOD, FittedHeadAdapter
from jjm.cli import main
from jjm.compile import compile_criterion, render_state
from jjm.engine import evaluate, evaluate_batch
from jjm.heads import head as heads
from jjm.heads.artifact import ArtifactStore
from jjm.heads.cache import FeatureCache, cache_key
from jjm.heads.cascade import escalation_curve, threshold_for_accuracy
from jjm.heads.encoders import DEFAULT_PREFIX_TEMPLATE, EncodeStats, INPUT_VERSION, resolve_layer
from jjm.heads.fit import accept_validation, fit_head
from jjm.heads.runtime import Featurizer
from jjm.heads.teachers import parse_letter, teacher_prompt
from jjm.profiles import Profile, load_profiles, load_teachers

WORDS = ("charged", "twice", "refund", "invoice", "crash", "error", "login", "broken", "hello", "weather")


class KeywordEncoder:
    """Deterministic stand-in for a model: counts of a few words, plus hashed noise."""

    kind = "fake"
    runtime_id = "fake"
    runtime_version = "1"
    dtype = None
    early_exit = "not_used"

    def __init__(self, tag="kw-1"):
        self.tag = tag
        self.calls = 0

    def identity(self):
        return {"kind": self.kind, "input_version": INPUT_VERSION, "model": self.tag, "revision": None,
                "prefix_template": DEFAULT_PREFIX_TEMPLATE}

    def compute_settings(self):
        return {}

    def model_identity(self):
        return {"id": f"fake/{self.tag}", "revision": None, "revision_status": "unknown", "tokenizer_revision": None}

    def count_tokens(self, prefix, body):
        return len((prefix + body).split())

    def encode(self, items):
        self.calls += 1
        rows = []
        for _, body in items:
            text = body.lower()
            rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
            rows.append(np.array([text.count(w) for w in WORDS] + list(rng.normal(0, 0.05, 6)), dtype=np.float32))
        return np.stack(rows), EncodeStats(items=len(items), forward_passes=1)


BILLING = ["I was charged twice", "refund my invoice please", "charged twice again", "the invoice is wrong refund",
           "why was I charged", "refund refund", "invoice charged twice", "please refund the charge"]
TECH = ["the app crash on start", "error on login", "login broken", "crash and error", "broken login page",
        "error error crash", "the export is broken", "login crash"]
OTHER = ["hello there", "what is the weather", "hello", "nice weather", "hello weather", "just saying hello",
         "weather report", "hello again"]


def examples():
    texts, labels = [], []
    for group, label in ((BILLING, "billing"), (TECH, "technical"), (OTHER, "other")):
        for t in group:
            texts.append(render_state({"content": {"ticket": t}, "evidence": []}))
            labels.append(label)
    return texts, labels


def criterion():
    return route_request()["criteria"][0]


def head_profiles(tmp_path, encoder=None, cache=True):
    encoder = encoder or KeywordEncoder()
    featurizer = Featurizer(encoder, tmp_path / "cache" if cache else None)
    adapter = FittedHeadAdapter(featurizer, ArtifactStore(tmp_path / "heads"), max_context_tokens=None)
    profiles = load_profiles(None)
    profiles["routing-head"] = Profile("routing-head", "test-1", "real", "fitted-head", {}, lambda: adapter)
    return profiles, adapter


def head_request(ticket="I was charged twice for the same subscription."):
    request = route_request()
    request["profile_id"] = "routing-head"
    request["requirements"]["allowed_methods"] = [METHOD]
    request["state"]["content"] = {"ticket": ticket}
    return request


def fit_and_promote(adapter, method="softmax"):
    texts, labels = examples()
    return fit_head(adapter.featurizer, adapter.store, compile_criterion(criterion()), texts, labels, method=method,
                    label_source={"kind": "human"}, promote=True)


def test_softmax_head_learns_a_separable_task():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(3), 40)
    X = rng.normal(0, 1, (120, 8)) + np.eye(3, 8)[y] * 4
    head, cv = heads.fit("softmax", X, y, 3)
    assert heads.classification_report(cv["oof_weights"], y, 3)["accuracy"] > 0.95
    w = head.weights(X)
    assert np.allclose(w.sum(1), 1.0)
    again, _ = heads.fit("softmax", X, y, 3)
    assert np.array_equal(again.W, head.W)


def test_ordinal_weights_are_a_distribution_centred_on_the_prediction():
    w = heads.ordinal_weights(np.array([0.0, 2.0, 4.4]), 0.5, 5)
    assert np.allclose(w.sum(1), 1.0)
    assert list(w.argmax(1)) == [0, 2, 4]


def test_ordinal_head_orders_levels():
    rng = np.random.default_rng(2)
    y = np.repeat(np.arange(5), 30)
    X = np.column_stack([y + rng.normal(0, 0.3, len(y)), rng.normal(0, 1, (len(y), 4))])
    head, cv = heads.fit("ordinal", X, y, 5)
    assert cv["sigma"] > 0
    assert heads.classification_report(cv["oof_weights"], y, 5)["accuracy"] > 0.7


def test_folds_are_stratified_and_cover_every_example():
    y = np.array([0] * 10 + [1] * 5)
    parts = heads.folds(y, 5)
    assert sorted(np.concatenate(parts).tolist()) == list(range(15))
    assert all(np.sum(y[p] == 1) == 1 for p in parts)


def test_layer_resolution():
    assert resolve_layer(None, 28) == 28
    assert resolve_layer(0.5, 28) == 14
    assert resolve_layer(7, 28) == 7
    with pytest.raises(errors.RequestError):
        resolve_layer(29, 28)


def test_feature_cache_round_trip(tmp_path):
    cache = FeatureCache(tmp_path)
    keys = [cache_key("sha256:x", "p", "a"), cache_key("sha256:x", "p", "b")]
    cache.put_many(keys, np.array([[1, 2], [3, 4]], dtype=np.float32))
    got = cache.get_many(keys + [cache_key("sha256:y", "p", "a")])
    assert got[0].tolist() == [1, 2] and got[1].tolist() == [3, 4] and got[2] is None
    assert cache_key("sha256:x", "pa", "b") != cache_key("sha256:x", "p", "ab")


def test_featurizer_reuses_identical_inputs(tmp_path):
    enc = KeywordEncoder()
    f = Featurizer(enc, tmp_path)
    items = [("p", "charged twice"), ("p", "charged twice"), ("p", "login")]
    first = f.features(items)
    assert first.encoded == 3 and enc.calls == 1
    second = f.features(items)
    assert second.cache_hits == 3 and enc.calls == 1
    fresh = Featurizer(enc, tmp_path)
    assert fresh.features(items).cache_hits == 3


def test_no_promoted_head_abstains_before_any_model_call(tmp_path):
    profiles, adapter = head_profiles(tmp_path)
    result = evaluate(head_request(), profiles, clock=fixed_clock)["results"][0]
    assert result["status"] == "abstained"
    assert result["acceptance"]["reason_codes"] == ["ARTIFACT_MISMATCH"]
    assert adapter.encoder.calls == 0
    assert result["provenance"]["validation"]["status"] == "not_fitted"
    assert result["provenance"]["probability_calibration"]["status"] == "not_assessed"


def test_fitted_head_serves_a_valid_envelope(tmp_path):
    profiles, adapter = head_profiles(tmp_path)
    fit = fit_and_promote(adapter)
    assert not fit.reused
    response = evaluate(head_request(), profiles, clock=fixed_clock)
    result = response["results"][0]
    assert result["status"] == "evaluated"
    assert result["prediction"]["selected"] == "billing"
    assert result["prediction"]["semantics"] == "normalized_readout"
    prov = result["provenance"]
    assert prov["method"] == METHOD
    assert prov["runtime"]["numeric_path"] == "hidden_state_head"
    assert prov["validation"] == {"status": "fitted_unvalidated", "artifact_id": fit.artifact["artifact_id"],
                                  "validation_report_id": None, "scope_match": False}
    assert prov["probability_calibration"]["status"] == "not_assessed"
    assert prov["diagnostics"]["feature_cache"] == "miss"
    again = evaluate(head_request(), profiles, clock=fixed_clock)["results"][0]
    assert again["provenance"]["diagnostics"]["feature_cache"] == "hit"
    assert again["provenance"]["runtime"]["prompt_evaluations"] == 0
    assert again["prediction"]["weights"] == result["prediction"]["weights"]
    assert fit.artifact["report"]["heldout"]["n"] > 0
    assert fit.artifact["validation_status"] == "fitted_unvalidated"


def test_an_accepted_report_satisfies_a_matching_scope(tmp_path):
    profiles, adapter = head_profiles(tmp_path)
    fit = fit_and_promote(adapter)
    request = head_request()
    request["requirements"]["validation"] = "accepted_in_scope"
    refused = evaluate(request, profiles, clock=fixed_clock)["results"][0]
    assert refused["status"] == "abstained"
    assert refused["acceptance"]["reason_codes"] == ["VALIDATION_NOT_ACCEPTED"]
    accept_validation(adapter.store, fit.artifact, "other-scope")
    wrong = evaluate(request, profiles, clock=fixed_clock)["results"][0]
    assert wrong["status"] == "abstained"
    assert wrong["provenance"]["validation"]["status"] == "out_of_scope"
    assert wrong["provenance"]["validation"]["scope_match"] is False
    accept_validation(adapter.store, fit.artifact, request["scope_id"])
    ordinary = evaluate(head_request(), profiles, clock=fixed_clock)["results"][0]
    assert ordinary["prediction"]["semantics"] == "normalized_readout"
    assert ordinary["provenance"]["validation"]["status"] == "accepted_in_scope"
    result = evaluate(request, profiles, clock=fixed_clock)["results"][0]
    assert result["status"] == "evaluated"
    assert result["acceptance"]["status"] == "pass"
    assert result["prediction"]["semantics"] == "normalized_readout"
    assert result["prediction"]["weights"] == ordinary["prediction"]["weights"]
    assert result["prediction"]["selected"] == "billing"
    val = result["provenance"]["validation"]
    assert val["status"] == "accepted_in_scope" and val["scope_match"] is True
    assert val["validation_report_id"] == fit.artifact["report"]["heldout"]["report_id"]
    assert result["provenance"]["probability_calibration"]["status"] == "not_assessed"
    request["requirements"]["probability_calibration"] = "assessed_in_scope"
    uncalibrated = evaluate(request, profiles, clock=fixed_clock)["results"][0]
    assert uncalibrated["status"] == "abstained"
    assert uncalibrated["acceptance"]["reason_codes"] == ["PROBABILITY_CALIBRATION_NOT_ASSESSED"]


def test_accept_refuses_a_head_with_no_holdout(tmp_path):
    _, adapter = head_profiles(tmp_path)
    texts, labels = [], []
    for label, word in (("billing", "charged"), ("technical", "crash")):
        for n in range(2):
            texts.append(render_state({"content": {"ticket": f"{word} {n}"}, "evidence": []}))
            labels.append(label)
    fit = fit_head(adapter.featurizer, adapter.store, compile_criterion(criterion()), texts, labels,
                   label_source={"kind": "human"}, promote=True)
    assert fit.artifact["report"]["heldout"]["n"] == 0
    with pytest.raises(errors.RequestError, match="no held-out report"):
        accept_validation(adapter.store, fit.artifact, "fixture.export-workflow")


def test_same_examples_reuse_the_stored_head(tmp_path):
    _, adapter = head_profiles(tmp_path)
    first = fit_and_promote(adapter)
    second = fit_and_promote(adapter)
    assert second.reused and second.artifact["artifact_id"] == first.artifact["artifact_id"]
    texts, labels = examples()
    labels[0] = "technical"
    changed = fit_head(adapter.featurizer, adapter.store, compile_criterion(criterion()), texts, labels,
                       label_source={"kind": "human"})
    assert not changed.reused and changed.artifact["artifact_id"] != first.artifact["artifact_id"]


def test_a_head_is_not_served_for_another_encoder_or_criterion(tmp_path):
    profiles, adapter = head_profiles(tmp_path)
    fit_and_promote(adapter)
    other_profiles, _ = head_profiles(tmp_path, KeywordEncoder("kw-2"))
    result = evaluate(head_request(), other_profiles, clock=fixed_clock)["results"][0]
    assert result["acceptance"]["reason_codes"] == ["ARTIFACT_MISMATCH"]
    request = head_request()
    request["criteria"][0]["question"] = "Which queue should take this ticket?"
    result = evaluate(request, profiles, clock=fixed_clock)["results"][0]
    assert result["acceptance"]["reason_codes"] == ["ARTIFACT_MISMATCH"]


def test_tampered_head_is_refused(tmp_path):
    profiles, adapter = head_profiles(tmp_path)
    art = fit_and_promote(adapter).artifact
    path = next((tmp_path / "heads").rglob(f"{art['artifact_id']}.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["b"][0] += 1.0
    path.write_text(json.dumps(data), encoding="utf-8")
    fresh, _ = head_profiles(tmp_path)
    result = evaluate(head_request(), fresh, clock=fixed_clock)["results"][0]
    assert result["acceptance"]["reason_codes"] == ["ARTIFACT_MISMATCH"]


def test_fit_refuses_labels_outside_the_criterion(tmp_path):
    _, adapter = head_profiles(tmp_path)
    texts, labels = examples()
    labels[0] = "sales"
    with pytest.raises(errors.RequestError):
        fit_head(adapter.featurizer, adapter.store, compile_criterion(criterion()), texts, labels,
                 label_source={"kind": "human"})


def test_batch_prefetch_encodes_once_and_each_envelope_reports_the_hit(tmp_path):
    profiles, adapter = head_profiles(tmp_path, cache=False)
    fit_and_promote(adapter)
    calls = adapter.encoder.calls
    requests = [head_request(t) for t in ("refund my invoice", "login is broken", "hello weather")]
    for i, r in enumerate(requests):
        r["request_id"] = f"batch-{i}"
    responses = evaluate_batch(requests, profiles, clock=fixed_clock)
    assert adapter.encoder.calls == calls + 1
    assert [r["results"][0]["prediction"]["selected"] for r in responses] == ["billing", "technical", "other"]
    assert all(r["results"][0]["provenance"]["diagnostics"]["feature_cache"] == "hit" for r in responses)


def test_teacher_prompt_and_letter_parsing():
    compiled = compile_criterion(criterion())
    messages = teacher_prompt("Snapshot", compiled)
    assert "A. Billing amounts" in messages[1]["content"] and "C. Neither" in messages[1]["content"]
    assert parse_letter("B", 3) == 1
    assert parse_letter(" (a) billing", 3) is None
    assert parse_letter("A.", 3) == 0
    assert parse_letter("D", 3) is None
    assert parse_letter("Billing", 3) is None


def test_escalation_curve_and_threshold():
    top = np.array([0.9, 0.8, 0.4, 0.3])
    head_ok = np.array([True, True, False, False])
    teacher_ok = np.array([True, False, True, True])
    curve = {c["min_top_mass"]: c for c in escalation_curve(top, head_ok, teacher_ok, [0.0, 0.5, 1.01])}
    assert curve[0.0]["accuracy"] == 0.5 and curve[0.0]["teacher_share"] == 0.0
    assert curve[0.5]["accuracy"] == 1.0 and curve[0.5]["teacher_share"] == 0.5
    assert curve[1.01]["accuracy"] == 0.75
    assert threshold_for_accuracy(top, head_ok, 1.0) == 0.8


def test_profile_file_accepts_head_profiles_and_teachers(tmp_path):
    cfg = {"profiles": [{"id": "route-emb", "adapter": "fitted-head", "artifacts": "heads", "cache": "cache",
                         "encoder": {"kind": "ollama", "model": "nomic-embed-text-v2-moe:latest"}},
                        {"id": "route-hf", "adapter": "fitted-head", "artifacts": "heads",
                         "encoder": {"kind": "hf", "model": {"id": "Qwen/Qwen3-Reranker-0.6B"}, "layer": 0.5}}],
           "teachers": [{"id": "local", "provider": "ollama", "model": "qwen3.5:9b"},
                        {"id": "hosted", "provider": "litellm", "model": "anthropic/claude-sonnet-4-6"}]}
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    profiles = load_profiles(str(path))
    assert profiles["route-emb"].config["artifacts"] == str((tmp_path / "heads").resolve())
    assert profiles["route-hf"].config["encoder"]["layer"] == 0.5
    assert profiles["route-emb"].generation != profiles["route-hf"].generation
    assert sorted(load_teachers(str(path))) == ["hosted", "local"]
    bad = copy.deepcopy(cfg)
    bad["profiles"][0]["encoder"]["prefix_template"] = "no placeholder"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(errors.RequestError):
        load_profiles(str(path))


def test_models_command_runs_without_loading_anything(capsys):
    assert main(["models", "--ollama-host", "http://127.0.0.1:9"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ollama"] == [] and out["ollama_problem"]
    assert isinstance(out["huggingface"], list)
