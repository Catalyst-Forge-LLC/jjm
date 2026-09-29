"""The AnyJev adapter through AnyJev's own FakeBackend: the real L0 pipeline, fake logits."""
import copy
import inspect

import numpy as np
import pytest
from anyjev import Decider
from anyjev.backends.fake import FakeBackend

from conftest import fixed_clock, fixture_request, profile_with, route_request
from jjm.adapters.anyjev_adapter import DECIDER_SETTINGS, METHOD, AnyJevAdapter
from jjm.compile import compile_criterion, render_state
from jjm.engine import evaluate

MODEL = {"id": "anyjev/fake-backend", "revision": None, "revision_status": "unknown", "tokenizer_revision": None}


def routing_content(state: str, option: str) -> float:
    if "charged" in state and option.startswith("Billing"):
        return 3.0
    return 0.0


def evidence_content(state: str, option: str) -> float:
    if "export-receipt" in state and option.startswith("Supported"):
        return 2.5
    if "export-receipt" not in state and option.startswith("Not observable"):
        return 2.5
    return 0.0


def adapter(content=routing_content, position_bias=None, max_context_tokens=None):
    backend = FakeBackend(content, position_bias=position_bias)
    return AnyJevAdapter(backend, execution_mode="synthetic", model=MODEL, runtime_id="anyjev-fake",
                         runtime_version="0.2.0", dtype=None, max_context_tokens=max_context_tokens)


def run(request, a):
    request = copy.deepcopy(request)
    request["profile_id"] = "test-profile"
    request["requirements"]["allowed_methods"] = [METHOD]
    return evaluate(request, profile_with(a), clock=fixed_clock)


def test_every_decider_setting_is_pinned():
    params = [p for p in inspect.signature(Decider.__init__).parameters if p not in ("self", "backend")]
    assert sorted(params) == sorted(DECIDER_SETTINGS), "AnyJev added or removed a Decider setting; pin it"


def test_choice_through_l0_permutation():
    result = run(route_request(), adapter())["results"][0]
    assert result["status"] == "evaluated"
    assert result["prediction"]["selected"] == "billing"
    prov = result["provenance"]
    assert prov["method"] == METHOD
    assert prov["upstream_level"] == "L0"
    assert prov["runtime"]["numeric_path"] == "logit_readout"
    assert prov["runtime"]["prompt_evaluations"] == 3
    assert prov["diagnostics"]["permutations"] == 3
    assert prov["diagnostics"]["prior_method"] == "none"
    assert result["prediction"]["semantics"] == "normalized_readout"


def test_position_bias_is_cancelled_by_all_rotations():
    plain = run(route_request(), adapter())["results"][0]["prediction"]["weights"]
    biased = run(route_request(), adapter(position_bias=[2.0, 0.0, -1.0]))["results"][0]["prediction"]["weights"]
    assert np.allclose([plain[k] for k in plain], [biased[k] for k in plain], atol=1e-9)


def test_assertion_compiles_to_three_outcomes_and_can_select_not_observable():
    request = fixture_request("04-not-observable")
    result = run(request, adapter(evidence_content))["results"][0]
    assert set(result["prediction"]["weights"]) == {"supported", "contradicted", "not_observable"}
    assert result["prediction"]["selected"] == "not_observable"
    request = fixture_request("02-supported")
    result = run(request, adapter(evidence_content))["results"][0]
    assert result["prediction"]["selected"] == "supported"
    assert result["acceptance"]["status"] == "pass"


def test_t11_t12_no_state_carries_between_requests():
    a = adapter()
    first = run(route_request(), a)["results"][0]["prediction"]
    for _ in range(3):
        run(fixture_request("02-supported"), a)
    second = run(route_request(), a)["results"][0]["prediction"]
    assert first == second


def test_context_limit_abstains_before_inference():
    a = adapter(max_context_tokens=1)  # the fake tokenizer counts any long text as 2 tokens
    result = run(route_request(), a)["results"][0]
    assert result["status"] == "abstained"
    assert result["acceptance"]["reason_codes"] == ["CONTEXT_LIMIT_EXCEEDED"]
    assert a.backend.calls == 0


def test_context_tokens_measures_the_longest_rotation():
    a = adapter()
    request = route_request()
    tokens = a.context_tokens(render_state(request["state"]), compile_criterion(request["criteria"][0]))
    assert isinstance(tokens, int) and tokens > 0


def test_backend_failure_is_a_bounded_error():
    class Broken(FakeBackend):
        def next_token_logprobs(self, prompts, token_ids):
            raise RuntimeError("CUDA out of memory with secret path C:/models")

    a = AnyJevAdapter(Broken(routing_content), execution_mode="synthetic", model=MODEL, runtime_id="anyjev-fake",
                      runtime_version="0.2.0", dtype=None, max_context_tokens=None)
    result = run(route_request(), a)["results"][0]
    assert result["status"] == "error"
    assert result["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in result["error"]["message"]


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_t06_nonfinite_logits_become_errors(bad):
    class Bad(FakeBackend):
        def next_token_logprobs(self, prompts, token_ids):
            out = super().next_token_logprobs(prompts, token_ids)
            out[0][0] = bad
            return out

    a = AnyJevAdapter(Bad(routing_content), execution_mode="synthetic", model=MODEL, runtime_id="anyjev-fake",
                      runtime_version="0.2.0", dtype=None, max_context_tokens=None)
    result = run(route_request(), a)["results"][0]
    assert result["status"] == "error"
    assert result["prediction"] is None


def test_shared_prefix_off_changes_identity_not_weights():
    auto = adapter()
    off = AnyJevAdapter(FakeBackend(routing_content), execution_mode="synthetic", model=MODEL,
                        runtime_id="anyjev-fake", runtime_version="0.2.0", dtype=None, max_context_tokens=None,
                        shared_prefix="off")
    assert off.decider_settings["shared_prefix"] is False
    a, b = run(route_request(), auto)["results"][0], run(route_request(), off)["results"][0]
    assert a["prediction"]["weights"] == b["prediction"]["weights"]
    assert b["provenance"]["runtime"]["prefix_reuse"] == "not_used"
    assert a["provenance"]["effective_config_digest"] != b["provenance"]["effective_config_digest"]
