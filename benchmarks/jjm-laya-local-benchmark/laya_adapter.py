"""Experimental choice-only JJM adapter for pinned Laya 0.4.8.

Use with the prompt audit in bench.py. This prototype does not preflight arbitrary
new option descriptions. State truncation is refused after upstream inference.

Public Laya values are rounded to four decimal places. We explicitly convert
that quantized vector to a normalized readout, preserving raw values and sum in
diagnostics. This is not probability calibration or validation acceptance.
"""
import math
from importlib.metadata import version
from jjm.adapters.base import AdapterOutput
from jjm import errors

METHOD = "laya:rounded-choice-readout"

def questions(compiled):
    return {"route": {"type": "choice", "instructions": compiled.criterion["question"],
                      "criteria": {o["id"]: o["description"] for o in compiled.criterion["options"]}}}

def readout(result, ids):
    if result["usage"].get("truncated") or result["usage"].get("options"):
        raise ValueError("Laya truncated the state or collapsed options. Refusing this comparison.")
    raw = result["answers"]["route"]["probabilities"]
    if set(raw) != set(ids):
        raise ValueError("Incomplete or unexpected Laya option labels")
    values = [float(raw[k]) for k in ids]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
        raise ValueError("Invalid Laya weights")
    total = sum(values)
    # Four-place rounding introduces at most 0.00005 error per option.
    if total <= 0 or abs(total - 1) > len(ids) * 0.00005 + 1e-7:
        raise ValueError("Laya mass error exceeds its rounding bound")
    weights = [v / total for v in values]
    selected = result["answers"]["route"]["choice"]
    if selected != ids[max(range(len(ids)), key=lambda i: weights[i])]:
        raise ValueError("Rounding changed the selected option or created an ambiguous tie")
    return weights, {"raw_probabilities": raw, "raw_sum": total,
                     "transform": "divide_four_decimal_weights_by_sum",
                     "upstream_confidence": result["answers"]["route"]["confidence"],
                     "usage": result["usage"]}

class LayaAdapter:
    adapter_id = "jjm-laya-experiment"
    adapter_version = "0.1"
    execution_mode = "real"
    method = METHOD
    upstream_level = None
    numeric_path = "provider_native"
    semantics = "normalized_readout"
    supported_kinds = ("choice",)
    max_choice_options = 26
    runtime_id = "laya"
    max_context_tokens = None

    def __init__(self, agent, model_id, revision, head_max_len=384):
        self.agent = agent
        self.model_id = model_id
        self.revision = revision
        self.head_max_len = head_max_len
        self.runtime_version = version("laya")
        self.dtype = str(agent.dtype)

    def model_identity(self):
        return {"id": self.model_id, "revision": self.revision,
                "revision_status": "operator_attested", "tokenizer_revision": self.revision}

    def settings_identity(self):
        return {"max_len": 512, "head_max_len": self.head_max_len,
                "rounding_transform": "explicit_normalization", "backend": "eager"}

    def context_tokens(self, state_text, compiled):
        return None  # No estimated token count masquerading as exact preflight.

    def evaluate(self, state_text, compiled):
        result = self.agent.predict(state_text, questions(compiled), max_len=512,
                                    head_max_len=self.head_max_len)
        try:
            weights, diagnostics = readout(result, list(compiled.outcome_ids))
        except ValueError as exc:
            raise errors.AdapterError(errors.INCOMPLETE_LABEL_DISTRIBUTION, str(exc)) from exc
        return AdapterOutput(weights, 1, 1, diagnostics=diagnostics)
