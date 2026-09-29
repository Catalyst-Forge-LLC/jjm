"""AnyJev adapter: JJM criteria through AnyJev's L0 letter readout.

Profile: permutation-only L0. Every Decider setting is pinned here, so a change of
upstream defaults cannot change JJM behavior. A fresh Decider is built for every
criterion, so no statistics carry over between requests or application scopes.
"""
from __future__ import annotations

from typing import Any, Optional

import anyjev
from anyjev import Decider, Question
from anyjev.calibrate.contextual import DEFAULT_PROBES
from anyjev.question import QuestionError
from anyjev.readout import LabelTokenError, build_prompt, render_chat_parts, resolve_labels

from jjm import errors
from jjm.adapters.base import AdapterOutput
from jjm.compile import SYSTEM_PROMPT, CompiledCriterion

ANYJEV_VERSION = "0.2.0"
METHOD = "anyjev:L0-permutation"

# Permutation-only L0: all K cyclic shifts, log-mean combine, no prior correction,
# no adaptive early stop, caller's option order, no L2 adaptation.
DECIDER_SETTINGS: dict[str, Any] = {
    "level": "L0",
    "prior": "none",
    "min_prior_n": 8,
    "prior_strength": 0.0,
    "max_permutations": None,
    "combine": "logmean",
    "cf_probes": tuple(DEFAULT_PROBES),
    "record_content_free": False,
    "system": SYSTEM_PROMPT,
    "shared_prefix": "auto",
    "shared_min_prefix_tokens": 256,
    "adaptive_shifts": False,
    "adaptive_min_shifts": 2,
    "adaptive_margin": None,
    "adaptive_stat": "logit",
    "adaptive_target": 0.01,
    "adaptive_wave": 2,
    "adaptive_order": "spread",
    "canonical_order": False,
    "adapt": False,
    "adapt_min_n": 30,
}


# Profile-selectable, compute path only: "off" scores each rotation as a full prompt
# instead of sharing one prefix KV cache across K copies, which can overflow a small GPU.
SHARED_PREFIX_MODES = {"auto": "auto", "off": False}


def settings_identity(shared_prefix: str = "auto") -> dict[str, Any]:
    out = dict(DECIDER_SETTINGS)
    out["shared_prefix"] = SHARED_PREFIX_MODES[shared_prefix]
    out["cf_probes"] = list(out["cf_probes"])
    out["system"] = "jjm.compile.SYSTEM_PROMPT"
    return out


def _question(compiled: CompiledCriterion) -> Question:
    try:
        return Question.choice(compiled.question_text, compiled.option_texts, name=compiled.id)
    except QuestionError as exc:
        raise errors.AdapterError(errors.TOO_MANY_OPTIONS if "at most" in str(exc) else errors.INVALID_REQUEST,
                                  f"AnyJev rejected the criterion: {exc}") from None


def _round(value: Any) -> Any:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


class AnyJevAdapter:
    adapter_id = "anyjev"
    adapter_version = f"jjm-anyjev-1/anyjev-{ANYJEV_VERSION}"
    method = METHOD
    upstream_level = "L0"
    numeric_path = "logit_readout"
    semantics = "normalized_readout"
    supported_kinds = ("choice", "assertion")
    max_choice_options = 26

    def __init__(self, backend: Any, *, execution_mode: str, model: dict[str, Any], runtime_id: str,
                 runtime_version: str, dtype: Optional[str], max_context_tokens: Optional[int],
                 shared_prefix: str = "auto"):
        if anyjev.__version__ != ANYJEV_VERSION:
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE,
                                      f"JJM is pinned to anyjev {ANYJEV_VERSION}; found {anyjev.__version__}")
        self.backend = backend
        self.execution_mode = execution_mode
        self._model = dict(model)
        self.runtime_id = runtime_id
        self.runtime_version = runtime_version
        self.dtype = dtype
        self.max_context_tokens = max_context_tokens
        if shared_prefix not in SHARED_PREFIX_MODES:
            raise errors.RequestError(errors.INVALID_REQUEST, f"shared_prefix must be one of {sorted(SHARED_PREFIX_MODES)}")
        self.shared_prefix = shared_prefix
        self.decider_settings = {**DECIDER_SETTINGS, "shared_prefix": SHARED_PREFIX_MODES[shared_prefix]}

    def settings_identity(self) -> dict[str, Any]:
        return settings_identity(self.shared_prefix)

    def model_identity(self) -> dict[str, Any]:
        return dict(self._model)

    def context_tokens(self, state_text: str, compiled: CompiledCriterion) -> Optional[int]:
        q = _question(compiled)
        tok = self.backend.tokenizer
        try:
            labels, _ = resolve_labels(tok, q)
        except LabelTokenError as exc:
            raise errors.AdapterError(errors.INCOMPLETE_LABEL_DISTRIBUTION, f"Answer labels are unusable: {exc}")
        longest = 0
        for shift in range(q.k):
            perm = [(j + shift) % q.k for j in range(q.k)]
            pre, suf = render_chat_parts(tok, build_prompt(state_text, q, perm, SYSTEM_PROMPT, labels))
            longest = max(longest, len(tok.encode(pre + suf, add_special_tokens=False)))
        return longest

    def evaluate(self, state_text: str, compiled: CompiledCriterion) -> AdapterOutput:
        q = _question(compiled)
        decider = Decider(self.backend, **self.decider_settings)
        fallbacks_before = getattr(self.backend, "shared_fallbacks", None)
        try:
            decision = decider.decide(state_text, [q])[0]
        except LabelTokenError as exc:
            raise errors.AdapterError(errors.INCOMPLETE_LABEL_DISTRIBUTION,
                                      f"Answer labels are not single tokens for this model: {exc}") from None
        except MemoryError:
            raise errors.AdapterError(errors.INTERNAL_ERROR, "The backend ran out of memory.", True) from None
        except Exception as exc:  # backend failures must become bounded item errors
            name = exc.__class__.__name__
            retry = "OutOfMemory" in name
            raise errors.AdapterError(errors.INTERNAL_ERROR, f"The backend failed with {name}.", retry) from None
        if decision.level != "L0":
            raise errors.AdapterError(errors.INTERNAL_ERROR,
                                      f"AnyJev returned level {decision.level}, expected L0.")
        diag = decision.diagnostics
        stats = decider.stats
        groups = int(stats["shared_groups"])
        if groups == 0:
            prefix_reuse = "not_used"
        elif fallbacks_before is None:
            prefix_reuse = "unknown"
        else:
            # the HF backend scores a group through full prompts when its token split is not exact
            fell_back = int(getattr(self.backend, "shared_fallbacks", 0)) - int(fallbacks_before)
            prefix_reuse = "used" if groups - fell_back > 0 else "not_used"
        return AdapterOutput(
            weights=[float(p) for p in decision.probs],
            prompt_evaluations=int(stats["flat_prompts"] + stats["shared_prompts"]),
            adapter_calls=int(stats["backend_calls"]),
            prefix_reuse=prefix_reuse,
            early_exit="not_used",
            diagnostics={
                "anyjev_version": ANYJEV_VERSION,
                "permutations": int(diag.get("permutations", 0)),
                "prior_method": str(diag.get("prior_method", "none")),
                "answer_mass": _round(diag.get("answer_mass")),
                "order_flip_raw": _round(diag.get("order_flip_raw")),
                "order_flip_l0": _round(diag.get("order_flip_l0")),
            },
        )
