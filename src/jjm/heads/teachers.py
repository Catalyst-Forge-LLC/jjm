"""Teachers label examples for fitting, or take the cases a head holds. They never run inside `evaluate`.

A teacher is any chat model: a local Ollama model directly, or any provider LiteLLM supports
(`pip install jjm-engine[providers]`). The teacher sees the criterion as lettered options and
answers one letter. Its answer is recorded as that teacher's label, never as ground truth.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from jjm import errors
from jjm.compile import SYSTEM_PROMPT, CompiledCriterion
from jjm.heads.encoders import _http_json, ollama_models

PROMPT_VERSION = "jjm-teacher-prompt-1"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LETTER = re.compile(r"^\W*([A-Z])(?![A-Za-z])")


def teacher_prompt(state_text: str, compiled: CompiledCriterion) -> list[dict[str, str]]:
    options = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(compiled.option_texts))
    user = (f"{state_text}\n\nQuestion: {compiled.question_text}\nOptions:\n{options}\n\n"
            f"Answer with the letter of one option.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_letter(text: str, k: int) -> Optional[int]:
    m = _LETTER.match(text or "")
    if not m:
        return None
    i = LETTERS.index(m.group(1))
    return i if i < k else None


@dataclass
class TeacherAnswer:
    index: Optional[int]          # None when the reply named no valid option
    letter_weights: Optional[list[float]]  # only when the provider reported every option letter's probability
    raw: str
    elapsed_ms: float


class OllamaTeacher:
    provider = "ollama"

    def __init__(self, model: str, *, host: str = "http://localhost:11434", keep_alive: str = "5m",
                 timeout_s: float = 300.0):
        self.model, self.host, self.keep_alive, self.timeout_s = model, host.rstrip("/"), keep_alive, timeout_s
        tags = {m["name"]: m for m in ollama_models(self.host)}
        if model not in tags:
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE,
                                        f"Ollama at {self.host} has no model {model!r}. Installed: {', '.join(sorted(tags))}")
        self.digest = tags[model]["digest"]
        self.thinks = "thinking" in tags[model].get("capabilities", [])

    def identity(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "revision": self.digest,
                "prompt_version": PROMPT_VERSION, "decoding": "temperature 0, thinking off"}

    def answer(self, state_text: str, compiled: CompiledCriterion) -> TeacherAnswer:
        body: dict[str, Any] = {"model": self.model, "stream": False, "keep_alive": self.keep_alive,
                                "messages": teacher_prompt(state_text, compiled),
                                "options": {"temperature": 0, "num_predict": 4}, "logprobs": True, "top_logprobs": 20}
        if self.thinks:
            body["think"] = False
        t0 = time.perf_counter()
        out = _http_json(f"{self.host}/api/chat", body, timeout=self.timeout_s)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if "error" in out:
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE, f"Ollama: {out['error']}")
        text = out.get("message", {}).get("content", "")
        weights = None
        lps = out.get("logprobs") or []
        if lps:
            top = {}
            for alt in lps[0].get("top_logprobs", []):
                tok = alt["token"].strip().upper()
                if len(tok) == 1 and tok in LETTERS[: compiled.k] and tok not in top:
                    top[tok] = math.exp(alt["logprob"])
            if len(top) == compiled.k:
                total = sum(top.values())
                weights = [top[LETTERS[i]] / total for i in range(compiled.k)]
        return TeacherAnswer(parse_letter(text, compiled.k), weights, text, elapsed)


class LiteLLMTeacher:
    """Any provider LiteLLM supports, named by its model string, for example 'anthropic/claude-sonnet-4-6'
    or 'ollama_chat/qwen3.5:9b'. Credentials come from the provider's usual environment variables."""

    provider = "litellm"

    def __init__(self, model: str, *, max_tokens: int = 4, timeout_s: float = 120.0,
                 extra: Optional[dict[str, Any]] = None):
        try:
            import litellm
        except ImportError:
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE,
                                        "LiteLLM teachers need the 'providers' extra (litellm)") from None
        self.litellm = litellm
        self.model, self.max_tokens, self.timeout_s = model, max_tokens, timeout_s
        self.extra = dict(extra or {})

    def identity(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "revision": None,
                "litellm_version": getattr(self.litellm, "__version__", None) or _pkg_version("litellm"),
                "prompt_version": PROMPT_VERSION, "decoding": "temperature 0"}

    def answer(self, state_text: str, compiled: CompiledCriterion) -> TeacherAnswer:
        t0 = time.perf_counter()
        resp = self.litellm.completion(model=self.model, messages=teacher_prompt(state_text, compiled),
                                       temperature=0, max_tokens=self.max_tokens, timeout=self.timeout_s,
                                       **self.extra)
        elapsed = (time.perf_counter() - t0) * 1000.0
        text = resp.choices[0].message.content or ""
        return TeacherAnswer(parse_letter(text, compiled.k), None, text, elapsed)


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def build_teacher(entry: dict[str, Any]):
    provider = entry.get("provider")
    if provider == "ollama":
        return OllamaTeacher(entry["model"], host=entry.get("host", "http://localhost:11434"),
                             keep_alive=entry.get("keep_alive", "5m"))
    if provider == "litellm":
        return LiteLLMTeacher(entry["model"], max_tokens=int(entry.get("max_tokens", 4)),
                              extra=entry.get("options"))
    raise errors.RequestError(errors.INVALID_REQUEST, f"teacher provider must be 'ollama' or 'litellm', not {provider!r}")
