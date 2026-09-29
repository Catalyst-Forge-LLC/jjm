"""Frozen encoders: text in, one feature vector per item out.

An encoder's identity is everything that changes its features: model, revision, input template,
depth, pooling, and truncation. Compute choices that should not change the features (batch size,
prefix reuse, device) belong to the effective config, not to the identity a head is bound to.
"""
from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from jjm import errors
from jjm.compile import CompiledCriterion
from jjm.digest import digest

INPUT_VERSION = "jjm-head-input-1"
DEFAULT_PREFIX_TEMPLATE = "Instruct: {question}\nQuery: "
POOLINGS = ("mean", "last")


def encoder_input(state_text: str, compiled: CompiledCriterion, prefix_template: str) -> tuple[str, str]:
    """(prefix, body). The prefix depends only on the criterion, so every state under one criterion shares it."""
    return prefix_template.format(question=compiled.question_text), state_text


@dataclass
class EncodeStats:
    items: int = 0
    forward_passes: int = 0
    prefix_reuse: str = "not_used"
    truncated: int = 0
    tokens: list[int] = field(default_factory=list)
    elapsed_ms: float = 0.0


class Encoder(Protocol):
    kind: str
    runtime_id: str
    runtime_version: str
    dtype: Optional[str]
    early_exit: str

    def identity(self) -> dict[str, Any]: ...

    def compute_settings(self) -> dict[str, Any]: ...

    def model_identity(self) -> dict[str, Any]: ...

    def count_tokens(self, prefix: str, body: str) -> Optional[int]: ...

    def encode(self, items: Sequence[tuple[str, str]]) -> tuple[np.ndarray, EncodeStats]: ...


def resolve_layer(layer: Any, total: int) -> int:
    """None means every layer; a float in (0, 1] is a fraction of depth; an int is a layer count."""
    if layer is None:
        return total
    if isinstance(layer, float):
        if not 0.0 < layer <= 1.0:
            raise errors.RequestError(errors.INVALID_REQUEST, "encoder layer as a fraction must be in (0, 1]")
        return max(1, round(layer * total))
    if isinstance(layer, int) and 1 <= layer <= total:
        return layer
    raise errors.RequestError(errors.INVALID_REQUEST, f"encoder layer must be 1..{total}, a fraction, or null")


class HFEncoder:
    """Hidden states of a Transformers decoder, pooled over tokens.

    Early exit: only the first `layer` blocks are kept, so later blocks never load or run.
    Prefix reuse: the criterion prefix is run once and its key/value cache is shared by every
    body in a batch. The prefix and body are always tokenized separately, so both paths feed the
    model the same token sequence.
    """

    kind = "hf"
    runtime_id = "transformers"

    def __init__(self, source: str, *, model_id: str, revision: Optional[str], device: str = "cuda",
                 dtype: str = "bfloat16", layer: Any = None, pooling: str = "mean", max_tokens: int = 512,
                 batch_size: int = 16, prefix_reuse: str = "off",
                 prefix_template: str = DEFAULT_PREFIX_TEMPLATE):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        if pooling not in POOLINGS:
            raise errors.RequestError(errors.INVALID_REQUEST, f"pooling must be one of {POOLINGS}")
        if prefix_reuse not in ("on", "off"):
            raise errors.RequestError(errors.INVALID_REQUEST, "prefix_reuse must be 'on' or 'off'")
        self.torch = torch
        self.runtime_version = str(transformers.__version__)
        self.model_id, self.revision, self.device, self.dtype = model_id, revision, device, dtype
        self.pooling, self.max_tokens, self.batch_size = pooling, int(max_tokens), int(batch_size)
        self.prefix_reuse, self.prefix_template = prefix_reuse, prefix_template
        self.tok = AutoTokenizer.from_pretrained(source, revision=revision)
        model = AutoModel.from_pretrained(source, revision=revision, dtype=getattr(torch, dtype))
        self.total_layers = int(model.config.num_hidden_layers)
        self.layer = resolve_layer(layer, self.total_layers)
        if self.layer < self.total_layers:
            model.layers = model.layers[: self.layer]
            model.config.num_hidden_layers = self.layer
        self.early_exit = "used" if self.layer < self.total_layers else "not_used"
        self.model = model.to(device).eval()
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.eos_id = self.tok.eos_token_id
        self._prefix_cache: dict[str, tuple[Any, Any, int]] = {}

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind, "input_version": INPUT_VERSION, "model": self.model_id, "revision": self.revision,
                "dtype": self.dtype, "layer": self.layer, "total_layers": self.total_layers, "pooling": self.pooling,
                "max_tokens": self.max_tokens, "prefix_template": self.prefix_template,
                "append_eos": self.pooling == "last"}

    def compute_settings(self) -> dict[str, Any]:
        return {"device": self.device, "batch_size": self.batch_size, "prefix_reuse": self.prefix_reuse}

    def model_identity(self) -> dict[str, Any]:
        status = "operator_attested" if self.revision else "unknown"
        return {"id": self.model_id, "revision": self.revision, "revision_status": status,
                "tokenizer_revision": self.revision}

    def _body_ids(self, body: str) -> tuple[list[int], bool]:
        ids = self.tok.encode(body, add_special_tokens=False)
        cut = len(ids) > self.max_tokens
        ids = ids[: self.max_tokens]
        if self.pooling == "last" and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids, cut

    def count_tokens(self, prefix: str, body: str) -> Optional[int]:
        return len(self.tok.encode(prefix, add_special_tokens=False)) + len(self._body_ids(body)[0])

    def _pool(self, hidden: Any, mask: Any) -> Any:
        if self.pooling == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(1) / m.sum(1)
        last = mask.sum(1) - 1
        return hidden[self.torch.arange(hidden.shape[0], device=hidden.device), last]

    def _pad(self, seqs: list[list[int]]) -> tuple[Any, Any]:
        torch = self.torch
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids.to(self.device), mask.to(self.device)

    def _prefix(self, prefix: str) -> tuple[Any, Any, int]:
        hit = self._prefix_cache.get(prefix)
        if hit is None:
            ids = self.tok.encode(prefix, add_special_tokens=False)
            t = self.torch.tensor([ids], device=self.device)
            out = self.model(input_ids=t, use_cache=True)
            hit = (out.past_key_values, out.last_hidden_state[0].float(), len(ids))
            if len(self._prefix_cache) >= 8:
                self._prefix_cache.pop(next(iter(self._prefix_cache)))
            self._prefix_cache[prefix] = hit
        return hit

    def _batch_with_prefix(self, prefix: str, bodies: list[list[int]]) -> Any:
        torch = self.torch
        cache, prefix_hidden, n_prefix = self._prefix(prefix)
        ids, mask = self._pad(bodies)
        batch_cache = copy.deepcopy(cache)
        batch_cache.batch_repeat_interleave(len(bodies))
        full_mask = torch.cat([torch.ones((len(bodies), n_prefix), dtype=mask.dtype, device=mask.device), mask], 1)
        out = self.model(input_ids=ids, attention_mask=full_mask, past_key_values=batch_cache, use_cache=True)
        hidden = torch.cat([prefix_hidden.unsqueeze(0).expand(len(bodies), -1, -1).to(out.last_hidden_state.dtype),
                            out.last_hidden_state], 1)
        return self._pool(hidden.float(), full_mask)

    def encode(self, items: Sequence[tuple[str, str]]) -> tuple[np.ndarray, EncodeStats]:
        torch = self.torch
        stats = EncodeStats(items=len(items))
        vecs: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, len(items), self.batch_size):
                chunk = items[s: s + self.batch_size]
                bodies = []
                for _, body in chunk:
                    ids, cut = self._body_ids(body)
                    bodies.append(ids)
                    stats.truncated += int(cut)
                prefixes = {p for p, _ in chunk}
                if self.prefix_reuse == "on" and len(prefixes) == 1:
                    pooled = self._batch_with_prefix(chunk[0][0], bodies)
                    stats.prefix_reuse = "used"
                    stats.tokens += [len(b) for b in bodies]
                else:
                    seqs = [self.tok.encode(p, add_special_tokens=False) + b for (p, _), b in zip(chunk, bodies)]
                    ids, mask = self._pad(seqs)
                    out = self.model(input_ids=ids, attention_mask=mask)
                    pooled = self._pool(out.last_hidden_state.float(), mask)
                    stats.tokens += [len(q) for q in seqs]
                stats.forward_passes += 1
                vecs.append(pooled.cpu().numpy().astype(np.float32))
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        stats.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return np.concatenate(vecs), stats


def _http_json(url: str, body: Optional[dict[str, Any]] = None, timeout: float = 120.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE, f"Could not reach {url}: {exc}") from None


def ollama_models(host: str) -> list[dict[str, Any]]:
    return _http_json(f"{host.rstrip('/')}/api/tags", timeout=10.0).get("models", [])


class OllamaEncoder:
    """Embeddings from a local Ollama server. Ollama pools and normalizes; depth is not selectable."""

    kind = "ollama"
    runtime_id = "ollama"
    dtype = None
    early_exit = "not_used"

    def __init__(self, model: str, *, host: str = "http://localhost:11434", digest_pin: Optional[str] = None,
                 batch_size: int = 32, prefix_template: str = DEFAULT_PREFIX_TEMPLATE, keep_alive: str = "5m"):
        self.model, self.host, self.batch_size = model, host.rstrip("/"), int(batch_size)
        self.prefix_template, self.keep_alive = prefix_template, keep_alive
        tags = {m["name"]: m for m in ollama_models(self.host)}
        entry = tags.get(model)
        if entry is None:
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE,
                                        f"Ollama at {self.host} has no model {model!r}. Installed: {', '.join(sorted(tags))}")
        if "embedding" not in entry.get("capabilities", ["embedding"]):
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE, f"Ollama model {model!r} does not produce embeddings")
        self.model_digest = entry["digest"]
        if digest_pin is not None and digest_pin != self.model_digest:
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE,
                                        f"Ollama model {model!r} has digest {self.model_digest}, not the pinned {digest_pin}")
        self.pinned = digest_pin is not None
        version = _http_json(f"{self.host}/api/version", timeout=10.0)
        self.runtime_version = str(version.get("version", "unknown"))

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind, "input_version": INPUT_VERSION, "model": self.model, "revision": self.model_digest,
                "prefix_template": self.prefix_template, "pooling": "provider", "truncate": "provider"}

    def compute_settings(self) -> dict[str, Any]:
        return {"host": self.host, "batch_size": self.batch_size}

    def model_identity(self) -> dict[str, Any]:
        return {"id": f"ollama/{self.model}", "revision": self.model_digest,
                "revision_status": "operator_attested" if self.pinned else "unknown", "tokenizer_revision": None}

    def count_tokens(self, prefix: str, body: str) -> Optional[int]:
        return None

    def encode(self, items: Sequence[tuple[str, str]]) -> tuple[np.ndarray, EncodeStats]:
        stats = EncodeStats(items=len(items))
        vecs = []
        t0 = time.perf_counter()
        for s in range(0, len(items), self.batch_size):
            chunk = [p + b for p, b in items[s: s + self.batch_size]]
            out = _http_json(f"{self.host}/api/embed", {"model": self.model, "input": chunk, "truncate": True,
                                                        "keep_alive": self.keep_alive})
            emb = out.get("embeddings")
            if not isinstance(emb, list) or len(emb) != len(chunk):
                raise errors.AdapterError(errors.INTERNAL_ERROR, "Ollama returned the wrong number of embeddings")
            vecs.append(np.asarray(emb, dtype=np.float32))
            stats.forward_passes += 1
        stats.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return np.concatenate(vecs), stats


def encoder_digest(encoder: Encoder) -> str:
    return digest(encoder.identity())
