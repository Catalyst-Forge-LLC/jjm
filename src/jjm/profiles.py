"""Allowlisted deployment profiles.

A request names a profile ID. It never carries model paths, URLs, or settings.
`synthetic` is built in. Real profiles come from an operator file passed with
`--profiles` or the `JJM_PROFILES` environment variable.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from jjm import errors
from jjm.adapters.base import Adapter
from jjm.contract import MAX_REQUEST_BYTES
from jjm.digest import digest

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
PROFILES_ENV = "JJM_PROFILES"
SYNTHETIC_PROFILE_ID = "synthetic"


@dataclass
class Profile:
    id: str
    generation: str
    execution_mode: str
    adapter_name: str
    config: dict[str, Any]
    _factory: Callable[[], Adapter] = field(repr=False)
    _adapter: Optional[Adapter] = field(default=None, repr=False)

    def adapter(self) -> Adapter:
        if self._adapter is None:
            self._adapter = self._factory()
        return self._adapter


def _synthetic_factory() -> Adapter:
    from jjm.adapters.synthetic import SyntheticAdapter

    return SyntheticAdapter()


def synthetic_profile() -> Profile:
    return Profile(SYNTHETIC_PROFILE_ID, "builtin-1", "synthetic", "synthetic", {"adapter": "synthetic"},
                   _synthetic_factory)


def _config_error(message: str) -> errors.RequestError:
    return errors.RequestError(errors.INVALID_REQUEST, f"Profile configuration: {message}")


def _hf_config(entry: dict[str, Any]) -> dict[str, Any]:
    allowed = {"id", "adapter", "model", "device", "dtype", "batch_size", "max_context_tokens", "allow_download",
               "shared_prefix"}
    extra = sorted(set(entry) - allowed)
    if extra:
        raise _config_error(f"profile {entry.get('id')!r} has unknown keys: {', '.join(extra)}")
    model = entry.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not ID_PATTERN.match(model["id"]):
        raise _config_error(f"profile {entry['id']!r} needs model.id like 'Qwen/Qwen3-1.7B'")
    extra = sorted(set(model) - {"id", "path", "revision"})
    if extra:
        raise _config_error(f"profile {entry['id']!r} model has unknown keys: {', '.join(extra)}")
    revision = model.get("revision")
    if revision is not None and not (isinstance(revision, str) and revision):
        raise _config_error(f"profile {entry['id']!r} model.revision must be a non-empty string or null")
    path = model.get("path")
    if path is not None and not isinstance(path, str):
        raise _config_error(f"profile {entry['id']!r} model.path must be a string or null")
    max_ctx = entry.get("max_context_tokens")
    if not (isinstance(max_ctx, int) and max_ctx > 0):
        raise _config_error(f"profile {entry['id']!r} needs max_context_tokens as a positive integer")
    shared_prefix = entry.get("shared_prefix", "auto")
    if shared_prefix not in ("auto", "off"):
        raise _config_error(f"profile {entry['id']!r} shared_prefix must be 'auto' or 'off'")
    return {
        "shared_prefix": shared_prefix,
        "adapter": "anyjev-hf",
        "model": {"id": model["id"], "path": path, "revision": revision},
        "device": str(entry.get("device", "cuda")),
        "dtype": str(entry.get("dtype", "bfloat16")),
        "batch_size": int(entry.get("batch_size", 8)),
        "max_context_tokens": max_ctx,
        "allow_download": bool(entry.get("allow_download", False)),
    }


def _hf_factory(cfg: dict[str, Any]) -> Callable[[], Adapter]:
    def build() -> Adapter:
        if not cfg["allow_download"]:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import transformers
            from anyjev.backends.hf import HFBackend
        except ImportError as exc:
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE,
                                      f"The Transformers route needs the 'hf' extra (torch, transformers): {exc}"
                                      ) from None
        from jjm.adapters.anyjev_adapter import AnyJevAdapter

        model = cfg["model"]
        source = model["path"] or model["id"]
        try:
            backend = HFBackend(source, device=cfg["device"], dtype=cfg["dtype"], batch_size=cfg["batch_size"],
                                trust_remote_code=False, revision=model["revision"])
        except OSError as exc:
            hint = "" if cfg["allow_download"] else " Downloads are off for this profile (allow_download: false)."
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE,
                                      f"Could not load model {model['id']!r} from {source!r}: {exc}.{hint}") from None
        revision_status = "operator_attested" if model["revision"] else "unknown"
        identity = {"id": model["id"], "revision": model["revision"], "revision_status": revision_status,
                    "tokenizer_revision": model["revision"]}
        return AnyJevAdapter(backend, execution_mode="real", model=identity, runtime_id="transformers",
                             runtime_version=str(transformers.__version__), dtype=cfg["dtype"],
                             max_context_tokens=cfg["max_context_tokens"], shared_prefix=cfg["shared_prefix"])

    return build


HEAD_ENCODER_KINDS = ("hf", "ollama")


def _model_ref(pid: str, model: Any) -> dict[str, Any]:
    if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not ID_PATTERN.match(model["id"]):
        raise _config_error(f"profile {pid!r} needs encoder.model.id like 'Qwen/Qwen3-Reranker-0.6B'")
    extra = sorted(set(model) - {"id", "path", "revision"})
    if extra:
        raise _config_error(f"profile {pid!r} encoder.model has unknown keys: {', '.join(extra)}")
    return {"id": model["id"], "path": model.get("path"), "revision": model.get("revision")}


def _head_config(entry: dict[str, Any], base: Path) -> dict[str, Any]:
    pid = entry["id"]
    extra = sorted(set(entry) - {"id", "adapter", "encoder", "artifacts", "cache", "max_context_tokens",
                                 "allow_download"})
    if extra:
        raise _config_error(f"profile {pid!r} has unknown keys: {', '.join(extra)}")
    enc = entry.get("encoder")
    if not isinstance(enc, dict) or enc.get("kind") not in HEAD_ENCODER_KINDS:
        raise _config_error(f"profile {pid!r} needs encoder.kind, one of {HEAD_ENCODER_KINDS}")
    if not isinstance(entry.get("artifacts"), str):
        raise _config_error(f"profile {pid!r} needs 'artifacts', the head store directory")
    from jjm.heads.encoders import DEFAULT_PREFIX_TEMPLATE

    template = enc.get("prefix_template", DEFAULT_PREFIX_TEMPLATE)
    if not isinstance(template, str) or "{question}" not in template:
        raise _config_error(f"profile {pid!r} encoder.prefix_template must contain {{question}}")
    if enc["kind"] == "hf":
        allowed = {"kind", "model", "device", "dtype", "layer", "pooling", "max_tokens", "batch_size", "prefix_reuse",
                   "prefix_template"}
        extra = sorted(set(enc) - allowed)
        if extra:
            raise _config_error(f"profile {pid!r} encoder has unknown keys: {', '.join(extra)}")
        layer = enc.get("layer")
        if isinstance(layer, bool) or (layer is not None and not isinstance(layer, (int, float))):
            raise _config_error(f"profile {pid!r} encoder.layer must be a layer count, a depth fraction, or null")
        encoder = {"kind": "hf", "model": _model_ref(pid, enc.get("model")), "device": str(enc.get("device", "cuda")),
                   "dtype": str(enc.get("dtype", "bfloat16")), "layer": layer,
                   "pooling": str(enc.get("pooling", "mean")), "max_tokens": int(enc.get("max_tokens", 512)),
                   "batch_size": int(enc.get("batch_size", 16)), "prefix_reuse": str(enc.get("prefix_reuse", "off")),
                   "prefix_template": template}
    else:
        allowed = {"kind", "model", "host", "digest", "batch_size", "prefix_template", "keep_alive"}
        extra = sorted(set(enc) - allowed)
        if extra:
            raise _config_error(f"profile {pid!r} encoder has unknown keys: {', '.join(extra)}")
        if not isinstance(enc.get("model"), str) or not enc["model"]:
            raise _config_error(f"profile {pid!r} needs encoder.model, an Ollama model name")
        encoder = {"kind": "ollama", "model": enc["model"], "host": str(enc.get("host", "http://localhost:11434")),
                   "digest": enc.get("digest"), "batch_size": int(enc.get("batch_size", 32)),
                   "prefix_template": template, "keep_alive": str(enc.get("keep_alive", "5m"))}
    max_ctx = entry.get("max_context_tokens")
    if max_ctx is not None and not (isinstance(max_ctx, int) and max_ctx > 0):
        raise _config_error(f"profile {pid!r} max_context_tokens must be a positive integer or null")
    cache = entry.get("cache")
    return {"adapter": "fitted-head", "encoder": encoder, "artifacts": str((base / entry["artifacts"]).resolve()),
            "cache": str((base / cache).resolve()) if isinstance(cache, str) else None, "max_context_tokens": max_ctx,
            "allow_download": bool(entry.get("allow_download", False))}


def build_encoder(cfg: dict[str, Any], allow_download: bool = False):
    enc = cfg
    if enc["kind"] == "ollama":
        from jjm.heads.encoders import OllamaEncoder

        return OllamaEncoder(enc["model"], host=enc["host"], digest_pin=enc["digest"], batch_size=enc["batch_size"],
                             prefix_template=enc["prefix_template"], keep_alive=enc["keep_alive"])
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from jjm.heads.encoders import HFEncoder
    except ImportError as exc:
        raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE, f"The Transformers encoder needs the 'hf' extra: {exc}"
                                    ) from None
    model = enc["model"]
    try:
        return HFEncoder(model["path"] or model["id"], model_id=model["id"], revision=model["revision"],
                         device=enc["device"], dtype=enc["dtype"], layer=enc["layer"], pooling=enc["pooling"],
                         max_tokens=enc["max_tokens"], batch_size=enc["batch_size"], prefix_reuse=enc["prefix_reuse"],
                         prefix_template=enc["prefix_template"])
    except (OSError, ImportError) as exc:
        hint = "" if allow_download else " Downloads are off for this profile (allow_download: false)."
        raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE, f"Could not load encoder {model['id']!r}: {exc}.{hint}"
                                    ) from None


def _head_factory(cfg: dict[str, Any]) -> Callable[[], Adapter]:
    def build() -> Adapter:
        from jjm.adapters.head_adapter import FittedHeadAdapter
        from jjm.heads.artifact import ArtifactStore
        from jjm.heads.runtime import Featurizer

        encoder = build_encoder(cfg["encoder"], cfg["allow_download"])
        featurizer = Featurizer(encoder, Path(cfg["cache"]) if cfg["cache"] else None)
        return FittedHeadAdapter(featurizer, ArtifactStore(Path(cfg["artifacts"])),
                                 max_context_tokens=cfg["max_context_tokens"])

    return build


def _tfidf_config(entry: dict[str, Any], base: Path) -> dict[str, Any]:
    allowed = {"id", "adapter", "artifacts", "ngram_range", "min_df", "max_features", "C", "max_iter"}
    if set(entry) - allowed:
        raise _config_error(f"TF-IDF has unknown keys: {sorted(set(entry) - allowed)}")
    if not isinstance(entry.get("artifacts"), str) or not entry["artifacts"]:
        raise _config_error("TF-IDF needs an artifacts directory")
    ngram = entry.get("ngram_range", [1, 2])
    if (not isinstance(ngram, list) or len(ngram) != 2 or any(type(v) is not int for v in ngram)
            or not 1 <= ngram[0] <= ngram[1] <= 3):
        raise _config_error("TF-IDF ngram_range must be two integers with 1 <= min <= max <= 3")
    settings = {"ngram_range": ngram, "min_df": entry.get("min_df", 1),
                "max_features": entry.get("max_features"), "C": entry.get("C", 4.0),
                "max_iter": entry.get("max_iter", 1000)}
    for name in ("min_df", "max_iter", "max_features"):
        value = settings[name]
        if name == "max_features" and value is None:
            continue
        if type(value) is not int or value < 1:
            raise _config_error(f"TF-IDF {name} must be a positive integer")
    value = settings["C"]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise _config_error("TF-IDF C must be finite and positive")
    return {"adapter": "tfidf", "artifacts": str((base / entry["artifacts"]).resolve()), "settings": settings}


def _tfidf_factory(cfg: dict[str, Any]) -> Callable[[], Adapter]:
    def build() -> Adapter:
        try:
            from jjm.adapters.tfidf_adapter import TfidfAdapter
        except ImportError:
            raise errors.ExecutionError(errors.BACKEND_UNAVAILABLE,
                                        "TF-IDF needs the 'text' extra: uv sync --extra text") from None
        from jjm.heads.artifact import ArtifactStore
        return TfidfAdapter(ArtifactStore(Path(cfg["artifacts"])), cfg["settings"])
    return build


def _read_file(path: Optional[str]) -> tuple[Optional[dict[str, Any]], Path]:
    path = path or os.environ.get(PROFILES_ENV)
    if not path:
        return None, Path.cwd()
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _config_error(f"file not found: {file}") from None
    except json.JSONDecodeError as exc:
        raise _config_error(f"{file} is not valid JSON: {exc.msg} at line {exc.lineno}") from None
    if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), list):
        raise _config_error(f"{file} must be an object with a 'profiles' list")
    return raw, file.resolve().parent


def load_profiles(path: Optional[str] = None) -> dict[str, Profile]:
    profiles = {SYNTHETIC_PROFILE_ID: synthetic_profile()}
    raw, base = _read_file(path)
    if raw is None:
        return profiles
    for entry in raw["profiles"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not ID_PATTERN.match(entry["id"]):
            raise _config_error("every profile needs an 'id' made of letters, digits, and _ . : / -")
        pid = entry["id"]
        if pid in profiles:
            raise _config_error(f"profile {pid!r} is defined twice or shadows a built-in profile")
        adapter = entry.get("adapter")
        if adapter == "anyjev-hf":
            cfg = _hf_config(entry)
            factory = _hf_factory(cfg)
        elif adapter == "fitted-head":
            cfg = _head_config(entry, base)
            factory = _head_factory(cfg)
        elif adapter == "tfidf":
            cfg = _tfidf_config(entry, base)
            factory = _tfidf_factory(cfg)
        else:
            raise _config_error(f"profile {pid!r} has adapter {adapter!r}; supported: 'anyjev-hf', 'fitted-head', 'tfidf'")
        generation = "cfg-" + digest(cfg)[7:19]
        profiles[pid] = Profile(pid, generation, "real", adapter, cfg, factory)
    return profiles


def load_teachers(path: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Teacher entries from the same operator file. Only fitting and escalation use them, never `evaluate`."""
    raw, _ = _read_file(path)
    out: dict[str, dict[str, Any]] = {}
    for entry in (raw or {}).get("teachers", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not ID_PATTERN.match(entry["id"]):
            raise _config_error("every teacher needs an 'id' made of letters, digits, and _ . : / -")
        if entry.get("provider") not in ("ollama", "litellm") or not isinstance(entry.get("model"), str):
            raise _config_error(f"teacher {entry['id']!r} needs provider 'ollama' or 'litellm' and a model name")
        if entry["id"] in out:
            raise _config_error(f"teacher {entry['id']!r} is defined twice")
        out[entry["id"]] = entry
    return out


def capabilities(profile: Profile) -> dict[str, Any]:
    """Descriptive capability record (schema 0.1). Does not load a model."""
    if profile.adapter_name == "synthetic":
        methods, max_ctx, semantics = ["mock:normalized"], None, ["normalized_readout"]
    elif profile.adapter_name == "fitted-head":
        from jjm.adapters.head_adapter import METHOD as HEAD_METHOD

        methods, max_ctx = [HEAD_METHOD], profile.config["max_context_tokens"]
        semantics = ["normalized_readout"]
    elif profile.adapter_name == "tfidf":
        methods, max_ctx, semantics = ["jjm:tfidf-logistic"], None, ["normalized_readout"]
    else:
        from jjm.adapters.anyjev_adapter import METHOD

        methods, max_ctx = [METHOD], profile.config["max_context_tokens"]
        semantics = ["normalized_readout"]
    return {
        "schema_version": "0.1",
        "profile_id": profile.id,
        "execution_mode": profile.execution_mode,
        "supported_kinds": ["choice"] if profile.adapter_name == "tfidf" else ["choice", "assertion"],
        "methods": methods,
        "max_choice_options": 26,
        "max_score_levels": 0,
        "input_forms": ["json"],
        "number_semantics": semantics,
        "evidence_selection": False,
        "can_verify_model_revision": False,
        "persistent_state_handles": False,
        "max_request_bytes": MAX_REQUEST_BYTES,
        "max_criteria": 32,
        "max_context_tokens": max_ctx,
    }
