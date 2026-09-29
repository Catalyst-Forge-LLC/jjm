"""Allowlisted deployment profiles.

A request names a profile ID. It never carries model paths, URLs, or settings.
`synthetic` is built in. Real profiles come from an operator file passed with
`--profiles` or the `JJM_PROFILES` environment variable.
"""
from __future__ import annotations

import json
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
    allowed = {"id", "adapter", "model", "device", "dtype", "batch_size", "max_context_tokens", "allow_download"}
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
    return {
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
                             max_context_tokens=cfg["max_context_tokens"])

    return build


def load_profiles(path: Optional[str] = None) -> dict[str, Profile]:
    profiles = {SYNTHETIC_PROFILE_ID: synthetic_profile()}
    path = path or os.environ.get(PROFILES_ENV)
    if not path:
        return profiles
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _config_error(f"file not found: {file}") from None
    except json.JSONDecodeError as exc:
        raise _config_error(f"{file} is not valid JSON: {exc.msg} at line {exc.lineno}") from None
    if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), list):
        raise _config_error(f"{file} must be an object with a 'profiles' list")
    for entry in raw["profiles"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not ID_PATTERN.match(entry["id"]):
            raise _config_error("every profile needs an 'id' made of letters, digits, and _ . : / -")
        pid = entry["id"]
        if pid in profiles:
            raise _config_error(f"profile {pid!r} is defined twice or shadows a built-in profile")
        if entry.get("adapter") != "anyjev-hf":
            raise _config_error(f"profile {pid!r} has adapter {entry.get('adapter')!r}; supported: 'anyjev-hf'")
        cfg = _hf_config(entry)
        generation = "cfg-" + digest(cfg)[7:19]
        profiles[pid] = Profile(pid, generation, "real", "anyjev-hf", cfg, _hf_factory(cfg))
    return profiles


def capabilities(profile: Profile) -> dict[str, Any]:
    """Descriptive capability record (schema 0.1). Does not load a model."""
    if profile.adapter_name == "synthetic":
        methods, max_ctx = ["mock:normalized"], None
    else:
        from jjm.adapters.anyjev_adapter import METHOD

        methods, max_ctx = [METHOD], profile.config["max_context_tokens"]
    return {
        "schema_version": "0.1",
        "profile_id": profile.id,
        "execution_mode": profile.execution_mode,
        "supported_kinds": ["choice", "assertion"],
        "methods": methods,
        "max_choice_options": 26,
        "max_score_levels": 0,
        "input_forms": ["json"],
        "number_semantics": ["normalized_readout"],
        "evidence_selection": False,
        "can_verify_model_revision": False,
        "persistent_state_handles": False,
        "max_request_bytes": MAX_REQUEST_BYTES,
        "max_criteria": 32,
        "max_context_tokens": max_ctx,
    }
