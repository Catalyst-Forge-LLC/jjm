"""What models this machine can use, without loading or downloading any of them.

Hugging Face: repositories in the local cache, read from each snapshot's config.json.
Ollama: the models a running server reports, with the capabilities it declares.
Roles are suggestions from declared architecture or capability, not a quality claim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from jjm import errors


def _hf_role(repo: str, config: dict[str, Any]) -> list[str]:
    name = repo.lower()
    arch = " ".join(config.get("architectures") or [])
    if "rerank" in name:
        return ["encoder", "zero-shot relevance"]
    if "embed" in name:
        return ["encoder"]
    if "CausalLM" in arch or "ConditionalGeneration" in arch:
        return ["encoder", "readout"]
    if arch:
        return ["encoder"]
    return []


def hf_models(cache_dir: Optional[str] = None) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return []
    try:
        info = scan_cache_dir(cache_dir)
    except Exception:
        return []
    out = []
    for repo in sorted(info.repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model":
            continue
        for rev in sorted(repo.revisions, key=lambda r: r.commit_hash):
            config: dict[str, Any] = {}
            cfg = Path(rev.snapshot_path) / "config.json"
            if cfg.exists():
                try:
                    config = json.loads(cfg.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    config = {}
            out.append({"source": "huggingface", "id": repo.repo_id, "revision": rev.commit_hash,
                        "path": str(rev.snapshot_path), "size_bytes": int(rev.size_on_disk),
                        "architectures": config.get("architectures") or [],
                        "hidden_size": config.get("hidden_size"), "layers": config.get("num_hidden_layers"),
                        "roles": _hf_role(repo.repo_id, config)})
    return out


def ollama_list(host: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    from jjm.heads.encoders import ollama_models

    try:
        models = ollama_models(host)
    except errors.JJMError as exc:
        return [], exc.message
    out = []
    for m in sorted(models, key=lambda m: m["name"]):
        caps = m.get("capabilities") or []
        roles = (["encoder"] if "embedding" in caps else []) + (["teacher"] if "completion" in caps else [])
        details = m.get("details") or {}
        out.append({"source": "ollama", "id": m["name"], "revision": m.get("digest"), "size_bytes": m.get("size"),
                    "parameters": details.get("parameter_size"), "quantization": details.get("quantization_level"),
                    "context_length": details.get("context_length"), "capabilities": caps, "roles": roles})
    return out, None


def discover(*, ollama_host: str = "http://localhost:11434", hf_cache: Optional[str] = None) -> dict[str, Any]:
    ollama, problem = ollama_list(ollama_host)
    return {"huggingface": hf_models(hf_cache), "ollama": ollama,
            "ollama_host": ollama_host, "ollama_problem": problem,
            "remote_teachers": "Any LiteLLM model string, configured under 'teachers' with provider 'litellm'."}
