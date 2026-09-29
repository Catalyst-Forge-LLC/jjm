"""Head artifacts: immutable, content-addressed, and bound to exactly what produced them.

The binding is the criterion digest, the encoder identity digest, and the head method. The
training digest covers the examples and their labels. A new fit with the same binding and the
same training digest reuses the stored artifact instead of refitting.

Promotion is a separate, explicit step: a profile serves only the artifact promoted for a
binding, never whichever fit happened last. Status is always `fitted_unvalidated` here, because
cross-validation used to choose the head is not a held-out validation (spec 08).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jjm import errors
from jjm.digest import digest

ARTIFACT_VERSION = "jjm-head-1"


def binding(criterion_digest: str, encoder_identity: dict[str, Any], method: str) -> dict[str, Any]:
    return {"artifact_version": ARTIFACT_VERSION, "criterion_digest": criterion_digest,
            "encoder": encoder_identity, "encoder_digest": digest(encoder_identity), "method": method}


def training_digest(examples: list[tuple[str, str]], label_source: dict[str, Any]) -> str:
    """Examples are (body text digest, outcome id) pairs in a canonical order."""
    return digest({"examples": sorted(examples), "label_source": label_source})


def _short(d: str, n: int = 24) -> str:
    return d.split(":", 1)[1][:n]


def build_artifact(*, bind: dict[str, Any], outcome_ids: list[str], training: dict[str, Any],
                   report: dict[str, Any], payload: dict[str, Any], created_at: Optional[str] = None) -> dict[str, Any]:
    payload_digest = digest(payload)
    art = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_id": "head-" + _short(payload_digest, 20),
        "binding": bind,
        "binding_digest": digest(bind),
        "outcome_ids": list(outcome_ids),
        "training": training,
        "report": report,
        "validation_status": "fitted_unvalidated",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "payload_digest": payload_digest,
        "payload": payload,
    }
    return art


class ArtifactStore:
    """<root>/<binding>/<artifact_id>.json, plus <root>/<binding>/promoted.json naming the served artifact."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, binding_digest: str) -> Path:
        return self.root / _short(binding_digest)

    def save(self, art: dict[str, Any]) -> Path:
        d = self._dir(art["binding_digest"])
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{art['artifact_id']}.json"
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(art, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        return path

    def load(self, binding_digest: str, artifact_id: str) -> dict[str, Any]:
        path = self._dir(binding_digest) / f"{artifact_id}.json"
        try:
            art = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise errors.AdapterError(errors.ARTIFACT_MISMATCH, f"Head artifact {artifact_id} is missing") from None
        if digest(art["payload"]) != art["payload_digest"] or digest(art["binding"]) != binding_digest:
            raise errors.AdapterError(errors.ARTIFACT_MISMATCH, f"Head artifact {artifact_id} failed its digest check")
        return art

    def find_training(self, binding_digest: str, train_digest: str) -> Optional[dict[str, Any]]:
        d = self._dir(binding_digest)
        if not d.is_dir():
            return None
        for path in sorted(d.glob("head-*.json")):
            art = json.loads(path.read_text(encoding="utf-8"))
            if art["training"]["training_digest"] == train_digest:
                return self.load(binding_digest, art["artifact_id"])
        return None

    def promote(self, art: dict[str, Any]) -> None:
        d = self._dir(art["binding_digest"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "promoted.tmp"
        tmp.write_text(json.dumps({"artifact_id": art["artifact_id"], "binding_digest": art["binding_digest"]}),
                       encoding="utf-8")
        os.replace(tmp, d / "promoted.json")

    def promoted(self, binding_digest: str) -> Optional[dict[str, Any]]:
        path = self._dir(binding_digest) / "promoted.json"
        if not path.exists():
            return None
        ref = json.loads(path.read_text(encoding="utf-8"))
        if ref.get("binding_digest") != binding_digest:
            raise errors.AdapterError(errors.ARTIFACT_MISMATCH, "Promoted head reference names another binding")
        return self.load(binding_digest, ref["artifact_id"])

    def list(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.is_dir():
            return out
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            promoted = None
            if (d / "promoted.json").exists():
                promoted = json.loads((d / "promoted.json").read_text(encoding="utf-8"))["artifact_id"]
            for path in sorted(d.glob("head-*.json")):
                art = json.loads(path.read_text(encoding="utf-8"))
                out.append({"artifact_id": art["artifact_id"], "binding_digest": art["binding_digest"],
                            "method": art["binding"]["method"], "encoder": art["binding"]["encoder"]["model"],
                            "criterion_digest": art["binding"]["criterion_digest"],
                            "examples": art["training"]["n"], "label_source": art["training"]["label_source"],
                            "cv": art["report"].get("cv"), "created_at": art["created_at"],
                            "promoted": art["artifact_id"] == promoted})
        return out
