"""SHA-256 over RFC 8785 canonical JSON (spec 06)."""
from __future__ import annotations

import hashlib
from typing import Any

import rfc8785


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def text_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
