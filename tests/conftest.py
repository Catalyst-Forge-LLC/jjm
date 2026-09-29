from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from jjm.profiles import Profile, load_profiles

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "specs" / "jjm-spec-pack"
EXAMPLES = PACK / "examples"
FIXED_NOW = datetime(2026, 9, 25, 18, 0, 1, tzinfo=timezone.utc)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_cases() -> list[dict[str, Any]]:
    return load_json(EXAMPLES / "index.json")["cases"]


def fixture_request(case_id: str, profile_id: str = "synthetic") -> dict[str, Any]:
    case = next(c for c in fixture_cases() if c["id"] == case_id)
    request = load_json(PACK / case["request"])
    request["profile_id"] = profile_id
    return request


def route_request(**overrides: Any) -> dict[str, Any]:
    request = fixture_request("01-routing")
    request["requirements"]["min_top_mass"] = None
    for key, value in overrides.items():
        request[key] = value
    return request


def fixed_clock() -> datetime:
    return FIXED_NOW


def profile_with(adapter: Any, profile_id: str = "test-profile") -> dict[str, Profile]:
    profiles = load_profiles(None)
    profiles[profile_id] = Profile(profile_id, "test-1", adapter.execution_mode, "test", {}, lambda: adapter)
    return profiles


@pytest.fixture
def profiles() -> dict[str, Profile]:
    return load_profiles(None)


@pytest.fixture
def clone():
    return copy.deepcopy
