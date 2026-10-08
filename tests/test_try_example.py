"""The public try-it files parse. No model download."""
from __future__ import annotations

import json
from pathlib import Path

from jjm.compile import compile_criterion
from jjm.profiles import load_profiles

ROOT = Path("examples/try")


def test_try_example_has_two_teams_and_enough_labels():
    criterion = json.loads((ROOT / "team.criterion.json").read_text(encoding="utf-8"))
    compiled = compile_criterion(criterion)
    assert list(compiled.outcome_ids) == ["billing", "technical"]
    labels = []
    for line in (ROOT / "examples.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert "content" in row["state"]
        labels.append(row["label"])
    assert labels.count("billing") >= 2
    assert labels.count("technical") >= 2
    request = json.loads((ROOT / "request.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "profile.json").read_text(encoding="utf-8"))
    assert request["profile_id"] == profile["profiles"][0]["id"]
    assert request["criteria"][0]["options"] == criterion["options"]
    assert profile["profiles"][0]["allow_download"] is True
    loaded = load_profiles(str(ROOT / "profile.json"))
    assert "try-embed" in loaded
