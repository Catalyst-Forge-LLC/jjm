"""Fold assignment and the systemone response parser. No model and no export."""
from __future__ import annotations

import numpy as np

from trials.fit_rank.head_run import outer_folds
from trials.routing.systemone import parse_choice, question


def test_softmax_on_a_planted_rank_signal_recovers_the_level():
    from trials.fit_rank.probe_eval import cv_softmax, softmax_report

    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
    X = np.zeros((len(y), 3))
    X[:, 0] = y
    weights = cv_softmax(X, y, outer=4, repeats=1, k=5)
    assert weights[0].shape == (len(y), 5)
    assert np.allclose(weights[0].sum(1), 1.0)
    assert softmax_report("planted", weights, y, 0.0)["exact"] == 1.0


def test_small_outcomes_stay_in_every_fit_and_each_other_row_is_scored_once():
    labels = ["perfect"] * 10 + ["strong"] * 6 + ["long_shot"] * 2 + ["stretch"] * 3
    digests = [f"{i:03d}" for i in range(len(labels))]
    folds = outer_folds(labels, digests, folds=5, min_train=2)
    assert len(folds) == 5
    seen = []
    for train, test in folds:
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(range(len(labels)))
        for label in ("perfect", "strong", "long_shot", "stretch"):
            in_train = sum(1 for i in train if labels[i] == label)
            assert in_train >= 2
        seen.extend(test)
        assert all(labels[i] != "long_shot" for i in test)
    assert sorted(seen) == [i for i, label in enumerate(labels) if label != "long_shot"]
    assert len(seen) == len(set(seen))


def test_probe_head_reads_the_dedicated_prompt():
    import json
    from pathlib import Path

    from trials.fit_rank.head_run import state_texts

    record = json.loads(Path("tests/data/fit_export.sample.jsonl").read_text(encoding="utf-8").splitlines()[0])
    text = state_texts([record], "probe")[0]
    assert text.startswith("<Instruct>:")
    assert "sonnet" not in text.lower()
    assert "Director of Platform" in text


def test_fit_texts_carry_the_job_and_not_the_rank():
    import json
    from pathlib import Path

    from trials.fit_rank.same_text import probe_text, rendered_text

    record = json.loads(Path("tests/data/fit_export.sample.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for text in (probe_text(record), rendered_text(record)):
        assert "sonnet" not in text.lower()
        assert "Director of Platform" in text
        assert len(text) <= 48_000


def test_systemone_choice_parser_accepts_a_team_and_rejects_anything_else():
    body = question()
    assert list(body["criteria"])[0] == "cards"
    assert parse_choice({"answers": {"team": {"choice": "payments"}}}) == "payments"
    assert parse_choice({"answers": {"team": {"choice": "not-a-team"}}}) is None
    assert parse_choice({}) is None
    assert parse_choice({"answers": {"rank": {"choice": "strong"}}}, "rank", ["perfect", "strong"]) == "strong"
