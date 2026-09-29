"""The fit criterion. Frozen by the owner on 2026-09-28 as version 1.

Changing any question or description text changes the criterion digest, so runs made
under different wording are never mixed in one report. A change needs a new version.
"""
from __future__ import annotations

from typing import Any

CRITERION_STATUS = "frozen"
CRITERION_VERSION = "1"

QUESTION = ("How well does this job listing fit the candidate, judged from the candidate evidence "
            "against the listing's core requirements, seniority, and stated work arrangement?")

# Exec Foundry rank -> (option id, description). Rank 1 is the strongest fit.
LEVELS: dict[int, tuple[str, str]] = {
    1: ("perfect", "Perfect Fit: the candidate evidence meets essentially every core requirement at the "
                   "stated seniority, and the role matches the candidate's direction and work preferences."),
    2: ("strong", "Strong Fit: the candidate evidence meets most core requirements with only minor gaps, "
                  "and the role is consistent with the candidate's direction."),
    3: ("good", "Good Fit: the candidate evidence meets several core requirements but shows clear gaps in "
                "experience, domain, or seniority that an employer would notice."),
    4: ("stretch", "Stretch: the candidate evidence meets only a few core requirements; the role needs "
                   "significant experience, skills, or seniority the evidence does not show."),
    5: ("long_shot", "Long Shot: the candidate evidence meets few or none of the core requirements, or the "
                     "role conflicts with the candidate's stated direction or work preferences."),
}
RANK_BY_OPTION = {option: rank for rank, (option, _) in LEVELS.items()}
CANDIDATE_EVIDENCE_ID = "candidate-evidence"


def fit_criterion() -> dict[str, Any]:
    return {
        "id": "fit-rank",
        "version": CRITERION_VERSION,
        "kind": "choice",
        "question": QUESTION,
        # No candidate material means a deterministic abstention, not a middle rank.
        "required_evidence_ids": [CANDIDATE_EVIDENCE_ID],
        "accept_outcomes": [],
        "options": [{"id": option, "description": text} for _, (option, text) in sorted(LEVELS.items())],
    }
