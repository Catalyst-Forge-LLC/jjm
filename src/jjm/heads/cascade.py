"""Escalation outside the judgment: the head answers when it is sure enough, a teacher takes the rest.

JJM itself only holds a result whose top mass is under the request's `min_top_mass`. Sending
held cases to a teacher is the application's choice, so it lives here as a helper and never
inside `evaluate`, which has no hidden provider fallback.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def escalation_curve(top_mass: np.ndarray, head_correct: np.ndarray, teacher_correct: np.ndarray,
                     thresholds: Optional[list[float]] = None) -> list[dict[str, Any]]:
    """Accuracy and teacher share at each `min_top_mass`, when every held case goes to the teacher."""
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    out = []
    for t in thresholds:
        held = top_mass < t
        correct = np.where(held, teacher_correct, head_correct)
        out.append({"min_top_mass": t, "teacher_share": float(held.mean()), "accuracy": float(correct.mean()),
                    "head_accuracy_when_kept": float(head_correct[~held].mean()) if (~held).any() else None})
    return out


def threshold_for_accuracy(top_mass: np.ndarray, head_correct: np.ndarray, target: float) -> Optional[float]:
    """The lowest top-mass bar at which the kept cases reach the target accuracy on this data."""
    for t in np.unique(np.concatenate([[0.0], top_mass])):
        kept = top_mass >= t
        if kept.any() and head_correct[kept].mean() >= target:
            return float(t)
    return None
