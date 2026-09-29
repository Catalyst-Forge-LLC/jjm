"""Agreement between two 1-5 rank lists. Pure functions, seeded bootstrap."""
from __future__ import annotations

import math
import random
from typing import Callable, Optional, Sequence

RANKS = (1, 2, 3, 4, 5)


def confusion(predicted: Sequence[int], reference: Sequence[int]) -> list[list[int]]:
    """rows = reference rank 1..5, columns = predicted rank 1..5"""
    matrix = [[0] * 5 for _ in RANKS]
    for p, r in zip(predicted, reference):
        matrix[r - 1][p - 1] += 1
    return matrix


def exact(predicted: Sequence[int], reference: Sequence[int]) -> float:
    return sum(p == r for p, r in zip(predicted, reference)) / len(reference)


def within_one(predicted: Sequence[int], reference: Sequence[int]) -> float:
    return sum(abs(p - r) <= 1 for p, r in zip(predicted, reference)) / len(reference)


def mean_signed_offset(predicted: Sequence[int], reference: Sequence[int]) -> float:
    """Positive means JJM ranks jobs as weaker fits than the reference does."""
    return sum(p - r for p, r in zip(predicted, reference)) / len(reference)


def quadratic_weighted_kappa(predicted: Sequence[int], reference: Sequence[int]) -> Optional[float]:
    n = len(reference)
    observed = confusion(predicted, reference)
    ref_hist = [sum(row) for row in observed]
    pred_hist = [sum(observed[i][j] for i in range(5)) for j in range(5)]
    num = den = 0.0
    for i in range(5):
        for j in range(5):
            weight = ((i - j) ** 2) / 16.0
            num += weight * observed[i][j]
            den += weight * ref_hist[i] * pred_hist[j] / n
    if den == 0:
        return None
    return 1.0 - num / den


def bootstrap_ci(metric: Callable[[Sequence[int], Sequence[int]], Optional[float]], predicted: Sequence[int],
                 reference: Sequence[int], *, resamples: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Optional[tuple[float, float]]:
    n = len(reference)
    if n < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        value = metric([predicted[i] for i in idx], [reference[i] for i in idx])
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return None
    values.sort()
    lo = values[int(math.floor(alpha / 2 * (len(values) - 1)))]
    hi = values[int(math.ceil((1 - alpha / 2) * (len(values) - 1)))]
    return (lo, hi)


def agreement(predicted: Sequence[int], reference: Sequence[int], *, seed: int = 0) -> dict:
    if not reference:
        return {"n": 0}
    out = {
        "n": len(reference),
        "exact": exact(predicted, reference),
        "within_one": within_one(predicted, reference),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(predicted, reference),
        "mean_signed_offset": mean_signed_offset(predicted, reference),
        "confusion_rows_reference_cols_predicted": confusion(predicted, reference),
    }
    for name, fn in (("exact", exact), ("within_one", within_one), ("quadratic_weighted_kappa",
                                                                    quadratic_weighted_kappa)):
        out[f"{name}_ci95"] = bootstrap_ci(fn, predicted, reference, seed=seed)
    return out


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
