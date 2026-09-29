"""Small heads over frozen features. numpy only, deterministic: fixed folds, zero initialization.

softmax: multinomial logistic regression on standardized features, for unordered outcomes.
ordinal: ridge regression onto the option index, turned into weights by a normal around the
prediction with the cross-validated residual spread. For outcomes whose order means something.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

METHODS = ("softmax", "ordinal")
SOFTMAX_L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
RIDGE_ALPHA_GRID = tuple(float(a) for a in np.logspace(-1, 5, 13))
ITERATIONS = 400
LEARNING_RATE = 0.05
FOLDS = 5
SEED = 0


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(0), X.std(0) + 1e-6


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def folds(y: np.ndarray, k: int = FOLDS, seed: int = SEED) -> list[np.ndarray]:
    """Stratified folds: each outcome's examples are dealt round-robin after a seeded shuffle."""
    rng = np.random.default_rng(seed)
    out: list[list[int]] = [[] for _ in range(k)]
    slot = 0
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        for i in idx:
            out[slot % k].append(int(i))
            slot += 1
    return [np.array(sorted(f), dtype=int) for f in out]


def _fit_softmax(Z: np.ndarray, y: np.ndarray, k: int, l2: float) -> tuple[np.ndarray, np.ndarray]:
    n, d = Z.shape
    Y = np.eye(k)[y]
    W, b = np.zeros((d, k)), np.zeros(k)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, ITERATIONS + 1):
        G = (_softmax(Z @ W + b) - Y) / n
        gW, gb = Z.T @ G + l2 * W, G.sum(0)
        mW, vW = b1 * mW + (1 - b1) * gW, b2 * vW + (1 - b2) * gW * gW
        mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb * gb
        c1, c2 = 1 - b1 ** t, 1 - b2 ** t
        W -= LEARNING_RATE * (mW / c1) / (np.sqrt(vW / c2) + eps)
        b -= LEARNING_RATE * (mb / c1) / (np.sqrt(vb / c2) + eps)
    return W, b


def _fit_ridge(Z: np.ndarray, t: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    n, d = Z.shape
    tm = float(t.mean())
    if n <= d:
        coef = np.linalg.solve(Z @ Z.T + alpha * np.eye(n), t - tm)
        return Z.T @ coef, tm
    return np.linalg.solve(Z.T @ Z + alpha * np.eye(d), Z.T @ (t - tm)), tm


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def ordinal_weights(pred: np.ndarray, sigma: float, k: int) -> np.ndarray:
    """Probability mass of N(pred, sigma) in each index bin; the outer bins are open-ended."""
    edges = np.arange(k - 1) + 0.5
    cdf = _normal_cdf((edges[None, :] - pred[:, None]) / sigma)
    lower = np.concatenate([np.zeros((len(pred), 1)), cdf], axis=1)
    upper = np.concatenate([cdf, np.ones((len(pred), 1))], axis=1)
    w = np.clip(upper - lower, 0.0, 1.0)
    return w / w.sum(1, keepdims=True)


@dataclass
class Head:
    method: str
    k: int
    mu: np.ndarray
    sd: np.ndarray
    W: np.ndarray            # softmax: (d, k); ordinal: (d, 1)
    b: np.ndarray            # softmax: (k,); ordinal: (1,)
    sigma: float | None      # ordinal only
    setting: float           # the L2 strength or ridge alpha chosen by cross-validation

    def weights(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        if self.method == "softmax":
            return _softmax(Z @ self.W + self.b)
        return ordinal_weights(Z @ self.W[:, 0] + self.b[0], float(self.sigma), self.k)

    def payload(self) -> dict[str, Any]:
        return {"method": self.method, "k": self.k, "setting": self.setting, "sigma": self.sigma,
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "W": self.W.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "Head":
        return cls(p["method"], int(p["k"]), np.array(p["mu"]), np.array(p["sd"]), np.array(p["W"]),
                   np.array(p["b"]), p["sigma"], float(p["setting"]))


def _train(method: str, X: np.ndarray, y: np.ndarray, k: int, setting: float, sigma: float | None) -> Head:
    mu, sd = _standardize(X)
    Z = (X - mu) / sd
    if method == "softmax":
        W, b = _fit_softmax(Z, y, k, setting)
        return Head(method, k, mu, sd, W, b, None, setting)
    w, b0 = _fit_ridge(Z, y.astype(np.float64), setting)
    return Head(method, k, mu, sd, w[:, None], np.array([b0]), sigma, setting)


def cross_validate(method: str, X: np.ndarray, y: np.ndarray, k: int) -> dict[str, Any]:
    """Pick the regularization by fold log loss (softmax) or squared error (ordinal).

    Returns the chosen setting and the out-of-fold weights for it. Because those folds also chose
    the setting, the report is a selection estimate, not a held-out evaluation.
    """
    if method not in METHODS:
        raise ValueError(f"head method must be one of {METHODS}")
    parts = folds(y)
    grid = SOFTMAX_L2_GRID if method == "softmax" else RIDGE_ALPHA_GRID
    best: tuple[float, float, np.ndarray] | None = None
    for setting in grid:
        oof = np.zeros((len(y), k)) if method == "softmax" else np.zeros(len(y))
        for test in parts:
            train = np.setdiff1d(np.arange(len(y)), test)
            if len(np.unique(y[train])) < 2:
                continue
            h = _train(method, X[train], y[train], k, setting, 1.0)
            if method == "softmax":
                oof[test] = h.weights(X[test])
            else:
                Z = (X[test] - h.mu) / h.sd
                oof[test] = Z @ h.W[:, 0] + h.b[0]
        if method == "softmax":
            loss = float(-np.mean(np.log(np.clip(oof[np.arange(len(y)), y], 1e-12, None))))
        else:
            loss = float(np.mean((oof - y) ** 2))
        if best is None or loss < best[0]:
            best = (loss, setting, oof)
    assert best is not None
    loss, setting, oof = best
    sigma = None
    if method == "ordinal":
        sigma = max(float(np.sqrt(loss)), 1e-3)
        oof = ordinal_weights(oof, sigma, k)
    return {"setting": setting, "sigma": sigma, "oof_weights": oof, "loss": loss}


def fit(method: str, X: np.ndarray, y: np.ndarray, k: int) -> tuple[Head, dict[str, Any]]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    cv = cross_validate(method, X, y, k)
    head = _train(method, X, y, k, cv["setting"], cv["sigma"])
    return head, cv


def classification_report(weights: np.ndarray, y: np.ndarray, k: int) -> dict[str, Any]:
    pred = weights.argmax(1)
    f1s = []
    for c in range(k):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "log_loss": float(-np.mean(np.log(np.clip(weights[np.arange(len(y)), y], 1e-12, None)))),
        "mean_top_mass": float(weights.max(1).mean()),
    }
