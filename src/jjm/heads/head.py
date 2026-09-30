"""Small heads over frozen features. numpy only, deterministic: fixed folds, zero initialization.

softmax: multinomial logistic regression on standardized features, for unordered outcomes,
fit by L-BFGS from a zero start.
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
LBFGS_ITERATIONS = 30
LBFGS_HISTORY = 8
FOLDS = 5
SEED = 0
# The penalty is chosen on at most this many stratified rows. The final head still fits on every row.
SELECTION_ROWS = 2048


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


def _softmax_loss_grad(theta: np.ndarray, Z: np.ndarray, y: np.ndarray, k: int, l2: float
                       ) -> tuple[float, np.ndarray]:
    """Multinomial log loss plus L2 on the weights. The bias is not penalized."""
    n, d = Z.shape
    W = theta[: d * k].reshape(d, k)
    b = theta[d * k:]
    P = _softmax(Z @ W + b)
    G = (P - np.eye(k)[y]) / n
    loss = float(-np.log(np.clip(P[np.arange(n), y], 1e-12, None)).mean() + 0.5 * l2 * np.sum(W * W))
    grad = np.concatenate([(Z.T @ G + l2 * W).ravel(), G.sum(0)])
    return loss, grad


def _lbfgs(loss_grad, theta: np.ndarray) -> np.ndarray:
    """Limited-memory BFGS from a fixed start. Convex loss, so this is the same minimum Adam was walking toward."""
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho: list[float] = []
    loss, grad = loss_grad(theta)
    for _ in range(LBFGS_ITERATIONS):
        if np.linalg.norm(grad) <= 1e-8 * (1.0 + np.linalg.norm(theta)):
            break
        q = grad.copy()
        alpha = []
        for s, yv, r in zip(reversed(s_hist), reversed(y_hist), reversed(rho)):
            a = r * float(np.dot(s, q))
            alpha.append(a)
            q -= a * yv
        gamma = 1.0
        if y_hist:
            gamma = float(np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1]))
        direction = gamma * q
        for s, yv, r, a in zip(s_hist, y_hist, rho, reversed(alpha)):
            beta = r * float(np.dot(yv, direction))
            direction += s * (a - beta)
        direction *= -1.0
        slope = float(np.dot(grad, direction))
        if slope >= 0.0:
            direction = -grad
            slope = float(np.dot(grad, direction))
        step = 1.0
        accepted = None
        for _attempt in range(12):
            trial = theta + step * direction
            trial_loss, trial_grad = loss_grad(trial)
            if trial_loss <= loss + 1e-4 * step * slope:
                accepted = (trial, trial_loss, trial_grad)
                break
            step *= 0.5
        if accepted is None:
            break
        new_theta, new_loss, new_grad = accepted
        s = new_theta - theta
        yv = new_grad - grad
        ys = float(np.dot(yv, s))
        if ys > 1e-12:
            if len(s_hist) == LBFGS_HISTORY:
                s_hist.pop(0)
                y_hist.pop(0)
                rho.pop(0)
            s_hist.append(s)
            y_hist.append(yv)
            rho.append(1.0 / ys)
        theta, loss, grad = new_theta, new_loss, new_grad
    return theta


def _fit_softmax(Z: np.ndarray, y: np.ndarray, k: int, l2: float) -> tuple[np.ndarray, np.ndarray]:
    d = Z.shape[1]
    theta = _lbfgs(lambda t: _softmax_loss_grad(t, Z, y, k, l2), np.zeros(d * k + k))
    return theta[: d * k].reshape(d, k), theta[d * k:]


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


def _selection_index(y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Stratified rows for choosing the penalty. Every outcome that exists is kept."""
    if len(y) <= cap:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    raw = cap * counts / len(y)
    alloc = np.maximum(1, np.floor(raw).astype(int))
    while int(alloc.sum()) > cap:
        alloc[int(np.argmax(alloc))] -= 1
    order = np.argsort(-(raw - alloc))
    cursor = 0
    while int(alloc.sum()) < cap:
        alloc[order[cursor % len(order)]] += 1
        cursor += 1
    chosen: list[int] = []
    for label, take, count in zip(labels, alloc, counts):
        idx = np.flatnonzero(y == label)
        chosen.extend(rng.choice(idx, size=int(min(take, count)), replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)


def cross_validate(method: str, X: np.ndarray, y: np.ndarray, k: int) -> dict[str, Any]:
    """Pick the regularization by fold log loss (softmax) or squared error (ordinal).

    Returns the chosen setting and the out-of-fold weights for it. The folds also chose the
    setting, so the report is a selection estimate, not a held-out evaluation. On a large set the
    choice uses a stratified subset; the caller fits the final head on every row.
    """
    if method not in METHODS:
        raise ValueError(f"head method must be one of {METHODS}")
    rows = _selection_index(y, SELECTION_ROWS, SEED)
    X, y = X[rows], y[rows]
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
    return {"setting": setting, "sigma": sigma, "oof_weights": oof, "loss": loss, "rows": rows}


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
