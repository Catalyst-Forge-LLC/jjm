"""Score the fast-judgment feature sets against Sonnet's ranks with honest cross-validation.

    uv run python -m trials.fit_rank.probe_eval runs/feat-probe-4b.npz runs/feat-embed-0.6b.npz ...

Zero-shot scores (reranker P(yes), cosine, coverage) are correlated directly.
Learned heads are ridge regressions on standardized features, predicting rank as a number.
The ridge strength is chosen by an inner 5-fold split inside each outer training fold,
the outer loop is 10-fold, and all of it repeats over 3 shuffles, so no job is ever
scored by a head that saw it. numpy only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from trials.fit_rank import metrics

ALPHAS = np.logspace(-1, 5, 13)


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    for r, x in ((ra, np.asarray(a)), (rb, np.asarray(b))):
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, positive) -> float:
    pos, neg = score[positive], score[~positive]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def _ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = y.mean()
    K = Z @ Z.T
    coef = np.linalg.solve(K + alpha * np.eye(len(Z)), y - ym)
    w = Z.T @ coef
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def _folds(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def cv_softmax(X, y_rank, *, outer=10, repeats=3, seed=0, k=5) -> list[np.ndarray]:
    """Out-of-fold softmax weights on the same outer splits as the ridge.

    y_rank is 1..k. Each repeat scores every row once. The penalty is chosen inside the
    training fold, the same way the fitted head chooses it.
    """
    from jjm.heads.head import fit

    y = np.asarray(y_rank, dtype=int) - 1
    if y.min() < 0 or y.max() >= k:
        raise ValueError(f"ranks must be 1..{k}")
    out = []
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        weights = np.zeros((len(y), k))
        for test in _folds(len(y), outer, rng):
            train = np.setdiff1d(np.arange(len(y)), test)
            head, _ = fit("softmax", X[train], y[train], k)
            weights[test] = head.weights(X[test])
        out.append(weights)
    return out


def softmax_report(name: str, weights: list[np.ndarray], y_rank: np.ndarray, ms: float) -> dict:
    """Average the per-repeat metrics. Expected rank is the probability-weighted level."""
    levels = np.arange(1, weights[0].shape[1] + 1, dtype=float)
    rows = []
    for w in weights:
        expected = w @ levels
        picked = levels[np.argmax(w, axis=1)].astype(int)
        row = report(name, expected, y_rank, ms)
        row["exact"] = metrics.exact(picked, y_rank)
        row["within_one"] = metrics.within_one(picked, y_rank)
        row["qwk"] = metrics.quadratic_weighted_kappa(list(picked), list(y_rank.astype(int)))
        row["rank_counts"] = {str(n): int(np.sum(picked == n)) for n in range(1, len(levels) + 1)}
        rows.append(row)
    averaged = _avg(rows)
    averaged["rank_counts"] = rows[0]["rank_counts"]
    averaged["repeats"] = len(rows)
    return averaged


def cv_predict(X, y, *, outer=10, inner=5, repeats=3, seed=0) -> np.ndarray:
    preds = np.zeros((repeats, len(y)))
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        for test in _folds(len(y), outer, rng):
            train = np.setdiff1d(np.arange(len(y)), test)
            errs = np.zeros(len(ALPHAS))
            for val in _folds(len(train), inner, rng):
                tr = np.setdiff1d(np.arange(len(train)), val)
                for j, a in enumerate(ALPHAS):
                    f = _ridge_fit(X[train[tr]], y[train[tr]], a)
                    errs[j] += ((f(X[train[val]]) - y[train[val]]) ** 2).sum()
            best = ALPHAS[int(np.argmin(errs))]
            preds[rep, test] = _ridge_fit(X[train], y[train], best)(X[test])
    return preds


def report(name: str, score: np.ndarray, y: np.ndarray, ms: float, *, lower_is_better_rank: bool = True) -> dict:
    """score is on the rank scale (1 = strongest) when learned; zero-shot scores are oriented first."""
    s = score if lower_is_better_rank else -score
    out = {"name": name, "spearman": spearman(s, y), "auc_top2": auc(-s, y <= 2), "ms": ms}
    rounded = np.clip(np.rint(s), 1, 5).astype(int) if lower_is_better_rank else None
    if rounded is not None:
        out.update(exact=metrics.exact(rounded, y), within_one=metrics.within_one(rounded, y),
                   qwk=metrics.quadratic_weighted_kappa(list(rounded), list(y)))
    return out


def evaluate_file(path: Path) -> list[dict]:
    d = np.load(path)
    y = d["sonnet"].astype(float)
    yi = d["sonnet"].astype(int)
    ms = float(np.median(d["ms"]))
    rows = []
    if "last" in d:
        for kind in ("last", "mean"):
            for j, layer in enumerate(d["layers"]):
                p = cv_predict(d[kind][:, j], y)
                r = [report(f"{path.stem} {kind}@L{layer}", p[k], yi, ms) for k in range(len(p))]
                rows.append(_avg(r))
    if "p_yes" in d:
        rows.append(report(f"{path.stem} zero-shot P(yes)", d["p_yes"], yi, ms, lower_is_better_rank=False))
        rows.append(_avg([report(f"{path.stem} head on hidden", p, yi, ms) for p in cv_predict(d["hidden"], y)]))
    if "listing" in d:
        for key in ("cosine", "coverage_mean", "coverage_p25"):
            rows.append(report(f"{path.stem} zero-shot {key}", d[key], yi, ms, lower_is_better_rank=False))
        rows.append(_avg([report(f"{path.stem} head on listing embedding", p, yi, ms)
                          for p in cv_predict(d["listing"], y)]))
        X = np.column_stack([d["listing"], d["cosine"], d["coverage_mean"], d["coverage_p25"], d["coverage_min"]])
        rows.append(_avg([report(f"{path.stem} head on embedding + coverage", p, yi, ms) for p in cv_predict(X, y)]))
    return rows


def _avg(rows: list[dict]) -> dict:
    out = dict(rows[0])
    for k, v in rows[0].items():
        if isinstance(v, float) and k != "ms":
            vals = [r[k] for r in rows if r.get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None
    return out


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    softmax_only = "--softmax" in argv
    paths = [Path(p) for p in argv if p != "--softmax"]
    y = np.load(paths[0])["sonnet"].astype(int)
    base = np.full(len(y), 2)
    print(f"n={len(y)}; always-2 baseline: exact {metrics.exact(base, y):.3f}, within1 "
          f"{metrics.within_one(base, y):.3f}, spearman 0, qwk 0")
    print(f"{'method':52} {'spearman':>8} {'auc12':>6} {'exact':>6} {'within1':>7} {'qwk':>6} {'ms/job':>7}")
    for path in paths:
        rows = []
        if softmax_only:
            d = np.load(path)
            if "hidden" not in d:
                raise SystemExit(f"{path} has no hidden states")
            weights = cv_softmax(d["hidden"], d["sonnet"])
            rows = [softmax_report(f"{path.stem} softmax on hidden", weights, d["sonnet"].astype(int),
                                   float(np.median(d["ms"])))]
        else:
            rows = evaluate_file(path)
        for r in rows:
            f = lambda k: f"{r[k]:.3f}" if r.get(k) is not None else "  -  "
            print(f"{r['name']:52} {f('spearman'):>8} {f('auc_top2'):>6} {f('exact'):>6} {f('within_one'):>7} "
                  f"{f('qwk'):>6} {r['ms']:7.0f}")
            if r.get("rank_counts"):
                print(f"  picks {r['rank_counts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
