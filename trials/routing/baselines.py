"""Same Banking77 split as the embedding head: a plain classifier, TF-IDF, and an operating point.

The embedding features are runs/routing/feat-embed.npz and the published head's test weights are
runs/routing/weights-embed.npy. Both are gitignored. scikit-learn is used only here; it is not a
JJM dependency.

    uv run --with scikit-learn python -m trials.routing.baselines

The operating point chooses a minimum top weight on a stratified 20% of the test set, then
measures coverage and error on the other 80%. That 80% was not used to pick the threshold.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trials.routing import data
from trials.routing.bench import OUT, _features

TARGETS = (0.01, 0.02, 0.05)
TUNE_FRACTION = 0.2
SEED = 0


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


def _fit_logistic(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, float, float]:
    print("fitting logistic regression on the cached embeddings", flush=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(solver="saga", max_iter=200, tol=1e-3)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = clf.predict(X_test)
    predict_us = (time.perf_counter() - t0) / len(X_test) * 1e6
    print(f"embedding logistic fit {fit_s:.1f}s", flush=True)
    return pred, fit_s, predict_us


def _tfidf(train_text: list[str], test_text: list[str], y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    print("fitting TF-IDF logistic regression", flush=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    t0 = time.perf_counter()
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    vector_s = time.perf_counter() - t0
    clf = LogisticRegression(solver="saga", max_iter=200, tol=1e-3)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = clf.predict(X_test)
    predict_us = (time.perf_counter() - t0) / len(test_text) * 1e6
    return pred, vector_s + fit_s, predict_us


def _split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    tune: list[int] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        n_tune = max(1, int(round(TUNE_FRACTION * len(idx))))
        tune.extend(rng.choice(idx, size=n_tune, replace=False).tolist())
    tune_idx = np.array(sorted(tune), dtype=int)
    eval_idx = np.setdiff1d(np.arange(len(y)), tune_idx)
    return tune_idx, eval_idx


def _kept(top: np.ndarray, correct: np.ndarray, idx: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    keep = top[idx] >= threshold
    n_kept = int(keep.sum())
    if n_kept == 0:
        return {"n": int(len(idx)), "n_kept": 0, "coverage": 0.0, "error_on_kept": None}
    return {"n": int(len(idx)), "n_kept": n_kept, "coverage": n_kept / len(idx),
            "error_on_kept": float(1.0 - correct[idx][keep].mean())}


def _choose(top: np.ndarray, correct: np.ndarray, idx: np.ndarray, target_error: float) -> float | None:
    scores = top[idx]
    thresholds = np.unique(np.concatenate([[0.0], np.quantile(scores, np.linspace(0.0, 1.0, 201))]))
    for threshold in thresholds:
        stats = _kept(top, correct, idx, float(threshold))
        error = stats["error_on_kept"]
        if error is not None and error <= target_error + 1e-9:
            return float(threshold)
    return None


def operating_point(weights: np.ndarray, y: np.ndarray) -> dict[str, object]:
    top = weights.max(axis=1)
    correct = weights.argmax(axis=1) == y
    tune_idx, eval_idx = _split(y)
    rows = []
    for target in TARGETS:
        threshold = _choose(top, correct, tune_idx, target)
        row: dict[str, object] = {"target_error_on_kept": target, "threshold": threshold}
        if threshold is None:
            row["tune"] = None
            row["eval"] = None
        else:
            row["tune"] = _kept(top, correct, tune_idx, threshold)
            row["eval"] = _kept(top, correct, eval_idx, threshold)
        rows.append(row)
    return {"seed": SEED, "tune_fraction": TUNE_FRACTION, "tune_n": int(len(tune_idx)),
            "eval_n": int(len(eval_idx)),
            "eval_accuracy_if_all_handled": _accuracy(weights.argmax(1)[eval_idx], y[eval_idx]),
            "points": rows}


def main() -> None:
    X_train, X_test, y_train, y_test = _features("embed")
    weights = np.load(OUT / "weights-embed.npy")
    if weights.shape != (len(y_test), len(data.TEAM_IDS)):
        raise SystemExit(f"weights-embed.npy has shape {weights.shape}, expected {(len(y_test), len(data.TEAM_IDS))}")
    head_pred = weights.argmax(1)
    point = operating_point(weights, y_test)
    print(json.dumps({"published_head": _accuracy(head_pred, y_test), "operating_point": point}, indent=2), flush=True)
    same_encoder_pred, same_fit_s, same_us = _fit_logistic(X_train, y_train, X_test)
    train_rows, test_rows = data.load("train"), data.load("test")
    tfidf_pred, tfidf_fit_s, tfidf_us = _tfidf(
        [row["text"] for row in train_rows], [row["text"] for row in test_rows], y_train)
    out = {
        "task": "Banking77 intents mapped to eight teams",
        "test_n": int(len(y_test)),
        "published_head": {"accuracy": _accuracy(head_pred, y_test), "source": "runs/routing/weights-embed.npy"},
        "same_encoder_logistic": {"accuracy": _accuracy(same_encoder_pred, y_test), "fit_seconds": same_fit_s,
                                  "predict_microseconds_per_item": same_us,
                                  "note": "sklearn LogisticRegression (saga, C=1) on the same cached embeddings, standardized on the training set."},
        "tfidf_logistic": {"accuracy": _accuracy(tfidf_pred, y_test), "fit_seconds": tfidf_fit_s,
                           "predict_microseconds_per_item": tfidf_us,
                           "note": "Word and bigram TF-IDF on the customer message, then sklearn LogisticRegression (saga, C=1)."},
        "operating_point": point,
    }
    path = OUT / "baselines.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
