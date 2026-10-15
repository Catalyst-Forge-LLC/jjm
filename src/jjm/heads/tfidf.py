"""Sparse TF-IDF plus logistic regression, persisted as data-only JSON.

Only training rows fit the vocabulary, IDF, and classifier. Hyperparameters
are operator settings; the holdout is never used to select them.
"""
from __future__ import annotations

import time
import warnings
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from jjm import errors
from jjm.compile import CompiledCriterion, RENDERING_DIGEST
from jjm.digest import digest, text_digest
from jjm.heads.artifact import ArtifactStore, binding, build_artifact, training_digest
from jjm.heads.fit import FitResult, HOLDOUT_FRACTION, holdout_positions, label_counts
from jjm.heads.head import classification_report

VERSION = "jjm-tfidf-1"
METHOD = "jjm:tfidf-logistic"


def identity(settings: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "tfidf", "model": "jjm/tfidf-logistic", "version": VERSION,
            "rendering_digest": RENDERING_DIGEST, "scikit_learn": sklearn.__version__,
            "settings": settings}


def vectorizer(settings: dict[str, Any], **kwargs) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=tuple(settings["ngram_range"]), min_df=settings["min_df"],
                           max_features=settings["max_features"], sublinear_tf=True,
                           lowercase=True, norm="l2", dtype=np.float64, **kwargs)


def classifier(settings: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(C=settings["C"], max_iter=settings["max_iter"], solver="lbfgs", random_state=0)


class TextModel:
    def __init__(self, vector: TfidfVectorizer, model: LogisticRegression):
        self.vector = vector
        self.model = model

    def weights(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.predict_proba(self.vector.transform(texts))

    def payload(self) -> dict[str, Any]:
        return {"version": VERSION, "vocabulary": self.vector.vocabulary_,
                "idf": self.vector.idf_.tolist(), "classes": self.model.classes_.tolist(),
                "coef": self.model.coef_.tolist(), "intercept": self.model.intercept_.tolist()}

    @classmethod
    def from_payload(cls, payload: dict[str, Any], settings: dict[str, Any], k: int) -> TextModel:
        try:
            vocabulary = payload["vocabulary"]
            if (payload["version"] != VERSION or not isinstance(vocabulary, dict) or not vocabulary
                    or not all(isinstance(term, str) and type(i) is int for term, i in vocabulary.items())
                    or sorted(vocabulary.values()) != list(range(len(vocabulary)))
                    or payload["classes"] != list(range(k))):
                raise ValueError("invalid vocabulary or classes")
            n = len(vocabulary)
            idf = np.asarray(payload["idf"], dtype=np.float64)
            coef = np.asarray(payload["coef"], dtype=np.float64)
            intercept = np.asarray(payload["intercept"], dtype=np.float64)
            rows = 1 if k == 2 else k
            if (idf.shape != (n,) or coef.shape != (rows, n) or intercept.shape != (rows,)
                    or not np.isfinite(idf).all() or not (idf > 0).all()
                    or not np.isfinite(coef).all() or not np.isfinite(intercept).all()):
                raise ValueError("invalid dimensions or numeric values")
            vector = vectorizer(settings, vocabulary=vocabulary)
            vector.idf_ = idf
            model = classifier(settings)
            model.classes_ = np.arange(k)
            model.coef_, model.intercept_, model.n_features_in_ = coef, intercept, n
            return cls(vector, model)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise errors.AdapterError(errors.ARTIFACT_MISMATCH, "Invalid TF-IDF artifact payload.") from None


def fit_text(store: ArtifactStore, settings: dict[str, Any], compiled: CompiledCriterion,
             texts: Sequence[str], labels: Sequence[str], *, method: str = "softmax",
             label_source: dict[str, Any], promote: bool = False) -> FitResult:
    ids = list(compiled.outcome_ids)
    if compiled.kind != "choice" or method != "softmax":
        raise errors.RequestError(errors.UNSUPPORTED_KIND, "TF-IDF supports choice criteria and --method softmax.")
    if len(texts) != len(labels) or any(not isinstance(y, str) or y not in ids for y in labels):
        raise errors.RequestError(errors.INVALID_REQUEST, "Every example needs one declared outcome label.")
    counts = label_counts(labels, ids)
    if len(ids) < 2 or any(n < 2 for n in counts.values()):
        raise errors.RequestError(errors.INVALID_REQUEST, f"TF-IDF needs 2+ examples for every outcome; counts {counts}")
    if len(set(texts)) != len(texts):
        raise errors.RequestError(errors.INVALID_REQUEST, "Deduplicate rendered states before fitting TF-IDF.")
    train_at, hold_at = holdout_positions(texts, labels)
    # Canonical ordering makes reuse independent of the JSONL row order.
    train_at = sorted(train_at.tolist(), key=lambda i: (text_digest(texts[i]), labels[i]))
    hold_keys = sorted((text_digest(texts[i]), labels[i]) for i in hold_at.tolist())
    split = {"holdout_fraction": HOLDOUT_FRACTION, "order": "text digest within each outcome", "holdout": hold_keys}
    bind = binding(compiled.criterion_digest, identity(settings), METHOD)
    train_digest = training_digest([(text_digest(t), y) for t, y in zip(texts, labels)], label_source, split)
    existing = store.find_training(digest(bind), train_digest)
    if existing is not None:
        TextModel.from_payload(existing["payload"], settings, len(ids))
        if promote:
            store.promote(existing)
        return FitResult(existing, True, promote, 0.0)
    y = np.array([ids.index(label) for label in labels], dtype=int)
    vector = vectorizer(settings)
    started = time.perf_counter()
    try:
        X = vector.fit_transform([texts[i] for i in train_at])
        feature_ms = (time.perf_counter() - started) * 1000
        model = classifier(settings)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X, y[train_at])
    except ConvergenceWarning:
        raise errors.RequestError(errors.INVALID_REQUEST, "TF-IDF fit did not converge; increase profile max_iter.") from None
    except ValueError as exc:
        raise errors.RequestError(errors.INVALID_REQUEST, f"TF-IDF cannot fit these examples/settings: {exc}") from None
    fitted = TextModel(vector, model)
    description = "stratified 20% by text digest within each outcome, before vocabulary, IDF, and classifier fit"
    held = {"n": 0, "split": description, "note": "Too few rows for a holdout."}
    if len(hold_at):
        held = {**classification_report(fitted.weights([texts[i] for i in hold_at]), y[hold_at], len(ids)),
                "split": description, "note": "Fixed operator settings; no selection on holdout. Not yet accepted."}
    training = {"training_digest": train_digest, "n": len(labels), "fit_count": len(train_at),
                "label_source": label_source, "label_counts": counts, "split": description, "feature_dim": X.shape[1]}
    art = build_artifact(bind=bind, outcome_ids=ids, training=training,
                         report={"cv": None, "heldout": held}, payload=fitted.payload())
    if held["n"]:
        held["report_id"] = "report-" + digest({"payload_digest": art["payload_digest"], "holdout": hold_keys})[7:27]
    store.save(art)
    if promote:
        store.promote(art)
    return FitResult(art, False, promote, feature_ms)
