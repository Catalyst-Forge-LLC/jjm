"""Fit, reuse, and promote heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from jjm import errors
from jjm.compile import CompiledCriterion
from jjm.digest import digest, text_digest
from jjm.heads import head as heads
from jjm.heads.artifact import ArtifactStore, binding, build_artifact, training_digest
from jjm.heads.runtime import Featurizer

MIN_PER_OUTCOME = 2


@dataclass
class FitResult:
    artifact: dict[str, Any]
    reused: bool
    promoted: bool
    feature_ms: float


def label_counts(labels: Sequence[str], outcome_ids: Sequence[str]) -> dict[str, int]:
    return {o: int(sum(1 for x in labels if x == o)) for o in outcome_ids}


def fit_head(featurizer: Featurizer, store: ArtifactStore, compiled: CompiledCriterion,
             state_texts: Sequence[str], labels: Sequence[str], *, method: str = "softmax",
             label_source: dict[str, Any], promote: bool = False) -> FitResult:
    """Fit a head on (state, outcome id) pairs, or reuse the stored head fitted on exactly these pairs.

    label_source says who produced the labels, for example {"kind": "human"} or
    {"kind": "teacher", "teacher": {...identity...}}. Agreement with a teacher stays agreement
    with that teacher; the artifact never calls teacher labels ground truth.
    """
    if len(state_texts) != len(labels):
        raise errors.RequestError(errors.INVALID_REQUEST, "every example needs exactly one label")
    ids = list(compiled.outcome_ids)
    unknown = sorted({x for x in labels if x not in ids})
    if unknown:
        raise errors.RequestError(errors.INVALID_REQUEST, f"labels are not outcomes of this criterion: {unknown}")
    counts = label_counts(labels, ids)
    thin = [o for o, c in counts.items() if 0 < c < MIN_PER_OUTCOME]
    if thin or sum(1 for c in counts.values() if c) < 2:
        raise errors.RequestError(errors.INVALID_REQUEST,
                                  f"need at least two outcomes with {MIN_PER_OUTCOME}+ examples each; counts {counts}")
    bind = binding(compiled.criterion_digest, featurizer.encoder.identity(), method)
    bind_digest = digest(bind)
    train = training_digest([(text_digest(t), y) for t, y in zip(state_texts, labels)], label_source)
    existing = store.find_training(bind_digest, train)
    if existing is not None:
        if promote:
            store.promote(existing)
        return FitResult(existing, True, promote, 0.0)
    run = featurizer.features(featurizer.inputs(state_texts, compiled))
    y = np.array([ids.index(x) for x in labels], dtype=int)
    head, cv = heads.fit(method, run.features, y, len(ids))
    held_for_selection = cv["rows"]
    report = {"cv": {**heads.classification_report(cv["oof_weights"], y[held_for_selection], len(ids)),
                     "folds": heads.FOLDS, "selected_setting": cv["setting"],
                     "selection_rows": int(len(held_for_selection)),
                     "note": "Cross-validation also chose the setting"
                             + (f" on {len(held_for_selection)} stratified rows" if len(held_for_selection) < len(y)
                                else "")
                             + ", so this is a selection estimate, not a held-out validation."}}
    training = {"training_digest": train, "n": len(labels), "label_source": label_source,
                "label_counts": counts, "split": f"stratified {heads.FOLDS}-fold, seed {heads.SEED}",
                "feature_dim": int(run.features.shape[1])}
    art = build_artifact(bind=bind, outcome_ids=ids, training=training, report=report, payload=head.payload())
    store.save(art)
    if promote:
        store.promote(art)
    return FitResult(art, False, promote, run.elapsed_ms)


def head_from_artifact(art: dict[str, Any]) -> heads.Head:
    return heads.Head.from_payload(art["payload"])


def served_artifact(store: ArtifactStore, featurizer: Featurizer, compiled: CompiledCriterion,
                    method: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The promoted artifact whose binding matches this criterion and encoder exactly, if any."""
    for m in ([method] if method else list(heads.METHODS)):
        art = store.promoted(digest(binding(compiled.criterion_digest, featurizer.encoder.identity(), m)))
        if art is not None:
            if art["outcome_ids"] != list(compiled.outcome_ids):
                raise errors.AdapterError(errors.ARTIFACT_MISMATCH, "Head outcomes differ from the criterion's")
            return art
    return None
