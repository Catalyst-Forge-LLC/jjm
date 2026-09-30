"""Fit, reuse, and promote heads."""
from __future__ import annotations

import re
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
HOLDOUT_FRACTION = 0.2
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_LABEL_KIND = {"human": "human", "teacher": "teacher_model", "deterministic": "deterministic_outcome",
               "provider": "teacher_model", "synthetic": "synthetic"}


@dataclass
class FitResult:
    artifact: dict[str, Any]
    reused: bool
    promoted: bool
    feature_ms: float


def label_counts(labels: Sequence[str], outcome_ids: Sequence[str]) -> dict[str, int]:
    return {o: int(sum(1 for x in labels if x == o)) for o in outcome_ids}


def holdout_positions(texts: Sequence[str], labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Rows held out before the penalty is chosen and before the head is fit.

    Within each outcome the rows are ordered by text digest, and the first slice is held out.
    An outcome with only the minimum training rows stays entirely in the fit, so a small set
    can have an empty holdout.
    """
    by_label: dict[str, list[tuple[str, int]]] = {}
    for i, (text, label) in enumerate(zip(texts, labels)):
        by_label.setdefault(label, []).append((text_digest(text), i))
    train, hold = [], []
    for rows in by_label.values():
        rows.sort()
        spare = len(rows) - MIN_PER_OUTCOME
        n_hold = 0 if spare <= 0 else min(spare, max(1, int(round(HOLDOUT_FRACTION * len(rows)))))
        for n, (_, i) in enumerate(rows):
            (hold if n < n_hold else train).append(i)
    return np.array(sorted(train), dtype=int), np.array(sorted(hold), dtype=int)


def fit_head(featurizer: Featurizer, store: ArtifactStore, compiled: CompiledCriterion,
             state_texts: Sequence[str], labels: Sequence[str], *, method: str = "softmax",
             label_source: dict[str, Any], promote: bool = False) -> FitResult:
    """Fit a head on (state, outcome id) pairs, or reuse the stored head fitted on exactly these pairs.

    label_source says who produced the labels, for example {"kind": "human"} or
    {"kind": "teacher", "teacher": {...identity...}}. Agreement with a teacher stays agreement
    with that teacher; the artifact never calls teacher labels ground truth.

    A slice of each outcome is scored and then left out of both the penalty choice and the fit.
    That score is stored and stays unaccepted.
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
    train_at, hold_at = holdout_positions(state_texts, labels)
    hold_keys = sorted((text_digest(state_texts[i]), labels[i]) for i in hold_at.tolist())
    split = {"holdout_fraction": HOLDOUT_FRACTION, "order": "text digest within each outcome",
             "holdout": hold_keys}
    bind = binding(compiled.criterion_digest, featurizer.encoder.identity(), method)
    bind_digest = digest(bind)
    train = training_digest([(text_digest(t), y) for t, y in zip(state_texts, labels)], label_source, split)
    existing = store.find_training(bind_digest, train)
    if existing is not None:
        if promote:
            store.promote(existing)
        return FitResult(existing, True, promote, 0.0)
    run = featurizer.features(featurizer.inputs(state_texts, compiled))
    y = np.array([ids.index(x) for x in labels], dtype=int)
    head, cv = heads.fit(method, run.features[train_at], y[train_at], len(ids))
    held_for_selection = cv["rows"]
    if len(hold_at):
        heldout = {**heads.classification_report(head.weights(run.features[hold_at]), y[hold_at], len(ids)),
                   "split": (f"stratified {HOLDOUT_FRACTION:.0%} by text digest within each outcome, "
                             "taken before the penalty choice and the fit"),
                   "note": "These rows were not used to choose the penalty or to fit the head. "
                           "This report is not accepted until an explicit accept."}
    else:
        heldout = {"n": 0, "split": "no holdout; an outcome had only the minimum training rows",
                   "note": "There were not enough labelled rows to hold any out."}
    report = {"cv": {**heads.classification_report(cv["oof_weights"], y[train_at][held_for_selection], len(ids)),
                     "folds": heads.FOLDS, "selected_setting": cv["setting"],
                     "selection_rows": int(len(held_for_selection)),
                     "note": "Cross-validation also chose the setting"
                             + (f" on {len(held_for_selection)} stratified training rows"
                                if len(held_for_selection) < len(train_at) else "")
                             + ", so this is a selection estimate, not a held-out validation."},
              "heldout": heldout}
    training = {"training_digest": train, "n": len(labels), "fit_count": int(len(train_at)),
                "label_source": label_source, "label_counts": counts, "split": heldout["split"],
                "feature_dim": int(run.features.shape[1])}
    art = build_artifact(bind=bind, outcome_ids=ids, training=training, report=report, payload=head.payload())
    if heldout["n"]:
        art["report"]["heldout"]["report_id"] = "report-" + digest(
            {"payload_digest": art["payload_digest"], "holdout": hold_keys})[7:27]
    store.save(art)
    if promote:
        store.promote(art)
    return FitResult(art, False, promote, run.elapsed_ms)


def accept_validation(store: ArtifactStore, art: dict[str, Any], scope_id: str) -> dict[str, Any]:
    """Accept the held-out report of a promoted head for one scope. The head file is not rewritten."""
    if not isinstance(scope_id, str) or not _ID.match(scope_id) or len(scope_id) > 128:
        raise errors.RequestError(errors.INVALID_REQUEST, "scope_id must be 1 to 128 letters, digits, and _ . : / -")
    promoted = store.promoted(art["binding_digest"])
    if promoted is None or promoted["artifact_id"] != art["artifact_id"]:
        raise errors.RequestError(errors.INVALID_REQUEST, "Promote the head before accepting its report.")
    held = art["report"].get("heldout") or {}
    if not held.get("n") or not held.get("report_id"):
        raise errors.RequestError(errors.INVALID_REQUEST, "This head has no held-out report to accept.")
    return store.accept(art, scope_id, held["report_id"])


def contract_manifest(art: dict[str, Any], provenance: dict[str, Any], scope_id: str) -> dict[str, Any]:
    """The artifact manifest validate_pair requires for a calibrated result. Built at serve time."""
    src = art["training"]["label_source"]
    dataset = src.get("dataset_id") if isinstance(src.get("dataset_id"), str) and _ID.match(src["dataset_id"]) else "unspecified"
    revision = src.get("dataset_revision") if isinstance(src.get("dataset_revision"), str) and src["dataset_revision"] else "unspecified"
    description = src.get("description") if isinstance(src.get("description"), str) and src["description"] else (
        f"Labels recorded as {src.get('kind', 'unknown')}.")
    held = art["report"]["heldout"]
    return {
        "schema_version": "0.1",
        "artifact_id": art["artifact_id"],
        "artifact_version": art["artifact_version"],
        "execution_mode": "real",
        "created_at": art["created_at"],
        "payload_digest": art["payload_digest"],
        "method": provenance["method"],
        "criterion_digest": provenance["criterion_digest"],
        "effective_config_digest": provenance["effective_config_digest"],
        "model": provenance["model"],
        "rendering_digest": provenance["rendering_digest"],
        "label_source": {"kind": _LABEL_KIND.get(src.get("kind"), "mixed"), "description": description,
                         "dataset_id": dataset, "dataset_revision": revision},
        "fit_count": int(art["training"]["fit_count"]),
        "heldout_count": int(held["n"]),
        "split_description": held["split"],
        "validation_status": "validated",
        "validation_report_id": held["report_id"],
        "scope_id": scope_id,
        "promoted": True,
    }


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
