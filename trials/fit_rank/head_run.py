"""Score the production fit export through the fitted-head engine.

Sonnet's ranks are teacher labels: agreement with that teacher, not ground truth.
Each scored job is judged by a head that did not train on it. An outcome with only
the fitter's minimum rows stays in every fit and is left out of the scored set.

    uv run python -m trials.fit_rank.head_run --input export.jsonl --profiles jjm-profiles.local.json

The trial profile raises the token cap. The routing profiles cap at 256, and every
job in this export renders to several thousand tokens.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from jjm.compile import compile_criterion, render_state
from jjm.digest import text_digest
from trials.fit_rank import metrics
from trials.fit_rank.build import build_request
from trials.fit_rank.criterion import LEVELS, RANK_BY_OPTION
from trials.fit_rank.probe_eval import auc, spearman
from trials.fit_rank.run import read_export

FOLDS = 5
MIN_TRAIN = 2
PROFILE_ID = "fit-rank-head"
MAX_TOKENS = 8192


def outer_folds(labels: Sequence[str], digests: Sequence[str], *, folds: int = FOLDS,
                min_train: int = MIN_TRAIN) -> list[tuple[list[int], list[int]]]:
    """Hold out different rows each fold. Rows are ordered by text digest within each outcome.

    An outcome with `min_train` or fewer rows is never held out. A fold never holds out so many
    of one outcome that fewer than `min_train` would remain.
    """
    if len(labels) != len(digests):
        raise ValueError("labels and digests must be the same length")
    if folds < 2:
        raise ValueError("need at least two folds")
    grouped: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        grouped.setdefault(label, []).append(i)
    test: list[list[int]] = [[] for _ in range(folds)]
    for idxs in grouped.values():
        ordered = sorted(idxs, key=lambda i: digests[i])
        if len(ordered) <= min_train:
            continue
        cap = len(ordered) - min_train
        buckets: list[list[int]] = [[] for _ in range(folds)]
        for n, i in enumerate(ordered):
            buckets[n % folds].append(i)
        if any(len(b) > cap for b in buckets):
            raise ValueError(f"an outcome of {len(ordered)} rows cannot fill {folds} folds and still keep {min_train}")
        for fold, bucket in enumerate(buckets):
            test[fold].extend(bucket)
    all_idx = list(range(len(labels)))
    return [(sorted(set(all_idx) - set(held)), sorted(held)) for held in test]


def _expected_rank(weights: dict[str, float]) -> float:
    return sum(RANK_BY_OPTION[option] * weight for option, weight in weights.items())


def _trial_profile(source: Path, out_dir: Path) -> Path:
    raw = json.loads(source.read_text(encoding="utf-8"))
    found = next((p for p in raw["profiles"] if p.get("id") == "route-rerank"), None)
    if found is None:
        raise SystemExit(f"{source} has no route-rerank profile to copy the reranker from")
    model = found["encoder"]["model"]
    profile = {
        "id": PROFILE_ID,
        "adapter": "fitted-head",
        "encoder": {
            "kind": "hf",
            "model": model,
            "pooling": "last",
            "batch_size": 2,
            "max_tokens": MAX_TOKENS,
            "prefix_reuse": "off",
        },
        "artifacts": "heads",
        "cache": "cache",
        "max_context_tokens": MAX_TOKENS,
        "allow_download": False,
    }
    path = out_dir / "profile.json"
    path.write_text(json.dumps({"profiles": [profile]}, indent=2) + "\n", encoding="utf-8")
    return path


def state_texts(records: list[dict[str, Any]], text: str) -> list[str]:
    """The words the encoder reads. `probe` is the dedicated prompt; `rendered` is the JJM state."""
    if text == "probe":
        from trials.fit_rank.same_text import probe_text
        return [probe_text(record) for record in records]
    if text != "rendered":
        raise ValueError(f"unknown text {text!r}")
    return [render_state(req["state"]) for req in _requests(records)]


def _requests(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built = []
    for record in records:
        request = build_request(record, profile_id=PROFILE_ID)
        request["requirements"]["allowed_methods"] = ["jjm:hidden-state-head"]
        built.append(request)
    return built


def _score_probe(records: list[dict[str, Any]], profiles_path: Path, out_dir: Path) -> dict[str, Any]:
    """Fit the same softmax head on the dedicated prompt, and score each job with a head that did not train on it.

    The engine would re-render a JJM state, so this path scores the head's weights on the prompt features directly.
    """
    from jjm.heads.fit import fit_head, head_from_artifact
    from jjm.profiles import load_profiles

    labeled = [r for r in records if r.get("sonnet_rank")]
    texts = state_texts(labeled, "probe")
    labels = [LEVELS[r["sonnet_rank"]][0] for r in labeled]
    folds = outer_folds(labels, [text_digest(t) for t in texts])
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(str(_trial_profile(profiles_path, out_dir)))
    adapter = profiles[PROFILE_ID].adapter()
    compiled = compile_criterion(_requests(labeled[:1])[0]["criteria"][0])
    label_source = {"kind": "teacher",
                    "description": "Sonnet fit rank on this export. Agreement with that teacher, not ground truth."}
    features = adapter.featurizer.features(adapter.featurizer.inputs(texts, compiled))
    encoded = features.encode
    ids = list(compiled.outcome_ids)
    scored: list[dict[str, Any]] = []
    fit_seconds = []
    for fold, (train_at, test_at) in enumerate(folds):
        if not test_at:
            continue
        t0 = time.perf_counter()
        fit = fit_head(adapter.featurizer, adapter.store, compiled,
                       [texts[i] for i in train_at], [labels[i] for i in train_at],
                       label_source=label_source, promote=True)
        fit_seconds.append(time.perf_counter() - t0)
        weights = head_from_artifact(fit.artifact).weights(features.features[np.array(test_at)])
        for row, i in enumerate(test_at):
            picked = ids[int(np.argmax(weights[row]))]
            scored.append({
                "job_id": labeled[i]["job_id"],
                "fold": fold,
                "sonnet_rank": labeled[i]["sonnet_rank"],
                "jjm_rank": RANK_BY_OPTION[picked],
                "expected_rank": _expected_rank({option: float(weights[row, n]) for n, option in enumerate(ids)}),
                "artifact_id": fit.artifact["artifact_id"],
            })
        print(f"fold {fold}: train {len(train_at)} test {len(test_at)} fit {fit_seconds[-1]:.1f}s", flush=True)
    kept = scored
    predicted = [r["jjm_rank"] for r in kept]
    reference = [r["sonnet_rank"] for r in kept]
    expected_a = np.asarray([r["expected_rank"] for r in kept], dtype=float)
    reference_a = np.asarray(reference, dtype=int)
    summary = {
        "encoder": "Qwen3-Reranker-0.6B last-token hidden state, softmax head on the dedicated prompt",
        "text": "probe",
        "labels": "Sonnet ranks, treated as a teacher. Not ground truth.",
        "jobs_with_sonnet_rank": len(labeled),
        "scored": len(kept),
        "unscored_because_outcome_too_small": len(labeled) - len(scored),
        "truncated": int(encoded.truncated) if encoded is not None else None,
        "encode_tokens_max": int(max(encoded.tokens)) if encoded is not None and encoded.tokens else None,
        "folds": FOLDS,
        "fit_seconds": fit_seconds,
        "encode_ms": None if encoded is None else encoded.elapsed_ms,
        "agreement_argmax": metrics.agreement(predicted, reference),
        "spearman_expected_rank": float(spearman(expected_a, reference_a)) if kept else None,
        "auc_sonnet_1_or_2": float(auc(-expected_a, reference_a <= 2)) if kept else None,
        "rank_counts": {str(n): predicted.count(n) for n in range(1, 6)},
        "note": "The head's criterion prefix sits in front of the dedicated prompt. The body is that prompt.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "rows.json").write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    return summary


def run(records: list[dict[str, Any]], profiles_path: Path, out_dir: Path, *, text: str = "rendered") -> dict[str, Any]:
    if text == "probe":
        return _score_probe(records, profiles_path, out_dir)
    from jjm.engine import evaluate_batch
    from jjm.heads.fit import fit_head
    from jjm.profiles import load_profiles

    labeled = [r for r in records if r.get("sonnet_rank")]
    requests = _requests(labeled)
    texts = [render_state(req["state"]) for req in requests]
    labels = [LEVELS[r["sonnet_rank"]][0] for r in labeled]
    digests = [text_digest(t) for t in texts]
    folds = outer_folds(labels, digests)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(str(_trial_profile(profiles_path, out_dir)))
    adapter = profiles[PROFILE_ID].adapter()
    compiled = compile_criterion(requests[0]["criteria"][0])
    label_source = {"kind": "teacher",
                    "description": "Sonnet fit rank on this export. Agreement with that teacher, not ground truth."}
    scored: list[dict[str, Any]] = []
    encode_ms = 0.0
    fit_seconds = []
    for fold, (train_at, test_at) in enumerate(folds):
        if not test_at:
            continue
        t0 = time.perf_counter()
        fit = fit_head(adapter.featurizer, adapter.store, compiled,
                       [texts[i] for i in train_at], [labels[i] for i in train_at],
                       label_source=label_source, promote=True)
        fit_seconds.append(time.perf_counter() - t0)
        encode_ms += fit.feature_ms
        adapter._served.clear()
        adapter._refused.clear()
        reqs = [requests[i] for i in test_at]
        t1 = time.perf_counter()
        responses = evaluate_batch(reqs, profiles)
        batch_ms = (time.perf_counter() - t1) * 1000.0
        for record, req, response in zip((labeled[i] for i in test_at), reqs, responses):
            result = response["results"][0]
            prediction = result["prediction"]
            scored.append({
                "job_id": record["job_id"],
                "fold": fold,
                "status": result["status"],
                "sonnet_rank": record["sonnet_rank"],
                "jjm_rank": RANK_BY_OPTION[prediction["selected"]] if prediction else None,
                "expected_rank": _expected_rank(prediction["weights"]) if prediction else None,
                "artifact_id": result["provenance"]["validation"].get("artifact_id"),
                "truncated": bool((result["provenance"].get("diagnostics") or {}).get("truncated")),
                "elapsed_ms": response["elapsed_ms"],
                "batch_ms_per_request": batch_ms / len(reqs),
            })
            if scored[-1]["artifact_id"] != fit.artifact["artifact_id"]:
                raise SystemExit("a held-out job was scored by a different head than the one just fitted")
        print(f"fold {fold}: train {len(train_at)} test {len(test_at)} "
              f"fit {fit_seconds[-1]:.1f}s encode {fit.feature_ms:.0f} ms", flush=True)
    kept = [r for r in scored if r["jjm_rank"] is not None]
    predicted = [r["jjm_rank"] for r in kept]
    reference = [r["sonnet_rank"] for r in kept]
    expected = [r["expected_rank"] for r in kept]
    agreement = metrics.agreement(predicted, reference)
    expected_a = np.asarray(expected, dtype=float)
    reference_a = np.asarray(reference, dtype=int)
    summary = {
        "encoder": "Qwen3-Reranker-0.6B last-token hidden state, engine softmax head",
        "labels": "Sonnet ranks, treated as a teacher. Not ground truth.",
        "jobs_with_sonnet_rank": len(labeled),
        "scored": len(kept),
        "unscored_because_outcome_too_small": len(labeled) - len(scored),
        "errors_or_abstentions": len(scored) - len(kept),
        "truncated": sum(r["truncated"] for r in scored),
        "folds": FOLDS,
        "fit_seconds": fit_seconds,
        "encode_ms_sum_across_folds": encode_ms,
        "agreement_argmax": agreement,
        "spearman_expected_rank": float(spearman(expected_a, reference_a)) if kept else None,
        "auc_sonnet_1_or_2": float(auc(-expected_a, reference_a <= 2)) if kept else None,
        "warm_elapsed_ms_p50": metrics.percentile([r["elapsed_ms"] for r in kept], 0.5),
        "batch_ms_per_request_p50": metrics.percentile([r["batch_ms_per_request"] for r in kept], 0.5),
        "rank_counts": {str(n): predicted.count(n) for n in range(1, 6)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "rows.json").write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--text", choices=("rendered", "probe"), default="rendered")
    args = parser.parse_args(argv)
    out = args.out or Path("runs/fit-rank-head-probe" if args.text == "probe" else "runs/fit-rank-head")
    summary = run(read_export(args.input), args.profiles, out, text=args.text)
    print(json.dumps({k: v for k, v in summary.items() if k != "agreement_argmax"}, indent=2))
    print(json.dumps(summary["agreement_argmax"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
