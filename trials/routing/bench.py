"""Banking77 routing benchmark: accuracy against human labels, and speed, for every JJM route.

    uv run python -m trials.routing.bench encode   --profiles P.json --profile route-rerank --name rerank
    uv run python -m trials.routing.bench heads    --name rerank
    uv run python -m trials.routing.bench teacher  --profiles P.json --teacher qwen35 --split test
    uv run python -m trials.routing.bench teacher  --profiles P.json --teacher qwen35 --split train
    uv run python -m trials.routing.bench distill  --name rerank --teacher qwen35
    uv run python -m trials.routing.bench cascade  --name rerank --teacher qwen35
    uv run python -m trials.routing.bench readout  --profiles P.json --profile local-advisory
    uv run python -m trials.routing.bench rerank-zeroshot --model-path <reranker snapshot>
    uv run python -m trials.routing.bench speed    --model-path <reranker snapshot>
    uv run python -m trials.routing.bench engine   --profiles P.json --profile route-rerank
    uv run python -m trials.routing.bench report

Results go to runs/routing/, which git ignores. "Accuracy" is always against the human
labels. Slow methods run on SUBSET: 50 test messages per team, 400 in all, seed 0.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from jjm.compile import render_state
from jjm.heads import head as heads
from jjm.heads.cascade import escalation_curve, threshold_for_accuracy
from trials.routing import data

OUT = data.ROOT / "runs" / "routing"
SUBSET_PER_TEAM = 50
TEACHER_TRAIN_N = 800
CURVE = (4, 8, 16, 32, 64, 128, 256)
CURVE_SEEDS = (0, 1, 2)


def _write(name: str, value: Any) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return path


def _read(name: str) -> Any:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q)) if xs else float("nan")


def subset_indices(y_test: np.ndarray) -> np.ndarray:
    return data.stratified(y_test, SUBSET_PER_TEAM, seed=0)


def teacher_train_indices(n: int) -> np.ndarray:
    return np.sort(np.random.default_rng(0).choice(n, size=TEACHER_TRAIN_N, replace=False))


def _profiles(path: str):
    from jjm.profiles import load_profiles

    return load_profiles(path)


def _per_team(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {t: float(np.mean(pred[y == i] == i)) for i, t in enumerate(data.TEAM_IDS) if np.any(y == i)}


# ---- encode ---------------------------------------------------------------
def cmd_encode(args) -> None:
    profile = _profiles(args.profiles)[args.profile]
    t0 = time.perf_counter()
    adapter = profile.adapter()
    load_ms = (time.perf_counter() - t0) * 1000.0
    f, c = adapter.featurizer, data.compiled()
    train, test = data.load("train"), data.load("test")
    out: dict[str, Any] = {}
    for split, rows in (("train", train), ("test", test)):
        run = f.features(f.inputs(data.texts(rows), c))
        out[split] = run.features
        print(f"{split}: {len(rows)} messages, {run.encoded} encoded, {run.cache_hits} from cache, "
              f"{run.elapsed_ms:.0f} ms")
        out[f"{split}_ms_per_item"] = run.elapsed_ms / max(1, run.encoded) if run.encoded else float("nan")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"feat-{args.name}.npz", X_train=out["train"], X_test=out["test"], y_train=data.labels(train),
             y_test=data.labels(test))
    _write(f"feat-{args.name}.json", {"profile": args.profile, "encoder": f.encoder.identity(),
                                      "compute": f.encoder.compute_settings(), "load_ms": load_ms,
                                      "dim": int(out["train"].shape[1]),
                                      "ms_per_item_train": out["train_ms_per_item"],
                                      "ms_per_item_test": out["test_ms_per_item"]})


def _features(name: str):
    d = np.load(OUT / f"feat-{name}.npz")
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"]


# ---- heads on human labels -------------------------------------------------
def cmd_heads(args) -> None:
    Xtr, Xte, ytr, yte = _features(args.name)
    k = len(data.TEAM_IDS)
    t0 = time.perf_counter()
    head, cv = heads.fit("softmax", Xtr, ytr, k)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    w = head.weights(Xte)
    head_us = (time.perf_counter() - t0) / len(Xte) * 1e6
    report = heads.classification_report(w, yte, k)
    np.save(OUT / f"weights-{args.name}.npy", w)
    curve = []
    for n in CURVE:
        accs = []
        for seed in CURVE_SEEDS:
            idx = data.stratified(ytr, n, seed)
            h, _ = heads.fit("softmax", Xtr[idx], ytr[idx], k)
            accs.append(float(np.mean(h.weights(Xte).argmax(1) == yte)))
        curve.append({"per_team": n, "examples": int(n * k), "accuracy_mean": float(np.mean(accs)),
                      "accuracy_min": float(np.min(accs)), "accuracy_max": float(np.max(accs))})
        print(f"{n:>4} per team: accuracy {np.mean(accs):.3f}")
    _write(f"heads-{args.name}.json", {"test": report, "per_team": _per_team(w.argmax(1), yte),
                                       "cv_selected_l2": cv["setting"], "fit_seconds": fit_s,
                                       "head_microseconds_per_item": head_us, "learning_curve": curve})
    print(f"full head: test accuracy {report['accuracy']:.3f}, macro F1 {report['macro_f1']:.3f}")


# ---- teachers ---------------------------------------------------------------
def cmd_teacher(args) -> None:
    from jjm.heads.teachers import build_teacher
    from jjm.profiles import load_teachers

    teacher = build_teacher(load_teachers(args.profiles)[args.teacher])
    rows = data.load(args.split)
    y = data.labels(rows)
    idx = subset_indices(y) if args.split == "test" else teacher_train_indices(len(rows))
    if args.limit:
        idx = idx[: args.limit]
    c = data.compiled()
    answers, ms = [], []
    teacher.answer(data.texts([rows[int(idx[0])]])[0], c)
    for n, i in enumerate(idx, 1):
        ans = teacher.answer(render_state(data.state(rows[int(i)]["text"])), c)
        answers.append({"index": int(i), "answer": ans.index, "letter_weights": ans.letter_weights,
                        "raw": ans.raw, "ms": ans.elapsed_ms})
        ms.append(ans.elapsed_ms)
        if n % 50 == 0:
            ok = np.mean([a["answer"] == y[a["index"]] for a in answers])
            print(f"[{n}/{len(idx)}] accuracy so far {ok:.3f}, median {statistics.median(ms):.0f} ms", flush=True)
    got = np.array([a["answer"] if a["answer"] is not None else -1 for a in answers])
    truth = y[idx]
    summary = {"teacher": teacher.identity(), "split": args.split, "n": len(idx),
               "accuracy": float(np.mean(got == truth)), "unparsed": int(np.sum(got < 0)),
               "letter_weights_complete": int(sum(a["letter_weights"] is not None for a in answers)),
               "per_team": _per_team(got, truth), "warm_ms_p50": _pct(ms, 50), "warm_ms_p95": _pct(ms, 95)}
    _write(f"teacher-{args.teacher}-{args.split}.json", {"summary": summary, "answers": answers})
    print(json.dumps(summary, indent=2))


# ---- head trained on teacher labels -----------------------------------------
def cmd_distill(args) -> None:
    Xtr, Xte, ytr, yte = _features(args.name)
    k = len(data.TEAM_IDS)
    answers = _read(f"teacher-{args.teacher}-train.json")["answers"]
    kept = [a for a in answers if a["answer"] is not None]
    idx = np.array([a["index"] for a in kept])
    y_teacher = np.array([a["answer"] for a in kept])
    out = {"teacher": args.teacher, "encoder": args.name, "train_examples": len(idx),
           "teacher_agreement_with_humans_on_train": float(np.mean(y_teacher == ytr[idx]))}
    for label, y in (("teacher_labels", y_teacher), ("human_labels_same_examples", ytr[idx])):
        present = np.unique(y)
        h, _ = heads.fit("softmax", Xtr[idx], y, k)
        w = h.weights(Xte)
        out[label] = {**heads.classification_report(w, yte, k), "outcomes_seen": int(len(present))}
        print(f"{label}: test accuracy {out[label]['accuracy']:.3f}")
    _write(f"distill-{args.name}-{args.teacher}.json", out)


# ---- escalation --------------------------------------------------------------
def cmd_cascade(args) -> None:
    _, _, _, yte = _features(args.name)
    w = np.load(OUT / f"weights-{args.name}.npy")
    idx = subset_indices(yte)
    answers = {a["index"]: a["answer"] for a in _read(f"teacher-{args.teacher}-test.json")["answers"]}
    idx = np.array([i for i in idx if int(i) in answers])
    top = w[idx].max(1)
    head_ok = w[idx].argmax(1) == yte[idx]
    teacher_ok = np.array([answers[int(i)] == yte[i] for i in idx])
    curve = escalation_curve(top, head_ok, teacher_ok)
    out = {"encoder": args.name, "teacher": args.teacher, "n": int(len(idx)),
           "head_accuracy": float(head_ok.mean()), "teacher_accuracy": float(teacher_ok.mean()),
           "curve": curve,
           "bar_for_95_on_kept": threshold_for_accuracy(top, head_ok, 0.95),
           "bar_for_98_on_kept": threshold_for_accuracy(top, head_ok, 0.98),
           "note": "Bars are read off this same subset, so they are descriptive, not a tuned deployment setting."}
    _write(f"cascade-{args.name}-{args.teacher}.json", out)
    for c in curve:
        print(f"min_top_mass {c['min_top_mass']:.2f}: teacher share {c['teacher_share']:.2f}, "
              f"accuracy {c['accuracy']:.3f}")


# ---- JJM readout (AnyJev) ------------------------------------------------------
def _request(message: str, i: int, profile_id: str, methods: list[str]) -> dict[str, Any]:
    return {"schema_version": "0.1", "request_id": f"route-{i}", "scope_id": "trial.banking77",
            "profile_id": profile_id,
            "state": {"id": f"msg-{i}", "revision": "1", "observed_at": "2026-09-29T00:00:00Z",
                      "content": {"message": message}, "evidence": []},
            "criteria": [data.criterion()],
            "requirements": {"validation": "not_required", "probability_calibration": "not_required",
                             "allowed_methods": methods, "min_top_mass": None,
                             "min_margin": None, "require_verified_model_revision": False,
                             "max_state_age_seconds": None},
            "timeout_ms": 600000}


def cmd_readout(args) -> None:
    from jjm.engine import evaluate

    profiles = _profiles(args.profiles)
    rows = data.load("test")
    y = data.labels(rows)
    idx = subset_indices(y)[: args.limit] if args.limit else subset_indices(y)
    t0 = time.perf_counter()
    profiles[args.profile].adapter()
    load_ms = (time.perf_counter() - t0) * 1000.0
    evaluate(_request(rows[int(idx[0])]["text"], -1, args.profile, []), profiles)
    preds, ms = [], []
    for n, i in enumerate(idx, 1):
        t = time.perf_counter()
        r = evaluate(_request(rows[int(i)]["text"], int(i), args.profile, []), profiles)["results"][0]
        ms.append((time.perf_counter() - t) * 1000.0)
        preds.append(data.TEAM_IDS.index(r["prediction"]["selected"]) if r["prediction"] else -1)
        if n % 50 == 0:
            print(f"[{n}/{len(idx)}] accuracy so far {np.mean(np.array(preds) == y[idx[:n]]):.3f}, "
                  f"median {statistics.median(ms):.0f} ms", flush=True)
    preds = np.array(preds)
    prov = r["provenance"]
    out = {"profile": args.profile, "model": prov["model"], "method": prov["method"], "n": int(len(idx)),
           "accuracy": float(np.mean(preds == y[idx])), "per_team": _per_team(preds, y[idx]),
           "load_ms": load_ms, "warm_ms_p50": _pct(ms, 50), "warm_ms_p95": _pct(ms, 95),
           "prompt_evaluations": prov["runtime"]["prompt_evaluations"]}
    _write(f"readout-{args.profile}.json", out)
    print(json.dumps(out, indent=2))


# ---- zero-shot reranker: one yes/no per team -------------------------------------
RERANK_SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
                 'Note that the answer can only be "yes" or "no".')


def cmd_rerank_zeroshot(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = data.load("test")
    y = data.labels(rows)
    idx = subset_indices(y)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda").eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n<|im_start|>user\n"
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    instruct = "Given a bank customer's message, judge whether this team should handle it"

    def prompts(message: str) -> list[str]:
        return [f"{pre}<Instruct>: {instruct}\n<Query>: {message}\n<Document>: {data.TEAMS[t][0]}{suf}"
                for t in data.TEAM_IDS]

    tok.padding_side = "left"
    preds, ms = [], []
    for n, i in enumerate(idx, 1):
        enc = tok(prompts(rows[int(i)]["text"]), return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, [no, yes]].float()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000.0)
        preds.append(int(torch.softmax(logits, -1)[:, 1].argmax()))
    preds = np.array(preds)
    out = {"model_path": args.model_path, "method": "reranker P(yes) per team, argmax, one batch of 8 prompts",
           "n": int(len(idx)), "accuracy": float(np.mean(preds == y[idx])), "per_team": _per_team(preds, y[idx]),
           "ms_p50": _pct(ms[1:], 50), "ms_p95": _pct(ms[1:], 95)}
    _write("rerank-zeroshot.json", out)
    print(json.dumps(out, indent=2))


# ---- encoder speed matrix -------------------------------------------------------------
def cmd_speed(args) -> None:
    import torch

    from jjm.heads.encoders import DEFAULT_PREFIX_TEMPLATE, HFEncoder, encoder_input

    rows = data.load("test")[: args.n]
    c = data.compiled()
    items = [encoder_input(t, c, DEFAULT_PREFIX_TEMPLATE) for t in data.texts(rows)]
    results = []
    for layer in (None, 0.5):
        for batch in (1, 8, 32, 64):
            for reuse in ("off", "on"):
                enc = HFEncoder(args.model_path, model_id=args.model_id, revision=args.revision, layer=layer,
                                pooling="mean", batch_size=batch, prefix_reuse=reuse)
                enc.encode(items[:batch])  # warm
                _, stats = enc.encode(items)
                results.append({"layer": enc.layer, "total_layers": enc.total_layers, "batch_size": batch,
                                "prefix_reuse": reuse, "items": len(items),
                                "ms_per_item": stats.elapsed_ms / len(items),
                                "items_per_second": len(items) / (stats.elapsed_ms / 1000.0)})
                print(results[-1], flush=True)
                del enc
                gc.collect()
                torch.cuda.empty_cache()
    _write("speed.json", {"model": args.model_id, "gpu": torch.cuda.get_device_name(0), "matrix": results})


# ---- the full JJM path -----------------------------------------------------------------
def cmd_engine(args) -> None:
    from jjm.engine import evaluate, evaluate_batch
    from jjm.heads.fit import fit_head

    profiles = _profiles(args.profiles)
    adapter = profiles[args.profile].adapter()
    train, test = data.load("train"), data.load("test")
    c = data.compiled()
    t0 = time.perf_counter()
    fit = fit_head(adapter.featurizer, adapter.store, c, data.texts(train), [r["team"] for r in train],
                   label_source={"kind": "human", "dataset": f"banking77@{data.COMMIT}", "mapping": "trials.routing.data.TEAMS"},
                   promote=True)
    fit_s = time.perf_counter() - t0
    refit_t = time.perf_counter()
    again = fit_head(adapter.featurizer, adapter.store, c, data.texts(train), [r["team"] for r in train],
                     label_source={"kind": "human", "dataset": f"banking77@{data.COMMIT}", "mapping": "trials.routing.data.TEAMS"},
                     promote=True)
    refit_s = time.perf_counter() - refit_t
    y = data.labels(test)
    idx = subset_indices(y)
    from jjm.adapters.head_adapter import METHOD

    # single requests with a cold feature memo: each envelope pays one encoder pass
    adapter.featurizer.memo.clear()
    cache = adapter.featurizer.cache
    adapter.featurizer.cache = None
    single_ms, single_ok, envelopes = [], [], []
    evaluate(_request("warm up", -1, args.profile, [METHOD]), profiles)
    for i in idx[: args.single]:
        t = time.perf_counter()
        resp = evaluate(_request(test[int(i)]["text"], int(i), args.profile, [METHOD]), profiles)
        single_ms.append((time.perf_counter() - t) * 1000.0)
        res = resp["results"][0]
        single_ok.append(res["prediction"]["selected"] == test[int(i)]["team"])
        envelopes.append(res["provenance"]["diagnostics"]["feature_cache"])
    # one batch through evaluate_batch
    adapter.featurizer.memo.clear()
    reqs = [_request(test[int(i)]["text"], int(i), args.profile, [METHOD]) for i in idx]
    t = time.perf_counter()
    responses = evaluate_batch(reqs, profiles)
    batch_ms = (time.perf_counter() - t) * 1000.0
    batch_ok = [r["results"][0]["prediction"]["selected"] == test[int(i)]["team"] for r, i in zip(responses, idx)]
    # the same batch again: every state is an exact repeat
    t = time.perf_counter()
    evaluate_batch(copy.deepcopy(reqs), profiles)
    repeat_ms = (time.perf_counter() - t) * 1000.0
    adapter.featurizer.cache = cache
    sample = responses[0]["results"][0]
    out = {"profile": args.profile, "artifact_id": fit.artifact["artifact_id"], "fit_seconds": fit_s,
           "refit_same_examples": {"reused": again.reused, "seconds": refit_s},
           "cv": fit.artifact["report"]["cv"],
           "single": {"n": len(single_ms), "accuracy": float(np.mean(single_ok)), "ms_p50": _pct(single_ms, 50),
                      "ms_p95": _pct(single_ms, 95), "feature_cache": sorted(set(envelopes))},
           "batch": {"n": len(reqs), "accuracy": float(np.mean(batch_ok)), "ms_per_request": batch_ms / len(reqs)},
           "repeat_batch": {"ms_per_request": repeat_ms / len(reqs)},
           "sample_provenance": {k: sample["provenance"][k] for k in ("method", "validation", "probability_calibration", "runtime", "model")}}
    _write(f"engine-{args.profile}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "sample_provenance"}, indent=2))


def cmd_report(args) -> None:
    lines = ["# Banking77 routing: results", ""]
    for path in sorted(OUT.glob("*.json")):
        lines.append(f"- `{path.name}`")
    (OUT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'index.md'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("encode"); s.add_argument("--profiles", required=True); s.add_argument("--profile", required=True)
    s.add_argument("--name", required=True); s.set_defaults(func=cmd_encode)
    s = sub.add_parser("heads"); s.add_argument("--name", required=True); s.set_defaults(func=cmd_heads)
    s = sub.add_parser("teacher"); s.add_argument("--profiles", required=True); s.add_argument("--teacher", required=True)
    s.add_argument("--split", choices=("test", "train"), required=True); s.add_argument("--limit", type=int)
    s.set_defaults(func=cmd_teacher)
    s = sub.add_parser("distill"); s.add_argument("--name", required=True); s.add_argument("--teacher", required=True)
    s.set_defaults(func=cmd_distill)
    s = sub.add_parser("cascade"); s.add_argument("--name", required=True); s.add_argument("--teacher", required=True)
    s.set_defaults(func=cmd_cascade)
    s = sub.add_parser("readout"); s.add_argument("--profiles", required=True); s.add_argument("--profile", required=True)
    s.add_argument("--limit", type=int); s.set_defaults(func=cmd_readout)
    s = sub.add_parser("rerank-zeroshot"); s.add_argument("--model-path", required=True)
    s.set_defaults(func=cmd_rerank_zeroshot)
    s = sub.add_parser("speed"); s.add_argument("--model-path", required=True)
    s.add_argument("--model-id", default="Qwen/Qwen3-Reranker-0.6B"); s.add_argument("--revision")
    s.add_argument("--n", type=int, default=512); s.set_defaults(func=cmd_speed)
    s = sub.add_parser("engine"); s.add_argument("--profiles", required=True); s.add_argument("--profile", required=True)
    s.add_argument("--single", type=int, default=100); s.set_defaults(func=cmd_engine)
    s = sub.add_parser("report"); s.set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
