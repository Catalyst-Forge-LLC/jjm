"""Reproducible Banking77 comparison. See README.md for scope and limitations."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from jjm.compile import render_state
from jjm.engine import evaluate, evaluate_batch
from jjm.profiles import Profile
from jjm.heads.fit import holdout_positions
import task_data as data
from laya_adapter import LayaAdapter, questions, readout

ROOT = Path(__file__).resolve().parent
PINS = json.loads((ROOT / "pins.json").read_text())
TEMPLATE = json.loads((ROOT / "request_template.json").read_text())

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed(fn):
    sync()
    start = time.perf_counter()
    value = fn()
    sync()
    return value, (time.perf_counter() - start) * 1000

def stats(times):
    return {"n": len(times), "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times))}

def wilson(success, n):
    if not n:
        return None
    z = 1.95996398454
    p = success / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(center - half), float(center + half)]

def metrics(weights, labels):
    pred = weights.argmax(1)
    correct = pred == labels
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "accuracy_wilson95": wilson(int(correct.sum()), len(labels)),
            "macro_f1": float(f1_score(labels, pred, labels=list(range(8)), average="macro", zero_division=0)),
            "log_loss_on_readout": float(log_loss(labels, weights, labels=list(range(8)))),
            "per_team_accuracy": {t: float(correct[labels == i].mean()) for i, t in enumerate(data.TEAM_IDS)},
            "warning": "Normalized readouts are not certified calibrated probabilities."}

def selection(weights, labels, threshold):
    take = weights.max(1) >= threshold
    n = int(take.sum())
    errors = int((weights.argmax(1)[take] != labels[take]).sum())
    return {"threshold": threshold, "accepted": n, "coverage": float(take.mean()),
            "errors": errors, "error_rate": errors/n if n else None,
            "error_wilson95": wilson(errors, n)}

def threshold_from_validation(weights, labels, target_error, min_accepted):
    # Empirical selection, not a statistical risk guarantee. Test remains untouched.
    candidates = sorted(set([0.0, *map(float, weights.max(1))]))
    for t in candidates:
        r = selection(weights, labels, t)
        if r["accepted"] >= min_accepted and r["error_rate"] <= target_error:
            return t
    return None

def sample(rows, per_class, seed):
    if not per_class:
        return list(rows)
    return [rows[i] for i in data.stratified(data.labels(rows), per_class, seed)]

def latency_sample(rows, n, seed):
    """Seeded proportional team sample, shuffled once and shared by every route."""
    n = min(n, len(rows))
    y = data.labels(rows)
    teams, counts = np.unique(y, return_counts=True)
    quotas = counts * n / len(rows)
    sizes = np.floor(quotas).astype(int)
    remainder = n - int(sizes.sum())
    sizes[np.argsort(-(quotas - sizes), kind="stable")[:remainder]] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for team, size in zip(teams, sizes):
        chosen.extend(rng.choice(np.flatnonzero(y == team), size=int(size), replace=False).tolist())
    rng.shuffle(chosen)
    return [rows[i] for i in chosen]


def hardware_metadata():
    info = {"cuda_runtime": torch.version.cuda, "torch_threads": torch.get_num_threads(),
            "cpu": platform.processor(), "cudnn_version": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        info.update(gpu_total_bytes=total, gpu_free_bytes_before_run=free,
                    compute_capability=[gpu.major, gpu.minor],
                    bfloat16_supported=torch.cuda.is_bf16_supported())
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,power.limit",
                                     "--format=csv"], capture_output=True, text=True, timeout=10, check=True)
            info["nvidia_smi"] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            info["nvidia_smi_error"] = str(exc)
    return info


def load_rows(args):
    rows = data.load("train")
    for i, row in enumerate(rows):
        row["row_id"] = f"train:{i}"
    validation = sample(rows, args.val_per_team, args.seed + 1)
    used = {r["row_id"] for r in validation}
    train = sample([r for r in rows if r["row_id"] not in used], args.train_per_team, args.seed)
    test = data.load("test")
    for i, row in enumerate(test):
        row["row_id"] = f"test:{i}"
    test = sample(test, args.test_per_team, args.seed + 2)
    # Reject exact cross-split duplicate states rather than allowing cached inputs or leakage.
    sets = [{r["text"] for r in group} for group in (train, validation, test)]
    if any(sets[i] & sets[j] for i, j in ((0,1),(0,2),(1,2))):
        raise RuntimeError("Exact text overlaps between splits. Resolve the split before benchmarking.")
    return train, validation, test

def request(row, adapter):
    req = copy.deepcopy(TEMPLATE)
    req.update(request_id="bench-" + row["row_id"], scope_id="banking77.benchmark", profile_id="bench")
    req["state"].update(id=row["row_id"], observed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                         content={"message": row["text"]})
    req["criteria"] = [data.criterion()]
    req["requirements"].update(min_top_mass=None, allowed_methods=[adapter.method])
    req["timeout_ms"] = 300000
    return req

def profile(adapter):
    return {"bench": Profile("bench", "experiment-1", "real", adapter.adapter_id, {}, lambda: adapter)}

def envelope_weights(envelopes):
    out = []
    for envelope in envelopes:
        result = envelope["results"][0]
        prediction = result["prediction"]
        if prediction is None:
            raise RuntimeError(json.dumps(result))
        out.append([prediction["weights"][t] for t in data.TEAM_IDS])
    return np.array(out)

def tfidf(args, train, validation, test):
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    # Match the actual JJM fit slice, including its deterministic internal holdout.
    fit_at, held_at = holdout_positions(data.texts(train), [r["team"] for r in train])
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True),
                             LogisticRegression(C=4, max_iter=1000, random_state=args.seed))
    _, fit_ms = timed(lambda: pipeline.fit([train[i]["text"] for i in fit_at], data.labels(train)[fit_at]))
    def predict(rows):
        return pipeline.predict_proba([r["text"] for r in rows])
    vp, _ = timed(lambda: predict(validation))
    tp, batch_ms = timed(lambda: predict(test))
    predict(test[:2])
    times = [timed(lambda r=r: predict([r]))[1] for r in latency_sample(test, args.latency_n, args.seed + 3)]
    return vp, tp, {"fit_ms": fit_ms, "task_labels_supplied": len(train), "actual_fit_rows": len(fit_at),
                    "internal_holdout_rows": len(held_at), "warm_batch1": stats(times),
                    "test_batch_ms": batch_ms, "test_rows_per_second": len(test)/(batch_ms/1000)}

def run_jjm(args, train, validation, test):
    from jjm.heads.encoders import HFEncoder
    from jjm.heads.runtime import Featurizer
    from jjm.heads.artifact import ArtifactStore
    from jjm.heads.fit import fit_head
    from jjm.adapters.head_adapter import FittedHeadAdapter
    dtype = "float32" if args.device == "cpu" else "bfloat16"
    encoder, load_ms = timed(lambda: HFEncoder(PINS["embedding_model"], model_id=PINS["embedding_model"],
        revision=PINS["embedding_revision"], device=args.device, dtype=dtype, pooling="last",
        batch_size=args.batch_size, max_tokens=256))
    # The encoder's token counter caps body length, so audit original tokens explicitly.
    for row in train + validation + test:
        if len(encoder.tok.encode(data.texts([row])[0], add_special_tokens=False)) > 256:
            raise RuntimeError("JJM state would be truncated")
    features = Featurizer(encoder, None)
    store = ArtifactStore(args.out / "heads")
    fit, fit_ms = timed(lambda: fit_head(features, store, data.compiled(), data.texts(train),
        [r["team"] for r in train], method="softmax", label_source={"kind": "human"}, promote=True))
    adapter = FittedHeadAdapter(features, store, max_context_tokens=None)
    profiles = profile(adapter)
    def predict(rows):
        features.memo.clear()
        return evaluate_batch([request(r, adapter) for r in rows], profiles)
    ve, _ = timed(lambda: predict(validation))
    te, batch_ms = timed(lambda: predict(test))
    predict(test[:2])
    times = []
    for row in latency_sample(test, args.latency_n, args.seed + 3):
        features.memo.clear()
        _, ms = timed(lambda: evaluate(request(row, adapter), profiles))
        times.append(ms)
    write(args.out / "jjm-envelope.json", te[0])
    info = {"load_ms_including_any_download": load_ms, "fit_ms": fit_ms, "feature_ms": fit.feature_ms,
            "task_labels_supplied": len(train), "actual_fit_rows": fit.artifact["training"]["fit_count"],
            "fit_reused": fit.reused, "dtype": dtype, "cache_policy": "no disk cache, memo cleared before each timed operation",
            "warm_batch1": stats(times), "test_batch_ms": batch_ms,
            "test_rows_per_second": len(test)/(batch_ms/1000),
            "validation_status": "fitted_unvalidated", "probability_calibration": "not_assessed"}
    return envelope_weights(ve), envelope_weights(te), info

def run_laya(args, train, validation, test):
    import laya
    from laya.common import build_head, render_options
    agent, load_ms = timed(lambda: laya.load(PINS["laya_model"], revision=PINS["laya_revision"],
                                            device=args.device, backend="eager"))
    comp = data.compiled()
    q = questions(comp)
    internal = agent._to_internal(q["route"])
    full_option_lengths = [len(agent.tok.encode(" " + x.replace(agent.tok.mask_token, " "), add_special_tokens=False)) for x in render_options(internal)]
    head_ids, _, head_stats = build_head(agent.tok, internal, args.head_max_len)
    write(args.out / "laya-token-budget.json", {"head_tokens": len(head_ids), "head_stats": head_stats,
          "full_option_token_lengths": full_option_lengths, "per_option_upstream_cap": 48,
          "max_len": 512, "head_max_len": args.head_max_len})
    if max(full_option_lengths) > 48:
        raise RuntimeError("A Laya option description exceeds upstream's 48-token cap")
    if head_stats["tokens_per_option"] is not None:
        raise RuntimeError("Laya's head budget shortens options. Increase --head-max-len.")
    instruction_length = len(agent.tok.encode("choice question: " + internal["ins"], add_special_tokens=False))
    if len(head_ids) != 3 + instruction_length + sum(full_option_lengths) + len(full_option_lengths):
        raise RuntimeError("Laya's question head would be shortened")
    def predict(rows):
        return agent.predict_batch(data.texts(rows), q, batch_size=args.batch_size,
                                   max_len=512, head_max_len=args.head_max_len)
    predict(test[:2])  # Warm-up outside throughput and latency measurements.
    vr, _ = timed(lambda: predict(validation))
    tr, batch_ms = timed(lambda: predict(test))
    vp = np.array([readout(r, data.TEAM_IDS)[0] for r in vr])
    tp = np.array([readout(r, data.TEAM_IDS)[0] for r in tr])
    adapter = LayaAdapter(agent, PINS["laya_model"], PINS["laya_revision"], args.head_max_len)
    profiles = profile(adapter)
    direct_times, wrapped_times, max_deltas = [], [], []
    first = None
    # Alternate order to reduce one-sided clock or thermal drift.
    for i, row in enumerate(latency_sample(test, args.latency_n, args.seed + 3)):
        text = data.texts([row])[0]
        direct = lambda: agent.predict(text, q, max_len=512, head_max_len=args.head_max_len)
        wrapped = lambda: evaluate(request(row, adapter), profiles)
        if i % 2:
            env, wm = timed(wrapped)
            raw, dm = timed(direct)
        else:
            raw, dm = timed(direct)
            env, wm = timed(wrapped)
        first = first or env
        delta = float(np.max(np.abs(np.array(readout(raw, data.TEAM_IDS)[0]) - envelope_weights([env])[0])))
        max_deltas.append(delta)
        direct_times.append(dm)
        wrapped_times.append(wm)
    write(args.out / "laya-through-jjm-envelope.json", first)
    write(args.out / "laya-raw-example.json", tr[0])
    if max(max_deltas) > 1e-6:
        raise RuntimeError("Laya direct and adapter predictions differ")
    info = {"load_ms_including_any_download": load_ms, "task_labels_used_for_model_fit": 0,
            "upstream_training_overlap_with_banking77": "unknown", "dtype": str(agent.dtype),
            "warm_batch1": stats(direct_times), "jjm_wrapper_warm_batch1": stats(wrapped_times),
            "paired_wrapper_delta_ms_median": float(np.median(np.array(wrapped_times)-direct_times)),
            "adapter_max_abs_weight_delta": max(max_deltas), "adapter_parity_n": len(max_deltas),
            "test_batch_ms": batch_ms, "test_rows_per_second": len(test)/(batch_ms/1000),
            "probability_calibration": "not_assessed_on_this_task", "state_truncation_count": 0}
    return vp, tp, info

def report(args, summary, predictions, test):
    if "jjm" in predictions and "laya" in predictions:
        from scipy.stats import binomtest
        y = data.labels(test)
        a = predictions["jjm"].argmax(1) == y
        b = predictions["laya"].argmax(1) == y
        aw = int((a & ~b).sum())
        bw = int((b & ~a).sum())
        summary["paired_comparison"] = {"jjm_only_correct": aw, "laya_only_correct": bw,
            "both_correct": int((a & b).sum()), "both_wrong": int((~a & ~b).sum()),
            "mcnemar_exact_two_sided_p": float(binomtest(aw, aw+bw).pvalue) if aw+bw else 1.0,
            "warning": "One task and seed. Pretrained data overlap unknown. No broad superiority claim."}
    write(args.out / "summary.json", summary)
    lines = ["# JJM / Laya Banking77 experiment", "", summary["run_scope"], "",
             "| Route | Test accuracy | Macro F1 | Warm batch 1 p50 / p95 ms | Batch rows/s |",
             "|---|---:|---:|---:|---:|"]
    for name, result in summary["routes"].items():
        m, perf = result["test"], result["performance"]
        latency = perf["warm_batch1"]
        lines.append(f"| {name} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} | {latency['p50_ms']:.1f} / {latency['p95_ms']:.1f} | {perf['test_rows_per_second']:.1f} |")
    lines += ["", "Timings are local observations on the recorded device. Batch throughput is not request latency.",
              "JJM and TF-IDF use supplied task labels. Laya receives no local model-fit labels, but uses validation labels for threshold selection.",
              "Thresholds are selected on a separate portion of the training split. The test split is never used to choose them.",
              "Readouts are not certified calibrated. JJM validation is left unaccepted. See summary.json and per-row JSONL for details."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--routes", nargs="+", choices=["tfidf", "jjm", "laya"], default=["tfidf", "jjm", "laya"])
    p.add_argument("--train-per-team", type=int, default=8, help="0 means all remaining train rows")
    p.add_argument("--val-per-team", type=int, default=2)
    p.add_argument("--test-per-team", type=int, default=4, help="0 means full test split")
    p.add_argument("--latency-n", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--head-max-len", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-error", type=float, default=0.02)
    p.add_argument("--min-val-accepted", type=int, default=20)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "cpu-smoke")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested, but this Python environment has no usable CUDA device")
    if min(args.val_per_team, args.latency_n, args.batch_size, args.threads) < 1 or min(args.train_per_team, args.test_per_team) < 0:
        p.error("Invalid row, batch or thread count")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Output directory is not empty. Use a new name to preserve run evidence.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, validation, test = load_rows(args)
    args.latency_n = min(args.latency_n, len(test))
    split_ids = {k: [r["row_id"] for r in v] for k, v in (("train",train),("validation",validation),("test",test))}
    write(args.out / "split.json", split_ids)
    latency = latency_sample(test, args.latency_n, args.seed + 3)
    latency_ids = [r["row_id"] for r in latency]
    write(args.out / "latency-sample.json", {"method": "seeded proportional team sample, shuffled",
          "seed": args.seed + 3, "row_ids": latency_ids,
          "team_counts": {t: sum(r["team"] == t for r in latency) for t in data.TEAM_IDS}})
    summary = {"pins": PINS, "arguments": {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "run_scope": "Smoke experiment only. Insufficient sample to rank products." if len(test) < 400 else "Single-task experiment. Interpret alongside label budget and data-overlap caveats.",
        "split_sha256": hashlib.sha256(json.dumps(split_ids,sort_keys=True).encode()).hexdigest(),
        "latency_sample_sha256": hashlib.sha256(json.dumps(latency_ids).encode()).hexdigest(),
        "harness_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ("bench.py", "task_data.py", "laya_adapter.py", "requirements.txt")},
        "rows": {k:len(v) for k,v in split_ids.items()}, "environment": {"platform":platform.platform(),
            "python":sys.version, "torch":torch.__version__, "cpu_count":os.cpu_count(),
            "hardware": hardware_metadata(),
            "device":torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
            "packages": {p:importlib.metadata.version(p) for p in ("laya","transformers","huggingface-hub","numpy","scipy","scikit-learn","jjm-engine","anyjev")}},
        "routes": {}}
    predictions = {}
    funcs = {"tfidf":tfidf,"jjm":run_jjm,"laya":run_laya}
    for name in args.routes:
        print(f"Running {name}: {len(train)} train, {len(validation)} validation, {len(test)} test",flush=True)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        vp, tp, perf = funcs[name](args, train, validation, test)
        if args.device == "cuda":
            perf["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        threshold = threshold_from_validation(vp, data.labels(validation), args.target_error, args.min_val_accepted)
        result = {"performance":perf, "validation":metrics(vp,data.labels(validation)), "test":metrics(tp,data.labels(test)),
                  "selective_test":selection(tp,data.labels(test),threshold) if threshold is not None else None,
                  "threshold_status":"selected_on_validation" if threshold is not None else "no_qualifying_validation_threshold"}
        summary["routes"][name] = result
        predictions[name] = tp
        with (args.out / f"{name}-predictions.jsonl").open("w",encoding="utf-8") as out:
            for split,rows,weights in (("validation",validation,vp),("test",test,tp)):
                for row,w in zip(rows,weights):
                    out.write(json.dumps({**row,"split":split,"weights":dict(zip(data.TEAM_IDS,map(float,w))),
                        "predicted":data.TEAM_IDS[int(w.argmax())]},ensure_ascii=False)+"\n")
        print(f"{name}: accuracy {result['test']['accuracy']:.3f}, p50 {perf['warm_batch1']['p50_ms']:.1f} ms",flush=True)
        report(args,summary,predictions,test)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Saved {args.out / 'REPORT.md'}",flush=True)

if __name__ == "__main__":
    main()
