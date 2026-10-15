"""Measure the integrated CPU backend on the frozen benchmark's full split.

Run from the repository root after uv sync --extra text:
  uv run python trials/routing/tfidf_backend.py --out runs/tfidf-backend-check
Requires the pinned CSV files downloaded by the original benchmark bootstrap.
Writes fresh artifacts and results; never rewrites the original benchmark.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from jjm.adapters.tfidf_adapter import TfidfAdapter
from jjm.compile import render_state
from jjm.engine import evaluate, evaluate_batch
from jjm.heads.artifact import ArtifactStore
from jjm.heads.head import classification_report
from jjm.heads.tfidf import METHOD
from jjm.profiles import Profile

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks/jjm-laya-local-benchmark"
SETTINGS = {"ngram_range": [1, 2], "min_df": 1, "max_features": None, "C": 4.0, "max_iter": 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory to preserve previous measurements")
    spec = importlib.util.spec_from_file_location("frozen_task_data", BENCH / "task_data.py")
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    split_path = BENCH / "results/gpu-full-20261015/split.json"
    split = json.loads(split_path.read_text())
    train_all, test_all = data.load("train"), data.load("test")
    def rows(ids, source):
        return [source[int(i.split(":")[1])] for i in ids]
    train, test = rows(split["train"], train_all), rows(split["test"], test_all)
    args.out.mkdir(parents=True)
    store = ArtifactStore(args.out / "heads")
    adapter = TfidfAdapter(store, SETTINGS)
    started = time.perf_counter()
    fit = adapter.fit(data.compiled(), data.texts(train), [r["team"] for r in train],
                      label_source={"kind": "human"}, promote=True)
    fit_ms = (time.perf_counter() - started) * 1000
    # Reload persisted data before scoring.
    adapter = TfidfAdapter(store, SETTINGS)
    started = time.perf_counter()
    art, model = adapter._serve(data.compiled())
    reload_ms = (time.perf_counter() - started) * 1000
    profiles = {"text": Profile("text", "measurement-1", "real", "tfidf", {}, lambda: adapter)}
    template = json.loads((ROOT / "examples/tfidf/request.json").read_text())
    template["profile_id"] = "text"
    template["criteria"] = [data.criterion()]
    requests = []
    for i, row in enumerate(test):
        req = copy.deepcopy(template)
        req["request_id"] = f"tfidf-test-{i}"
        req["state"]["content"] = {"message": row["text"]}
        requests.append(req)
    started = time.perf_counter()
    responses = evaluate_batch(requests, profiles)
    batch_ms = (time.perf_counter() - started) * 1000
    weights = np.array([[res["results"][0]["prediction"]["weights"][k] for k in data.TEAM_IDS] for res in responses])
    direct = model.weights([render_state(req["state"]) for req in requests])
    delta = float(np.max(np.abs(weights - direct)))
    assert delta < 1e-12
    # Reuse the original preselected, shuffled latency sample (no test-label tuning).
    sample_path = BENCH / "results/gpu-full-20261015/latency-sample.json"
    sample = json.loads(sample_path.read_text())
    sample_ids = sample["row_ids"]
    lookup = dict(zip(split["test"], requests))
    for req in list(lookup.values())[:5]:
        evaluate(req, profiles)
    latency, bare = [], []
    for row_id in sample_ids:
        req = lookup[row_id]
        started = time.perf_counter()
        evaluate(req, profiles)
        latency.append((time.perf_counter() - started) * 1000)
        text = render_state(req["state"])
        started = time.perf_counter()
        model.weights([text])
        bare.append((time.perf_counter() - started) * 1000)
    def stats(values):
        return {"n": len(values), "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95))}
    sources = ["src/jjm/heads/tfidf.py", "src/jjm/adapters/tfidf_adapter.py", "trials/routing/tfidf_backend.py"]
    report = {"purpose": "Integrated TF-IDF backend verification; separate from original raw-text baseline",
              "date": time.strftime("%Y-%m-%d"), "platform": platform.platform(), "python": platform.python_version(),
              "packages": {p: version(p) for p in ("jjm-engine", "scikit-learn", "numpy", "scipy")},
              "neural_modules_loaded": [p for p in ("torch", "transformers", "laya") if p in sys.modules],
              "settings": SETTINGS, "artifact_id": art["artifact_id"], "payload_digest": art["payload_digest"],
              "training": art["training"], "heldout": art["report"]["heldout"],
              "test": classification_report(weights, data.labels(test), len(data.TEAM_IDS)),
              "direct_vs_contract_max_weight_delta": delta, "fit_ms": fit_ms, "artifact_reload_ms": reload_ms,
              "warm_contract": stats(latency), "warm_bare_classifier": stats(bare),
              "evaluate_batch_ms": batch_ms, "evaluate_batch_rows_per_second": len(test) / (batch_ms / 1000),
              "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
              "latency_sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
              "sources": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              "note": "One run on a shared laptop; rendered JJM state and fixed C. No calibrated probabilities."}
    (args.out/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (args.out/"predictions.jsonl").open("w",encoding="utf-8") as f:
        for row_id, row, w in zip(split["test"], test, weights):
            f.write(json.dumps({"row_id": row_id, "expected": row["team"],
                                "selected": data.TEAM_IDS[int(w.argmax())], "weights": w.tolist()})+"\n")
    (args.out/"envelope.json").write_text(json.dumps(responses[0],indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
