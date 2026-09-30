"""Zero-shot Nimble and Tev1 on the fit jobs, reading the same words as the heads.

Two texts, same five options, same jobs:

- rendered: the JJM state the engine head read
- probe: the dedicated prompt (candidate as the query, listing as the document)

Sonnet's rank is the teacher, not ground truth. Nothing here is a JJM envelope.
The fitted head on the probe text is a separate GPU pass, so this script does not load it.

    uv run python -m trials.fit_rank.same_text --input export.jsonl --model nimble --model tev1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jjm.compile import render_state
from trials.fit_rank import metrics
from trials.fit_rank.build import build_request
from trials.fit_rank.criterion import LEVELS, QUESTION, RANK_BY_OPTION
from trials.fit_rank.features import FIT_INSTRUCTION, candidate_text, listing_text
from trials.fit_rank.probe_eval import auc, spearman
from trials.fit_rank.run import read_export
from trials.routing.systemone import PromptTooLong, ask, parse_choice

HOST = "http://127.0.0.1:11434"
# The endpoint rejects bodies over 64 KiB. Leave room for the question JSON.
STATE_MAX_CHARS = 48_000
OPTION_IDS = [option for _, (option, _) in sorted(LEVELS.items())]


def rendered_text(record: dict[str, Any]) -> str:
    return render_state(build_request(record, profile_id="fit-same")["state"])


def probe_text(record: dict[str, Any]) -> str:
    return (f"<Instruct>: {FIT_INSTRUCTION}\n<Query>: {candidate_text(record)}\n"
            f"<Document>: {listing_text(record)}")


def fit_questions() -> dict[str, Any]:
    return {"rank": {"type": "choice", "instructions": QUESTION,
                     "criteria": {option: text for _, (option, text) in sorted(LEVELS.items())}}}


def score_model(model: str, records: list[dict[str, Any]], text_of: Callable[[dict[str, Any]], str],
                *, host: str) -> dict[str, Any]:
    questions = fit_questions()
    rows, skipped = [], []
    warm = next((r for r in records if len(text_of(r)) <= STATE_MAX_CHARS), None)
    if warm is not None:
        try:
            ask(model, text_of(warm), host=host, questions=questions, name="rank")
        except PromptTooLong:
            pass
    for n, record in enumerate(records, 1):
        state = text_of(record)
        if len(state) > STATE_MAX_CHARS:
            skipped.append(record["job_id"])
            continue
        t = time.perf_counter()
        try:
            payload = ask(model, state, host=host, questions=questions, name="rank")
        except PromptTooLong:
            skipped.append(record["job_id"])
            continue
        elapsed = (time.perf_counter() - t) * 1000.0
        choice = parse_choice(payload, "rank", OPTION_IDS)
        probs = ((payload.get("answers") or {}).get("rank") or {}).get("probabilities") or {}
        expected = None
        if choice is not None and all(option in probs for option in OPTION_IDS):
            expected = sum(RANK_BY_OPTION[option] * float(probs[option]) for option in OPTION_IDS)
        rows.append({"job_id": record["job_id"], "sonnet_rank": record["sonnet_rank"],
                     "choice": choice, "jjm_rank": RANK_BY_OPTION[choice] if choice else None,
                     "expected_rank": expected, "ms": elapsed})
        if n % 25 == 0:
            got = [r["jjm_rank"] for r in rows if r["jjm_rank"] is not None]
            ref = [r["sonnet_rank"] for r in rows if r["jjm_rank"] is not None]
            exact = metrics.exact(got, ref) if ref else float("nan")
            print(f"[{model} {n}/{len(records)}] exact {exact:.3f} median {float(np.median([r['ms'] for r in rows])):.0f} ms "
                  f"skipped {len(skipped)}", flush=True)
    kept = [r for r in rows if r["jjm_rank"] is not None]
    predicted = [r["jjm_rank"] for r in kept]
    reference = [r["sonnet_rank"] for r in kept]
    expected = [r["expected_rank"] for r in kept if r["expected_rank"] is not None]
    expected_ref = [r["sonnet_rank"] for r in kept if r["expected_rank"] is not None]
    summary = {
        "model": model,
        "n_jobs": len(records),
        "scored": len(kept),
        "unparsed": sum(r["choice"] is None for r in rows),
        "skipped_over_size": len(skipped),
        "agreement": metrics.agreement(predicted, reference) if kept else {"n": 0},
        "warm_ms_p50": float(np.percentile([r["ms"] for r in rows], 50)) if rows else None,
        "warm_ms_p95": float(np.percentile([r["ms"] for r in rows], 95)) if rows else None,
        "note": "Sonnet is the teacher, not ground truth. Confidence on the endpoint is entropy.",
    }
    if expected:
        exp = np.asarray(expected, dtype=float)
        ref = np.asarray(expected_ref, dtype=int)
        summary["spearman_expected_rank"] = float(spearman(exp, ref))
        summary["auc_sonnet_1_or_2"] = float(auc(-exp, ref <= 2))
    return {"summary": summary, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--text", action="append", choices=("rendered", "probe"), default=None)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--out", type=Path, default=Path("runs/fit-rank-same-text"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    texts = args.text or ["rendered", "probe"]
    builders = {"rendered": rendered_text, "probe": probe_text}
    records = [r for r in read_export(args.input) if r.get("sonnet_rank")]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in texts:
        for model in args.model:
            slug = model.replace(":", "-").replace("/", "-")
            path = args.out / f"{slug}-{name}.json"
            if path.exists() and not args.force:
                print(f"=== {model} / {name} already at {path} ===", flush=True)
                continue
            print(f"=== {model} / {name} ===", flush=True)
            result = score_model(model, records, builders[name], host=args.host)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
