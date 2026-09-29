"""Run the fit-rank shadow trial over an Exec Foundry export.

    uv run python -m trials.fit_rank.run --input export.jsonl --profile synthetic
    uv run python -m trials.fit_rank.run --input export.jsonl --profiles my-profiles.json \
        --profile local-advisory --hardware "RTX 4000 Ada 12 GB, CUDA 12"

Writes envelopes.jsonl (one full JJM envelope per job), summary.json, and summary.md.
Nothing is sent to Exec Foundry, and nothing on its board changes.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jjm import ENGINE_VERSION, errors
from jjm.engine import evaluate
from jjm.profiles import load_profiles
from trials.fit_rank import metrics
from trials.fit_rank.build import RecordError, build_request, labels, validate_record
from trials.fit_rank.criterion import CRITERION_STATUS, CRITERION_VERSION, RANK_BY_OPTION


def read_export(path: Path) -> list[dict[str, Any]]:
    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecordError(f"{path.name} line {n}: not JSON ({exc.msg})") from None
        records.append(validate_record(record))
    ids = [r["job_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise RecordError("job_id values repeat in the export")
    return records


def _row(record: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    result = response["results"][0]
    p = result["prediction"]
    return {
        "job_id": record["job_id"],
        "status": result["status"],
        "acceptance": result["acceptance"]["status"],
        "reason_codes": result["acceptance"]["reason_codes"],
        "error": result["error"],
        "jjm_rank": RANK_BY_OPTION[p["selected"]] if p else None,
        "weights": p["weights"] if p else None,
        "top_mass": p["top_mass"] if p else None,
        "margin": p["margin"] if p else None,
        "elapsed_ms": response["elapsed_ms"],
        "context_tokens": result["provenance"]["diagnostics"].get("context_tokens"),
        "criterion_digest": result["provenance"]["criterion_digest"],
        **labels(record),
    }


def _pairs(rows: list[dict[str, Any]], a: str, b: str) -> tuple[list[int], list[int]]:
    keep = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
    return [r[a] for r in keep], [r[b] for r in keep]


def summarize(rows: list[dict[str, Any]], repeats: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    warm = [r["elapsed_ms"] for r in evaluated[1:]]
    comparisons = {}
    for name, (a, b) in {
        "jjm_vs_sonnet": ("jjm_rank", "sonnet_rank"),
        "jjm_vs_owner": ("jjm_rank", "owner_label"),
        "jjm_vs_board": ("jjm_rank", "board_rank"),
        "sonnet_vs_owner": ("sonnet_rank", "owner_label"),
        "sonnet_rerun_vs_sonnet": ("sonnet_rerun_rank", "sonnet_rank"),
    }.items():
        predicted, reference = _pairs(rows, a, b)
        comparisons[name] = metrics.agreement(predicted, reference)
    overrides = [r for r in rows if r["board_rank"] is not None and r["sonnet_rank"] is not None
                 and r["board_rank"] != r["sonnet_rank"]]
    predicted, reference = _pairs(overrides, "jjm_rank", "board_rank")
    comparisons["jjm_vs_owner_overrides"] = metrics.agreement(predicted, reference)
    tokens = [r["context_tokens"] for r in rows if r["context_tokens"] is not None]
    return {
        **meta,
        "counts": {
            "jobs": len(rows),
            "evaluated": len(evaluated),
            "abstained": sum(r["status"] == "abstained" for r in rows),
            "error": sum(r["status"] == "error" for r in rows),
            "abstain_reasons": sorted({c for r in rows if r["status"] == "abstained" for c in r["reason_codes"]}),
            "error_codes": sorted({r["error"]["code"] for r in rows if r["error"]}),
        },
        "comparisons": comparisons,
        "latency_ms": {
            "first_request": evaluated[0]["elapsed_ms"] if evaluated else None,
            "warm_n": len(warm),
            "warm_p50": metrics.percentile(warm, 0.50),
            "warm_p95": metrics.percentile(warm, 0.95),
            "note": "JJM wall time per job, model already loaded. Sonnet's latency also includes writing the "
                    "justification, so the two are not a like-for-like comparison.",
        },
        "context_tokens": {"n": len(tokens), "p50": metrics.percentile(tokens, 0.5),
                           "max": max(tokens) if tokens else None},
        "determinism": {
            "repeated": len(repeats),
            "same_selection": sum(x["same_selection"] for x in repeats),
            "max_abs_weight_diff": max((x["max_abs_diff"] for x in repeats), default=None),
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"[{_fmt(value[0])}, {_fmt(value[1])}]"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# Fit-rank shadow trial: {summary['run_id']}", ""]
    if summary["execution_mode"] == "synthetic":
        lines += ["**Synthetic profile. These numbers come from hashed weights and mean nothing about fit.**", ""]
    if summary["criterion_status"] != "frozen":
        lines += [f"Criterion wording is `{summary['criterion_status']}` ({summary['criterion_version']}). "
                  "Results are not comparable to runs under other wording.", ""]
    c = summary["counts"]
    lines += [f"Profile `{summary['profile_id']}` ({summary['execution_mode']}), model `{summary['model']['id']}`, "
              f"hardware: {summary['hardware'] or 'not stated'}.", "",
              f"{c['jobs']} jobs: {c['evaluated']} evaluated, {c['abstained']} abstained, {c['error']} errors.", "",
              "| Comparison | n | Exact | Within one | QWK | Mean offset |", "| --- | --- | --- | --- | --- | --- |"]
    for name, m in summary["comparisons"].items():
        if not m.get("n"):
            lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(f"| {name} | {m['n']} | {_fmt(m['exact'])} {_fmt(m['exact_ci95'])} | "
                     f"{_fmt(m['within_one'])} | {_fmt(m['quadratic_weighted_kappa'])} | "
                     f"{_fmt(m['mean_signed_offset'], 2)} |")
    lat = summary["latency_ms"]
    det = summary["determinism"]
    lines += ["", f"Model load: {_fmt(summary['model_load_ms'], 0)} ms. First request: "
              f"{_fmt(lat['first_request'], 0)} ms. Warm p50 {_fmt(lat['warm_p50'], 0)} ms, "
              f"p95 {_fmt(lat['warm_p95'], 0)} ms over {lat['warm_n']} jobs.", "", lat["note"], "",
              f"Determinism: {det['same_selection']} of {det['repeated']} repeated jobs kept their selection; "
              f"largest weight change {_fmt(det['max_abs_weight_diff'], 9)}.", ""]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--profile", default="synthetic")
    parser.add_argument("--profiles")
    parser.add_argument("--hardware", help="required for real profiles, e.g. 'RTX 4000 Ada 12 GB, CUDA 12'")
    parser.add_argument("--repeat", type=int, default=10, help="re-run this many evaluated jobs for determinism")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        records = read_export(args.input)[: args.limit]
        profiles = load_profiles(args.profiles)
        if args.profile not in profiles:
            raise RecordError(f"profile {args.profile!r} is not configured")
        profile = profiles[args.profile]
        if profile.execution_mode == "real" and not args.hardware:
            raise RecordError("--hardware is required for a real profile so latency is reported against it")
        t0 = time.monotonic()
        adapter = profile.adapter()
        load_ms = (time.monotonic() - t0) * 1000.0
    except (RecordError, errors.JJMError) as exc:
        print(f"fit-rank trial: {exc}", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("fit-rank-%Y%m%dT%H%M%SZ")
    out = args.out or Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows, envelopes = [], []
    for i, record in enumerate(records, 1):
        response = evaluate(build_request(record, profile_id=profile.id), profiles)
        envelopes.append({"job_id": record["job_id"], "envelope": response})
        rows.append(_row(record, response))
        print(f"[{i}/{len(records)}] {record['job_id']}: {rows[-1]['status']} rank={rows[-1]['jjm_rank']}",
              file=sys.stderr)
    repeats = []
    for row, record in [(r, rec) for r, rec in zip(rows, records) if r["status"] == "evaluated"][: args.repeat]:
        again = _row(record, evaluate(build_request(record, profile_id=profile.id, request_suffix="-repeat"),
                                      profiles))
        diff = max(abs(again["weights"][k] - row["weights"][k]) for k in row["weights"]) if again["weights"] else None
        repeats.append({"job_id": row["job_id"], "same_selection": again["jjm_rank"] == row["jjm_rank"],
                        "max_abs_diff": diff})
    meta = {
        "run_id": run_id,
        "engine_version": ENGINE_VERSION,
        "profile_id": profile.id,
        "profile_generation": profile.generation,
        "execution_mode": profile.execution_mode,
        "model": adapter.model_identity(),
        "hardware": args.hardware,
        "platform": f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
        "criterion_status": CRITERION_STATUS,
        "criterion_version": CRITERION_VERSION,
        "criterion_digests": sorted({r["criterion_digest"] for r in rows}),
        "model_load_ms": load_ms,
        "input": args.input.name,
    }
    summary = summarize(rows, repeats, meta)
    with (out / "envelopes.jsonl").open("w", encoding="utf-8") as fh:
        for item in envelopes:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"fit-rank trial: wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
