"""Score Banking77 through Ollama's /v1/systemone endpoint.

This is a zero-shot baseline. It does not go through a JJM envelope, and the
endpoint's confidence is entropy, not calibrated correctness.

    uv run python -m trials.routing.systemone --model tev1:0.8b --model tev1 --model nimble

The default slice is the same 400 test messages (50 per team, seed 0) used for
the other zero-shot rows. Pass --per-team 0 to score the full test split.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Collection, Optional

import numpy as np

from trials.routing import data

OUT = Path("runs/routing")
HOST = "http://127.0.0.1:11434"


def question() -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": data.QUESTION,
        "criteria": {team: data.TEAMS[team][0] for team in data.TEAM_IDS},
    }


def parse_choice(payload: dict[str, Any], name: str = "team",
                 allowed: Optional[Collection[str]] = None) -> Optional[str]:
    answer = (payload.get("answers") or {}).get(name) or {}
    choice = answer.get("choice")
    known = data.TEAM_IDS if allowed is None else allowed
    if not isinstance(choice, str) or choice not in known:
        return None
    return choice


def ask(model: str, state: str, *, host: str = HOST, timeout: float = 180.0,
        questions: Optional[dict[str, Any]] = None, name: str = "team") -> dict[str, Any]:
    body = json.dumps({"model": model, "state": state, "keep_alive": "30m",
                       "questions": questions if questions is not None else {name: question()}}).encode("utf-8")
    url = f"{host.rstrip('/')}/v1/systemone"
    last = ""
    for attempt in range(3):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code >= 500 and attempt < 2:
                time.sleep(2.0)
                continue
            raise SystemExit(f"{model}: /v1/systemone returned {exc.code}: {last}") from None
        except urllib.error.URLError as exc:
            if attempt < 2:
                time.sleep(2.0)
                continue
            raise SystemExit(f"Could not reach Ollama at {host}: {exc}") from None
    raise SystemExit(f"{model}: /v1/systemone failed: {last}")


def score(model: str, rows: list[dict[str, str]], indices: np.ndarray, *, host: str = HOST) -> dict[str, Any]:
    y = data.labels(rows)
    ask(model, data.texts([rows[int(indices[0])]])[0], host=host)
    answers, ms = [], []
    for n, i in enumerate(indices, 1):
        state = data.texts([rows[int(i)]])[0]
        t = time.perf_counter()
        payload = ask(model, state, host=host)
        elapsed = (time.perf_counter() - t) * 1000.0
        choice = parse_choice(payload)
        answers.append({"index": int(i), "choice": choice, "ms": elapsed,
                        "confidence": (payload.get("answers") or {}).get("team", {}).get("confidence")})
        ms.append(elapsed)
        if n % 50 == 0:
            got = [data.TEAM_IDS.index(a["choice"]) if a["choice"] else -1 for a in answers]
            print(f"[{model} {n}/{len(indices)}] accuracy {np.mean(np.array(got) == y[indices[:n]]):.3f} "
                  f"median {float(np.median(ms)):.0f} ms", flush=True)
    got = np.array([data.TEAM_IDS.index(a["choice"]) if a["choice"] else -1 for a in answers])
    truth = y[indices]
    return {
        "model": model,
        "n": int(len(indices)),
        "accuracy": float(np.mean(got == truth)),
        "unparsed": int(np.sum(got < 0)),
        "per_team": {team: float(np.mean(got[truth == i] == i)) for i, team in enumerate(data.TEAM_IDS)
                     if np.any(truth == i)},
        "warm_ms_p50": float(np.percentile(ms, 50)),
        "warm_ms_p95": float(np.percentile(ms, 95)),
        "note": "confidence on each answer is 1 - H/ln(N). It is not calibrated correctness.",
        "answers": answers,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--per-team", type=int, default=50, help="0 scores every test message")
    parser.add_argument("--host", default=HOST)
    args = parser.parse_args(argv)
    rows = data.load("test")
    y = data.labels(rows)
    indices = np.arange(len(rows)) if args.per_team == 0 else data.stratified(y, args.per_team, seed=0)
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.model:
        result = score(model, rows, indices, host=args.host)
        slug = model.replace(":", "-").replace("/", "-")
        path = OUT / f"systemone-{slug}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        shown = {k: v for k, v in result.items() if k != "answers"}
        print(json.dumps(shown, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
