"""Command line: JSON in, JSON out.

Exit codes: 0 an envelope was produced (read each result's acceptance, exit 0 is not
approval); 2 invalid request or configuration; 3 execution failure. JSON goes to
stdout and short diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from typing import Any, Optional, Sequence

from jjm import ENGINE_VERSION, errors
from jjm.contract import load_request
from jjm.profiles import capabilities, load_profiles

EXIT_OK, EXIT_INVALID, EXIT_EXECUTION = 0, 2, 3


def _emit(value: Any) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


def _fail(exc: errors.JJMError, exit_code: int) -> int:
    _emit({"schema_version": "0.1", "error": exc.to_dict()})
    print(f"jjm: {exc.code}: {exc.message}", file=sys.stderr)
    return exit_code


def _read_request(source: Optional[str]) -> bytes:
    if source in (None, "-"):
        return sys.stdin.buffer.read()
    try:
        with open(source, "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise errors.RequestError(errors.INVALID_REQUEST, f"Cannot read request file {source!r}: {exc.strerror}"
                                  ) from None


def cmd_evaluate(args: argparse.Namespace) -> int:
    from jjm.engine import evaluate

    request = load_request(_read_request(args.request))
    if args.profile and args.profile != request["profile_id"]:
        raise errors.RequestError(errors.INVALID_REQUEST,
                                  f"--profile {args.profile!r} differs from the request's profile_id "
                                  f"{request['profile_id']!r}")
    response = evaluate(request, load_profiles(args.profiles))
    _emit(response)
    held = sum(1 for r in response["results"] if r["acceptance"]["status"] != "pass")
    print(f"jjm: {len(response['results'])} result(s), {held} not passed for consumption", file=sys.stderr)
    return EXIT_OK


def cmd_evaluate_batch(args: argparse.Namespace) -> int:
    from jjm.engine import evaluate_batch

    lines = _read_request(args.requests).decode("utf-8").split("\n")
    requests = [load_request(line.encode("utf-8")) for line in lines if line.strip()]
    responses = evaluate_batch(requests, load_profiles(args.profiles))
    for response in responses:
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    held = sum(1 for resp in responses for r in resp["results"] if r["acceptance"]["status"] != "pass")
    print(f"jjm: {len(responses)} request(s), {held} result(s) not passed for consumption", file=sys.stderr)
    return EXIT_OK


def cmd_models(args: argparse.Namespace) -> int:
    from jjm.discover import discover

    _emit(discover(ollama_host=args.ollama_host))
    return EXIT_OK


def _head_profile(args: argparse.Namespace):
    profiles = load_profiles(args.profiles)
    profile = profiles.get(args.profile)
    if profile is None or profile.adapter_name != "fitted-head":
        heads = sorted(pid for pid, p in profiles.items() if p.adapter_name == "fitted-head")
        raise errors.RequestError(errors.INVALID_REQUEST,
                                  f"{args.profile!r} is not a fitted-head profile. Fitted-head profiles: {heads}")
    return profile


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise errors.RequestError(errors.INVALID_REQUEST, f"Cannot read {path!r}: {exc.strerror}") from None
    except json.JSONDecodeError as exc:
        raise errors.RequestError(errors.INVALID_REQUEST, f"{path!r} is not valid JSON: {exc.msg}") from None


def read_examples(path: str) -> list[dict[str, Any]]:
    """JSONL: {"state": {"content": ..., "evidence": [...]}, "label": "<outcome id>" or null}."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh.read().split("\n"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            state = row.get("state")
            if not isinstance(state, dict) or "content" not in state:
                raise errors.RequestError(errors.INVALID_REQUEST, f"{path} line {n}: needs state.content")
            state.setdefault("evidence", [])
            out.append({"state": state, "label": row.get("label")})
    return out


def cmd_fit(args: argparse.Namespace) -> int:
    from jjm.compile import compile_criterion, render_state
    from jjm.heads.fit import fit_head
    from jjm.profiles import load_teachers

    profile = _head_profile(args)
    compiled = compile_criterion(_read_json(args.criterion))
    examples = read_examples(args.examples)
    texts = [render_state(e["state"]) for e in examples]
    dropped = 0
    if args.teacher:
        from jjm.heads.teachers import build_teacher

        teachers = load_teachers(args.profiles)
        if args.teacher not in teachers:
            raise errors.RequestError(errors.INVALID_REQUEST,
                                      f"teacher {args.teacher!r} is not configured. Configured: {sorted(teachers)}")
        teacher = build_teacher(teachers[args.teacher])
        labels, kept = [], []
        for i, text in enumerate(texts, 1):
            ans = teacher.answer(text, compiled)
            if ans.index is None:
                dropped += 1
            else:
                kept.append(text)
                labels.append(compiled.outcome_ids[ans.index])
            if i % 50 == 0:
                print(f"jjm: teacher labelled {i}/{len(texts)}", file=sys.stderr)
        texts = kept
        label_source = {"kind": "teacher", "teacher": teacher.identity()}
    else:
        missing = sum(1 for e in examples if e["label"] is None)
        if missing:
            raise errors.RequestError(errors.INVALID_REQUEST,
                                      f"{missing} example(s) have no label. Label them, or pass --teacher.")
        labels = [e["label"] for e in examples]
        label_source = {"kind": args.label_source}
    adapter = profile.adapter()
    result = fit_head(adapter.featurizer, adapter.store, compiled, texts, labels, method=args.method,
                      label_source=label_source, promote=args.promote)
    art = result.artifact
    held = art["report"]["heldout"]
    _emit({"artifact_id": art["artifact_id"], "binding_digest": art["binding_digest"], "reused": result.reused,
           "promoted": result.promoted, "examples": art["training"]["n"], "fit_count": art["training"]["fit_count"],
           "teacher_unparsed": dropped, "label_source": art["training"]["label_source"],
           "label_counts": art["training"]["label_counts"], "cv": art["report"]["cv"],
           "heldout": {k: held[k] for k in ("n", "accuracy", "macro_f1", "report_id") if k in held},
           "feature_ms": round(result.feature_ms, 1)})
    return EXIT_OK


def cmd_accept(args: argparse.Namespace) -> int:
    from jjm.compile import compile_criterion
    from jjm.heads.fit import accept_validation, served_artifact

    profile = _head_profile(args)
    compiled = compile_criterion(_read_json(args.criterion))
    adapter = profile.adapter()
    art = served_artifact(adapter.store, adapter.featurizer, compiled)
    if art is None:
        raise errors.RequestError(errors.INVALID_REQUEST, "No promoted head matches this criterion and encoder.")
    _emit(accept_validation(adapter.store, art, args.scope))
    return EXIT_OK


def cmd_heads(args: argparse.Namespace) -> int:
    profile = _head_profile(args)
    from jjm.heads.artifact import ArtifactStore

    _emit(ArtifactStore(profile.config["artifacts"]).list())
    return EXIT_OK


def cmd_capabilities(args: argparse.Namespace) -> int:
    profiles = load_profiles(args.profiles)
    if args.profile not in profiles:
        raise errors.RequestError(errors.INVALID_REQUEST,
                                  f"Profile {args.profile!r} is not configured. Configured: {', '.join(sorted(profiles))}")
    _emit(capabilities(profiles[args.profile]))
    return EXIT_OK


def _package(name: str) -> Optional[str]:
    if importlib.util.find_spec(name) is None:
        return None
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "installed"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what is present. Never installs, downloads, or loads a model."""
    report: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "python": platform.python_version(),
        "packages": {name: _package(name) for name in ("anyjev", "numpy", "torch", "transformers", "jsonschema")},
        "cuda": None,
        "profiles": {},
        "problems": [],
    }
    if report["packages"]["anyjev"] != "0.2.0":
        report["problems"].append(f"anyjev 0.2.0 is required; found {report['packages']['anyjev']}")
    if report["packages"]["torch"]:
        try:
            import torch

            report["cuda"] = {"available": bool(torch.cuda.is_available()),
                              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        except Exception as exc:
            report["problems"].append(f"torch import failed: {exc.__class__.__name__}")
    try:
        profiles = load_profiles(args.profiles)
    except errors.JJMError as exc:
        report["problems"].append(exc.message)
        profiles = {}
    for pid, profile in sorted(profiles.items()):
        entry: dict[str, Any] = {"execution_mode": profile.execution_mode, "adapter": profile.adapter_name}
        if profile.adapter_name == "anyjev-hf":
            cfg = profile.config
            entry.update({"model": cfg["model"]["id"], "revision": cfg["model"]["revision"],
                          "allow_download": cfg["allow_download"], "device": cfg["device"], "dtype": cfg["dtype"]})
            if not (report["packages"]["torch"] and report["packages"]["transformers"]):
                report["problems"].append(f"profile {pid!r} needs the 'hf' extra (torch, transformers)")
            if cfg["model"]["revision"] is None:
                entry["note"] = "No pinned model revision: provenance reports revision_status 'unknown'."
        elif profile.adapter_name == "fitted-head":
            enc = profile.config["encoder"]
            entry.update({"encoder": enc["kind"], "model": enc["model"]["id"] if enc["kind"] == "hf" else enc["model"],
                          "artifacts": profile.config["artifacts"], "cache": profile.config["cache"]})
            if enc["kind"] == "hf" and not (report["packages"]["torch"] and report["packages"]["transformers"]):
                report["problems"].append(f"profile {pid!r} needs the 'hf' extra (torch, transformers)")
        report["profiles"][pid] = entry
    report["ok"] = not report["problems"]
    _emit(report)
    return EXIT_OK if report["ok"] else EXIT_INVALID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jjm", description="JJM local judgment runtime")
    parser.add_argument("--profiles", help="operator profile file (default: $JJM_PROFILES)")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="evaluate one request (JSON file or stdin)")
    ev.add_argument("--request", help="request file, or - for stdin (default)")
    ev.add_argument("--profile", help="refuse the request unless it names this profile")
    ev.set_defaults(func=cmd_evaluate)
    cap = sub.add_parser("capabilities", help="print a profile's capability record")
    cap.add_argument("--profile", required=True)
    cap.set_defaults(func=cmd_capabilities)
    doc = sub.add_parser("doctor", help="report installed pieces and configured profiles")
    doc.set_defaults(func=cmd_doctor)
    batch = sub.add_parser("evaluate-batch", help="evaluate a JSONL file of requests; one envelope per line out")
    batch.add_argument("--requests", help="JSONL file, or - for stdin (default)")
    batch.set_defaults(func=cmd_evaluate_batch)
    models = sub.add_parser("models", help="list local Hugging Face and Ollama models; loads nothing")
    models.add_argument("--ollama-host", default="http://localhost:11434")
    models.set_defaults(func=cmd_models)
    fit = sub.add_parser("fit", help="fit a head for one criterion, or reuse the one fitted on the same examples")
    fit.add_argument("--profile", required=True, help="a fitted-head profile")
    fit.add_argument("--criterion", required=True, help="criterion JSON (kind, question, options)")
    fit.add_argument("--examples", required=True, help="JSONL of {state, label}")
    fit.add_argument("--method", choices=("softmax", "ordinal"), default="softmax")
    fit.add_argument("--teacher", help="label every example with this configured teacher instead")
    fit.add_argument("--label-source", default="human", choices=("human", "deterministic", "provider", "synthetic"),
                     help="who produced the labels in the file (ignored with --teacher)")
    fit.add_argument("--promote", action="store_true", help="serve this head for its criterion and encoder")
    fit.set_defaults(func=cmd_fit)
    accept = sub.add_parser("accept", help="accept a promoted head's held-out report for one scope")
    accept.add_argument("--profile", required=True)
    accept.add_argument("--criterion", required=True, help="the criterion the head was fitted for")
    accept.add_argument("--scope", required=True, help="scope_id the report applies to")
    accept.set_defaults(func=cmd_accept)
    heads = sub.add_parser("heads", help="list stored heads for a fitted-head profile")
    heads.add_argument("--profile", required=True)
    heads.set_defaults(func=cmd_heads)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except errors.RequestError as exc:
        return _fail(exc, EXIT_INVALID)
    except errors.ExecutionError as exc:
        return _fail(exc, EXIT_EXECUTION)
    except errors.JJMError as exc:
        return _fail(exc, EXIT_EXECUTION)
    except KeyboardInterrupt:
        return _fail(errors.ExecutionError(errors.INTERNAL_ERROR, "Interrupted; no result was produced."),
                     EXIT_EXECUTION)
    except Exception as exc:  # never a traceback on stdout
        return _fail(errors.ExecutionError(errors.INTERNAL_ERROR, f"Unexpected {exc.__class__.__name__}."),
                     EXIT_EXECUTION)


if __name__ == "__main__":
    raise SystemExit(main())
