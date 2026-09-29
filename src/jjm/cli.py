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
