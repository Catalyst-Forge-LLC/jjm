"""Extract per-job features for the fast-judgment experiments. One forward pass per job.

    uv run python -m trials.fit_rank.features probe  --input export.jsonl --model-path <dir> --out runs/feat-4b.npz
    uv run python -m trials.fit_rank.features embed  --input export.jsonl --model-path <dir> --out runs/feat-embed.npz
    uv run python -m trials.fit_rank.features rerank --input export.jsonl --model-path <dir> --out runs/feat-rerank.npz

probe:  a causal LM reads JJM's fit prompt (first rotation only). Saves the last-token hidden
        state and the mean over tokens at four depths.
embed:  an embedding model (last-token pooling) embeds the listing, the candidate, and chunks
        of both. Saves the listing embedding, candidate-listing cosine, and requirement coverage.
rerank: a yes/no reranker judges "does this listing fit this candidate". Saves P(yes) and the
        final hidden state.

Every mode records per-job forward time. Nothing here is fitted; see probe_eval.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from trials.fit_rank.build import build_request
from trials.fit_rank.run import read_export

DEPTHS = (0.25, 0.5, 0.75, 1.0)
FIT_INSTRUCTION = ("Given an executive candidate's resume, profile, and work preferences, judge whether the job "
                   "listing is a strong fit for the candidate's experience, seniority, direction, and preferences")
RERANK_SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
                 'Note that the answer can only be "yes" or "no".')


def candidate_text(r: dict[str, Any]) -> str:
    parts = [(k, r.get(k)) for k in ("candidate_profile", "work_preferences", "location_instructions",
                                     "supplementary_context", "base_resume")]
    return "\n\n".join(f"{k.upper()}:\n{v}" for k, v in parts if v)


def listing_text(r: dict[str, Any]) -> str:
    hints = r.get("listing_hints") or {}
    head = " | ".join(x for x in (hints.get("title"), hints.get("company")) if x)
    return f"{head}\n\n{r['listing_text']}" if head else r["listing_text"]


def chunks(text: str, lo: int = 40, hi: int = 400) -> list[str]:
    out = []
    for piece in re.split(r"\n+|(?<=[.;!?])\s+(?=[A-Z])", text):
        piece = piece.strip(" \t-•*·")
        while len(piece) > hi:
            out.append(piece[:hi])
            piece = piece[hi:]
        if len(piece) >= lo:
            out.append(piece)
    return out


def _load(path: str, kind: str):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    cls = AutoModel if kind == "embed" else AutoModelForCausalLM
    model = cls.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    return tok, model


def _timed(fn):
    import torch

    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000.0


def probe(records, tok, model) -> dict[str, np.ndarray]:
    import torch
    from anyjev.readout import build_prompt, render_chat, resolve_labels

    from jjm.adapters.anyjev_adapter import _question
    from jjm.compile import SYSTEM_PROMPT, compile_criterion, render_state

    n_layers = model.config.num_hidden_layers
    layers = [max(1, round(f * n_layers)) for f in DEPTHS]
    last, mean, ms, ntok = [], [], [], []
    for i, r in enumerate(records, 1):
        req = build_request(r, profile_id="probe")
        q = _question(compile_criterion(req["criteria"][0]))
        labels, _ = resolve_labels(tok, q)
        text = render_chat(tok, build_prompt(render_state(req["state"]), q, list(range(q.k)), SYSTEM_PROMPT, labels))
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out, t = _timed(lambda: model(**enc, output_hidden_states=True, logits_to_keep=1))
        hs = out.hidden_states
        last.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in layers]))
        mean.append(np.stack([hs[l][0].float().mean(0).cpu().numpy() for l in layers]))
        ms.append(t)
        ntok.append(enc["input_ids"].shape[1])
        del out, hs
        if i % 25 == 0:
            print(f"[{i}/{len(records)}] {np.median(ms):.0f} ms median", flush=True)
    return {"last": np.stack(last).astype(np.float32), "mean": np.stack(mean).astype(np.float32),
            "layers": np.array(layers), "ms": np.array(ms), "tokens": np.array(ntok)}


def embed(records, tok, model) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    def run(texts: list[str], batch: int = 16) -> np.ndarray:
        vecs = []
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=8192,
                      return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, -1]
            vecs.append(F.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(vecs)

    def query(task: str, text: str) -> str:
        return f"Instruct: {task}\nQuery:{text}"

    cand = candidate_text(records[0])
    same_candidate = all(candidate_text(r) == cand for r in records)
    cand_vec = run([cand])[0]
    evidence = chunks(records[0].get("base_resume") or cand)
    evidence_vecs = run(evidence)
    listing_vecs, cos, cov_mean, cov_p25, cov_min, ms = [], [], [], [], [], []
    for i, r in enumerate(records, 1):
        if not same_candidate:
            cand_vec = run([candidate_text(r)])[0]
            evidence_vecs = run(chunks(r.get("base_resume") or candidate_text(r)))

        def one():
            lv = run([query(FIT_INSTRUCTION, listing_text(r))])[0]
            reqs = chunks(r["listing_text"]) or [r["listing_text"][:400]]
            rv = run([query("Given a job requirement, retrieve resume evidence that shows the candidate meets it",
                            x) for x in reqs])
            return lv, (rv @ evidence_vecs.T).max(axis=1)

        (lv, best), t = _timed(one)
        listing_vecs.append(lv)
        cos.append(float(lv @ cand_vec))
        cov_mean.append(float(best.mean()))
        cov_p25.append(float(np.percentile(best, 25)))
        cov_min.append(float(best.min()))
        ms.append(t)
        if i % 25 == 0:
            print(f"[{i}/{len(records)}] {np.median(ms):.0f} ms median", flush=True)
    return {"listing": np.stack(listing_vecs).astype(np.float32), "cosine": np.array(cos),
            "coverage_mean": np.array(cov_mean), "coverage_p25": np.array(cov_p25),
            "coverage_min": np.array(cov_min), "ms": np.array(ms), "same_candidate": np.array(same_candidate)}


def rerank(records, tok, model) -> dict[str, np.ndarray]:
    import torch

    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    prefix = f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    p_yes, hidden, ms, ntok = [], [], [], []
    for i, r in enumerate(records, 1):
        text = (f"{prefix}<Instruct>: {FIT_INSTRUCTION}\n<Query>: {candidate_text(r)}\n"
                f"<Document>: {listing_text(r)}{suffix}")
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out, t = _timed(lambda: model(**enc, output_hidden_states=True, logits_to_keep=1))
        logits = out.logits[0, -1, [no, yes]].float()
        p_yes.append(float(torch.softmax(logits, -1)[1]))
        hidden.append(out.hidden_states[-1][0, -1].float().cpu().numpy())
        ms.append(t)
        ntok.append(enc["input_ids"].shape[1])
        del out
        if i % 25 == 0:
            print(f"[{i}/{len(records)}] {np.median(ms):.0f} ms median", flush=True)
    return {"p_yes": np.array(p_yes), "hidden": np.stack(hidden).astype(np.float32), "ms": np.array(ms),
            "tokens": np.array(ntok)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("probe", "embed", "rerank"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    records = [r for r in read_export(args.input) if r.get("sonnet_rank")][: args.limit]
    t0 = time.monotonic()
    tok, model = _load(args.model_path, args.mode)
    load_ms = (time.monotonic() - t0) * 1000.0
    feats = {"probe": probe, "embed": embed, "rerank": rerank}[args.mode](records, tok, model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, job_ids=np.array([r["job_id"] for r in records]),
             sonnet=np.array([r["sonnet_rank"] for r in records]), load_ms=np.array(load_ms), **feats)
    print(f"wrote {args.out}: {len(records)} jobs, median {np.median(feats['ms']):.0f} ms per job, "
          f"load {load_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
