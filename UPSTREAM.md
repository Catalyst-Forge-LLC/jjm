# Upstream: AnyJev

JJM uses AnyJev as an unmodified, pinned dependency. It does not vendor or fork AnyJev's source. A fork happens only if a patch becomes unavoidable, and the patch goes upstream.

| Field | Value |
| ----- | ----- |
| Project | [nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev) |
| License | Apache-2.0 |
| Pinned release | `anyjev==0.2.0` (PyPI, uploaded 2026-09-28) |
| Wheel SHA-256 | `0749a4c709069840906678c76efad3c29dafb538be44d91fa46f15cc1bd8486f` |
| Source SHA-256 | `a3f1b3195b8643a8019275fed5f2fffdb385c204ec58684dfaad011e8a7bfecb` |
| Git commit | Not recorded. The PyPI release does not carry one, so envelopes report `upstream_commit: null`. |
| Pinned in | `pyproject.toml` and `uv.lock` (hash-checked) |

## Tests run (2026-09-28, Windows 10, Python 3.11.5, numpy 2.4.6, CPU)

AnyJev's own suite, from the 0.2.0 source distribution, against the installed wheel:

- 78 passed, 7 skipped. The skipped tests need `torch` or `transformers`, which are not installed in the core environment.
- 5 test modules could not be collected because the source distribution omits the repository's `bench/` and `demo/` folders: `test_demo_games`, `test_demo_jev_mode`, `test_maze`, `test_metrics`, `test_pools`. They were not run.

JJM's suite: `uv run pytest`, 87 passed. It includes AnyJev's L0 pipeline running on AnyJev's `FakeBackend`.

No real model has been run yet.

## Findings from spec 01, checked against 0.2.0 source

| Finding | Still true in 0.2.0? | What JJM does |
| ------- | -------------------- | ------------- |
| F01 head routing by option text | Yes. `Decider.route` matches kind plus option texts, and reordered option sets for random-listing heads. | JJM runs L0 only and never loads heads or calls `route`. Exact binding comes with fitted artifacts in release 0.2. |
| F02 vLLM fills a missing label with `-30.0` | Yes, `backends/vllm.py` line 76. | vLLM is out of scope for v1. JJM checks every returned distribution for coverage, finiteness, and sum, and turns a failure into an error instead of repairing it. |
| F03 `Decision.to_dict()` drops diagnostics | Yes. | The adapter reads the `Decision` object directly and copies an allowlist of diagnostics. |
| F04 mutable prior statistics and label-free adaptation | Yes: `_running`, `observe`, `adapt`. | A fresh `Decider` for every criterion. No statistics survive a request. `adapt` is off. |
| F05 levels form an ordered `require` ladder | Yes, `LEVEL_ORDER`. | JJM reports `upstream_level` as method metadata only. Acceptance never uses level order. |
| F06 vLLM path issues a one-token completion | Yes. | Not used. The Transformers route reports `numeric_path: logit_readout`. |
| F07 shared-prefix path with a token-boundary check and fallback | Yes, `shared_fallbacks` in `backends/hf.py`. | Prefix reuse is reported as `used` only when at least one group really shared its prefix. |

## New in 0.2.0 and pinned off

`adaptive_shifts` and `canonical_order` are opt-in in 0.2.0, and upstream says they become defaults in a later release. JJM passes every `Decider` setting explicitly (`src/jjm/adapters/anyjev_adapter.py`, `DECIDER_SETTINGS`). A test fails if AnyJev adds or removes a setting, so an upgrade cannot change behavior silently.

## Deliberate differences from AnyJev defaults

- Prior correction is off (`prior="none"`). Upstream's default is the batch prior at strength 0.75. JJM's L0 is permutation marginalization only.
- The system prompt is a pinned copy inside JJM's compiler, and it is part of `rendering_digest`.
- Model downloads are off unless a profile sets `allow_download: true`.
