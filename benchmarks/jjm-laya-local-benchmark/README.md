# JJM vs Laya local experiment

A runnable comparison of a JJM frozen Qwen embedding + softmax head, pretrained Laya, and a TF-IDF baseline. An experimental adapter also runs Laya through the actual JJM engine and checks prediction parity with direct Laya calls.

This package does not alter your JJM or Exec Foundry repositories. It downloads a separate, pinned JJM snapshot. Start with the public Banking77 task. Add private EF examples only on your own machine.

## Setup

Python 3.12 or newer is required by the pinned NumPy and SciPy versions; use 3.12 to match the baseline. From this extracted directory:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then:

```sh
python bootstrap.py
python -m pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q tests
```

The CUDA command is for a compatible NVIDIA setup. To run on CPU, install `torch==2.14.0+cpu` from `https://download.pytorch.org/whl/cpu` instead. The included execution used Torch 2.14.1+cpu. See the local GPU reports under `results/` for CUDA execution evidence; the supplied CPU baseline used a different Torch build. If the CUDA check prints `False`, resolve the local driver/PyTorch setup before interpreting GPU timings.

Initial model downloads total roughly 2 GB. Model loading and working memory need additional space. The runner loads one neural model at a time and records versions. No Ollama is required. After initial downloads, the model cache can be used offline with `HF_HUB_OFFLINE=1`.

## Run

CPU wiring check, 64 supplied training labels, 16 validation rows, 32 test rows:

```sh
python bench.py --device cpu --out results/my-cpu-smoke
```

First useful GPU comparison, 512 supplied training labels, 400 validation rows, 400 test rows:

```sh
python bench.py --device cuda --train-per-team 64 --val-per-team 50 --test-per-team 50 --latency-n 100 --batch-size 16 --out results/gpu-512-labels
```

Full-data comparison, reserving validation from the official training split and evaluating all 3,080 official test rows:

```sh
python bench.py --device cuda --train-per-team 0 --val-per-team 50 --test-per-team 0 --latency-n 200 --batch-size 32 --out results/gpu-full
```

Use a new output directory for each run. If a run fails, its completed route results remain for diagnosis. A smaller `--batch-size` can resolve GPU memory pressure. Use `--routes tfidf`, `--routes jjm`, or `--routes laya` to isolate a route. A single-route rerun is diagnostic, not a paired comparison unless split hashes match.

## What is controlled

- The exact JJM commit, Laya package, neural checkpoints, public CSV digests, team mapping, seed and row IDs are recorded. The original JJM question and eight option descriptions are reused.
- Laya receives the same rendered state, question and options. JJM's fitted head binds the criterion at fitting time. Its encoder prefix contains the question, while learned weights represent the fixed options. These are intentionally different model architectures.
- Laya receives no local labels for model fitting. JJM uses task labels and internally withholds 20% before head selection/fitting. TF-IDF fits the same actual subset as JJM. The report shows both supplied labels and actual fit counts.
- A separate validation slice comes from official training data. It selects an empirical rejection threshold. Test rows never choose thresholds or hyperparameters. If no threshold meets the requested error and minimum count, the report says so.
- Exact text overlap between train, validation and test is rejected. Upstream pretrained data overlap with Banking77 is unknown for both checkpoints. This is a deployment comparison, not a controlled pretraining study.
- Latency uses a seeded proportional sample across teams, shuffled and identical across routes. Its row IDs and team counts are saved separately; the accuracy splits are unchanged.
- Hardware, CUDA runtime, driver, package versions and harness file digests are recorded.
- Warm uncached batch-1 p50/p95, batched throughput, load time, fit time and GPU allocation are separate. JJM's feature memo is cleared before every timed operation and no disk feature cache is enabled. Model initialization includes any download time and is labeled accordingly.
- Laya uses the eager Python/HF backend. Its `head_max_len` is explicitly 384 instead of the checkpoint default 192, allowing the longer JJM option descriptions. Total length stays 512. Option token limits and state truncation are checked, with budget details saved.
- Laya's public API rounds to four decimal places. Both direct metrics and the adapter explicitly normalize that rounded vector. Raw weights, sum and transformation are retained in adapter diagnostics. Invalid mass, missing options, truncation or changed winner cause failure. This is not probability calibration.

## Outputs

Each run includes `REPORT.md`, `summary.json`, `split.json`, `latency-sample.json`, per-route prediction JSONL, token-budget audit and example JJM envelopes. JJM head artifacts are also retained. The summary includes accuracy, macro-F1, per-team accuracy, Wilson intervals, a paired discordance test, selective coverage/error and local timings. The smoke sample is too small for product conclusions.

The adapter implements choices only, has no batching prefetch hook, and is registered programmatically in this harness. It is not a new supported profile type in upstream JJM. Its batch-1 overhead is measured against direct Laya with alternating call order. Negative measured overhead is timing noise, not a speedup. Wrapper parity is checked on the latency sample, whose size is recorded.

The harness audits the fixed question and option lengths before inference. The adapter alone is not a general prompt-budget validator. Do not reuse it with new criteria without that audit. State truncation is detected and refused after upstream inference, not before consuming model compute.

The runner leaves JJM validation at `fitted_unvalidated` and probability calibration at `not_assessed`. A useful benchmark score does not automatically accept an artifact or certify probability calibration. Requiring assessed calibration produces abstention, which is covered by a contract test.

## Next experiments

1. Repeat the 512-label and full-data runs on your target GPU. Review errors and paired disagreements before changing prompts or labels.
2. Sweep batches 1, 8, 16 and 32 in fresh run directories. Hold checkpoints, split and prompt fixed. Compare throughput separately from request latency.
3. Once quality is established, test Laya's accelerated/ONNX backends as separate configurations with parity checks. This package does not enable them.
   A shorter option rubric is another candidate for reducing Laya's token cost. Choose it on validation data and freeze it before a new test evaluation. Shortening options can change predictions.
4. For Exec Foundry, build a task-specific held-out set with ordinal rubric and human-reviewed labels. Teacher agreement and human agreement should be reported separately. Long job descriptions require an explicit evidence/segmentation policy. This short-message result does not establish long-document quality.
5. The most useful JJM contribution may be one contract for fitted heads and Laya, explicit provenance, refusal rules and measured fallback behavior. Validate those benefits in a real EF workflow before expanding the integration.

## Sources and attribution

- JJM: https://github.com/Catalyst-Forge-LLC/jjm
- Laya: https://github.com/NandhaKishorM/laya and https://huggingface.co/convaiinnovations/laya
- Qwen embedding: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Banking77: https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data (PolyAI, CC BY 4.0)

`task_data.py` and the request template were adapted from the pinned JJM source. Its Apache-2.0 license and notice are included. Model weights and third-party packages are downloaded under their upstream licenses and are not bundled. The new experiment code is provided under Apache-2.0 as well.

The imported package checksum manifest is preserved as `SHA256SUMS.imported.json`. `SHA256SUMS.json` records the current harness and saved evidence. GPU environment versions are captured in `results/environment-cuda.freeze.txt`; the editable JJM source remains fixed by `pins.json`.
