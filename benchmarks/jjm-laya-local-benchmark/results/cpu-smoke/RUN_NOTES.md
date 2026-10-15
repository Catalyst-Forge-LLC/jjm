# Observed CPU smoke run

Executed in this workspace on 2026-10-15. This was real inference using the pinned neural weights, not mocked results. CPU reported AMD EPYC 9V74, with 9 visible logical CPUs and PyTorch limited to 4 threads. The environment had no usable NVIDIA GPU. Both neural routes used float32. The shared environment is not a controlled performance lab.

| Route | Correct / 32 | Median batch-1 latency | p95 batch-1 latency | Batch throughput |
|---|---:|---:|---:|---:|
| JJM + frozen Qwen embedding + fitted softmax | 25 | 318 ms | 529 ms | 2.92 rows/s |
| Laya pretrained, eager backend | 22 | 1,740 ms | 2,066 ms | 0.67 rows/s |
| TF-IDF + logistic regression | 16 | 0.45 ms | 3.55 ms | 8,083 rows/s |

JJM and TF-IDF received 64 training labels, of which 48 were used for fitting and 16 reserved by JJM's internal holdout. Laya received zero local fitting labels. An additional 16 training-split rows were reserved for external threshold validation. All routes were evaluated on the same 32 official test rows, four per team. There was no exact text overlap across these sets.

This is insufficient evidence of an accuracy lead. JJM alone was correct on four messages and Laya alone on one. The exact paired test gives p=0.375. The 95% Wilson accuracy intervals are roughly 61–89% for JJM and 51–82% for Laya. Neither route established the requested selective policy because there were fewer than the required 20 validation acceptances.

Laya's direct and JJM-wrapped weights matched exactly on all 16 checked messages. The median paired wrapper timing difference was +17.5 ms. With only 16 noisy samples this is an overhead observation, not a stable estimate. The wrapper does not accelerate Laya.

All eight options and the complete question fit in a 273-token Laya head when head_max_len=384. No evaluated state was truncated. JJM's body limit was also checked against original tokens. JJM encodes a shorter input containing the question and message, with the fixed options represented by fitted head weights. Laya reads the option descriptions in its prompt. That difference matters when interpreting speed.

Laya emitted a load warning about its upstream `choice:11+` temperature being clamped to 0.5. This task has eight choices and uses the `choice:6-10` entry, so that reported correction does not apply here. No claim of task-specific calibration is made.

After the run, the harness's prompt audit was tightened to reject shortened instructions or option descriptions even if options remained distinct. The final audit was checked against the same cached tokenizer. It confirmed complete preservation. Token counts in `post-run-prompt-audit.json` include the exact leading spaces used upstream. The earlier `laya-token-budget.json` counted option text without those spaces. Predictions and timings are unchanged.

Nine integrity tests passed. They cover bounded rounding normalization, malformed or truncated output refusal, test-independent threshold selection, and JJM contract/calibration behavior. The GPU execution path is prepared but untested here.

For the next run, use the 512-label / 400-test GPU command in README.md, then the full test split. Quality and latency in Exec Foundry remain unmeasured. A job-ranking pilot needs its own rubric, labels and long-input policy.
