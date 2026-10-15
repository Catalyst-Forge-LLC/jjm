# Local GPU benchmark — October 15, 2026

All three planned runs completed: the GPU smoke test, the 512-label comparison, and the full official Banking77 test split. JJM was the most accurate route in both substantive runs. With the larger training set, TF-IDF was close in accuracy and much faster. Laya performed poorly with this fixed eight-team rubric; wrapping it in JJM preserved its predictions but did not improve its accuracy.

## Main results

| Route | 512 supplied labels: correct / 400 | Accuracy | Full training: correct / 3,080 | Accuracy |
|---|---:|---:|---:|---:|
| JJM fitted Qwen head | 365 | 91.25% | 2,978 | 96.69% |
| Laya pretrained | 250 | 62.50% | 1,704 | 55.32% |
| TF-IDF + logistic regression | 323 | 80.75% | 2,946 | 95.65% |

The smaller run supplied 512 human-labelled training examples; JJM withheld 104 internally and fitted on 408. The full run supplied 9,603 examples after reserving 400 validation rows; JJM withheld 1,923 internally and fitted on 7,680. TF-IDF used the same actual fit rows in each run. Laya used zero local model-fit labels. All routes used the same separate validation rows to select rejection thresholds.

The 400-test run is balanced across eight teams. The full test follows the original 77-intent distribution mapped to those teams. Scores across these two sets are therefore not a learning curve for Laya. Test labels did not select prompts, hyperparameters, or thresholds.

| Full-test route | Macro F1 | 95% Wilson accuracy interval |
|---|---:|---|
| JJM fitted Qwen head | 0.9619 | 96.00%–97.26% |
| Laya pretrained | 0.5660 | 53.56%–57.07% |
| TF-IDF + logistic regression | 0.9508 | 94.87%–96.31% |

On the full set, JJM alone was correct on 78 rows where TF-IDF was wrong, and TF-IDF alone on 46 rows; the paired exact test gives p=0.00516. JJM's accuracy advantage was 1.04 percentage points.
Against Laya, JJM alone was correct on 1,299 rows and Laya alone on 25. The saved summary's p=0 is floating-point underflow, not a literal zero probability; the post-run log-space calculation gives log10(p)=-345.50.

## Latency, throughput, and setup cost

| Run / route | Warm single request p50 / p95 ms | Batch rows/s | Model load s | Fit incl. features s | Peak Torch GPU allocation GiB |
|---|---:|---:|---:|---:|---:|
| 512 labels; neural batch 16 / jjm | 25.88 / 51.61 | 310.7 | 2.76 | 2.06 | 1.49 |
| 512 labels; neural batch 16 / laya | 27.00 / 40.74 | 130.8 | 1.48 | none | 2.47 |
| 512 labels; neural batch 16 / tfidf | 0.41 / 0.50 | 100,918.4 | n/a | 0.10 | 0.00 |
| Full; neural batch 32 / jjm | 36.36 / 68.74 | 326.5 | 2.85 | 25.97 | 2.06 |
| Full; neural batch 32 / laya | 38.94 / 78.58 | 118.6 | 1.75 | none | 2.69 |
| Full; neural batch 32 / tfidf | 0.40 / 0.48 | 99,644.1 | n/a | 1.54 | 0.00 |

JJM times include the JJM contract; Laya times above are direct calls. TF-IDF runs on CPU and scores the test set in one vectorized call. The throughput column is not single-request latency. JJM feature memory was cleared before each timed operation, with no disk feature cache. JJM full fitting took about 26 seconds, including about 23.4 seconds encoding training examples.

These are single sequential measurements on a working laptop, not isolated repeated performance trials. The neural batch sizes and test distributions differ between runs, so the table does not establish a batch-size speedup. The smoke run downloaded Laya; the two substantive runs used HF_HUB_OFFLINE=1 and cached weights. Cold load/download time remains separate from warm inference.

## JJM wrapper and numerical checks

All 10 integrity tests passed, including the added representative latency-sampling regression. The installed environment passed `uv --no-cache pip check`. The GPU smoke decisions matched all 32 saved CPU-smoke decisions for each of the three routes. Dataset hashes, exact split separation, and prompt budgets passed. No model input truncation was reported.

| Run | Direct / wrapped Laya checks | Max weight delta | Median paired wrapper overhead ms |
|---|---:|---:|---:|
| gpu-smoke-20261015 | 16 | 0 | 0.285 |
| gpu-512-labels-20261015 | 100 | 0 | 0.631 |
| gpu-full-20261015 | 200 | 0 | 0.772 |

A separate post-run diagnostic found 2/400 common Laya rows changed winner between the batch-16 and batch-32 runs, with a maximum weight difference of 0.0218. Comparing single calls against saved full-run batches on the 200 preselected latency rows plus those two changed rows found 3/202 changed winners and maximum weight difference 0.0428. This establishes some sensitivity to batching in this CUDA bfloat16 configuration. It does not identify the cause. The direct/wrapper parity checks compare like-for-like single calls and remain exact. No numerical settings were changed in response to test results.

The upstream warning about clamping `choice:11+` temperature does not apply to this eight-choice task, which uses `choice:6-10`. The missing Triton warning concerns FLOP counting; this experiment intentionally uses the eager backend. Neither warning was suppressed or treated as evidence of calibration.

## Rejection policy selected on validation

| Run / route | Test coverage | Accepted / test | Errors among accepted | Accepted error rate |
|---|---:|---:|---:|---:|
| 512 labels / jjm | 76.00% | 304 / 400 | 9 | 2.96% |
| 512 labels / laya | No qualifying validation threshold | — | — | — |
| 512 labels / tfidf | 32.25% | 129 / 400 | 3 | 2.33% |
| Full / jjm | 90.03% | 2,773 / 3,080 | 24 | 0.87% |
| Full / laya | No qualifying validation threshold | — | — | — |
| Full / tfidf | 87.18% | 2,685 / 3,080 | 35 | 1.30% |

The target was at most 2% empirical validation error with at least 20 acceptances. It is not a guarantee on unseen data: the 512-label JJM policy had 2.96% test error among accepted rows. The full JJM policy accepted 90.03% with 0.87% test error; TF-IDF accepted 87.18% with 1.30% error. No Laya threshold met the validation rule. These are normalized readouts; calibration remains not assessed and the JJM head remains fitted_unvalidated.

## Error review

With this rubric, Laya often chose payments for unrelated teams. On the full set, 265 card cases, 190 fee/FX cases, 148 account cases, and 144 security cases were assigned to payments. It predicted payments 1,113 times although only 240 test rows belonged to that team. Full-set accuracy ranged from 24.17% on security to 92.5% on payments.

Examples where JJM was correct and Laya was wrong include “Is there a way to know when my card will arrive?” (cards → Laya payments) and “Where do you guys acquire your exchange rate?” (fees_fx → Laya payments). Laya also caught cases JJM missed, such as “I lost my wallet today with all my credit cards. Will you please block the card and send a replacement?” (security → JJM cards). Some short messages are ambiguous without the dataset intent; these are disagreements with the fixed mapped labels, not a fresh human adjudication.

Each substantive run now includes `disagreements.jsonl`; `comparison-analysis-20261015.json` contains pairwise tests and the largest confusion counts. The prompt, team mapping, checkpoint, and acceptance rules were held fixed throughout.

## Machine and reproducibility

- Windows; NVIDIA RTX 4000 Ada Generation Laptop GPU, 12,282 MiB VRAM, driver 596.58. About 10.2 GiB was free after DictaWhisper stopped. Host RAM is about 64 GiB.
- Isolated CPython 3.12.13; PyTorch 2.14.0+cu130, CUDA 13.0, cuDNN 92400. Both neural routes report bfloat16; PyTorch CPU threads=4.
- Laya 0.4.8; Transformers 4.57.6; huggingface-hub 0.36.2; NumPy 2.5.4; SciPy 1.18.1; scikit-learn 1.9.1. Full installed versions are in `environment-cuda.freeze.txt`.
- JJM snapshot `0192a6354b2ab7b5a2c4d69b4c2bfe785b775c14`; Laya model revision `7b928d828b7b0e022f929d9bd2e44165aa270148`; Qwen embedding revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Harness commit used for all runs: `0291b4c`. Summaries also record hashes of the harness files. Each run has immutable split IDs, a split digest, latency-sample IDs/counts, JSON predictions, envelopes, retained head artifacts, and a console log.
- Latency sampling was changed before any GPU run from a file prefix to a seeded proportional team sample shuffled identically for every route. The full-run latency sample covers all eight teams. The supplied CPU result remains unchanged.

To reproduce from this directory in PowerShell, use the existing `.venv` and a fresh output directory. The two measured commands were:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
.\.venv\Scripts\python.exe bench.py --device cuda --train-per-team 64 --val-per-team 50 --test-per-team 50 --latency-n 100 --batch-size 16 --out results/gpu-512-labels-NEW
.\.venv\Scripts\python.exe bench.py --device cuda --train-per-team 0 --val-per-team 50 --test-per-team 0 --latency-n 200 --batch-size 32 --out results/gpu-full-NEW
```

This establishes the behavior of these fixed configurations on Banking77. It does not establish Exec Foundry ranking quality, equal-supervision model superiority, calibrated probabilities, or the best possible Laya prompt. A shorter rubric or different precision/backend would be a separate configuration selected on validation data and measured without rewriting this baseline.
