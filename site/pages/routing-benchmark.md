---
title: Local text routing with JJM Laya and TF-IDF
description: A reproducible eight-team Banking77 comparison of a fitted Qwen head through JJM, pretrained Laya, and TF-IDF. Accuracy, latency, label budgets, and limits.
order: 2
---

<p class="kicker">Measured October 15, 2026 · Catalyst Forge · one laptop · public data</p>

**A Qwen embedding model with a fitted head, served through JJM, reached 96.69% accuracy on our eight-team Banking77 routing task. Pretrained Laya reached 55.32%. A CPU TF-IDF classifier reached 95.65% at 0.40 ms median latency.** The neural head led in accuracy; the sparse classifier made the strongest case for keeping a cheap baseline.

The supervision differs: JJM and TF-IDF learned from this task's labels; Laya received the rubric with no local fine-tuning. These results compare those deployment choices. They do not establish that JJM's runtime makes a model more accurate or that a fitted Laya model would lose.

[Download the reproducibility bundle](/benchmarks/routing-2026-10-15.zip) · [Try JJM's new CPU text backend](/tfidf)

## Accuracy at two label budgets

Banking77 contains customer messages labeled with 77 banking intents. We fixed a mapping from those intents to eight operational teams: cards, payments, transfers, top-ups, cash, fees and FX, security, and account. This is an **eight-class derived task**, not the standard 77-class Banking77 leaderboard.

<div class="compare" role="region" aria-label="Routing accuracy comparison" tabindex="0">
<table>
<thead><tr><th scope="col">Route</th><th scope="col">512 labels<br>400 test rows</th><th scope="col">Full training<br>3,080 test rows</th><th scope="col">Full macro F1</th></tr></thead>
<tbody>
<tr><th scope="row">Qwen + fitted head through JJM</th><td>91.25%<br>365 correct</td><td>96.69%<br>2,978 correct</td><td>0.9619</td></tr>
<tr><th scope="row">TF-IDF + logistic regression</th><td>80.75%<br>323 correct</td><td>95.65%<br>2,946 correct</td><td>0.9508</td></tr>
<tr><th scope="row">Pretrained Laya</th><td>62.50%<br>250 correct</td><td>55.32%<br>1,704 correct</td><td>0.5660</td></tr>
</tbody>
</table>
</div>

With 512 supplied labels, JJM withheld 104 internally and fitted on 408. With the full training pool, we first reserved 400 validation messages from the official 10,003 training rows, supplied the remaining 9,603, withheld 1,923 internally, and fitted on 7,680. TF-IDF used exactly the same actual fit rows. Laya used zero local fitting labels. All three routes shared the separate validation set used to select rejection thresholds.

The smaller test has 50 messages per team. The full test retains the official test set's intent distribution after mapping. Because these test distributions differ, the two columns are not a controlled learning curve. In particular, Laya's lower score in the second column is not evidence that more training labels hurt it.

The full-set Qwen/JJM accuracy has a 95% Wilson interval of 96.00%–97.26%. Against TF-IDF, it was uniquely correct on 78 rows and uniquely wrong on 46, a net 32 additional correct decisions: **1.04 percentage points**. The paired exact test gives p=0.00516. That quantifies this test-set difference, not its reliability on another domain. With the smaller label budget, its lead over TF-IDF was 10.50 points.

## Latency and fitting cost

These full-test measurements ran on Windows with an NVIDIA RTX 4000 Ada Generation Laptop GPU, 12 GB VRAM, and about 64 GB host RAM. Neural inference used CUDA bfloat16; TF-IDF used CPU.

<div class="compare" role="region" aria-label="Full-test timing comparison" tabindex="0">
<table>
<thead><tr><th scope="col">Route</th><th scope="col">Warm p50 / p95</th><th scope="col">Batch rows/s</th><th scope="col">Fit time</th></tr></thead>
<tbody>
<tr><th scope="row">Qwen + fitted head through JJM</th><td>36.36 / 68.74 ms</td><td>326.5</td><td>25.97 s</td></tr>
<tr><th scope="row">TF-IDF + logistic regression</th><td>0.40 / 0.48 ms</td><td>99,644.1</td><td>1.54 s</td></tr>
<tr><th scope="row">Pretrained Laya</th><td>38.94 / 78.58 ms</td><td>118.6</td><td>No local fit</td></tr>
</tbody>
</table>
</div>

JJM timings include its judgment contract. Laya timings are direct model calls. TF-IDF timings are for the bare classifier. Neural throughput used batches of 32; TF-IDF scored all test rows in one sparse vectorized call. **Rows per second in a batch is not single-request latency.**

JJM's fit time includes 23.40 seconds of feature extraction. Cached model loading took 2.85 seconds for Qwen and 1.75 seconds for Laya, separately from warm inference. Peak Torch GPU allocation was 2.06 GiB and 2.69 GiB respectively; these are allocation measurements, not total device-memory requirements.

Before the GPU runs, we replaced prefix-based latency sampling with a seeded, proportional team sample, shuffled identically for all routes. The 200 full-test timing rows cover all eight teams. JJM's feature memory was cleared before timed operations, and disk feature caching was disabled. These are single sequential runs on a working laptop, so timing variation across repeated runs remains unmeasured.

## Why TF-IDF is so competitive here

TF-IDF represents text as weighted counts of terms. Words and bigrams that occur in a message matter more when they are distinctive across the training corpus. A linear classifier learns which features predict each team. Sparse arithmetic makes this cheap on CPU.

It is already a standard text-classification baseline; scikit-learn's [sparse document classification example](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) illustrates the approach. Banking messages often name the object and problem directly: a card, a refund, an exchange rate, a transfer. Our interpretation is that this vocabulary and the larger labeled sample make the task particularly favorable to it.

TF-IDF does not gain semantic knowledge from a pretrained language model. A new synonym, another language, a changed domain, or a new category can require fresh examples. Changing the question does not teach the classifier what the new question means. The smaller-budget result shows a practical reason to retain an embedding route even when the sparse route is excellent with more labels.

## What JJM contributes

The measured predictive system is **Qwen3-Embedding-0.6B plus a task-trained head**. JJM supplies the contract around it: explicit options, a complete weight vector, provenance, artifact binding, scope validation, and consumption checks.

Earlier routing trials in this repository found roughly the same accuracy from a plain classifier on the same embeddings as from JJM's fitted head. This experiment therefore supports the choice of representation and fitted classifier for this task; it does not isolate a runtime accuracy advantage.

That distinction also makes TF-IDF useful inside JJM. An application can choose an inexpensive model while keeping the same judgment interface.

## The new TF-IDF backend in JJM

After preserving the comparison above, we added an optional TF-IDF backend. It fits the vocabulary, IDF, and logistic classifier only on the training slice, persists them as digest-checked JSON, and uses JJM's fit, promotion, acceptance, and evaluation commands. It supports fixed choice criteria and runs without a GPU or a language-model download.

A separate integration run used the same full split and reached **95.78% accuracy, 2,950/3,080 correct**, with **1.85 ms median / 2.20 ms p95 through the complete JJM contract** over the same 200 timing rows. Reloaded classifier weights and contract output matched exactly across all 3,080 rows. No PyTorch, Transformers, or Laya modules were loaded.

This is a distinct configuration: it classifies JJM's rendered state rather than raw message text and orders fitting rows canonically. It does not replace the original baseline. The initial batch API evaluates TF-IDF requests individually and measured 669 rows/s. Its throughput should not be confused with the original classifier's vectorized batch rate.

[Read the CPU backend guide and run it locally](/tfidf).

## Interpreting Laya's result

We tested Laya 0.4.8 with the English convaiinnovations/laya checkpoint and one fixed rubric. The complete question and options occupied 273 tokens; head_max_len was raised to 384 so they fit. No input truncation was reported. The adapter used the eager backend and the same declared option order throughout. Laya's public API rounds weights to four decimal places; direct scoring and the wrapper both explicitly normalized that rounded vector, recording the transformation. This was not probability calibration.

Laya's appeal includes typed choice, score, and yes/no-style decisions over configurable questions without generating prose. Its maintainers also explicitly emphasize domain fine-tuning and [document the limits of the pretrained models](https://github.com/NandhaKishorM/laya#honest-limits). Our test covers the specified English checkpoint without local fitting; it does not cover its multilingual models, newer routing configurations, or equal-budget fine-tuning.

With this rubric, Laya often selected payments for other teams: it predicted payments 1,113 times against 240 true payments cases. Examples included a card-arrival question assigned to payments and an exchange-rate-source question assigned to payments. It also caught cases the fitted Qwen head missed, including a lost-wallet request for blocking and replacing a card, which the mapped labels place in security.

These are disagreements with the fixed dataset labels, not fresh human adjudication. Short messages can cross operational boundaries. Both directions of disagreement and the full confusion counts are included in the bundle.

## Wrapper checks and numerical sensitivity

Wrapping Laya in JJM preserved every tested single-call weight exactly: 16 smoke pairs, 100 smaller-run pairs, and 200 full-run pairs. Median paired wrapper overhead in the full run was 0.77 ms.

A separate diagnostic found that 2 of 400 shared messages changed Laya's selected outcome between batch sizes 16 and 32, with maximum weight change 0.0218. Single-call evaluation versus saved full-run batches changed 3 of 202 outcomes: the 200 preselected latency rows plus the two changed cases. Maximum weight change was 0.0428.

This establishes some sensitivity to batching in this CUDA bfloat16 configuration. It does not identify the cause. The direct/wrapped checks used matching single-call settings, so their exact parity remains valid. We did not change numerical settings in response to test results.

## Rejecting uncertain decisions

On the separate validation set, we selected a top-weight threshold targeting at most 2% empirical error with at least 20 accepted examples. On the full test:

<div class="compare" role="region" aria-label="Validation-selected rejection policy" tabindex="0">
<table>
<thead><tr><th scope="col">Route</th><th scope="col">Test coverage</th><th scope="col">Accepted errors</th><th scope="col">Error rate</th></tr></thead>
<tbody>
<tr><th scope="row">Qwen + fitted head through JJM</th><td>90.03%</td><td>24 / 2,773</td><td>0.87%</td></tr>
<tr><th scope="row">TF-IDF baseline</th><td>87.18%</td><td>35 / 2,685</td><td>1.30%</td></tr>
<tr><th scope="row">Pretrained Laya</th><td colspan="3">No validation threshold qualified</td></tr>
</tbody>
</table>
</div>

The 2% target is not a guarantee: the smaller-label JJM run had 2.96% test error among accepted cases. These weights are normalized readouts, not calibrated probabilities. The benchmark head remains fitted_unvalidated; reporting a useful held-out score does not automatically accept it for an application's scope. This table concerns the original baseline, not the new integrated TF-IDF run.

## Reproduce and inspect

[Download the benchmark and evidence bundle](/benchmarks/routing-2026-10-15.zip). It includes the runner, pinned revisions, all 78 manifest-listed files, exact split and latency-sample IDs, predictions, envelopes, fitted-head artifacts, reports, and numerical diagnostics. It excludes environments, model weights, and downloaded CSVs. The bootstrap fetches the pinned public dependencies and data; see the included README for setup.

The data is [PolyAI's Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data), used under CC BY 4.0. The fixed intent-to-team mapping is in task_data.py. Source CSV hashes and split separation were checked; no exact duplicate texts were found across the selected splits.

The measured stack was Python 3.12.13, Torch 2.14.0+cu130, Transformers 4.57.6, Laya 0.4.8, and scikit-learn 1.9.1. Qwen and Laya used pinned model snapshots. JJM's benchmark snapshot is 0192a6354b2ab7b5a2c4d69b4c2bfe785b775c14. The complete pins and environment freeze are in the bundle; the newer CPU backend is a separate source snapshot linked from its guide.

After bootstrap, the substantive commands are:

~~~sh
python bench.py --device cuda --train-per-team 64 --val-per-team 50 --test-per-team 50 --latency-n 100 --batch-size 16 --out results/512-labels-NEW
python bench.py --device cuda --train-per-team 0 --val-per-team 50 --test-per-team 0 --latency-n 200 --batch-size 32 --out results/full-NEW
~~~

Run these using the isolated benchmark environment, with cached weights if reproducing warm-load conditions. Use fresh output directories. The bundle's SHA256SUMS.json verifies its contents. [Download SHA-256 checksums for the published archives](/benchmarks/SHA256SUMS.txt).

## What to test next

This study covers one English task, seed, mapping, and machine. Unknown pretraining overlap with Banking77 is a limitation for both pretrained models. Prompt wording and precision were not optimized on test labels.

Before making a broader competitive claim, repeat seeds and timing runs, compare a plain classifier on the same Qwen features, and evaluate validation-selected Laya rubrics and equal-budget fine-tuning on a fresh held-out test. A second domain with paraphrases and fewer lexical cues would test how well the cheap sparse route travels.

For a fixed routing task with enough labeled examples, this evidence makes TF-IDF a serious starting point. For this task with fewer labels, the fitted embedding head gave a substantial accuracy gain. JJM now lets either route use the same explicit contract.
