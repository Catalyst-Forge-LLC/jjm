# JJM experiment log

Last updated: 2026-09-30

This is a lab notebook for the judgment experiments. It is not a model card (that describes a model you ship) and not a test plan (that is a walkthrough of the product). Each entry records one comparison: the question, what each method was allowed to see, the result, and what that result does and does not support.

The raw tables stay in the trial write-ups. This log states the conclusion and points at them, so the numbers are not copied into a second place that can drift.

## Form

Each entry uses these fields.

| Field | What it holds |
| --- | --- |
| Question | The one comparison this entry is allowed to answer |
| Data | Which examples, which labels, and which rows were held out of the fit |
| Methods | What text each method read, and whether it saw the labels |
| Hardware | Machine, and the software versions that affect the numbers |
| Result | The headline numbers, and the file that holds the rest |
| Worked | What the result supports |
| Did not | What failed, or what the setup cannot support |
| Promising | The next measurement that is worth the time |
| Caveat | Teacher labels, subset vs full set, privacy, or anything else a reader could miss |

A later entry is appended below. It does not rewrite an earlier one.

## 2026-09-30 — Fitted head, zero-shot models, and two tasks

### Question

For a closed choice, is a frozen encoder plus a small fitted head more accurate and faster than asking a local model with no labels? The two tasks are bank-message routing, and job-fit rank on one person's production jobs.

### Data

- **Banking77** (PolyAI, CC-BY-4.0, commit `57ec275d`). 10,003 train and 3,080 test messages. Human intent labels, mapped here to 8 teams. Slow methods share one stratified subset: 50 test messages per team, 400 in all, seed 0. Data is in `data/banking77`, which git ignores.
- **Exec Foundry fit export.** 253 production jobs, 241 with a Sonnet rank. The export holds resume text and is gitignored (`exec-foundry/temp/jjm-fit-export.prod.jsonl`). Sonnet's rank is a teacher label, not ground truth. There are no owner labels yet. Candidate text is the same person on every job.

### Methods

- **Fitted head.** A frozen encoder (Hugging Face or Ollama) and a softmax head fit with L-BFGS. The head is reused when the training set matches, and it is served only after an explicit promote. Fitting holds out about 20% of each outcome and leaves the artifact `fitted_unvalidated` until `jjm accept --scope`.
- **Zero-shot decision models.** Ollama 0.35 `/v1/systemone`: Nimble 9B, Tev1 4B, Tev1 0.8B. No labels. Confidence on that API is entropy, not calibrated correctness. These calls are not JJM envelopes.
- **Other zero-shot.** qwen3.5:9b asked for a team, a reranker yes/no per team, and the AnyJev readout on Qwen3-1.7B and Qwen3-4B.
- **Fit-rank probe.** One forward pass per job, then a ridge head inside nested cross-validation, so no job is scored by a head that trained on it. The strong reranker row used a dedicated prompt: the candidate as the query and the listing as the document. That is a different text from the JJM rendered state.

### Hardware

RTX 4000 Ada laptop GPU, 12 GB, CUDA 13.0, bfloat16. Transformers 5.17. Banking77 zero-shot through `/v1/systemone` used Ollama 0.35.0. Earlier Ollama rows used 0.34.2. Decision models are in `Z:\models` (`OLLAMA_MODELS`). The tray app's model list is a separate store.

### Result

Routing, against human labels. Full detail: `trials/routing/README.md`.

| Route | Accuracy | Speed | Labels the method saw |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B + head | 97.1% full test; 96.5% on the 400 | about 5 ms per message in bulk; 11 ms through `evaluate_batch` | all 10,003 training messages |
| nomic-embed (Ollama) + head | 96.5% full test | about 30 ms in bulk | same |
| Qwen3-Reranker-0.6B + head | 94.3% full test | about 3 ms in bulk | same |
| Nimble 9B, zero-shot | 92.5% on the 400 | 482 ms median | none |
| Tev1 4B, zero-shot | 89.3% on the 400 | 389 ms median | none |
| qwen3.5:9b, zero-shot | 88.0% on the 400 | 390 ms median | none |
| Tev1 0.8B, zero-shot | 79.5% on the 400 | 370 ms median | none |
| Reranker yes/no per team | 77.8% on the 400 | 82 ms median | none |
| AnyJev readout, Qwen3-1.7B | 75.3% on the 400 | 229 ms median | none |

Labels the embedding head needs, mean of three draws, full test: 68 / 77 / 85 / 87 / 92 / 93 / 95 / 97.1% at 4, 8, 16, 32, 64, 128, 256, and all examples per team. At 64 per team (512 labels) it is about even with Nimble's 92.5% and ahead of the 9B's 88%. At 16 per team it is 85%, which is under the 9B.

A head fit on 800 teacher labels scored 82.7% on the test set. The same 800 human labels scored 92.6%. Sending the head's unsure cases to that teacher lowered accuracy. At a top-mass bar of 0.76, the cases the embedding head keeps are about 98% correct.

Fit rank, against Sonnet. Full detail: `trials/fit_rank/README.md`.

| Method | What it read | Rank correlation | Exact | Weighted kappa | ms/job |
| --- | --- | --- | --- | --- | --- |
| Always answer "Strong" | nothing | 0 | 0.564 | 0 | |
| JJM readout, Qwen3-4B | five JJM prompts | 0.43 | 0.27 | 0.19 | 5000 |
| Reranker P(yes), zero-shot | candidate as query, listing as document | 0.43 | | | 186 |
| Reranker hidden-state ridge | that same prompt | 0.52 | 0.640 | 0.45 | 186 |
| Engine softmax head | JJM rendered state | 0.26 | 0.498 of 239 | 0.13 | 365 encode, then about 16 ms cached |
| Ridge on those same rendered-state vectors | JJM rendered state | 0.14 | 0.553 | about 0 | 365 |

The engine run scored 239 of 241 jobs with a head that had not trained on that job. The two Long Shot jobs stayed in every fit. Nothing was truncated at 8,192 tokens. Always answering "Strong" on those 239 jobs is exact 0.569. The head's picks were Perfect 26, Strong 144, Good 69, and neither Stretch nor Long Shot.

### Worked

- On bank routing, with labeled history, the embedding head is the most accurate route measured and the fast one: 97% at a few milliseconds, against 92.5% for Nimble at about half a second and 88% for a 9B chat model at about 400 ms.
- The head needs on the order of 64 labels per team before it matches Nimble, and it keeps the speed after that.
- Embedding features beat reranker features at every label budget on this routing task.
- Batching is the large speed lever on short messages. A cached batch through the contract is about 7–11 ms per request. Fitting 10,000 examples is about 8 seconds. Asking to fit the same examples again returns the stored head.
- Zero-shot, Nimble is the strongest decision model tried here, and every one of its 400 replies parsed. Tev1 4B is close to qwen3.5:9b. Tev1 0.8B is not.
- On fit rank, a reranker hidden state plus a ridge, reading the dedicated prompt, orders jobs about as well as the 4B readout (correlation 0.52 against 0.43) and is much faster (186 ms against 5 s). That head was trained on Sonnet, so it is distillation.

### Did not

- The engine head on the JJM rendered fit state does not beat always answering "Strong" (exact 0.50 against 0.57). A ridge on the same vectors is weaker still (correlation 0.14). The rendered state does not carry the signal the dedicated prompt carried (correlation 0.52).
- Training the routing head on the 9B teacher's labels costs about 10 points against human labels. Escalating unsure cases to that teacher made accuracy worse.
- The 4B readout's picked rank is not usable on its own. It mostly says Perfect or Strong. Its expected rank still tracks Sonnet (correlation 0.43).
- Qwen3-1.7B as an AnyJev readout is too small for fit rank. On the local jobs it answered Strong for every one.
- These routing heads have not been accepted with `jjm accept`. Their envelopes stay `fitted_unvalidated`. Cross-validation chose the penalty, so it is not a held-out validation.

### Promising

- The next fair comparison is the same 241 fit jobs, the same five options, and the same text, for the fitted head, Nimble, and Tev1 4B. Run it on the rendered state and on the dedicated prompt. Sonnet stays the teacher, not the truth. That is the comparison that can show whether the head loses when everyone reads the same words.
- After that, Nimble on all 3,080 Banking77 test messages would put the zero-shot number on the same full set as the head.
- A second public intent set (CLINC150 is the close twin) is the check that the routing result is not special to Banking77. A longer document-match set is the twin of fit rank.
- Blind owner labels on a subset of the fit jobs are still the check that matters for that task. A head trained here will not transfer to another person without their labels.

### Caveat

- Banking77 gives the head thousands of human labels and gives Nimble none. Same messages, different information. The label curve is what makes that comparison fair.
- Fit rank gives the head Sonnet's ranks. Agreement with Sonnet is not correctness. The candidate text is one person, and the export's resume may be newer than the resume Sonnet saw.
- The production export holds resume text. It stays gitignored, and it should be deleted when the fit trial is over.
- The published 97.1% routing figure is the earlier optimizer. A refit with the current fitter landed within about half a point (96.8% on that check).
