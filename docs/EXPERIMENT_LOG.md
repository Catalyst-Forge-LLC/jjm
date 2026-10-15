# JJM experiment log

Last updated: 2026-10-15 (Wilson interval and held-aside wording appended)

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

## 2026-09-30 — Same words on the fit jobs

### Question

When the fitted head, Nimble, and Tev1 read the same fit jobs and the same five options, does the head still lose, and does the dedicated prompt change that?

### Data

The same 241 Sonnet-ranked production jobs as the entry above. Sonnet remains the teacher. The two Long Shot jobs stay in every head fit, so the head rows are 239 jobs. The decision models scored all 241.

### Methods

- **Rendered state.** The JJM state the engine head already read.
- **Dedicated prompt.** The candidate as the query and the listing as the document. Nimble and Tev1 received that string as the state. The fitted head received it as the body, with its usual criterion prefix in front. That is not the chat wrapper behind the earlier ridge correlation of 0.52.
- **Tev1.** The published model file caps context at 2,050 tokens and refuses a longer prompt. The run used a local alias with the same weights and an 8,192-token window, the same cap Nimble ships with. No job was truncated. Details: `trials/fit_rank/README.md`.

### Hardware

Same GPU as the entry above. Ollama 0.35.0 for Nimble and Tev1. The head encode used the local Qwen3-Reranker-0.6B.

### Result

Full table: `trials/fit_rank/README.md`, section "Same jobs, same words."

On the dedicated prompt, exact agreement with Sonnet was Nimble 0.45, the fitted head 0.46 (of 239), and Tev1 0.29. Always answering "Strong" is 0.56. Expected-rank correlation was about 0.60 for both decision models on either text, and 0.20 for the head on the dedicated prompt (0.26 on the rendered state). Nimble took about 1.1 seconds per job on that prompt, Tev1 about 0.7, and the head's encode about 0.4.

### Worked

- Giving Nimble and Tev1 the dedicated prompt instead of the rendered state raised exact agreement, from 0.24 to 0.45 for Nimble and from 0.16 to 0.29 for Tev1.
- Their probability-weighted rank tracks the teacher at about 0.60 whichever text they read. Every reply parsed.

### Did not

- No picked letter beat always answering "Strong."
- The softmax head on the dedicated prompt did not recover the earlier ridge result. It still mostly says Strong, and its probabilities order jobs more weakly than Nimble or Tev1.

### Promising

- The useful fit-rank signal in the decision models is the probability distribution, not the letter they pick.
- The ridge's 0.52 used a different encoding of this prompt (the reranker chat wrapper, last token at the assistant position). Scoring that same encoding with the softmax head is the remaining same-model check.
- Blind owner labels are still the check that is not agreement with Sonnet.

### Caveat

- Tev1's published context cap cannot read these jobs. The numbers above are the same weights at 8,192 tokens.
- The head's row has a criterion prefix the decision models do not. The body words match.
- Agreement with Sonnet is not correctness. One candidate. The export's resume may be newer than the resume Sonnet saw.

## 2026-09-30 — Softmax on the reranker hidden states

### Question

Does the softmax head recover the ridge result when both read the chat-wrapped reranker hidden states?

### Data

The same 241 Sonnet-ranked jobs. Features are the saved vectors in `runs/feat-rerank-0.6b.npz` (gitignored). Sonnet is the teacher.

### Methods

The ridge row and this row share those vectors and the same outer loop: 10 folds, 3 shuffles, no job scored by a head that trained on it. The ridge predicts a number and rounds it. The softmax picks the class with the highest weight. The penalty is chosen inside each training fold.

### Hardware

CPU only. The vectors were encoded earlier on the RTX 4000 Ada, 186 ms per job.

### Result

Full row: `trials/fit_rank/README.md`, "Reranker hidden-state head" and the softmax row under it.

Rank correlation 0.522, AUC 0.811, exact 0.609, within one 0.967, weighted kappa 0.420. Always answering "Strong" is exact 0.564. One shuffle picked Perfect 28, Strong 147, Good 65, Stretch 0, Long Shot 1.

### Worked

The softmax tracks Sonnet as well as the ridge does. Exact agreement beats always answering "Strong." The earlier miss was the text the encoder read, not the head.

### Did not

Exact agreement is a little under the ridge's rounded score (0.61 against 0.64). The head still barely uses Stretch.

### Promising

Serving this encoding through the fitted-head engine, with the reranker's chat wrapper as the encoder input, is the path that matches the result that worked.

### Caveat

Agreement with Sonnet is distillation, not correctness. One candidate.

## 2026-09-30 — Nimble on the full Banking77 test set

### Question

Does Nimble's 92.5% on 400 messages hold on all 3,080 test messages?

### Data

Banking77 test split, human labels mapped to 8 teams. The 400-message figure was 50 per team, seed 0.

### Methods

Nimble 9B, zero-shot, through Ollama `/v1/systemone`. No labels. Not a JJM envelope. Confidence on that API is entropy.

### Hardware

RTX 4000 Ada, Ollama 0.35.0, Nimble on GPU with an 8,192-token window.

### Result

91.6% accuracy, 3,080 of 3,080 replies parsed, median 694 ms. The embedding head on this same full set is 97.1% at a few milliseconds. Weakest team: transfers, 83%. Detail: `trials/routing/README.md`.

### Worked

The subset result was not a fluke. Nimble stays near 92% on the full set, and every reply parsed.

### Did not

It does not catch the fitted head, and it is about 100 times slower per message once the head is fitted.

### Promising

A second public intent set is the check that this routing gap is not special to Banking77.

### Caveat

The head saw the 10,003 training labels. Nimble saw none. The label curve in the routing write-up is the fair budget comparison: about 64 labels per team before the head matches Nimble.

## 2026-10-15 — Same embeddings, a plain classifier, and an operating point

### Question

On the eight-team Banking77 mapping, does the published embedding head beat a plain classifier that sees the same labels? And if a cutoff is chosen on held-aside messages, what share of the remaining messages can be handled at a stated error rate?

### Data

Banking77, human intents mapped to eight teams. Train 10,003, test 3,080. Cached embeddings and the published head's test weights are `runs/routing/feat-embed.npz` and `runs/routing/weights-embed.npy`. The operating point uses a stratified 20% of the test set (616 messages, seed 0) only to choose the cutoff. The other 2,464 are scored after that choice.

### Methods

- The published Qwen3-Embedding-0.6B softmax head, weights already saved.
- scikit-learn logistic regression (saga, C=1) on those same embeddings, after standardizing on the training set.
- Word and bigram TF-IDF on the customer message, then the same logistic regression. No encoder.
- A minimum top weight. Targets of 1%, 2%, and 5% error among handled messages, chosen on the 616 and scored on the 2,464.

### Hardware

The classifier fits ran on CPU. The embeddings were the existing GPU cache. scikit-learn is not a JJM dependency.

### Result

Logistic regression on the same embeddings: 97.2% of 3,080. The published head: 97.1%. TF-IDF: 94.9%, fit in 0.3 seconds. Detail is `runs/routing/baselines.json` and `trials/routing/README.md`.

On the untouched 2,464, a top weight of 0.91 handles 91.7% (2,259) with 0.84% of those wrong. That cutoff was chosen for a 1% error target. The 2% target did not hold: the untouched error was 2.8% while handling 99.8%. Handling every message on that slice is 3.0% wrong.

### Worked

A plain classifier on the same encoder matches the head. The useful gap against TF-IDF is about 2 points, on this eight-team task, with the same human labels. A cutoff chosen aside from the scored messages can hold a 1% error among the messages the head handles.

### Did not

The tenth of a point between the head and the logistic regression is not a ranking. That fit stopped at its iteration limit. The 2% target, chosen on 616 messages, was too optimistic on the rest. This does not compare SetFit, and it does not say the head beats a classifier that fine-tunes the encoder.

### Promising

The same cutoff procedure on a second task, and a real application that sends the unhandled share to review.

### Caveat

The head and both classifiers saw the 10,003 human training labels. This is not the zero-shot comparison. The mapping is eight teams, not 77 intents.

## 2026-10-15 — The 0.84% is an observation

### Question

What does the 0.84% error among handled messages support?

### Data

The same operating point as the entry above. 19 of the 2,259 handled messages on the 2,464 were wrong.

### Methods

A 95% Wilson interval on 19 errors in 2,259 handled cases.

### Hardware

No new run.

### Result

The interval is about 0.54% to 1.31%. The 2,464 were held aside from choosing the cutoff. They are part of the test set already scored for accuracy, so they are not an untouched test.

### Worked

The point estimate on this slice is 0.84%, under 1%. The interval runs to about 1.31%, so it does not establish that the rate stays under 1%.

### Did not

The interval is not a guarantee that a future batch stays under 1%. Calling the 2,464 untouched overstated how new they were.

### Promising

The same cutoff on messages that never entered an earlier accuracy table.

### Caveat

Nineteen errors is a small count. The interval is wide for that reason.
