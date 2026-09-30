# Banking77 routing trial

Which of eight teams should take a bank customer's message? The trial measures every JJM route against
human labels, and how fast each one is, on one fixed task.

## Data

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) by PolyAI, CC-BY-4.0: 13,083 real
customer messages, each labelled by people with one of 77 intents. The data is pinned to commit
`57ec275d` and checked by sha256 on load. It is not redistributed here: `data/` is gitignored, and
`data.py` has the download URL.

The 77 intents are grouped into eight teams: cards, payments, transfers, top_up, cash, fees_fx,
security, and account. The grouping is ours and is fixed in `data.py`. Every message keeps its human
label, mapped through that table. Some intents could sit with either of two teams, for example fees on
card payments, so a little of the remaining error is our grouping, not the model.

The criterion is a `choice` with the eight team descriptions as options, compiled by JJM as usual.
Heads are fitted on the 10,003 training messages and scored on all 3,080 test messages. The slow
methods (the teacher and the zero-shot readouts) run on a fixed stratified subset of 400 test
messages, 50 per team.

## Setup

RTX 4000 Ada laptop GPU (12 GB), CUDA 13.0, bfloat16, transformers 5.17, Ollama 0.34.2. Models are
from the local Hugging Face cache and Ollama, pinned by revision or digest in `jjm-profiles.local.json`.

## Results

### Accuracy against human labels

| Route | Accuracy | Speed per message | Notes |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B + head | **97.1%** (full test) | 4.7 ms in bulk | macro F1 0.967 |
| nomic-embed-text-v2 (Ollama) + head | 96.5% (full test) | 30 ms in bulk | macro F1 0.961 |
| Qwen3-Reranker-0.6B + head | 94.3% (full test) | 3.2 ms in bulk | macro F1 0.937 |
| Qwen3-Reranker-0.6B, half depth, + head | 94.1% (full test) | 3.0 ms in bulk | 14 of 28 layers |
| qwen3.5:9b teacher, zero-shot (Ollama) | 88.0% (subset) | 390 ms median | every reply parsed |
| Qwen3-Reranker-0.6B yes/no per team, zero-shot | 77.8% (subset) | 82 ms median | 8 prompts in one batch |
| JJM AnyJev readout, Qwen3-1.7B, zero-shot | 75.3% (subset) | 229 ms median | 8 rotations |

On the 400-message subset, the Qwen3-Embedding head scores 96.5%, so the subset is slightly harder
than the full test set.

### How many labels a head needs

Test accuracy by labelled examples per team (mean of three draws):

| Per team | 4 | 8 | 16 | 32 | 64 | 128 | 256 | all (~1,250) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding | 68% | 77% | 85% | 87% | 92% | 93% | 95% | 97.1% |
| nomic-embed | 65% | 75% | 83% | 86% | 91% | 92% | 94% | 96.5% |
| Reranker | 45% | 57% | 65% | 77% | 84% | 87% | 91% | 94.3% |
| Reranker, half depth | 35% | 46% | 61% | 71% | 77% | 81% | 89% | 94.1% |

With 16 labels per team (128 in all), the embedding head already beats the 9B teacher's zero-shot
answers. The reranker's features are built for yes/no relevance and need more labels to separate
eight teams. Half depth costs little with all the labels and a lot with few.

### Training on teacher labels instead of human labels

800 random training messages, labelled once by qwen3.5:9b (85% agreement with the human labels) and
once by people:

| Encoder | Head on teacher labels | Head on human labels, same 800 |
| --- | --- | --- |
| Qwen3-Embedding | 82.7% | 92.6% |
| Reranker | 79.8% | 87.4% |

A head learns its labeller, errors included, and ends up a little below that labeller. The label
source is recorded in each artifact for this reason.

### Escalating unsure cases to the teacher

The head answers when its top weight clears a bar, and the teacher takes the rest. On the subset:

| Bar | Share sent to teacher | Accuracy (embedding head) | Accuracy (reranker head) |
| --- | --- | --- | --- |
| none | 0% | 96.5% | 92.8% |
| 0.6 | 2–3% | 95.8% | 93.5% |
| 0.9 | 7–14% | 95.5% | 92.7% |
| 0.99 | 22–41% | 94.3% | 91.0% |
| everything | 100% | 88.0% | 88.0% |

With this teacher, escalation does not pay. On the 30 messages the embedding head was unsure about
(top weight under 0.9), the head was right on 67% and the teacher on 53%. Escalation only helps when
the teacher is better than the head on the hard cases, so it needs a stronger teacher, or a human.
The head's top weight does sort its errors well: at a 0.76 bar, the messages it keeps are 98%
correct, and the rest can go to a person.

### Speed

Qwen3-Reranker-0.6B encoder alone, 512 test messages, milliseconds per message:

| Batch | Full depth | Full + prefix reuse | Half depth | Half + prefix reuse |
| --- | --- | --- | --- | --- |
| 1 | 90.5 | 114.7 | 39.4 | 43.9 |
| 8 | 12.0 | 12.2 | 5.2 | 5.3 |
| 32 | 3.5 | 2.8 | 1.6 | 1.5 |
| 64 | 2.5 | 1.9 | 1.4 | 1.3 |

- Batching is the largest lever: 36 times faster at 64 than at 1.
- Half depth halves the cost at small batch sizes.
- Prefix reuse helps only in large batches, and slows single messages. These messages are short (the
  shared instruction is a large part of each input), so this is the best case for it; long states
  will gain less.
- In bulk runs through the featurizer, the two depths took the same time (about 3 ms). There,
  tokenizing and writing the cache cost as much as the model.

Through the full JJM contract (`evaluate` and `evaluate_batch`) with the Qwen3-Embedding head:

| Path | Accuracy | Time per request |
| --- | --- | --- |
| `evaluate`, one request at a time, nothing cached | 96% (100 requests) | 97 ms median, 227 ms p95 |
| `evaluate_batch`, 400 requests | 96.5% | 11 ms |
| the same batch again, every feature cached | — | 7 ms |

- Fitting on all 10,003 messages took about 2 minutes with the first optimizer, features already
  cached. The fitter now chooses its penalty on a stratified subset and then fits once, which takes
  about 8 seconds on this machine and landed within half a point of that run. Asking to fit the same
  examples again returned the stored head in 0.35 s instead of refitting.
- Cached features load from disk at about 0.24 ms per message.
- The 7 ms for a fully cached batch is JJM's own envelope work (validation, digests, checks), so that
  is now the largest cost per request in a batch.

Each envelope reports `calibration.status: fitted_unvalidated` with the artifact id, and
`numeric_path: hidden_state_head`. Inside a batch, an envelope whose features were prefetched reports
`prompt_evaluations: 0`, because the shared batch pass is not its own.

## What this says

- For a closed routing question with labelled history, a small frozen encoder with a fitted head is
  the strongest and fastest route here: 97% at a few milliseconds, against 88% for a 9B model asked
  directly at about 400 ms.
- Encoder choice matters more with few labels. An embedding model beats the reranker at every size
  here.
- Zero-shot readouts on 0.6–1.7B models land near 75–78% on this task. They are for when there are
  no labels at all.
- A teacher is useful for labels when people's labels are missing, at a cost of about 10 points
  here. It is not a useful fallback unless it is better than the head on the head's hard cases.
- The subset thresholds above are read off the same data they describe. They are not tuned deployment
  settings, and none of these heads has a held-out validation report.

## Reproduce

```text
uv run python -m trials.routing.bench encode   --profiles jjm-profiles.local.json --profile route-embed --name embed
uv run python -m trials.routing.bench heads    --name embed
uv run python -m trials.routing.bench teacher  --profiles jjm-profiles.local.json --teacher qwen35 --split test
uv run python -m trials.routing.bench teacher  --profiles jjm-profiles.local.json --teacher qwen35 --split train
uv run python -m trials.routing.bench distill  --name embed --teacher qwen35
uv run python -m trials.routing.bench cascade  --name embed --teacher qwen35
uv run python -m trials.routing.bench readout  --profiles jjm-profiles.local.json --profile local-advisory
uv run python -m trials.routing.bench rerank-zeroshot --model-path <reranker snapshot>
uv run python -m trials.routing.bench speed    --model-path <reranker snapshot> --revision <sha>
uv run python -m trials.routing.bench engine   --profiles jjm-profiles.local.json --profile route-embed
```

Profiles for the four encoders are in `examples/jjm-profiles.routing.example.json`. Results land in
`runs/routing/`, which git ignores.
