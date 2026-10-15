# TF-IDF for fixed text categories

Use this CPU backend when you have labeled examples, stable categories, and short
text with useful vocabulary: support routing, document categories, or triage
queues. It needs neither a GPU nor a downloaded language model.

It learns weighted words and bigrams and fits logistic regression. It does not
interpret a new rubric zero-shot. Synonyms, language changes, new categories, and
domain drift need evaluation and often new labels.

## Run the example

From the repository root, with Python 3.10+ and uv:

~~~sh
uv sync --extra text
uv run jjm --profiles examples/tfidf/profile.json fit --profile route-text --criterion examples/tfidf/criterion.json --examples examples/tfidf/examples.jsonl --promote
uv run jjm --profiles examples/tfidf/profile.json evaluate --request examples/tfidf/request.json
uv run jjm --profiles examples/tfidf/profile.json heads --profile route-text
~~~

The eight invented tickets demonstrate the lifecycle; they do not establish
accuracy. Artifacts go under ignored runs/tfidf/heads. The returned method is
jjm:tfidf-logistic. The doctor and capabilities commands recognize this profile.

For your own data, keep the same state structure in training and evaluation.
Each JSONL row has a state and an outcome ID in label:

~~~json
{"state":{"content":{"ticket":"I was charged twice"},"evidence":[]},"label":"billing"}
~~~

Copy and edit the example criterion and profile, then fit with your own JSONL.
This initial backend supports choice criteria with two to 26 options and the
default softmax method. It requires at least two distinct rendered states for
**every** declared outcome. Duplicate rendered states are rejected; resolve
conflicting labels and deduplicate first. Two examples are far too few to assess
most routing tasks.

## Fitting and serving

A deterministic, stratified 20% holdout is selected by rendered-state digest
before the vocabulary, IDF, or classifier is fitted. Each outcome retains at
least two fit rows, so tiny datasets can have no holdout. Only actual fit rows
determine features, including IDF. Hyperparameters are fixed profile settings;
the CV report is null. If you tune after inspecting a holdout, reserve another
untouched evaluation set.

~~~json
{
  "profiles": [{
    "id": "route-text",
    "adapter": "tfidf",
    "artifacts": "../../runs/tfidf/heads",
    "ngram_range": [1, 2],
    "min_df": 1,
    "max_features": null,
    "C": 4.0,
    "max_iter": 1000
  }]
}
~~~

Artifact paths resolve relative to the profile file. Defaults are shown above.
The ngram range allows inclusive word lengths from 1 through 3; min_df is a
positive document count; max_features is null or a positive integer. C must be
finite and positive. Text is lowercased, term frequency is sublinear, and vectors
use L2 normalization. The solver is lbfgs. Failure to converge refuses the fit;
increase max_iter and refit.

Artifacts contain vocabulary, IDF, classes, coefficients, and intercepts as
digest-checked JSON, with no executable pickle. They bind to the criterion,
rendering identity, feature settings, and scikit-learn version. Changed criteria
or settings need a matching promoted artifact; a scikit-learn upgrade also
requires refitting. Repeating a fit with the same labeled states reuses the
artifact. Promotion requires --promote. A missing or invalid promoted artifact
abstains with ARTIFACT_MISMATCH. A long-lived process caches the loaded artifact;
recreate its profiles/adapters after promotion in a different process.

The full JJM contract applies: outcome order, method allowlists, validation
scope, thresholds, provenance, and calibration requirements. After inspecting
the held-out report and deciding it is appropriate for a scope, record acceptance:

~~~sh
uv run jjm --profiles examples/tfidf/profile.json accept --profile route-text --criterion examples/tfidf/criterion.json --scope example.support
~~~

The example's tiny report only demonstrates this command. Acceptance never
happens during fitting and does not assess probability calibration. Weights stay
normalized_readout. Requiring assessed_in_scope calibration still abstains.

An input with no recognized features produces the classifier's intercept-based
distribution. The adapter reports all_features_unknown when the entire rendered
input has zero active features, but common state-format words can still match.
This is not a domain-shift detector or automatic rejection policy. Choose and
evaluate your thresholds on separate validation data.

## Measured integration check

On October 15, 2026, the integrated backend scored 2,950 of 3,080 Banking77
eight-team test messages correctly (**95.78%**, macro F1 **0.9528**), after fitting
7,680 rows and withholding 1,923. On the same laptop as the neural benchmark,
200 warm requests through JJM had **1.85 ms p50 / 2.20 ms p95**. Fit and artifact
creation took 1.58 seconds; artifact reload took 0.57 seconds. No torch,
transformers, or laya modules were loaded.

This is separate from the original **95.65%, 0.40 ms** raw-text TF-IDF baseline.
The integrated backend receives rendered JJM state, sorts fit rows canonically,
and includes the contract. Its reloaded classifier and JJM outputs matched
exactly on all 3,080 rows. The batch command currently evaluates each TF-IDF
request separately; its measured 669 rows/s is not the original vectorized
classifier's roughly 100,000 rows/s.

Reproduce after downloading the pinned CSVs using the benchmark bootstrap:

~~~sh
uv run python trials/routing/tfidf_backend.py --out runs/tfidf-backend-NEW
~~~

See [the saved integration results](../../trials/routing/results/tfidf-backend-20261015/summary.json)
and [the original benchmark](../../benchmarks/jjm-laya-local-benchmark/results/LOCAL_GPU_REPORT.md).
These are single-task, single-machine measurements, not latency guarantees or
evidence of calibrated probabilities.
