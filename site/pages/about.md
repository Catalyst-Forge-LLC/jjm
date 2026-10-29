---
title: "Turn fixed-choice questions into decisions your app can use."
description: "Use JJM when an application must choose among fixed options, such as assigning support messages to teams. Fit a local decision model on labeled examples, then receive a selected option, weights, provenance, and acceptance checks your application can use to route or request review."
order: 1
---

<section class="value-hero" aria-label="JJM overview"><div class="value-intro"><div class="value-headline"><p class="value-kind">Local judgment runtime for developers · alpha</p><h1>Turn fixed-choice questions into decisions your app can use.</h1></div><div class="value-pitch"><p class="value-lede">Use JJM when an application must choose among fixed options, such as assigning support messages to teams. Fit a local decision model on labeled examples, then receive a selected option, weights, provenance, and acceptance checks your application can use to route or request review.</p><div class="cta-row"><a class="cta cta-primary" href="#try-a-fitted-judgment">Try the local example</a><a class="cta cta-secondary" href="#judgment-example">See a response and routing decision</a></div><p class="value-limit">Alpha. Fit and evaluate on your own labeled cases. Weights are not calibrated probabilities.</p></div></div>
<figure class="value-proof worked-example" id="judgment-example" tabindex="-1">
<figcaption>Teach a local model your support categories</figcaption>
<p class="example-context">The runnable eight-ticket walkthrough uses invented training examples and two teams. It demonstrates fitting and evaluation, not production routing accuracy.</p>
<section class="example-step" aria-labelledby="example-step-1"><h2 id="example-step-1"><span>1 · Your task</span> Choose billing or technical support</h2><blockquote><p>The settings screen closes itself as soon as it opens.</p></blockquote><p>The application defines two categories: <strong>billing</strong> for charges and refunds, and <strong>technical</strong> for defects and failed operations.</p></section>
<section class="example-step example-report" aria-labelledby="example-step-2"><h2 id="example-step-2"><span>2 · Fit and evaluate</span> Learn from labeled tickets</h2><p><code>jjm fit</code> fits a small head on eight example tickets. <code>jjm evaluate</code> applies it to the new message and returns the selected team, a weight for each team, and the method and model used.</p><p><a href="#try-a-fitted-judgment">Run the exact commands below</a>.</p></section>
<section class="example-step" aria-labelledby="example-step-3"><h2 id="example-step-3"><span>3 · Your application decides</span> Route a result or ask for review</h2><p>Your code reads the selection alongside <code>acceptance</code> and provenance. With a backend evaluated for your task, route results your policy accepts, send held results for review, and refuse rejected results.</p><p class="example-payoff">The app gets explicit fields for a defined decision instead of parsing an open-ended paragraph.</p></section>
<p class="value-note">The tutorial model remains <code>fitted_unvalidated</code> until scope acceptance. Eight rows are enough to demonstrate the commands, not to establish reliable routing. Weights are not calibrated probabilities.</p>
</figure></section>

## Try a fitted judgment

Eight invented tickets, two teams. The first run downloads one small encoder and fits on CPU. Eight rows show the command. They are not a benchmark.

```bash
uv sync --extra hf
uv run jjm --profiles examples/try/profile.json fit \
    --profile try-embed --criterion examples/try/team.criterion.json \
    --examples examples/try/examples.jsonl --promote
uv run jjm --profiles examples/try/profile.json evaluate --request examples/try/request.json
```

The first run downloads `Qwen/Qwen3-Embedding-0.6B`. It requires Python 3.10+ and uv, and writes the head and feature cache under `runs/try/`. The envelope remains `fitted_unvalidated` until scope acceptance; the tutorial request deliberately does not require that acceptance. [Full walkthrough and validation rules](https://github.com/Catalyst-Forge-LLC/jjm#try-a-fitted-judgment).

## What you might hand it

- A support note, to sort into the teams you already use.
- A receipt, a log, or a short passage, to see whether it settles the question.
- An application or a review, to place into the ratings you defined, once you have labelled a few.
- A tool result or a document snippet, to return as one of the answers you allowed.

## What you send

A JSON request with a snapshot and explicit criteria. A choice has two to 26 options, or buckets, you define. An assertion has three outcomes: the supplied evidence supports the claim, contradicts it, or does not settle it.

<div class="snippet">
  <div class="snippet-label">Request</div>
  <pre><code class="hljs language-json">{
  <span class="hljs-attr">"state"</span>: {
    <span class="hljs-attr">"content"</span>: {
      <span class="hljs-attr">"ticket"</span>: <span class="hljs-string">"I was charged twice for the same subscription."</span>
    }
  },
  <span class="hljs-attr">"criteria"</span>: [{
    <span class="hljs-attr">"id"</span>: <span class="hljs-string">"route-ticket"</span>,
    <span class="hljs-attr">"kind"</span>: <span class="hljs-string">"choice"</span>,
    <span class="hljs-attr">"question"</span>: <span class="hljs-string">"Which team should own this ticket?"</span>,
    <span class="hljs-attr">"options"</span>: [
      { <span class="hljs-attr">"id"</span>: <span class="hljs-string">"billing"</span> },
      { <span class="hljs-attr">"id"</span>: <span class="hljs-string">"technical"</span> },
      { <span class="hljs-attr">"id"</span>: <span class="hljs-string">"other"</span> }
    ]
  }],
  <span class="hljs-attr">"requirements"</span>: { <span class="hljs-attr">"min_top_mass"</span>: <span class="hljs-number">0.5</span> }
}</code></pre>
</div>

Shortened from [`examples/route-ticket.json`](https://github.com/Catalyst-Forge-LLC/jjm/blob/main/examples/route-ticket.json). The file also names the scope, the profile, and the evidence list.

## What you get back

- **A weight** over every option you allowed.
- **Provenance.** The method, the model, and digests of the inputs.
- **Your check.** Pass, hold, or reject against requirements you declared. A pass means those requirements were met. The next action stays with your application.

Weights are a normalized readout. Accepting a held-out report records that scope. It does not change the weights, and it is not a check that a weight of 0.95 occurs about 95% of the time.

For a download-free view of the envelope, use the separate [contract demonstration](#check-the-contract-with-no-model). Its synthetic weights do not classify tickets.

## New routing benchmark

On our eight-team Banking77 task, Qwen embeddings with a fitted head through JJM
reached **96.69%**, pretrained Laya **55.32%**, and TF-IDF **95.65%**. JJM and
TF-IDF used local labels; Laya did not. The bare TF-IDF classifier took 0.40 ms
median on CPU.

[Read the full comparison, methods, limits, and reproducible results](/routing-benchmark).
You can also [try TF-IDF as a JJM backend](/tfidf), with the same judgment contract.

## What has been tried

Banking77, 77 intents mapped to eight teams, scored against the human labels, on one laptop GPU.

<div class="compare">
<table>
  <thead>
    <tr><th></th><th>Right</th><th>Time per note</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">Fitted head</th><td>97.1% of 3,080</td><td>about 5 ms in bulk</td></tr>
    <tr><th scope="row">Nimble, a 9B decision model</th><td>91.6% of 3,080</td><td>694 ms</td></tr>
    <tr><th scope="row">A 9B chat model, asked to pick</th><td>88% of 400</td><td>390 ms</td></tr>
  </tbody>
</table>
</div>

The head had seen the human training labels. Nimble and the chat model had seen none. About 64 labels per team, 512 in all, is where the head matches Nimble. The 97.1% is the earlier fit on the full test set. A refit scored 96.8%. In bulk that run was a few milliseconds per note, about 5 ms. Through the full contract, a batch of 400 took about 11 ms each, and one request at a time with nothing cached took 97 ms median on 100 requests. The other times in the table are medians.

On the same labels, a logistic regression on the same embeddings scored 97.2% of the 3,080, and a word-and-bigram TF-IDF classifier scored 94.9%. The tenth of a point is not a ranking. TF-IDF never calls the encoder.

A cutoff was chosen on 616 of the test messages and scored on the 2,464 held aside from that choice. At a top weight of 0.91 the head handled 2,259 of those 2,464. 19 of the 2,259 were wrong, 0.84%. A 95% Wilson interval for that rate is about 0.54% to 1.31%. That is an observation on this slice, not a guarantee of staying under 1%. A cutoff chosen for 2% error did not hold there: 2.8% of the handled messages were wrong.

Fit rank was one candidate. Agreement with the teacher moved with how the job was written, and agreement is not correctness.

The figures and the hardware are in the [experiment log](https://github.com/Catalyst-Forge-LLC/jjm/blob/main/docs/EXPERIMENT_LOG.md).

## Limits

- The eight-ticket walkthrough has been run on the author's machine. It downloads one small encoder and fits those eight rows.
- Routing was measured on Banking77, eight teams, human labels, on one laptop GPU. The fitted head used the training labels. On those same labels it matches a plain classifier on the same embeddings, and it is about 2 points ahead of TF-IDF. About 64 labels per team is where it lines up with Nimble, which had seen no labels.
- Fit rank was measured on one candidate. Agreement with the teacher is not correctness, and it moved with how the job was written.

## Check the contract, with no model

<figure class="value-proof worked-example" id="contract-response" tabindex="-1">
<figcaption>Optional contract demonstration: inspect the response without a model</figcaption>
<p class="example-context">Contract demonstration using the documented no-model response. These weights come from a hash, not a classifier. They must not route real tickets.</p>
<section class="example-step" aria-labelledby="contract-step-1">
<h2 id="contract-step-1"><span>1 · Request</span> Your app defines the available teams</h2>
<blockquote><p>I was charged twice for the same subscription.</p></blockquote><p><strong>Allowed options:</strong> billing, technical, other.</p><pre><code>"requirements": {
  "min_top_mass": 0.5
}</code></pre><p>Your application supplies the question and the threshold.</p>
</section>
<section class="example-step example-report" aria-labelledby="contract-step-2">
<h2 id="contract-step-2"><span>2 · Response excerpt</span> Weights, selection, and acceptance are explicit</h2>
<pre><code>"prediction": {
  "weights": {
    "billing": 0.892,
    "technical": 0.005,
    "other": 0.103
  },
  "selected": "billing"
},
"acceptance": { "status": "pass" },
"provenance": {
  "method": "mock:normalized",
  "model": { "id": "jjm/synthetic-no-model" }
}</code></pre>
</section>
<section class="example-step" aria-labelledby="contract-step-3">
<h2 id="contract-step-3"><span>3 · Application decision</span> Route only when your policy allows it</h2>
<p><strong>This response:</strong> Hold it. The provenance says synthetic, even though the threshold check passed.</p><p><strong>With an evaluated, fitted backend:</strong> Your app can route a passing result to its selected team, send <code>hold</code> to human review, and refuse <code>reject</code>.</p><p class="example-payoff">Your code gets a consistent decision envelope instead of parsing an open-ended paragraph.</p>
</section>
<p class="value-note">Excerpt shortened and weights rounded. “Pass” means declared checks passed, not that the selected answer is correct. Weights are not calibrated probabilities.</p>
<p class="example-links"><a href="#what-you-send">Inspect the request</a> · <a href="#try-a-fitted-judgment">Try the fitted example</a></p>
</figure>


The synthetic path needs no download. Its weights are a hash of the input, so they carry no meaning. They show the envelope.

```bash
uv sync
uv run jjm evaluate --request examples/route-ticket.json
```

JJM depends on [AnyJev](https://github.com/nokia-applied-research/AnyJev), Apache-2.0. JJM is not affiliated with TypeSafe AI or Jev.

Built by [Catalyst Forge LLC](https://www.catalystforge.com).
