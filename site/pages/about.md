---
title: Just Judgment Machine
description: An open-source local judgment runtime. Alpha. Your application sends the state and the options, and keeps the decision.
order: 1
---

## PE:TL;DR

The payoff is a fast, solid judgment an application can use. Fit it on cases you have already labeled, in the buckets you use. JJM does not try to talk. It makes the judgment: a weight for each option you allowed, the method and the model that produced it, and a pass, hold, or reject against checks you declared. Your application decides what happens next.

<div class="cta-row">
  <a class="cta cta-primary" href="https://github.com/Catalyst-Forge-LLC/jjm">View the source</a>
  <a class="cta cta-secondary" href="#limits">Read the limits</a>
</div>

<p class="kicker">Alpha · Apache-2.0 · runs on your machine</p>

## New routing benchmark

On our eight-team Banking77 task, Qwen embeddings with a fitted head through JJM
reached **96.69%**, pretrained Laya **55.32%**, and TF-IDF **95.65%**. JJM and
TF-IDF used local labels; Laya did not. The bare TF-IDF classifier took 0.40 ms
median on CPU.

[Read the full comparison, methods, limits, and reproducible results](/routing-benchmark).
You can also [try TF-IDF as a JJM backend](/tfidf), with the same judgment contract.

## What you might hand it

- A support note, to sort into the teams you already use.
- A receipt, a log, or a short passage, to see whether it settles the question.
- An application or a review, to place into the ratings you defined, once you have labelled a few.
- A tool result or a document snippet, to return as one of the answers you allowed.

## What you send

A JSON request with a snapshot and explicit criteria. A choice has two to 26 options you define. An assertion has three outcomes: the supplied evidence supports the claim, contradicts it, or does not settle it.

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

<div class="snippet">
  <div class="snippet-label">Return</div>
  <pre><code class="hljs language-json">{
  <span class="hljs-attr">"results"</span>: [{
    <span class="hljs-attr">"criterion_id"</span>: <span class="hljs-string">"route-ticket"</span>,
    <span class="hljs-attr">"prediction"</span>: {
      <span class="hljs-attr">"weights"</span>: {
        <span class="hljs-attr">"billing"</span>: <span class="hljs-number">0.892</span>,
        <span class="hljs-attr">"technical"</span>: <span class="hljs-number">0.005</span>,
        <span class="hljs-attr">"other"</span>: <span class="hljs-number">0.103</span>
      },
      <span class="hljs-attr">"semantics"</span>: <span class="hljs-string">"normalized_readout"</span>,
      <span class="hljs-attr">"selected"</span>: <span class="hljs-string">"billing"</span>
    },
    <span class="hljs-attr">"acceptance"</span>: { <span class="hljs-attr">"status"</span>: <span class="hljs-string">"pass"</span> },
    <span class="hljs-attr">"provenance"</span>: {
      <span class="hljs-attr">"method"</span>: <span class="hljs-string">"mock:normalized"</span>,
      <span class="hljs-attr">"model"</span>: { <span class="hljs-attr">"id"</span>: <span class="hljs-string">"jjm/synthetic-no-model"</span> }
    }
  }]
}</code></pre>
</div>

Shortened from the no-model command below, with the weights rounded. They are a hash of the input, so this pass is not a judgment that the ticket is billing. The full envelope also includes the digests, the checks, and that notice.

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

## Try a fitted judgment

Eight invented tickets, two teams. The first run downloads one small encoder and fits on CPU. Eight rows show the command. They are not a benchmark.

```bash
uv sync --extra hf
uv run jjm --profiles examples/try/profile.json fit \
    --profile try-embed --criterion examples/try/team.criterion.json \
    --examples examples/try/examples.jsonl --promote
uv run jjm --profiles examples/try/profile.json evaluate --request examples/try/request.json
```

The same walkthrough is in the repository README.

## Check the contract, with no model

The synthetic path needs no download. Its weights are a hash of the input, so they carry no meaning. They show the envelope.

```bash
uv sync
uv run jjm evaluate --request examples/route-ticket.json
```

JJM depends on [AnyJev](https://github.com/nokia-applied-research/AnyJev), Apache-2.0. JJM is not affiliated with TypeSafe AI or Jev.

Built by [Catalyst Forge LLC](https://www.catalystforge.com).
