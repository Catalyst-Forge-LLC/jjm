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

## What you might hand it

- A support note, to sort into the teams you already use.
- A receipt, a log, or a short passage, to see whether it settles the question.
- An application or a review, to place into the ratings you defined, once you have labelled a few.
- A tool result or a document snippet, to return as one of the answers you allowed.

## What you send

A JSON request with a snapshot and explicit criteria. A choice has two to 26 options you define. An assertion has three outcomes: the supplied evidence supports the claim, contradicts it, or does not settle it.

<div class="snippet">
  <div class="snippet-label">Request</div>
  <pre><code>{
  <span class="k">"state"</span>: {
    <span class="k">"content"</span>: {
      <span class="k">"ticket"</span>: <span class="s">"I was charged twice for the same subscription."</span>
    }
  },
  <span class="k">"criteria"</span>: [{
    <span class="k">"id"</span>: <span class="s">"route-ticket"</span>,
    <span class="k">"kind"</span>: <span class="s">"choice"</span>,
    <span class="k">"question"</span>: <span class="s">"Which team should own this ticket?"</span>,
    <span class="k">"options"</span>: [
      { <span class="k">"id"</span>: <span class="s">"billing"</span> },
      { <span class="k">"id"</span>: <span class="s">"technical"</span> },
      { <span class="k">"id"</span>: <span class="s">"other"</span> }
    ]
  }],
  <span class="k">"requirements"</span>: { <span class="k">"min_top_mass"</span>: <span class="n">0.5</span> }
}</code></pre>
</div>

Shortened from [`examples/route-ticket.json`](https://github.com/Catalyst-Forge-LLC/jjm/blob/main/examples/route-ticket.json). The file also names the scope, the profile, and the evidence list.

## What you get back

- **A weight** over every option you allowed.
- **Provenance.** The method, the model, and digests of the inputs.
- **Your check.** Pass, hold, or reject against requirements you declared. A pass means those requirements were met. The next action stays with your application.

Weights are a normalized readout. A request that requires a validated scope abstains until that scope has an accepted held-out report. That status records the report. It is the same head output.

<div class="snippet">
  <div class="snippet-label">Return</div>
  <pre><code>{
  <span class="k">"results"</span>: [{
    <span class="k">"criterion_id"</span>: <span class="s">"route-ticket"</span>,
    <span class="k">"prediction"</span>: {
      <span class="k">"weights"</span>: {
        <span class="k">"billing"</span>: <span class="n">0.892</span>,
        <span class="k">"technical"</span>: <span class="n">0.005</span>,
        <span class="k">"other"</span>: <span class="n">0.103</span>
      },
      <span class="k">"semantics"</span>: <span class="s">"normalized_readout"</span>,
      <span class="k">"selected"</span>: <span class="s">"billing"</span>
    },
    <span class="k">"acceptance"</span>: { <span class="k">"status"</span>: <span class="s">"pass"</span> },
    <span class="k">"provenance"</span>: {
      <span class="k">"method"</span>: <span class="s">"mock:normalized"</span>,
      <span class="k">"model"</span>: { <span class="k">"id"</span>: <span class="s">"jjm/synthetic-no-model"</span> }
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

The head had seen the human training labels. The other two had seen none. About 64 labels per team, 512 in all, is where the head matches Nimble. The 97.1% is the earlier fit on the full test set. A refit scored 96.8%. In bulk that run was a few milliseconds per note, about 5 ms. Through the full contract, a batch of 400 took about 11 ms each, and one request at a time with nothing cached took 97 ms median on 100 requests. The other times in the table are medians.

Fit rank was one candidate. Agreement with the teacher moved with how the job was written, and agreement is not correctness.

The figures and the hardware are in the [experiment log](https://github.com/Catalyst-Forge-LLC/jjm/blob/main/docs/EXPERIMENT_LOG.md).

## Limits

- The eight-ticket walkthrough has been run on the author's machine. It downloads one small encoder and fits those eight rows.
- Routing was measured on Banking77, eight teams, human labels, on one laptop GPU. The fitted head used the training labels. About 64 labels per team is where it lines up with Nimble on that set.
- Fit rank was measured on one candidate. Agreement with the teacher is not correctness, and it moved with how the job was written.

## Try it with no model

The synthetic path needs no download. Its weights are a hash of the input, so they carry no meaning. They show the envelope.

```bash
uv sync
uv run jjm evaluate --request examples/route-ticket.json
```

The fitted-head walkthrough is in the repository README. JJM depends on [AnyJev](https://github.com/nokia-applied-research/AnyJev), Apache-2.0. JJM is not affiliated with TypeSafe AI or Jev.

Built by [Catalyst Forge LLC](https://www.catalystforge.com).
