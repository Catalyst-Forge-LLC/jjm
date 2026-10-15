---
title: Just Judgment Machine
description: An open-source local judgment runtime. Alpha. Your application sends the state and the options, and keeps the decision.
order: 1
---

## PE:TL;DR

Plain English. You have notes, and a set of buckets you are willing to use. Show JJM a few notes that are already in the right bucket. After that it sorts new notes on your own computer, quickly, and writes down how it got there. The next step stays yours.

Your application sends a state and the options it allows. JJM returns a weight for each option, and a record of what produced it. Your application decides what happens next.

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

## What you get back

- **A weight** over every option you allowed.
- **Provenance.** The method, the model, and digests of the inputs.
- **Your check.** Pass, hold, or reject against requirements you declared. A pass means those requirements were met. The next action stays with your application.

Weights are a normalized readout. A request that requires a validated scope abstains until that scope has an accepted held-out report. That status records the report. It is the same head output.

## What has been tried

On Banking77, mapped to eight teams, a frozen embedding model plus a small head fitted on the human labels scored 97.1% on the 3,080 test messages, at a few milliseconds each. About 64 labels per team is where that head matches Nimble, a 9B decision model, on the same set. On fit rank for one candidate, the chat-wrapped reranker encoding tracked the teacher's ranks.

The figures and the hardware are in the [experiment log](https://github.com/Catalyst-Forge-LLC/jjm/blob/main/docs/EXPERIMENT_LOG.md).

## Limits

- The eight-ticket walkthrough has been run on the author's machine. It downloads one small encoder and fits those eight rows.
- Routing was measured on Banking77, eight teams, human labels, on one laptop GPU. The fitted head used the training labels. About 64 labels per team is where it lines up with Nimble on that set.
- Fit rank was measured on one candidate. The chat-wrapped reranker encoding is the input that tracked the teacher's ranks.

## Try it with no model

The synthetic path needs no download. Its weights are a hash of the input, so they carry no meaning. They show the envelope.

```bash
uv sync
uv run jjm evaluate --request examples/route-ticket.json
```

The fitted-head walkthrough is in the repository README. JJM depends on [AnyJev](https://github.com/nokia-applied-research/AnyJev), Apache-2.0. JJM is not affiliated with TypeSafe AI or Jev.

Built by [Catalyst Forge LLC](https://www.catalystforge.com).
