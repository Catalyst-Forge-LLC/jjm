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

<p class="kicker">Alpha · Apache-2.0 · local only · not a package yet</p>

## What you send

A JSON request with a snapshot and explicit criteria. A choice has two to 26 options you define. An assertion has three outcomes: the supplied evidence supports the claim, contradicts it, or does not settle it.

## What you get back

- **A weight** over every option you allowed.
- **Provenance.** The method, the model, and digests of the inputs.
- **Your check.** Pass, hold, or reject against requirements you declared. A pass means those requirements were met. The next action stays with your application.

Weights are a normalized readout. A request that requires a validated scope abstains until that scope has an accepted held-out report. That status records the report. It is the same head output.

## What has been tried

On a public routing set, a frozen embedding model plus a small head fitted on human labels was more accurate, and much faster, than asking a 9B decision model that had no labels. The fair point on that set is about 64 labels per team before the head matches the 9B model. Fit ranking on one person's jobs did not beat always answering the most common rank, except when the encoder saw a dedicated prompt. Agreement with another model is agreement, not correctness.

The figures, the hardware, and the misses are in the [experiment log](https://github.com/Catalyst-Forge-LLC/jjm/blob/main/docs/EXPERIMENT_LOG.md).

## Limits

- Local only. This site does not run a model, and there is no account.
- The eight-ticket walkthrough downloads one small encoder and has been run on the author's machine.
- The package name in the repository is `jjm-engine`. It is not on PyPI. Clone the repository and use uv.
- The production fit notes are one candidate. That export is not in the repository.

## Try it with no model

The synthetic path needs no download. Its weights are a hash of the input, so they carry no meaning. They show the envelope.

```bash
uv sync
uv run jjm evaluate --request examples/route-ticket.json
```

The fitted-head walkthrough is in the repository README. JJM depends on [AnyJev](https://github.com/nokia-applied-research/AnyJev), Apache-2.0. JJM is not affiliated with TypeSafe AI or Jev.

Built by [Catalyst Forge LLC](https://www.catalystforge.com).
