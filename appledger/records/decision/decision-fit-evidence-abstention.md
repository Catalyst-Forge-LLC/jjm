---
format_version: 0.1.0
id: decision-fit-evidence-abstention
kind: decision
title: Fit trial abstains on missing candidate evidence before the model runs
record_status: active
created_at: 2026-09-29T03:15:00Z
updated_at: 2026-09-29T03:15:00Z
recorded_by:
  id: cursor-agent
  type: agent
visibility: internal
relations:
  - type: affects
    target: app-36f50a0b-90a2-4e99-a024-197c6cba62c8
  - type: supersedes
    target: decision-fit-rank-shadow
claims: []
data:
  status: accepted
  choice: The fit criterion declares candidate evidence required. A job with no resume, profile, supplementary context, or work preferences abstains with MISSING_REQUIRED_EVIDENCE before any model runs. This replaces the not_observable assertion in brief section 2a and in decision-fit-rank-shadow. The five level descriptions are frozen as criterion version 1.
  rationale: The check is deterministic and free, and the model is never asked to guess a middle rank from the listing alone.
  alternatives:
    - A model-judged assertion that returns not_observable, as the brief first described
  authority: "Owner, 2026-09-28: yes on freezing the five fit; the evidence change is good"
---

# Fit evidence abstention

Amends brief section 2a. Everything else in `decision-fit-rank-shadow` still stands.
