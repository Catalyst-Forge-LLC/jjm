---
format_version: 0.1.0
id: decision-v1-local-runtime
kind: decision
title: v1 is a local Python judgment path
record_status: active
created_at: 2026-09-28T22:20:00Z
updated_at: 2026-09-28T22:20:00Z
recorded_by:
  id: cursor-agent
  type: agent
visibility: internal
relations:
  - type: affects
    target: app-36f50a0b-90a2-4e99-a024-197c6cba62c8
claims: []
data:
  status: proposed
  choice: Keep AnyJev's Python inference, add a strict local CLI, and ship synthetic choice plus one Transformers route before any site or hosted API.
  rationale: The spec pack and the owner both treat JJM as a local replacement for a hosted judgment call. A website or a JavaScript rewrite would not show that.
  alternatives:
    - Rewrite inference in JavaScript
    - Start with the website
    - Call a hosted provider and wrap it
  authority: Draft brief, pending owner lock
---

# v1 runtime

Proposed in `docs/PHASE_1_BRIEF.md` as D1 and D2. Not locked.
