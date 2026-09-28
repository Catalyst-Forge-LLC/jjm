---
format_version: 0.1.0
id: decision-stateless-no-accounts
kind: decision
title: v1 has no accounts or database
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
  choice: Each judgment is a stateless request. v1 has no PocketBase, login, or hosted inference endpoint.
  rationale: The caller already owns the state. Persistence would turn JJM into an application platform.
  alternatives:
    - PocketBase plus accounts
    - A hosted api.jjm.dev
  authority: Draft brief, pending owner lock
---

# Stateless v1

Proposed in `docs/PHASE_1_BRIEF.md` as D4. Not locked.
