---
format_version: 0.1.0
id: decision-filepress-landing
kind: decision
title: JJM.dev uses the FilePress landing layout
record_status: active
created_at: 2026-09-28T20:05:00Z
updated_at: 2026-09-28T20:05:00Z
recorded_by:
  id: cursor-agent
  type: agent
visibility: internal
relations:
  - type: affects
    target: app-36f50a0b-90a2-4e99-a024-197c6cba62c8
claims:
  - id: landing-layout
    statement: The owner directed JJM.dev to follow the FilePress landing design spec.
    basis: declared
    status: supported
    evidence_refs: []
    scope:
      repository_id: repo-home
      limitations:
        - The spec pack still names a SvelteKit site. This decision is the later instruction.
data:
  status: accepted
  choice: Use the FilePress landing layout from catalyst-forge/docs/filepress-landing-design-spec.md. Keep JJM's wordmark, colors, and typefaces. Adopt that spec's page shape, header, and chrome.
  rationale: The owner named that file as the site style guide. The guide says each product keeps its own accent, logo, and type, and shares the Finetuna landing hierarchy.
  alternatives:
    - Custom SvelteKit landing as written in the spec pack
  authority: Project owner, 2026-09-28
---

# Site style

JJM.dev follows the FilePress landing design spec. The segmented-J wordmark stays. Colors and type are JJM's, not Finetuna's teal and orange.

The home page uses a short eyebrow, a short display heading, a larger opening paragraph, smaller body text, uppercase section labels, a filled pill and a ghost pill, a washed page background, and a header that is a different surface from the page. Dark mode repeats designed tokens. It does not invert the light theme automatically.
