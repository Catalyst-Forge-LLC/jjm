# ForgeTrail Hooks

Enforces ForgeTrail rules and guards at the host tool level, rather than relying only on prompt requests.

## What is in this directory

- `guard-shell.mjs`: runs pre-commit verification (`pnpm run verify`), prevents npm/yarn when `pnpm-lock.yaml` is present, allows `ingotvault` (asks before `--force-with-lease`), asks before any `git push`, and prompts on destructive git or filesystem operations.
- `guard-edit.mjs`: guards `.env*` secrets files and `specs/completed/**` or `specs/canonical/**` records against accidental mutation.
- `session-start.mjs`: on session start, points the session at `appledger/`. A writable `workflow_tracking.json` is reported as a legacy conflict, not the live phase.
- `validate-tracking.mjs`: after editing `workflow_tracking.json`, runs structural validation. A pointer is accepted. A legacy writable document is a conflict.
- `validate-tracking-core.mjs`: standalone zero-dependency tracking validator.
- `session-stop.mjs`: on session stop, reminds the agent to update the appledger session record. It does not ask for a new JSON session entry.
- `cursor-hooks.json`: standard Cursor hooks configuration.
- `claude-settings-hooks.json`: standard Claude Code configuration fragment.

## Host setup

### Cursor

Copy or link `cursor-hooks.json` to `.cursor/hooks.json` in your repository root. The hook scripts reside in `.forgetrail/hooks/`.

### Claude Code

Add the contents of `claude-settings-hooks.json` to `.claude/settings.json` in your project root or user configuration.

## Verification

Run any script with test input via stdin:

```bash
echo '{"command": "npm install lodash"}' | node .forgetrail/hooks/guard-shell.mjs
# Output: {"permission":"deny", ...} (when pnpm-lock.yaml is present)

echo '{"command": "git push"}' | node .forgetrail/hooks/guard-shell.mjs
# Output: {"permission":"ask", ...}
```
