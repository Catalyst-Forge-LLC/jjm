#!/usr/bin/env node

/**
 * ForgeTrail Shell Guard
 * Hook event: beforeShellExecution (Cursor) / PreToolUse (Claude Code)
 * Enforces Lite §0, verify-before-commit, pnpm consistency, and git push guards.
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { execSync } from "node:child_process";

function readStdin() {
  try {
    const raw = readFileSync(0, "utf-8");
    if (raw.trim()) return JSON.parse(raw);
  } catch {}
  return {};
}

function hasPnpmLock(cwd) {
  let curr = resolve(cwd || process.cwd());
  for (let i = 0; i < 6; i++) {
    if (existsSync(join(curr, "pnpm-lock.yaml"))) return true;
    const parent = resolve(curr, "..");
    if (parent === curr) break;
    curr = parent;
  }
  return false;
}

function hasVerifyScript(cwd) {
  try {
    const pkgPath = join(cwd || process.cwd(), "package.json");
    if (existsSync(pkgPath)) {
      const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
      return Boolean(pkg.scripts && pkg.scripts.verify);
    }
  } catch {}
  return false;
}

function main() {
  const input = readStdin();
  const command = (input.command || input.tool_input?.command || "").trim();
  const cwd = input.cwd || process.cwd();

  if (!command) {
    console.log(JSON.stringify({ permission: "allow" }));
    return;
  }

  // 1. Verify-before-commit: run whenever package.json has a verify script (unless skipped via FORGETRAIL_SKIP_VERIFY=1 or --no-verify)
  if (/\bgit\s+commit\b/.test(command)) {
    if (
      process.env.FORGETRAIL_SKIP_VERIFY !== "1" &&
      !/\s+--no-verify\b/.test(command) &&
      hasVerifyScript(cwd)
    ) {
      try {
        execSync("pnpm run verify", { cwd, stdio: "pipe" });
      } catch (err) {
        console.log(
          JSON.stringify({
            permission: "deny",
            user_message:
              "ForgeTrail Verification: 'pnpm run verify' failed. Code checks must pass before committing.",
            agent_message:
              "Pre-commit verification failed. Run pnpm verify and fix errors before committing.",
          })
        );
        return;
      }
    }
  }

  // Local vault updates go through the ingotvault command. Force still asks.
  if (/\bingotvault\b/.test(command)) {
    if (/\s--force-with-lease\b/.test(command)) {
      console.log(
        JSON.stringify({
          permission: "ask",
          user_message:
            "ForgeTrail Safety: ingotvault --force-with-lease rewrites mirror history. Confirm to proceed.",
          agent_message:
            "ingotvault --force-with-lease requires explicit user confirmation.",
        })
      );
      return;
    }
    console.log(JSON.stringify({ permission: "allow" }));
    return;
  }

  // 2. Check for git push (Do not push to upstream remote unless explicitly requested by user)
  if (/\bgit\s+push\b/.test(command)) {
    if (/\s+(-f|--force|--force-with-lease)\b/.test(command)) {
      console.log(
        JSON.stringify({
          permission: "ask",
          user_message:
            "ForgeTrail Safety: Force-pushing to remote can overwrite git history. Please confirm if you wish to proceed.",
          agent_message:
            "Force-push requires explicit user confirmation.",
        })
      );
      return;
    }

    console.log(
      JSON.stringify({
        permission: "ask",
        user_message:
          "ForgeTrail Safety: git push requires explicit user approval. Local vault updates use ingotvault --repo .",
        agent_message:
          "git push requires explicit user approval. Mirror this repo with ingotvault --repo . instead of git push backup.",
      })
    );
    return;
  }

  // 3. Check for npm/yarn when pnpm-lock exists (Lite §0 Rule 3)
  if (/\b(npm\s+(i|install|add)|yarn(\s+add|\s+install)?)\b/.test(command)) {
    if (hasPnpmLock(cwd)) {
      console.log(
        JSON.stringify({
          permission: "deny",
          user_message:
            "ForgeTrail Lite §0: This repository uses pnpm (pnpm-lock.yaml detected). Do not run npm or yarn install.",
          agent_message:
            "Package manager is locked to pnpm. Run pnpm instead.",
        })
      );
      return;
    }
  }

  // 4. Check for destructive git operations
  if (/\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f)/.test(command)) {
    console.log(
      JSON.stringify({
        permission: "ask",
        user_message:
          "ForgeTrail Safety: Destructive git operation (" +
          command +
          ") will permanently discard uncommitted changes. Confirm?",
        agent_message: "Destructive git operation requires user approval.",
      })
    );
    return;
  }

  // 5. Check for dangerous rm -rf outside build/temp
  if (/\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/.test(command)) {
    const isSafe =
      /\b(temp|build|\.svelte-kit|\.turbo|dist|\.wrangler|node_modules)[/\\]?/.test(
        command
      );
    if (!isSafe) {
      console.log(
        JSON.stringify({
          permission: "ask",
          user_message:
            "ForgeTrail Safety: Recursive file deletion outside temporary/build directories. Confirm?",
          agent_message:
            "Dangerous recursive deletion requires user approval.",
        })
      );
      return;
    }
  }

  console.log(JSON.stringify({ permission: "allow" }));
}

main();
