#!/usr/bin/env node

/**
 * ForgeTrail Session Stop Check
 * Hook event: stop (Cursor)
 * Reminds the agent to update the AppLedger session record when work happened after it was last written.
 */

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

function walkUp(startDir, found) {
  let curr = resolve(startDir || process.cwd());
  for (let i = 0; i < 6; i++) {
    const hit = found(curr);
    if (hit) return hit;
    const parent = resolve(curr, "..");
    if (parent === curr) break;
    curr = parent;
  }
  return null;
}

function findLedger(startDir) {
  return walkUp(startDir, (dir) => (existsSync(join(dir, "appledger", "manifest.yaml")) ? dir : null));
}

function findProjectTracking(startDir) {
  return walkUp(startDir, (dir) => {
    const candidate = join(dir, ".forgetrail", "workflow_tracking.json");
    if (existsSync(candidate)) return candidate;
    return null;
  });
}

function isPointer(data) {
  return data?.status === "pointer" && typeof data.record === "string" && data.record.replaceAll("\\", "/").includes("appledger");
}

function readHookInput() {
  try {
    const raw = readFileSync(0, "utf-8");
    return raw.trim() ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function latestSessionMs(root) {
  const dir = join(root, "appledger", "records", "session");
  if (!existsSync(dir)) return 0;
  let latest = 0;
  for (const name of readdirSync(dir)) {
    if (!name.endsWith(".md")) continue;
    latest = Math.max(latest, statSync(join(dir, name)).mtimeMs);
  }
  return latest;
}

function git(root, args) {
  return execFileSync("git", args, { cwd: root, encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] });
}

const NOT_PROJECT_WORK = ["appledger/", ".forgetrail/", ".cursor/", ".claude/"];

function outsideLedger(path) {
  const p = path.replaceAll("\\", "/");
  return p !== "" && !NOT_PROJECT_WORK.some((prefix) => p.startsWith(prefix));
}

/** True when project files changed after the newest session record, or when that cannot be told. Ledger and host-config paths do not count. */
function workSinceSession(root) {
  const sessionMs = latestSessionMs(root);
  if (sessionMs === 0) return true;
  try {
    const since = `@${Math.floor(sessionMs / 1000)}`;
    const hasCommits = (() => {
      try {
        git(root, ["rev-parse", "--verify", "--quiet", "HEAD"]);
        return true;
      } catch {
        return false;
      }
    })();
    if (hasCommits) {
      const committed = git(root, ["log", `--since=${since}`, "--name-only", "--format="]).split("\n").filter(outsideLedger);
      if (committed.length > 0) return true;
    }
    const pending = git(root, ["status", "--porcelain", "--untracked-files=all"])
      .split("\n")
      .map((line) => line.slice(3).split(" -> ").pop()?.replace(/^"|"$/g, "") ?? "")
      .filter(outsideLedger);
    return pending.some((path) => {
      try {
        return statSync(join(root, path)).mtimeMs > sessionMs;
      } catch {
        return true;
      }
    });
  } catch {
    return true;
  }
}

function main() {
  const input = readHookInput();
  if ((input.loop_count ?? 0) > 0 || (input.status && input.status !== "completed")) {
    console.log(JSON.stringify({}));
    return;
  }

  const ledgerRoot = findLedger(process.cwd());
  const trackingPath = findProjectTracking(process.cwd());
  let legacy = false;
  if (trackingPath) {
    try {
      legacy = !isPointer(JSON.parse(readFileSync(trackingPath, "utf-8")));
    } catch {
      legacy = true;
    }
  }

  if (legacy) {
    console.log(
      JSON.stringify({
        followup_message:
          "ForgeTrail reminder: workflow_tracking.json is a legacy file. Do not append sessions to it. Run `appledger migrate preview`, then apply, and record this session in appledger/.",
      })
    );
    return;
  }

  if (ledgerRoot && workSinceSession(ledgerRoot)) {
    console.log(
      JSON.stringify({
        followup_message:
          "ForgeTrail reminder: files changed after the latest appledger session record. Update that record (what was accomplished, left_off, next_steps) before ending.",
      })
    );
    return;
  }

  console.log(JSON.stringify({}));
}

main();
