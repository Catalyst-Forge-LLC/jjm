#!/usr/bin/env node

/**
 * ForgeTrail Session Stop Check
 * Hook event: stop (Cursor)
 * Reminds the agent to update the AppLedger session, not a tracking JSON file.
 */

import { existsSync, readFileSync } from "node:fs";
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

function main() {
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

  if (ledgerRoot) {
    console.log(
      JSON.stringify({
        followup_message:
          "ForgeTrail reminder: update the appledger session record (what was accomplished, left_off, next_steps) before ending. Do not write workflow_tracking.json.",
      })
    );
    return;
  }

  console.log(JSON.stringify({}));
}

main();
