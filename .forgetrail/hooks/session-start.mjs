#!/usr/bin/env node

/**
 * ForgeTrail Session Start Context Injector
 * Hook event: sessionStart (Cursor)
 * Points the session at appledger/. A legacy workflow_tracking.json is a conflict, not the live phase.
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
  return walkUp(startDir, (dir) => {
    const manifest = join(dir, "appledger", "manifest.yaml");
    if (existsSync(manifest)) return dir;
    return null;
  });
}

function findProjectTracking(startDir) {
  return walkUp(startDir, (dir) => {
    const candidate = join(dir, ".forgetrail", "workflow_tracking.json");
    if (existsSync(candidate)) return candidate;
    const rootStarter = join(dir, "workflow_tracking.json");
    const methodology = existsSync(join(dir, "WORKFLOW.md")) && existsSync(join(dir, "content"));
    if (existsSync(rootStarter) && !methodology) return rootStarter;
    return null;
  });
}

function isPointer(data) {
  return data?.status === "pointer" && typeof data.record === "string" && data.record.replaceAll("\\", "/").includes("appledger");
}

function main() {
  const ledgerRoot = findLedger(process.cwd());
  const trackingPath = findProjectTracking(process.cwd());
  const lines = [];

  if (ledgerRoot) {
    lines.push("=== ForgeTrail Context ===");
    lines.push("Project record: appledger/");
    lines.push("Read appledger/profiles/forgetrail.yaml for the current phase.");
    lines.push("Read the latest session record for left_off and next_steps.");
  }

  if (trackingPath) {
    try {
      const tracking = JSON.parse(readFileSync(trackingPath, "utf-8"));
      if (isPointer(tracking)) {
        if (!ledgerRoot) {
          lines.push("=== ForgeTrail Context ===");
          lines.push("workflow_tracking.json is a pointer to appledger/. Do not write decisions, sessions, or phase status there.");
        }
      } else {
        lines.push("=== Legacy tracking conflict ===");
        lines.push("workflow_tracking.json is not the system of record.");
        lines.push("Do not add decisions, sessions, or phase status to it.");
        lines.push("Run `appledger migrate preview`, then apply once, and write only in appledger/.");
      }
    } catch {
      lines.push("workflow_tracking.json could not be read. Do not recreate it. Use appledger/.");
    }
  }

  if (lines.length === 0) {
    console.log(JSON.stringify({}));
    return;
  }

  lines.push("========================================");
  console.log(JSON.stringify({ additional_context: lines.join("\n") }));
}

main();
