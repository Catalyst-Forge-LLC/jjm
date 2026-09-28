#!/usr/bin/env node

/**
 * ForgeTrail Tracking Validator Hook
 * Hook event: afterFileEdit (Cursor) / PostToolUse (Claude Code)
 * Automatically validates .forgetrail/workflow_tracking.json upon edit.
 */

import { readFileSync, existsSync } from "node:fs";
import { validateTrackingData, formatValidationResult } from "./validate-tracking-core.mjs";

function readStdin() {
  try {
    const raw = readFileSync(0, "utf-8");
    if (raw.trim()) return JSON.parse(raw);
  } catch {}
  return {};
}

function main() {
  const payload = readStdin();
  const filePath =
    payload.path ||
    payload.file_path ||
    payload.target_file ||
    payload.input?.path ||
    payload.tool_input?.path ||
    "";

  if (!filePath.endsWith("workflow_tracking.json")) {
    console.log(JSON.stringify({}));
    return;
  }

  if (!existsSync(filePath)) {
    console.log(JSON.stringify({}));
    return;
  }

  try {
    const content = JSON.parse(readFileSync(filePath, "utf-8"));
    const result = validateTrackingData(content);

    if (result.issues.length > 0 || result.warnings.length > 0) {
      const formatted = formatValidationResult(result);
      console.log(
        JSON.stringify({
          additional_context: `=== ForgeTrail Tracking Validation Notice ===\n${formatted}\n=============================================`,
        })
      );
      return;
    }
  } catch (err) {
    console.log(
      JSON.stringify({
        additional_context: `ForgeTrail Tracking Syntax Error: Failed to parse ${filePath} as valid JSON: ${err.message}`,
      })
    );
    return;
  }

  console.log(JSON.stringify({}));
}

main();
