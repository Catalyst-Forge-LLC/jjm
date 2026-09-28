#!/usr/bin/env node

/**
 * ForgeTrail Edit Guard
 * Hook event: preToolUse (Cursor) / PreToolUse (Claude Code)
 * Guards against accidental mutation of secrets (.env*) and completed/canonical specs.
 */

import { readFileSync } from "node:fs";

function readStdin() {
  try {
    const raw = readFileSync(0, "utf-8");
    if (raw.trim()) return JSON.parse(raw);
  } catch {}
  return {};
}

function extractTargetPaths(tool, input) {
  if (!input) return [];
  const paths = [];
  if (typeof input.path === "string") paths.push(input.path);
  if (typeof input.file_path === "string") paths.push(input.file_path);
  if (typeof input.target_file === "string") paths.push(input.target_file);
  return paths;
}

function main() {
  const payload = readStdin();
  const tool = payload.tool || payload.tool_name || "";
  const input = payload.input || payload.tool_input || {};

  // Only check write/edit tools
  const isEditTool =
    /^(Write|Edit|StrReplace|WriteFile|file_editor|edit_file)$/i.test(tool);

  if (!isEditTool) {
    console.log(JSON.stringify({ permission: "allow" }));
    return;
  }

  const paths = extractTargetPaths(tool, input);

  for (const path of paths) {
    const normalized = path.replace(/\\/g, "/");

    // 1. Guard environment secrets (.env, .env.local, etc. but allow .env.example)
    if (/(^|\/)\.env(\.[a-zA-Z0-9_-]+)?$/.test(normalized)) {
      if (!normalized.endsWith(".env.example")) {
        console.log(
          JSON.stringify({
            permission: "ask",
            user_message:
              "ForgeTrail Safety: Modifying environment secrets file (" +
              path +
              "). Please confirm before proceeding.",
            agent_message:
              "Editing environment secrets file requires user confirmation.",
          })
        );
        return;
      }
    }

    // 2. Guard completed and canonical delivery specs
    if (/(^|\/)specs\/(completed|canonical)\//.test(normalized)) {
      console.log(
        JSON.stringify({
          permission: "ask",
          user_message:
            "ForgeTrail Safety: Completed delivery specs (" +
            path +
            ") are historical records. Modify only if intentionally updating links or summaries.",
          agent_message:
            "Editing completed/canonical spec requires user confirmation.",
        })
      );
      return;
    }
  }

  console.log(JSON.stringify({ permission: "allow" }));
}

main();
