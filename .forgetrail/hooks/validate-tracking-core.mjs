/**
 * Structural validation for .forgetrail/workflow_tracking.json (MCP full + Lite shapes).
 * Standalone zero-dependency module for ForgeTrail hooks and MCP.
 */

const VALID_MCP_PHASES = [
  "1-architecture",
  "2-scaffolding",
  "3-stabilization",
  "4-feature-iteration",
  "5-refactoring",
  "6-strategic-review",
  "7-hardening",
];

const VALID_LITE_PHASES = ["1", "2", "3", "4", "5", "6", "7"];

const VALID_GOTCHA_CATEGORIES = new Set([
  "environment",
  "integration",
  "data-model",
  "ui",
  "performance",
  "ai-output",
  "security",
  "tooling",
  "other",
]);

const VALID_DECISION_STATUS = new Set(["active", "superseded", "revisit"]);

const VALID_PHASE_STATUS = new Set([
  "not_started",
  "pending",
  "in_progress",
  "completed",
  "revisiting",
]);

function isLiteSchema(data) {
  return data && data.schemaVersion === "lite-1";
}

const LEGACY_TRACKING_ISSUE =
  "Legacy writable tracking is not the system of record. Do not add decisions, sessions, or phase status to this file. Run `appledger migrate preview`, then apply once, and write only in appledger/.";

const POINTER_WARNING =
  "This file is a pointer to appledger/. Do not add decisions, sessions, or phase status here.";

function isPointer(data) {
  return data.status === "pointer" && typeof data.record === "string" && data.record.replaceAll("\\", "/").includes("appledger");
}

function isLegacyTracking(data) {
  return (
    "currentPhase" in data ||
    "phases" in data ||
    "decisions" in data ||
    "sessions" in data ||
    "schemaVersion" in data
  );
}

export function validateTrackingData(data) {
  const issues = [];
  const warnings = [];

  if (!data || typeof data !== "object") {
    issues.push("Tracking data is empty or not an object.");
    return { issues, warnings };
  }

  if (isPointer(data)) {
    warnings.push(POINTER_WARNING);
    return { issues, warnings };
  }

  if (isLegacyTracking(data)) {
    issues.push(LEGACY_TRACKING_ISSUE);
  }

  const project = data.project;
  if (!project || typeof project !== "object") {
    issues.push("Missing top-level project object.");
  } else if (!project.name || typeof project.name !== "string" || !project.name.trim()) {
    issues.push("project.name is missing or empty.");
  }

  if (!data.currentPhase) {
    issues.push("Missing currentPhase.");
  }

  const lite = isLiteSchema(data);
  const currentPhase = data.currentPhase;

  if (currentPhase) {
    if (lite) {
      const phaseStr = String(currentPhase);
      if (!VALID_LITE_PHASES.includes(phaseStr)) {
        issues.push(
          `Lite schema currentPhase "${phaseStr}" is not a valid phase number (1–7).`
        );
      }
    } else if (
      typeof currentPhase === "string" &&
      !VALID_MCP_PHASES.includes(currentPhase)
    ) {
      issues.push(
        `currentPhase "${currentPhase}" is not a recognized MCP phase id (e.g. "1-architecture").`
      );
    }
  }

  const phases = data.phases;
  if (!phases || typeof phases !== "object") {
    issues.push("Missing phases object.");
    return { issues, warnings };
  }

  const phaseKeys = Object.keys(phases);
  if (phaseKeys.length === 0) {
    issues.push("No phases defined in tracking.");
  }

  if (!lite && currentPhase && typeof currentPhase === "string" && !phaseKeys.includes(currentPhase)) {
    issues.push(`currentPhase "${currentPhase}" is not a key in phases.`);
  }

  for (const [key, rawPhase] of Object.entries(phases)) {
    if (!rawPhase || typeof rawPhase !== "object") {
      issues.push(`phases["${key}"] is not an object.`);
      continue;
    }
    const phase = rawPhase;

    if (phase.status && !VALID_PHASE_STATUS.has(String(phase.status))) {
      warnings.push(
        `phases["${key}"].status "${phase.status}" is not a standard value (not_started | in_progress | completed | revisiting).`
      );
    }

    if (lite) {
      if (phase.exitCriteria !== undefined && (typeof phase.exitCriteria !== "object" || phase.exitCriteria === null)) {
        issues.push(`phases["${key}"].exitCriteria should be an object of boolean flags (Lite schema).`);
      }
    } else {
      if (phase.exitCriteriaMet !== undefined && !Array.isArray(phase.exitCriteriaMet)) {
        issues.push(`phases["${key}"].exitCriteriaMet should be an array.`);
      }
      if (phase.exitCriteriaRemaining !== undefined && !Array.isArray(phase.exitCriteriaRemaining)) {
        issues.push(`phases["${key}"].exitCriteriaRemaining should be an array.`);
      }
      const met = phase.exitCriteriaMet ?? [];
      const remaining = phase.exitCriteriaRemaining ?? [];
      if (Array.isArray(met) && Array.isArray(remaining)) {
        const overlap = met.filter((item) => remaining.includes(item));
        if (overlap.length > 0) {
          warnings.push(
            `phases["${key}"] has exit criteria in both Met and Remaining: ${overlap.slice(0, 3).join("; ")}${overlap.length > 3 ? "…" : ""}`
          );
        }
      }
    }

    const subagentRuns = phase.subagentRuns;
    if (subagentRuns !== undefined && !Array.isArray(subagentRuns)) {
      issues.push(`phases["${key}"].subagentRuns should be an array when present.`);
    }
  }

  const decisions = data.decisions;
  if (decisions !== undefined) {
    if (!Array.isArray(decisions)) {
      issues.push("decisions should be an array.");
    } else {
      const seenIds = new Set();
      for (let i = 0; i < decisions.length; i++) {
        const d = decisions[i];
        if (!d || typeof d !== "object") {
          issues.push(`decisions[${i}] is not an object.`);
          continue;
        }
        if (!d.decision || typeof d.decision !== "string") {
          issues.push(`decisions[${i}] missing decision text.`);
        }
        if (!d.rationale || typeof d.rationale !== "string") {
          warnings.push(`decisions[${i}] missing rationale (recommended for Phase 1 handoff).`);
        }
        if (d.id && typeof d.id === "string") {
          if (seenIds.has(d.id)) {
            issues.push(`Duplicate decisions id "${d.id}".`);
          }
          seenIds.add(d.id);
        }
        if (d.status && !VALID_DECISION_STATUS.has(String(d.status))) {
          warnings.push(`decisions[${i}].status "${d.status}" is not standard (active | superseded | revisit).`);
        }
        if (
          d.alternatives_considered !== undefined &&
          !Array.isArray(d.alternatives_considered)
        ) {
          warnings.push(`decisions[${i}].alternatives_considered should be an array when present.`);
        }
      }
    }
  }

  const gotchas = data.gotchas;
  if (gotchas !== undefined) {
    if (!Array.isArray(gotchas)) {
      issues.push("gotchas should be an array.");
    } else {
      for (let i = 0; i < gotchas.length; i++) {
        const g = gotchas[i];
        if (!g || typeof g !== "object") {
          issues.push(`gotchas[${i}] is not an object.`);
          continue;
        }
        if (!g.issue || typeof g.issue !== "string") {
          warnings.push(`gotchas[${i}] missing issue description.`);
        }
        if (!g.resolution || typeof g.resolution !== "string") {
          warnings.push(`gotchas[${i}] missing resolution.`);
        }
        if (g.category && !VALID_GOTCHA_CATEGORIES.has(String(g.category))) {
          warnings.push(
            `gotchas[${i}].category "${g.category}" is not a standard category (environment, integration, …).`
          );
        }
      }
    }
  }

  const sessions = data.sessions;
  if (sessions !== undefined && !Array.isArray(sessions)) {
    issues.push("sessions should be an array when present.");
  }

  return { issues, warnings };
}

export function formatValidationResult(result) {
  const { issues, warnings } = result;

  if (issues.length === 0 && warnings.length === 0) {
    return "No workflow_tracking.json is required. Project state lives in appledger/.";
  }

  if (issues.length === 0 && warnings.every((item) => item.includes("pointer"))) {
    return "Pointer accepted. Project state lives in appledger/.\n\n" + warnings.map((item) => `- ${item}`).join("\n");
  }

  const parts = [];
  if (issues.length > 0) {
    parts.push("Issues (fix these):\n" + issues.map((i) => `- ${i}`).join("\n"));
  }
  if (warnings.length > 0) {
    parts.push("Warnings:\n" + warnings.map((w) => `- ${w}`).join("\n"));
  }

  return parts.join("\n\n");
}
