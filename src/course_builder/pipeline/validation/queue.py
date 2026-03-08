"""Correction Queue Step.

Converts ValidationReport into actionable CorrectionQueue.
Persists queue to JSONL for resumability.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from course_builder.domain.validation import (
    CorrectionAction,
    CorrectionQueue,
    ValidationIssue,
    ValidationReport,
    ValidationResult,
)
from course_builder.pipeline.base import PipelineContext, PipelineStep


def _generate_action_id() -> str:
    """Generate a unique action ID."""
    return f"action_{uuid.uuid4().hex[:8]}"


def _build_enhanced_prompt(issue: ValidationIssue, result: ValidationResult) -> str:
    """Build an enhanced regeneration prompt based on the issue."""
    prompts = []

    prompts.append(f"REGENERATION REQUIRED for {result.entity_type} '{result.entity_id}'")
    prompts.append(f"Path: {' > '.join(result.entity_path)}")
    prompts.append("")
    prompts.append("ISSUE FOUND:")
    prompts.append(f"  - Rule: {issue.rule_name}")
    prompts.append(f"  - Severity: {issue.severity.upper()}")
    prompts.append(f"  - Description: {issue.description}")

    if issue.field_path:
        prompts.append(f"  - Field: {issue.field_path}")

    if issue.current_value:
        prompts.append(f"  - Current value: {issue.current_value}")

    if issue.suggested_fix:
        prompts.append("")
        prompts.append(f"SUGGESTED FIX: {issue.suggested_fix}")

    if issue.source_evidence:
        prompts.append("")
        prompts.append(f"SOURCE EVIDENCE: {issue.source_evidence}")

    prompts.append("")
    prompts.append("Please regenerate this content addressing the issue above.")
    prompts.append("Ensure the new content:")
    prompts.append("  1. Fixes the identified problem")
    prompts.append("  2. Maintains educational quality")
    prompts.append("  3. Is grounded in source material")

    return "\n".join(prompts)


def _determine_action_type(severity: str) -> str:
    """Determine action type based on severity."""
    if severity == "minor":
        return "auto_fix"
    elif severity == "major":
        return "regenerate"
    elif severity == "critical":
        return "regenerate"
    return "skip"


def _severity_to_priority(severity: str) -> int:
    """Convert severity to priority (higher = more urgent)."""
    priorities = {
        "critical": 100,
        "major": 50,
        "minor": 10,
        "passed": 0,
    }
    return priorities.get(severity, 0)


def _generate_auto_fix(issue: ValidationIssue, result: ValidationResult) -> dict:
    """Generate field corrections for auto-fixable issues."""
    corrections = {}

    # Handle specific rule types that can be auto-fixed
    if issue.rule_name == "options_count_valid":
        # Can't really auto-fix option count, needs regeneration
        return {}

    if issue.rule_name == "no_duplicate_options":
        # Can't really auto-fix duplicates, needs regeneration
        return {}

    if issue.rule_name == "chunk_ids_present":
        # Minor issue, can be flagged for later
        return {}

    if issue.suggested_fix and issue.field_path:
        # If we have a suggested fix for a specific field, use it
        corrections[issue.field_path] = issue.suggested_fix

    return corrections


def _issue_to_action(
    issue: ValidationIssue,
    result: ValidationResult,
    auto_fix_minor: bool = True,
) -> CorrectionAction | None:
    """Convert a validation issue to a correction action."""
    if issue.severity == "passed":
        return None

    action_type = _determine_action_type(issue.severity)

    if action_type == "skip":
        return None

    # For minor issues with auto_fix enabled, try to generate fixes
    field_corrections = {}
    regenerate_prompt = None

    if issue.severity == "minor" and auto_fix_minor:
        field_corrections = _generate_auto_fix(issue, result)
        if not field_corrections:
            # Can't auto-fix, need to regenerate
            action_type = "regenerate"
            regenerate_prompt = _build_enhanced_prompt(issue, result)
    else:
        # Major/critical always regenerate
        action_type = "regenerate"
        regenerate_prompt = _build_enhanced_prompt(issue, result)

    # For critical issues, also flag for manual review
    if issue.severity == "critical":
        action_type = "manual_review"

    return CorrectionAction(
        action_id=_generate_action_id(),
        entity_type=result.entity_type,
        entity_id=result.entity_id,
        entity_path=result.entity_path,
        action_type=action_type,
        field_corrections=field_corrections,
        regenerate_prompt=regenerate_prompt,
        priority=_severity_to_priority(issue.severity),
        status="pending",
        created_at=datetime.now(),
    )


class CorrectionQueueStep(PipelineStep):
    """Converts ValidationReport into actionable CorrectionQueue.

    For each issue in the report:
    - MINOR with auto_fix → Create auto_fix action
    - MAJOR → Create regenerate action with enhanced prompt
    - CRITICAL → Create regenerate action + flag for manual review

    Persists queue to JSONL file for resumability.
    """

    def __init__(
        self,
        corrections_dir: str = "corrections",
        auto_fix_minor: bool = True,
    ):
        self.corrections_dir = corrections_dir
        self.auto_fix_minor = auto_fix_minor

    def run(self, context: PipelineContext) -> PipelineContext:
        report: ValidationReport = context.get("validation_report")

        if not report:
            print("No validation report found, skipping correction queue generation")
            return context

        cert_name = report.certification_name
        print(f"Generating correction queue for '{cert_name}'...")

        # Create output directory
        output_dir = Path(self.corrections_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize queue
        queue = CorrectionQueue(
            certification_name=cert_name,
            source_version=report.skeleton_version,
            target_version=report.skeleton_version + 1,
            created_at=datetime.now(),
            actions=[],
        )

        # Process each validation result
        actions_by_type = {"auto_fix": 0, "regenerate": 0, "manual_review": 0}
        processed_issues = set()

        for result in report.results:
            if result.overall_status == "passed":
                continue

            for issue in result.issues:
                if issue.severity == "passed":
                    continue

                # Avoid duplicate actions for the same entity/issue
                issue_key = (result.entity_id, issue.rule_name)
                if issue_key in processed_issues:
                    continue
                processed_issues.add(issue_key)

                action = _issue_to_action(issue, result, self.auto_fix_minor)
                if action:
                    queue.actions.append(action)
                    actions_by_type[action.action_type] = actions_by_type.get(action.action_type, 0) + 1

        # Sort actions by priority (highest first)
        queue.actions.sort(key=lambda a: -a.priority)

        # Save queue to JSONL file
        cert_slug = cert_name.replace(" ", "_").replace("/", "_")
        queue_file = output_dir / f"{cert_slug}_corrections_v{report.skeleton_version}_to_v{report.skeleton_version + 1}.jsonl"
        queue.save(queue_file)

        # Save state file
        state_file = output_dir / f"{cert_slug}_state.json"

        state = {
            "certification_name": cert_name,
            "current_version": report.skeleton_version,
            "last_validation": report.validated_at.isoformat(),
            "pending_corrections": queue.pending_count,
            "applied_corrections": queue.applied_count,
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        print(f"  Correction queue generated:")
        print(f"    - Total actions: {len(queue.actions)}")
        print(f"    - Auto-fix: {actions_by_type['auto_fix']}")
        print(f"    - Regenerate: {actions_by_type['regenerate']}")
        print(f"    - Manual review: {actions_by_type['manual_review']}")
        print(f"  Queue saved to: {queue_file}")

        context["correction_queue"] = queue
        context["correction_queue_path"] = str(queue_file)
        return context
