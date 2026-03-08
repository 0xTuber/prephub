"""Validation and correction domain models.

This module contains models for validation and correction:
- ValidationIssue: A single validation issue
- ValidationResult: Validation result for an entity
- ValidationReport: Complete validation report
- CorrectionAction, CorrectionQueue: Correction management
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ValidationIssue(BaseModel):
    """A single validation issue found."""

    issue_id: str
    severity: str  # "passed", "minor", "major", "critical"
    rule_name: str  # e.g., "content_not_empty", "answer_grounded"
    description: str
    field_path: str  # e.g., "content", "options[2]", "explanation"
    current_value: Any | None = None
    suggested_fix: str | None = None
    source_evidence: str | None = None  # Retrieved chunk supporting/refuting


class ValidationResult(BaseModel):
    """Validation result for a single entity."""

    entity_type: str  # "skeleton", "module", "topic", "subtopic", "lab", "capsule", "item"
    entity_id: str
    entity_path: list[str]  # ["module_01", "topic_02"] for navigation
    overall_status: str  # worst severity found
    issues: list[ValidationIssue] = []
    validated_at: datetime
    validator_model: str | None = None  # Which LLM reviewed it


class ValidationReport(BaseModel):
    """Complete validation report for a skeleton."""

    certification_name: str
    skeleton_version: int
    validated_at: datetime
    total_entities: int
    passed_count: int
    minor_count: int
    major_count: int
    critical_count: int
    results: list[ValidationResult] = []

    def get_entities_by_severity(self, severity: str) -> list[ValidationResult]:
        """Get all validation results with a specific severity."""
        return [r for r in self.results if r.overall_status == severity]


class CorrectionAction(BaseModel):
    """A correction to be applied."""

    action_id: str
    entity_type: str
    entity_id: str
    entity_path: list[str]
    action_type: str  # "auto_fix", "regenerate", "manual_review"
    field_corrections: dict[str, Any] = {}  # field -> new value (for auto_fix)
    regenerate_prompt: str | None = None  # Enhanced prompt for regeneration
    priority: int = 0  # Higher = more urgent
    status: str = "pending"  # "pending", "in_progress", "applied", "failed", "skipped"
    created_at: datetime
    applied_at: datetime | None = None
    error: str | None = None


class CorrectionQueue(BaseModel):
    """Persistent correction queue with state tracking."""

    certification_name: str
    source_version: int  # Skeleton version being corrected
    target_version: int  # Version after corrections applied
    created_at: datetime
    actions: list[CorrectionAction] = []

    @property
    def pending_count(self) -> int:
        return sum(1 for a in self.actions if a.status == "pending")

    @property
    def applied_count(self) -> int:
        return sum(1 for a in self.actions if a.status == "applied")

    def save(self, path: Path) -> None:
        """Save queue to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            # Write header line with queue metadata
            header = {
                "_type": "queue_header",
                "certification_name": self.certification_name,
                "source_version": self.source_version,
                "target_version": self.target_version,
                "created_at": self.created_at.isoformat(),
            }
            f.write(json.dumps(header) + "\n")
            # Write each action as a line
            for action in self.actions:
                f.write(action.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: Path) -> "CorrectionQueue":
        """Load queue from JSONL file."""
        actions = []
        header = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("_type") == "queue_header":
                    header = data
                else:
                    actions.append(CorrectionAction.model_validate(data))

        if not header:
            raise ValueError(f"Invalid queue file: no header found in {path}")

        return cls(
            certification_name=header["certification_name"],
            source_version=header["source_version"],
            target_version=header["target_version"],
            created_at=datetime.fromisoformat(header["created_at"]),
            actions=actions,
        )

    def append_action(self, action: CorrectionAction, path: Path | None = None) -> None:
        """Append action to queue and optionally persist immediately."""
        self.actions.append(action)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(action.model_dump_json() + "\n")


class Step6Output(BaseModel):
    """Summary output for Step 6."""

    certification_name: str
    input_version: int
    output_version: int
    total_entities_validated: int
    passed_count: int
    minor_count: int
    major_count: int
    critical_count: int
    corrections_applied: int
    corrections_failed: int
    validation_status: str  # "passed", "corrected", "needs_review"
