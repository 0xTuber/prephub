"""Tests for Step 6.2: Correction Queue Step."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    CorrectionAction,
    CorrectionQueue,
    ValidationIssue,
    ValidationReport,
    ValidationResult,
)
from pipeline.step6.correction_queue import (
    CorrectionQueueStep,
    _build_enhanced_prompt,
    _determine_action_type,
    _issue_to_action,
    _severity_to_priority,
)


def _make_validation_report() -> ValidationReport:
    """Create a validation report with various issues."""
    return ValidationReport(
        certification_name="Test Certification",
        skeleton_version=1,
        validated_at=datetime.now(),
        total_entities=10,
        passed_count=5,
        minor_count=2,
        major_count=2,
        critical_count=1,
        results=[
            ValidationResult(
                entity_type="item",
                entity_id="item_01",
                entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_01"],
                overall_status="critical",
                issues=[
                    ValidationIssue(
                        issue_id="issue_001",
                        severity="critical",
                        rule_name="content_not_empty",
                        description="Item content is missing",
                        field_path="content",
                    ),
                ],
                validated_at=datetime.now(),
            ),
            ValidationResult(
                entity_type="item",
                entity_id="item_02",
                entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_02"],
                overall_status="major",
                issues=[
                    ValidationIssue(
                        issue_id="issue_002",
                        severity="major",
                        rule_name="answer_grounded",
                        description="Answer not found in source material",
                        field_path="correct_answer_index",
                        suggested_fix="Regenerate with better grounding",
                    ),
                ],
                validated_at=datetime.now(),
            ),
            ValidationResult(
                entity_type="item",
                entity_id="item_03",
                entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_03"],
                overall_status="minor",
                issues=[
                    ValidationIssue(
                        issue_id="issue_003",
                        severity="minor",
                        rule_name="chunk_ids_present",
                        description="No chunk IDs in source reference",
                        field_path="source_reference.chunk_ids",
                    ),
                ],
                validated_at=datetime.now(),
            ),
            ValidationResult(
                entity_type="item",
                entity_id="item_04",
                entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_04"],
                overall_status="passed",
                issues=[],
                validated_at=datetime.now(),
            ),
        ],
    )


class TestActionTypeMapping:
    """Tests for action type determination."""

    def test_minor_maps_to_auto_fix(self):
        assert _determine_action_type("minor") == "auto_fix"

    def test_major_maps_to_regenerate(self):
        assert _determine_action_type("major") == "regenerate"

    def test_critical_maps_to_regenerate(self):
        assert _determine_action_type("critical") == "regenerate"

    def test_passed_maps_to_skip(self):
        assert _determine_action_type("passed") == "skip"


class TestSeverityPriority:
    """Tests for severity to priority mapping."""

    def test_critical_highest_priority(self):
        assert _severity_to_priority("critical") == 100

    def test_major_medium_priority(self):
        assert _severity_to_priority("major") == 50

    def test_minor_low_priority(self):
        assert _severity_to_priority("minor") == 10

    def test_passed_zero_priority(self):
        assert _severity_to_priority("passed") == 0


class TestEnhancedPromptGeneration:
    """Tests for enhanced prompt generation."""

    def test_builds_prompt_with_issue_details(self):
        issue = ValidationIssue(
            issue_id="test",
            severity="major",
            rule_name="answer_grounded",
            description="Answer not found in source",
            field_path="correct_answer_index",
            suggested_fix="Regenerate with better grounding",
        )
        result = ValidationResult(
            entity_type="item",
            entity_id="item_01",
            entity_path=["domain_1", "topic_1"],
            overall_status="major",
            issues=[issue],
            validated_at=datetime.now(),
        )

        prompt = _build_enhanced_prompt(issue, result)

        assert "REGENERATION REQUIRED" in prompt
        assert "item_01" in prompt
        assert "answer_grounded" in prompt
        assert "MAJOR" in prompt
        assert "SUGGESTED FIX" in prompt

    def test_includes_source_evidence_if_present(self):
        issue = ValidationIssue(
            issue_id="test",
            severity="major",
            rule_name="test_rule",
            description="Test description",
            field_path="test",
            source_evidence="Quote from source material",
        )
        result = ValidationResult(
            entity_type="item",
            entity_id="item_01",
            entity_path=["domain_1"],
            overall_status="major",
            issues=[issue],
            validated_at=datetime.now(),
        )

        prompt = _build_enhanced_prompt(issue, result)

        assert "SOURCE EVIDENCE" in prompt
        assert "Quote from source material" in prompt


class TestIssueToAction:
    """Tests for converting issues to actions."""

    def test_passed_returns_none(self):
        issue = ValidationIssue(
            issue_id="test",
            severity="passed",
            rule_name="test",
            description="",
            field_path="",
        )
        result = ValidationResult(
            entity_type="item",
            entity_id="item_01",
            entity_path=[],
            overall_status="passed",
            issues=[],
            validated_at=datetime.now(),
        )

        action = _issue_to_action(issue, result)
        assert action is None

    def test_critical_creates_manual_review_action(self):
        issue = ValidationIssue(
            issue_id="test",
            severity="critical",
            rule_name="content_not_empty",
            description="Content is missing",
            field_path="content",
        )
        result = ValidationResult(
            entity_type="item",
            entity_id="item_01",
            entity_path=["domain_1"],
            overall_status="critical",
            issues=[issue],
            validated_at=datetime.now(),
        )

        action = _issue_to_action(issue, result)

        assert action is not None
        assert action.action_type == "manual_review"
        assert action.priority == 100

    def test_major_creates_regenerate_action(self):
        issue = ValidationIssue(
            issue_id="test",
            severity="major",
            rule_name="answer_grounded",
            description="Answer not grounded",
            field_path="correct_answer_index",
        )
        result = ValidationResult(
            entity_type="item",
            entity_id="item_01",
            entity_path=["domain_1"],
            overall_status="major",
            issues=[issue],
            validated_at=datetime.now(),
        )

        action = _issue_to_action(issue, result)

        assert action is not None
        assert action.action_type == "regenerate"
        assert action.regenerate_prompt is not None


class TestCorrectionQueuePersistence:
    """Tests for CorrectionQueue save/load."""

    def test_save_and_load_queue(self):
        queue = CorrectionQueue(
            certification_name="Test",
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[
                CorrectionAction(
                    action_id="action_001",
                    entity_type="item",
                    entity_id="item_01",
                    entity_path=["domain_1"],
                    action_type="regenerate",
                    priority=50,
                    status="pending",
                    created_at=datetime.now(),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_queue.jsonl"
            queue.save(path)

            loaded = CorrectionQueue.load(path)

            assert loaded.certification_name == "Test"
            assert loaded.source_version == 1
            assert loaded.target_version == 2
            assert len(loaded.actions) == 1
            assert loaded.actions[0].action_id == "action_001"

    def test_pending_and_applied_counts(self):
        queue = CorrectionQueue(
            certification_name="Test",
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[
                CorrectionAction(
                    action_id="1",
                    entity_type="item",
                    entity_id="item_01",
                    entity_path=[],
                    action_type="auto_fix",
                    status="pending",
                    created_at=datetime.now(),
                ),
                CorrectionAction(
                    action_id="2",
                    entity_type="item",
                    entity_id="item_02",
                    entity_path=[],
                    action_type="regenerate",
                    status="applied",
                    created_at=datetime.now(),
                ),
                CorrectionAction(
                    action_id="3",
                    entity_type="item",
                    entity_id="item_03",
                    entity_path=[],
                    action_type="regenerate",
                    status="pending",
                    created_at=datetime.now(),
                ),
            ],
        )

        assert queue.pending_count == 2
        assert queue.applied_count == 1


class TestCorrectionQueueStep:
    """Tests for the CorrectionQueueStep."""

    def test_generates_queue_from_report(self):
        report = _make_validation_report()

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionQueueStep(corrections_dir=tmpdir)
            ctx = PipelineContext(validation_report=report)

            result = step.run(ctx)

            assert "correction_queue" in result
            queue = result["correction_queue"]
            assert isinstance(queue, CorrectionQueue)
            # Should have actions for non-passed issues
            assert len(queue.actions) >= 1

    def test_sorts_by_priority(self):
        report = _make_validation_report()

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionQueueStep(corrections_dir=tmpdir)
            ctx = PipelineContext(validation_report=report)

            result = step.run(ctx)

            queue = result["correction_queue"]
            # Actions should be sorted by priority (highest first)
            priorities = [a.priority for a in queue.actions]
            assert priorities == sorted(priorities, reverse=True)

    def test_saves_queue_file(self):
        report = _make_validation_report()

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionQueueStep(corrections_dir=tmpdir)
            ctx = PipelineContext(validation_report=report)

            result = step.run(ctx)

            # Check that queue file was created
            queue_path = Path(result["correction_queue_path"])
            assert queue_path.exists()

    def test_skips_without_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionQueueStep(corrections_dir=tmpdir)
            ctx = PipelineContext()

            result = step.run(ctx)

            assert "correction_queue" not in result

    def test_no_duplicates_for_same_issue(self):
        # Create report with duplicate issues for same entity
        report = ValidationReport(
            certification_name="Test",
            skeleton_version=1,
            validated_at=datetime.now(),
            total_entities=1,
            passed_count=0,
            minor_count=0,
            major_count=1,
            critical_count=0,
            results=[
                ValidationResult(
                    entity_type="item",
                    entity_id="item_01",
                    entity_path=["domain_1"],
                    overall_status="major",
                    issues=[
                        ValidationIssue(
                            issue_id="issue_1",
                            severity="major",
                            rule_name="same_rule",
                            description="Same issue",
                            field_path="content",
                        ),
                        ValidationIssue(
                            issue_id="issue_2",
                            severity="major",
                            rule_name="same_rule",  # Same rule name
                            description="Same issue again",
                            field_path="content",
                        ),
                    ],
                    validated_at=datetime.now(),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionQueueStep(corrections_dir=tmpdir)
            ctx = PipelineContext(validation_report=report)

            result = step.run(ctx)

            queue = result["correction_queue"]
            # Should only have one action for the same entity/rule combination
            assert len(queue.actions) == 1
