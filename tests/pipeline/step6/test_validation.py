"""Tests for Step 6.1: Hierarchical Validation Step."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    Capsule,
    CapsuleItem,
    CourseModule,
    CourseOverview,
    CourseSkeleton,
    CourseTopic,
    ItemSourceReference,
    Lab,
    LearningObjective,
    SubTopic,
    ValidationReport,
)
from pipeline.step6.validation import (
    HierarchicalValidationStep,
    _validate_entity_structural,
)
from pipeline.step6.rules import Severity, get_structural_rules


def _make_valid_skeleton() -> CourseSkeleton:
    """Create a valid skeleton with all required fields."""
    return CourseSkeleton(
        certification_name="Test Certification",
        exam_code="TEST-001",
        overview=CourseOverview(
            target_audience="Test audience",
            course_description="Test course description",
        ),
        domain_modules=[
            CourseModule(
                domain_name="Domain 1",
                domain_weight_pct=100.0,
                overview="Domain overview text",
                topics=[
                    CourseTopic(
                        name="Topic 1",
                        description="Topic description",
                        learning_objectives=[
                            LearningObjective(
                                objective="Test objective",
                                bloom_level="Apply",
                            ),
                        ],
                        subtopics=[
                            SubTopic(
                                name="Subtopic 1",
                                description="Subtopic description",
                                labs=[
                                    Lab(
                                        lab_id="lab_01",
                                        title="Lab 1",
                                        objective="Lab objective text here",
                                        lab_type="guided",
                                        capsules=[
                                            Capsule(
                                                capsule_id="cap_01",
                                                title="Capsule 1",
                                                learning_goal="Capsule learning goal here",
                                                capsule_type="conceptual",
                                                items=[
                                                    CapsuleItem(
                                                        item_id="item_01",
                                                        item_type="Multiple Choice",
                                                        title="Item 1",
                                                        learning_target="Test learning target",
                                                        difficulty="intermediate",
                                                        content="What is the correct answer?",
                                                        options=["A", "B", "C", "D"],
                                                        correct_answer_index=0,
                                                        explanation="A is correct because...",
                                                        source_reference=ItemSourceReference(
                                                            summary="Source summary " * 5,
                                                            citations=[],
                                                            chunk_ids=["chunk_1"],
                                                        ),
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _make_skeleton_with_invalid_item() -> CourseSkeleton:
    """Create a skeleton with an invalid item (missing content)."""
    skeleton = _make_valid_skeleton()
    skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0].content = None
    skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0].options = None
    return skeleton


def _make_empty_skeleton() -> CourseSkeleton:
    """Create a skeleton with no modules (critical failure)."""
    return CourseSkeleton(
        certification_name="Empty Test",
        overview=CourseOverview(),
        domain_modules=[],
    )


class TestStructuralValidation:
    """Tests for structural validation rules."""

    def test_valid_item_passes(self):
        skeleton = _make_valid_skeleton()
        item = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0]

        result = _validate_entity_structural(
            item, "item", "item_01", ["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_01"]
        )

        assert result.overall_status == "passed"
        assert len(result.issues) == 0

    def test_invalid_item_fails(self):
        skeleton = _make_skeleton_with_invalid_item()
        item = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0]

        result = _validate_entity_structural(
            item, "item", "item_01", ["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_01"]
        )

        assert result.overall_status == "critical"
        assert len(result.issues) > 0
        rule_names = [i.rule_name for i in result.issues]
        assert "content_not_empty" in rule_names
        assert "options_present" in rule_names

    def test_empty_skeleton_critical(self):
        skeleton = _make_empty_skeleton()

        result = _validate_entity_structural(skeleton, "skeleton", "skeleton", [])

        assert result.overall_status == "critical"
        assert any(i.rule_name == "skeleton_has_domains" for i in result.issues)

    def test_module_without_topics_critical(self):
        module = CourseModule(
            domain_name="Empty Module",
            topics=[],
        )

        result = _validate_entity_structural(module, "module", "empty_module", ["empty_module"])

        assert result.overall_status == "critical"
        assert any(i.rule_name == "module_not_empty" for i in result.issues)

    def test_invalid_lab_type(self):
        lab = Lab(
            lab_id="lab_01",
            title="Test Lab",
            objective="Test objective text",
            lab_type="invalid_type",
            capsules=[
                Capsule(
                    capsule_id="cap_01",
                    title="Capsule 1",
                    learning_goal="Test learning goal",
                    capsule_type="conceptual",
                    items=[],
                ),
            ],
        )

        result = _validate_entity_structural(lab, "lab", "lab_01", ["module", "topic", "subtopic", "lab_01"])

        # lab_type_valid should fail with minor severity
        assert any(i.rule_name == "lab_type_valid" for i in result.issues)

    def test_duplicate_options_detected(self):
        item = CapsuleItem(
            item_id="item_01",
            item_type="Multiple Choice",
            title="Test Item",
            learning_target="Test target",
            content="Test question?",
            options=["Same", "Same", "C", "D"],  # Duplicate
            correct_answer_index=0,
            explanation="Test explanation text",
        )

        result = _validate_entity_structural(
            item, "item", "item_01", ["module", "topic", "subtopic", "lab", "capsule", "item_01"]
        )

        assert any(i.rule_name == "no_duplicate_options" for i in result.issues)


class TestHierarchicalValidationStep:
    """Tests for the HierarchicalValidationStep."""

    def test_validates_skeleton_structurally(self):
        skeleton = _make_valid_skeleton()

        step = HierarchicalValidationStep(skip_llm_review=True)

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            ctx = PipelineContext(
                course_skeleton=skeleton,
                collection_name="test",
            )
            result = step.run(ctx)

        assert "validation_report" in result
        report = result["validation_report"]
        assert isinstance(report, ValidationReport)
        assert report.certification_name == "Test Certification"
        assert report.total_entities > 0

    def test_stops_at_critical_skeleton(self):
        skeleton = _make_empty_skeleton()

        step = HierarchicalValidationStep(skip_llm_review=True)

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            ctx = PipelineContext(
                course_skeleton=skeleton,
                collection_name="test",
            )
            result = step.run(ctx)

        report = result["validation_report"]
        # Should only have skeleton result since it failed critically
        assert report.critical_count >= 1
        # Should have stopped early - only skeleton validated
        assert report.total_entities == 1

    def test_counts_severities_correctly(self):
        skeleton = _make_skeleton_with_invalid_item()

        step = HierarchicalValidationStep(skip_llm_review=True)

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            ctx = PipelineContext(
                course_skeleton=skeleton,
                collection_name="test",
            )
            result = step.run(ctx)

        report = result["validation_report"]
        # Should have at least one critical (the invalid item)
        assert report.critical_count >= 1

    def test_preserves_context(self):
        skeleton = _make_valid_skeleton()

        step = HierarchicalValidationStep(skip_llm_review=True)

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            ctx = PipelineContext(
                course_skeleton=skeleton,
                collection_name="test",
                custom_key="custom_value",
            )
            result = step.run(ctx)

        assert result["custom_key"] == "custom_value"
        assert result["course_skeleton"] == skeleton


class TestValidationRules:
    """Tests for individual validation rules."""

    def test_content_not_empty_rule(self):
        rules = get_structural_rules("item")
        rule = next(r for r in rules if r.name == "content_not_empty")

        # Test with empty content
        item = MagicMock()
        item.content = None
        passed, desc, _ = rule.check_fn(item)
        assert not passed

        # Test with short content
        item.content = "short"
        passed, desc, _ = rule.check_fn(item)
        assert not passed

        # Test with valid content
        item.content = "This is a valid question with enough content"
        passed, desc, _ = rule.check_fn(item)
        assert passed

    def test_options_count_valid_rule(self):
        rules = get_structural_rules("item")
        rule = next(r for r in rules if r.name == "options_count_valid")

        item = MagicMock()
        item.item_type = "Multiple Choice"

        # Test with wrong count
        item.options = ["A", "B", "C"]
        passed, desc, _ = rule.check_fn(item)
        assert not passed

        # Test with correct count
        item.options = ["A", "B", "C", "D"]
        passed, desc, _ = rule.check_fn(item)
        assert passed

    def test_correct_index_valid_rule(self):
        rules = get_structural_rules("item")
        rule = next(r for r in rules if r.name == "correct_index_valid")

        item = MagicMock()
        item.options = ["A", "B", "C", "D"]

        # Test with invalid index
        item.correct_answer_index = 5
        passed, desc, _ = rule.check_fn(item)
        assert not passed

        # Test with valid index
        item.correct_answer_index = 2
        passed, desc, _ = rule.check_fn(item)
        assert passed

    def test_domain_weights_sum_rule(self):
        rules = get_structural_rules("skeleton")
        rule = next(r for r in rules if r.name == "domain_weights_sum")

        skeleton = MagicMock()

        # Test with correct sum
        module1 = MagicMock()
        module1.domain_weight_pct = 60.0
        module2 = MagicMock()
        module2.domain_weight_pct = 40.0
        skeleton.domain_modules = [module1, module2]

        passed, desc, _ = rule.check_fn(skeleton)
        assert passed

        # Test with incorrect sum
        module1.domain_weight_pct = 80.0
        module2.domain_weight_pct = 80.0

        passed, desc, _ = rule.check_fn(skeleton)
        assert not passed


class TestSeverityOrdering:
    """Tests for severity comparison."""

    def test_severity_ordering(self):
        assert Severity.PASSED < Severity.MINOR
        assert Severity.MINOR < Severity.MAJOR
        assert Severity.MAJOR < Severity.CRITICAL

    def test_severity_max(self):
        severities = [Severity.PASSED, Severity.MINOR, Severity.CRITICAL, Severity.MAJOR]
        assert max(severities) == Severity.CRITICAL
