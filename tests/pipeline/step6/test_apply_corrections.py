"""Tests for Step 6.3: Correction Application Step."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    Capsule,
    CapsuleItem,
    CorrectionAction,
    CorrectionQueue,
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
from pipeline.step6.apply_corrections import (
    CorrectionApplicationStep,
    _apply_auto_fix,
    _find_entity_in_skeleton,
    _quick_validate_item,
)


def _make_mock_engine():
    """Create a mock generation engine for testing."""
    engine = MagicMock()
    engine.engine_type = "mock"
    engine.model_name = "mock-model"
    engine.generate.return_value = MagicMock(
        text='{"stem": "Test question?", "options": ["A", "B", "C", "D"], "correct_index": 0, "explanation": "Test explanation"}'
    )
    return engine


def _make_skeleton_with_items() -> CourseSkeleton:
    """Create a skeleton with items for testing."""
    return CourseSkeleton(
        certification_name="Test Certification",
        exam_code="TEST-001",
        version=1,
        overview=CourseOverview(
            target_audience="Test audience",
            course_description="Test description",
        ),
        domain_modules=[
            CourseModule(
                domain_name="Domain 1",
                domain_weight_pct=100.0,
                overview="Domain overview",
                topics=[
                    CourseTopic(
                        name="Topic 1",
                        description="Topic description",
                        learning_objectives=[
                            LearningObjective(objective="Test objective"),
                        ],
                        subtopics=[
                            SubTopic(
                                name="Subtopic 1",
                                description="Subtopic description",
                                labs=[
                                    Lab(
                                        lab_id="lab_01",
                                        title="Lab 1",
                                        objective="Lab objective text",
                                        lab_type="guided",
                                        capsules=[
                                            Capsule(
                                                capsule_id="cap_01",
                                                title="Capsule 1",
                                                learning_goal="Capsule learning goal",
                                                capsule_type="conceptual",
                                                items=[
                                                    CapsuleItem(
                                                        item_id="item_01",
                                                        item_type="Multiple Choice",
                                                        title="Item 1",
                                                        learning_target="Test target",
                                                        difficulty="intermediate",
                                                        content="Original question?",
                                                        options=["A", "B", "C", "D"],
                                                        correct_answer_index=0,
                                                        explanation="Original explanation",
                                                        source_reference=ItemSourceReference(
                                                            summary="Test summary " * 5,
                                                            citations=[],
                                                            chunk_ids=["chunk_1"],
                                                        ),
                                                    ),
                                                    CapsuleItem(
                                                        item_id="item_02",
                                                        item_type="Multiple Choice",
                                                        title="Item 2",
                                                        learning_target="Test target 2",
                                                        content=None,  # Invalid - no content
                                                        options=None,
                                                        correct_answer_index=None,
                                                        explanation=None,
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


def _make_correction_queue(skeleton: CourseSkeleton) -> CorrectionQueue:
    """Create a correction queue for the skeleton."""
    return CorrectionQueue(
        certification_name=skeleton.certification_name,
        source_version=skeleton.version,
        target_version=skeleton.version + 1,
        created_at=datetime.now(),
        actions=[
            CorrectionAction(
                action_id="action_001",
                entity_type="item",
                entity_id="item_02",
                entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_02"],
                action_type="regenerate",
                regenerate_prompt="Content is missing. Please regenerate.",
                priority=100,
                status="pending",
                created_at=datetime.now(),
            ),
        ],
    )


class TestFindEntityInSkeleton:
    """Tests for finding entities in skeleton by path."""

    def test_finds_item_by_path(self):
        skeleton = _make_skeleton_with_items()

        entity, parent_list, idx = _find_entity_in_skeleton(
            skeleton,
            "item",
            ["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_01"],
        )

        assert entity is not None
        assert entity.item_id == "item_01"
        assert parent_list is not None
        assert idx == 0

    def test_finds_capsule_by_path(self):
        skeleton = _make_skeleton_with_items()

        entity, parent_list, idx = _find_entity_in_skeleton(
            skeleton,
            "capsule",
            ["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01"],
        )

        assert entity is not None
        assert entity.capsule_id == "cap_01"

    def test_finds_lab_by_path(self):
        skeleton = _make_skeleton_with_items()

        entity, parent_list, idx = _find_entity_in_skeleton(
            skeleton,
            "lab",
            ["domain_1", "topic_1", "subtopic_1", "lab_01"],
        )

        assert entity is not None
        assert entity.lab_id == "lab_01"

    def test_returns_none_for_invalid_path(self):
        skeleton = _make_skeleton_with_items()

        entity, parent_list, idx = _find_entity_in_skeleton(
            skeleton,
            "item",
            ["nonexistent", "path"],
        )

        assert entity is None


class TestApplyAutoFix:
    """Tests for applying auto-fix corrections."""

    def test_applies_field_correction(self):
        skeleton = _make_skeleton_with_items()
        action = CorrectionAction(
            action_id="test",
            entity_type="item",
            entity_id="item_01",
            entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_01"],
            action_type="auto_fix",
            field_corrections={"explanation": "New explanation"},
            status="pending",
            created_at=datetime.now(),
        )

        success = _apply_auto_fix(skeleton, action)

        assert success
        item = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0]
        assert item.explanation == "New explanation"

    def test_handles_nested_field_path(self):
        skeleton = _make_skeleton_with_items()
        action = CorrectionAction(
            action_id="test",
            entity_type="item",
            entity_id="item_01",
            entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_01"],
            action_type="auto_fix",
            field_corrections={"source_reference.summary": "New summary"},
            status="pending",
            created_at=datetime.now(),
        )

        success = _apply_auto_fix(skeleton, action)

        assert success
        item = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0]
        assert item.source_reference.summary == "New summary"

    def test_returns_false_for_invalid_entity(self):
        skeleton = _make_skeleton_with_items()
        action = CorrectionAction(
            action_id="test",
            entity_type="item",
            entity_id="nonexistent",
            entity_path=["invalid", "path"],
            action_type="auto_fix",
            field_corrections={"content": "New content"},
            status="pending",
            created_at=datetime.now(),
        )

        success = _apply_auto_fix(skeleton, action)

        assert not success


class TestQuickValidateItem:
    """Tests for quick item validation."""

    def test_valid_item_passes(self):
        item = CapsuleItem(
            item_id="test",
            item_type="Multiple Choice",
            title="Test",
            learning_target="Test target",
            content="Valid question content here?",
            options=["A", "B", "C", "D"],
            correct_answer_index=0,
            explanation="Valid explanation text",
        )

        assert _quick_validate_item(item)

    def test_invalid_item_fails(self):
        item = CapsuleItem(
            item_id="test",
            item_type="Multiple Choice",
            title="Test",
            learning_target="Test target",
            content=None,  # Invalid
            options=None,  # Invalid
            correct_answer_index=None,
            explanation=None,
        )

        assert not _quick_validate_item(item)


SAMPLE_REGENERATION_JSON = json.dumps({
    "stem": "Regenerated question content?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 1,
    "explanation": "Option B is correct because...",
})


def _make_mock_genai_response(text: str):
    return SimpleNamespace(text=text)


class TestCorrectionApplicationStep:
    """Tests for the CorrectionApplicationStep."""

    def test_applies_corrections_without_llm(self):
        skeleton = _make_skeleton_with_items()

        # Create auto-fix only queue
        queue = CorrectionQueue(
            certification_name=skeleton.certification_name,
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[
                CorrectionAction(
                    action_id="action_001",
                    entity_type="item",
                    entity_id="item_01",
                    entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_01"],
                    action_type="auto_fix",
                    field_corrections={"explanation": "Updated explanation"},
                    priority=10,
                    status="pending",
                    created_at=datetime.now(),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Provide a mock engine since auto_fix doesn't need LLM
            mock_engine = _make_mock_engine()
            step = CorrectionApplicationStep(
                engine=mock_engine,
                corrections_dir=tmpdir,
                output_dir=tmpdir,
            )

            ctx = PipelineContext(
                course_skeleton=skeleton,
                correction_queue=queue,
            )

            result = step.run(ctx)

            updated_skeleton = result["course_skeleton"]
            assert updated_skeleton.version == 2
            item = updated_skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0]
            assert item.explanation == "Updated explanation"

    def test_regeneration_with_mock_engine(self):
        """Test that regeneration actions use the provided engine."""
        skeleton = _make_skeleton_with_items()
        queue = _make_correction_queue(skeleton)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock engine that returns valid regenerated content
            mock_engine = _make_mock_engine()

            step = CorrectionApplicationStep(
                engine=mock_engine,
                corrections_dir=tmpdir,
                output_dir=tmpdir,
            )

            ctx = PipelineContext(
                course_skeleton=skeleton,
                correction_queue=queue,
            )

            result = step.run(ctx)

            # Engine's generate method should have been called for regeneration
            assert mock_engine.generate.called
            # The action should be processed (applied or failed based on validation)
            queue_result = result["correction_queue"]
            assert queue_result.actions[0].status in ("applied", "failed")

    def test_bumps_version(self):
        skeleton = _make_skeleton_with_items()
        assert skeleton.version == 1

        queue = CorrectionQueue(
            certification_name=skeleton.certification_name,
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[],  # Empty queue
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionApplicationStep(
                corrections_dir=tmpdir,
                output_dir=tmpdir,
            )

            ctx = PipelineContext(
                course_skeleton=skeleton,
                correction_queue=queue,
            )

            with patch.dict("os.environ", {"GOOGLE_API_KEY": ""}):
                result = step.run(ctx)

            assert result["course_skeleton"].version == 2

    def test_saves_skeleton_file(self):
        skeleton = _make_skeleton_with_items()
        queue = CorrectionQueue(
            certification_name=skeleton.certification_name,
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionApplicationStep(
                corrections_dir=tmpdir,
                output_dir=tmpdir,
            )

            ctx = PipelineContext(
                course_skeleton=skeleton,
                correction_queue=queue,
            )

            with patch.dict("os.environ", {"GOOGLE_API_KEY": ""}):
                result = step.run(ctx)

            # Check files exist
            cert_slug = skeleton.certification_name.replace(" ", "_")
            assert (Path(tmpdir) / f"{cert_slug}_skeleton_v2.json").exists()
            assert (Path(tmpdir) / f"{cert_slug}_skeleton_latest.json").exists()

    def test_loads_queue_from_file(self):
        skeleton = _make_skeleton_with_items()
        queue = CorrectionQueue(
            certification_name=skeleton.certification_name,
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[
                CorrectionAction(
                    action_id="action_001",
                    entity_type="item",
                    entity_id="item_01",
                    entity_path=["domain_1", "topic_1", "subtopic_1", "lab_01", "cap_01", "item_01"],
                    action_type="auto_fix",
                    field_corrections={"explanation": "From file"},
                    priority=10,
                    status="pending",
                    created_at=datetime.now(),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save queue to file
            queue_path = Path(tmpdir) / "test_queue.jsonl"
            queue.save(queue_path)

            # Provide a mock engine
            mock_engine = _make_mock_engine()
            step = CorrectionApplicationStep(
                engine=mock_engine,
                corrections_dir=tmpdir,
                output_dir=tmpdir,
            )

            # Don't include queue in context, only the path
            ctx = PipelineContext(
                course_skeleton=skeleton,
                correction_queue_path=str(queue_path),
            )

            result = step.run(ctx)

            # Should have loaded and applied the correction
            item = result["course_skeleton"].domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0]
            assert item.explanation == "From file"

    def test_sets_validation_status(self):
        skeleton = _make_skeleton_with_items()

        # Test "passed" status (no corrections)
        queue = CorrectionQueue(
            certification_name=skeleton.certification_name,
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionApplicationStep(
                corrections_dir=tmpdir,
                output_dir=tmpdir,
            )

            ctx = PipelineContext(
                course_skeleton=skeleton,
                correction_queue=queue,
            )

            with patch.dict("os.environ", {"GOOGLE_API_KEY": ""}):
                result = step.run(ctx)

            assert result["course_skeleton"].validation_status == "passed"

    def test_preserves_context(self):
        skeleton = _make_skeleton_with_items()
        queue = CorrectionQueue(
            certification_name=skeleton.certification_name,
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionApplicationStep(
                corrections_dir=tmpdir,
                output_dir=tmpdir,
            )

            ctx = PipelineContext(
                course_skeleton=skeleton,
                correction_queue=queue,
                custom_key="preserved",
            )

            with patch.dict("os.environ", {"GOOGLE_API_KEY": ""}):
                result = step.run(ctx)

            assert result["custom_key"] == "preserved"

    def test_returns_step6_output(self):
        skeleton = _make_skeleton_with_items()
        queue = CorrectionQueue(
            certification_name=skeleton.certification_name,
            source_version=1,
            target_version=2,
            created_at=datetime.now(),
            actions=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CorrectionApplicationStep(
                corrections_dir=tmpdir,
                output_dir=tmpdir,
            )

            # Include validation report for full output
            report = ValidationReport(
                certification_name=skeleton.certification_name,
                skeleton_version=1,
                validated_at=datetime.now(),
                total_entities=10,
                passed_count=8,
                minor_count=1,
                major_count=1,
                critical_count=0,
                results=[],
            )

            ctx = PipelineContext(
                course_skeleton=skeleton,
                correction_queue=queue,
                validation_report=report,
            )

            with patch.dict("os.environ", {"GOOGLE_API_KEY": ""}):
                result = step.run(ctx)

            assert "step6_output" in result
            output = result["step6_output"]
            assert output.input_version == 1
            assert output.output_version == 2
            assert output.total_entities_validated == 10
