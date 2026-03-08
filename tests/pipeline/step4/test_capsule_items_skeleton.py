import json
from types import SimpleNamespace
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
    Lab,
    LearningObjective,
    QuestionTypeGuide,
    SubTopic,
)
from pipeline.step4.capsule_items_skeleton import CapsuleItemsSkeletonStep


def _make_skeleton_with_capsules() -> CourseSkeleton:
    """Create a minimal skeleton with capsules for testing."""
    return CourseSkeleton(
        certification_name="AWS Solutions Architect Associate",
        exam_code="SAA-C03",
        overview=CourseOverview(
            target_audience="Cloud architects",
            course_description="AWS SAA preparation",
        ),
        question_type_guides=[
            QuestionTypeGuide(
                question_type_name="Multiple Choice",
                detailed_structure="A stem with 4 options, one correct",
            ),
            QuestionTypeGuide(
                question_type_name="Multiple Response",
                detailed_structure="Select 2-3 correct answers from 5-6 options",
            ),
        ],
        domain_modules=[
            CourseModule(
                domain_name="Design Secure Architectures",
                domain_weight_pct=30.0,
                topics=[
                    CourseTopic(
                        name="IAM Policies",
                        learning_objectives=[
                            LearningObjective(
                                objective="Configure IAM policies",
                                bloom_level="Apply",
                                relevant_question_types=["Multiple Choice"],
                            ),
                        ],
                        subtopics=[
                            SubTopic(
                                name="Policy Structure",
                                description="JSON policy documents",
                                labs=[
                                    Lab(
                                        lab_id="lab_01",
                                        title="Creating Basic IAM Policies",
                                        objective="Create a custom IAM policy",
                                        lab_type="guided",
                                        capsules=[
                                            Capsule(
                                                capsule_id="cap_01",
                                                title="Understanding IAM Policy Syntax",
                                                learning_goal="Identify all policy components",
                                                capsule_type="conceptual",
                                                assessment_criteria=[
                                                    "Can name elements",
                                                    "Can explain effects",
                                                ],
                                                common_errors=[
                                                    "Confusing Action/Resource"
                                                ],
                                            ),
                                            Capsule(
                                                capsule_id="cap_02",
                                                title="Writing Your First Policy",
                                                learning_goal="Write a correct policy",
                                                capsule_type="procedural",
                                                assessment_criteria=[
                                                    "Policy validates"
                                                ],
                                                common_errors=["Invalid JSON"],
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


SAMPLE_ITEMS_JSON = json.dumps(
    {
        "items": [
            {
                "item_id": "item_01",
                "item_type": "Multiple Choice",
                "title": "IAM Policy Structure Overview",
                "learning_target": "Identify the main components of an IAM policy document",
                "difficulty": "beginner",
            },
            {
                "item_id": "item_02",
                "item_type": "Multiple Choice",
                "title": "Policy Effect Types",
                "learning_target": "Distinguish between Allow and Deny effects",
                "difficulty": "beginner",
            },
            {
                "item_id": "item_03",
                "item_type": "Multiple Response",
                "title": "Policy Elements",
                "learning_target": "Select all required elements in an IAM policy",
                "difficulty": "intermediate",
            },
            {
                "item_id": "item_04",
                "item_type": "Multiple Choice",
                "title": "Action vs Resource",
                "learning_target": "Correctly match actions to resources",
                "difficulty": "intermediate",
            },
            {
                "item_id": "item_05",
                "item_type": "Multiple Choice",
                "title": "Policy Evaluation",
                "learning_target": "Predict the outcome of policy evaluation",
                "difficulty": "advanced",
            },
        ]
    }
)


def _make_mock_response(text: str):
    return SimpleNamespace(text=text)


def _build_side_effect(
    items_responses=None,
    items_errors=None,
):
    """Build a side_effect function for item generation."""
    if items_responses is None:
        items_responses = {}
    if items_errors is None:
        items_errors = {}

    def side_effect(*, model, contents, config):
        # Check for specific capsule titles in the prompt
        for capsule_title, error in items_errors.items():
            if capsule_title in contents:
                raise error

        for capsule_title, resp in items_responses.items():
            if capsule_title in contents:
                return _make_mock_response(resp)

        # Default response for any capsule
        return _make_mock_response(SAMPLE_ITEMS_JSON)

    return side_effect


def _run_step(
    skeleton=None,
    items_responses=None,
    items_errors=None,
    max_workers=2,
    target_item_count=5,
) -> tuple[PipelineContext, MagicMock]:
    if skeleton is None:
        skeleton = _make_skeleton_with_capsules()

    step = CapsuleItemsSkeletonStep(
        max_workers=max_workers, target_item_count=target_item_count
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _build_side_effect(
        items_responses=items_responses,
        items_errors=items_errors,
    )

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch(
            "pipeline.step4.capsule_items_skeleton.genai.Client",
            return_value=mock_client,
        ),
        patch("pipeline.step4.capsule_items_skeleton.time.sleep"),
    ):
        ctx = PipelineContext(
            certification_name="AWS SAA-C03",
            course_skeleton=skeleton,
        )
        result = step.run(ctx)
        return result, mock_client


class TestCapsuleItemsSkeletonStep:
    def test_generates_items_for_all_capsules(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        # Check that all capsules have items
        for module in skeleton.domain_modules:
            for topic in module.topics:
                for subtopic in topic.subtopics:
                    for lab in subtopic.labs:
                        for capsule in lab.capsules:
                            assert len(capsule.items) > 0, (
                                f"Capsule '{capsule.capsule_id}' should have items"
                            )

    def test_item_skeleton_fields_populated(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        item = (
            skeleton.domain_modules[0]
            .topics[0]
            .subtopics[0]
            .labs[0]
            .capsules[0]
            .items[0]
        )
        assert isinstance(item, CapsuleItem)
        assert item.item_id == "item_01"
        assert item.item_type == "Multiple Choice"  # From exam question types
        assert item.title == "IAM Policy Structure Overview"
        assert "components" in item.learning_target
        assert item.difficulty == "beginner"

    def test_content_fields_are_none(self):
        """Content fields should be None - filled in Step 5."""
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        item = (
            skeleton.domain_modules[0]
            .topics[0]
            .subtopics[0]
            .labs[0]
            .capsules[0]
            .items[0]
        )
        assert item.content is None
        assert item.options is None
        assert item.correct_answer_index is None
        assert item.explanation is None
        assert item.source_reference is None

    def test_item_types_from_exam_format(self):
        """Item types should come from exam question types."""
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        items = (
            skeleton.domain_modules[0]
            .topics[0]
            .subtopics[0]
            .labs[0]
            .capsules[0]
            .items
        )

        item_types = {i.item_type for i in items}
        # Should use exam question types
        assert "Multiple Choice" in item_types or "Multiple Response" in item_types

    def test_handles_code_fences_in_response(self):
        fenced = f"```json\n{SAMPLE_ITEMS_JSON}\n```"
        ctx, _ = _run_step(
            items_responses={
                "Understanding IAM Policy Syntax": fenced,
                "Writing Your First Policy": fenced,
            }
        )
        skeleton = ctx["course_skeleton"]

        item = (
            skeleton.domain_modules[0]
            .topics[0]
            .subtopics[0]
            .labs[0]
            .capsules[0]
            .items[0]
        )
        assert isinstance(item, CapsuleItem)
        assert item.item_id == "item_01"

    def test_partial_failure_continues(self):
        ctx, _ = _run_step(
            items_errors={
                "Writing Your First Policy": RuntimeError("API timeout"),
            },
        )
        skeleton = ctx["course_skeleton"]

        capsules = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules

        # First capsule should have items
        assert len(capsules[0].items) > 0

        # Second capsule should have no items (failed)
        assert len(capsules[1].items) == 0

        # Failed results should be tracked
        assert len(ctx["failed_capsule_items"]) == 1
        assert ctx["failed_capsule_items"][0].capsule_id == "cap_02"
        assert "API timeout" in ctx["failed_capsule_items"][0].error

    def test_no_google_search_tool(self):
        _, mock_client = _run_step()
        from google.genai import types

        # Items step should NOT use Google Search
        for call in mock_client.models.generate_content.call_args_list:
            config = call.kwargs["config"]
            assert isinstance(config, types.GenerateContentConfig)
            # No tools or empty tools list
            assert config.tools is None or len(config.tools) == 0

    def test_prompt_includes_capsule_context(self):
        _, mock_client = _run_step()

        # Find a call and check prompt content
        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]

        # Should include capsule info
        assert (
            "Understanding IAM Policy Syntax" in contents
            or "Writing Your First Policy" in contents
        )
        assert "conceptual" in contents or "procedural" in contents

    def test_preserves_existing_context(self):
        skeleton = _make_skeleton_with_capsules()
        step = CapsuleItemsSkeletonStep(max_workers=2)
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = _build_side_effect()

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.capsule_items_skeleton.genai.Client",
                return_value=mock_client,
            ),
            patch("pipeline.step4.capsule_items_skeleton.time.sleep"),
        ):
            ctx = PipelineContext(
                certification_name="AWS SAA-C03",
                course_skeleton=skeleton,
                some_key="preserved",
            )
            result = step.run(ctx)

        assert result["some_key"] == "preserved"
        assert "course_skeleton" in result

    def test_empty_capsules_handled(self):
        skeleton = CourseSkeleton(
            certification_name="Test Cert",
            exam_code="TC-100",
            overview=CourseOverview(),
            domain_modules=[
                CourseModule(
                    domain_name="Test Domain",
                    topics=[
                        CourseTopic(
                            name="Test Topic",
                            subtopics=[
                                SubTopic(
                                    name="No Capsules",
                                    labs=[
                                        Lab(
                                            lab_id="lab_01",
                                            title="Empty Lab",
                                            objective="Test",
                                            lab_type="guided",
                                            capsules=[],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        ctx, _ = _run_step(skeleton=skeleton)
        assert "course_skeleton" in ctx
        assert len(ctx.get("failed_capsule_items", [])) == 0

    def test_target_item_count_in_prompt(self):
        _, mock_client = _run_step(target_item_count=7)

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]
        assert "7 practice question skeletons" in contents

    def test_question_types_in_prompt(self):
        _, mock_client = _run_step()

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]
        # Should include question types section
        assert "EXAM QUESTION TYPES" in contents
        assert "Multiple Choice" in contents

    def test_learning_objectives_in_prompt(self):
        _, mock_client = _run_step()

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]
        # Should include learning objectives
        assert "Learning Objectives" in contents
        assert "Configure IAM policies" in contents

    def test_difficulty_variety(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        items = (
            skeleton.domain_modules[0]
            .topics[0]
            .subtopics[0]
            .labs[0]
            .capsules[0]
            .items
        )

        difficulties = {i.difficulty for i in items if i.difficulty}
        # Should have variety in difficulty
        assert len(difficulties) >= 2
