import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    Capsule,
    CourseModule,
    CourseOverview,
    CourseSkeleton,
    CourseTopic,
    Lab,
    LearningObjective,
    QuestionTypeGuide,
    SubTopic,
)
from pipeline.step4.capsules_skeleton import CapsulesSkeletonStep


def _make_skeleton_with_labs() -> CourseSkeleton:
    """Create a minimal skeleton with labs for testing."""
    return CourseSkeleton(
        certification_name="AWS Solutions Architect Associate",
        exam_code="SAA-C03",
        overview=CourseOverview(
            target_audience="Cloud architects",
            course_description="AWS SAA preparation",
        ),
        question_type_guides=[
            QuestionTypeGuide(question_type_name="multiple choice"),
            QuestionTypeGuide(question_type_name="multiple response"),
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
                                relevant_question_types=["multiple choice"],
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
                                        success_criteria=[
                                            "Policy validates",
                                            "Permissions minimal",
                                        ],
                                        real_world_application="Secure access",
                                    ),
                                    Lab(
                                        lab_id="lab_02",
                                        title="Debugging IAM Denials",
                                        objective="Diagnose IAM issues",
                                        lab_type="challenge",
                                        success_criteria=["Fix the policy"],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


SAMPLE_CAPSULES_JSON = json.dumps(
    {
        "capsules": [
            {
                "capsule_id": "cap_01",
                "title": "Understanding IAM Policy Syntax",
                "description": "Learn the JSON structure of IAM policies",
                "learning_goal": "Identify all components of an IAM policy document",
                "capsule_type": "conceptual",
                "estimated_duration_minutes": 15.0,
                "prerequisites_within_lab": [],
                "assessment_criteria": [
                    "Can name all policy elements",
                    "Can explain effect types",
                ],
                "common_errors": [
                    "Confusing Action with Resource",
                    "Missing required fields",
                ],
            },
            {
                "capsule_id": "cap_02",
                "title": "Writing Your First Policy",
                "description": "Hands-on policy creation",
                "learning_goal": "Write a syntactically correct IAM policy",
                "capsule_type": "procedural",
                "estimated_duration_minutes": 20.0,
                "prerequisites_within_lab": ["cap_01"],
                "assessment_criteria": ["Policy validates in simulator"],
                "common_errors": ["Invalid JSON syntax", "Wrong ARN format"],
            },
            {
                "capsule_id": "cap_03",
                "title": "Real-World Policy Examples",
                "description": "Analyze production policies",
                "learning_goal": "Understand common policy patterns",
                "capsule_type": "case_study",
                "estimated_duration_minutes": 25.0,
                "prerequisites_within_lab": ["cap_01"],
                "assessment_criteria": ["Can explain each policy's purpose"],
                "common_errors": [],
            },
            {
                "capsule_id": "cap_04",
                "title": "Policy Practice Exercises",
                "description": "Test your policy skills",
                "learning_goal": "Apply policy knowledge independently",
                "capsule_type": "practice",
                "estimated_duration_minutes": 30.0,
                "prerequisites_within_lab": ["cap_02", "cap_03"],
                "assessment_criteria": ["Complete all exercises correctly"],
                "common_errors": ["Over-permissive policies"],
            },
        ]
    }
)


def _make_mock_response(text: str):
    return SimpleNamespace(text=text)


def _build_side_effect(
    capsules_responses=None,
    capsules_errors=None,
):
    """Build a side_effect function for capsule generation."""
    if capsules_responses is None:
        capsules_responses = {}
    if capsules_errors is None:
        capsules_errors = {}

    def side_effect(*, model, contents, config):
        # Check for specific lab titles in the prompt
        for lab_title, error in capsules_errors.items():
            if lab_title in contents:
                raise error

        for lab_title, resp in capsules_responses.items():
            if lab_title in contents:
                return _make_mock_response(resp)

        # Default response for any lab
        return _make_mock_response(SAMPLE_CAPSULES_JSON)

    return side_effect


def _run_step(
    skeleton=None,
    capsules_responses=None,
    capsules_errors=None,
    max_workers=2,
    target_capsule_count=4,
) -> tuple[PipelineContext, MagicMock]:
    if skeleton is None:
        skeleton = _make_skeleton_with_labs()

    step = CapsulesSkeletonStep(
        max_workers=max_workers, target_capsule_count=target_capsule_count
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _build_side_effect(
        capsules_responses=capsules_responses,
        capsules_errors=capsules_errors,
    )

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch(
            "pipeline.step4.capsules_skeleton.genai.Client",
            return_value=mock_client,
        ),
        patch("pipeline.step4.capsules_skeleton.time.sleep"),
    ):
        ctx = PipelineContext(
            certification_name="AWS SAA-C03",
            course_skeleton=skeleton,
        )
        result = step.run(ctx)
        return result, mock_client


class TestCapsulesSkeletonStep:
    def test_generates_capsules_for_all_labs(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        # Check that all labs have capsules
        for module in skeleton.domain_modules:
            for topic in module.topics:
                for subtopic in topic.subtopics:
                    for lab in subtopic.labs:
                        assert len(lab.capsules) > 0, (
                            f"Lab '{lab.lab_id}' should have capsules"
                        )

    def test_capsule_fields_populated_correctly(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        capsule = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0]
        assert isinstance(capsule, Capsule)
        assert capsule.capsule_id == "cap_01"
        assert capsule.title == "Understanding IAM Policy Syntax"
        assert capsule.learning_goal == "Identify all components of an IAM policy document"
        assert capsule.capsule_type == "conceptual"
        assert capsule.estimated_duration_minutes == 15.0
        assert len(capsule.assessment_criteria) == 2
        assert len(capsule.common_errors) == 2

    def test_capsule_prerequisites_populated(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        capsules = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules
        # cap_02 should have cap_01 as prerequisite
        cap_02 = next((c for c in capsules if c.capsule_id == "cap_02"), None)
        assert cap_02 is not None
        assert "cap_01" in cap_02.prerequisites_within_lab

    def test_handles_code_fences_in_response(self):
        fenced = f"```json\n{SAMPLE_CAPSULES_JSON}\n```"
        ctx, _ = _run_step(
            capsules_responses={
                "Creating Basic IAM Policies": fenced,
                "Debugging IAM Denials": fenced,
            }
        )
        skeleton = ctx["course_skeleton"]

        capsule = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0]
        assert isinstance(capsule, Capsule)
        assert capsule.capsule_id == "cap_01"

    def test_partial_failure_continues(self):
        ctx, _ = _run_step(
            capsules_errors={
                "Debugging IAM Denials": RuntimeError("API timeout"),
            },
        )
        skeleton = ctx["course_skeleton"]

        labs = skeleton.domain_modules[0].topics[0].subtopics[0].labs

        # First lab should have capsules
        assert len(labs[0].capsules) > 0

        # Second lab should have no capsules (failed)
        assert len(labs[1].capsules) == 0

        # Failed results should be tracked
        assert len(ctx["failed_capsules"]) == 1
        assert ctx["failed_capsules"][0].lab_id == "lab_02"
        assert "API timeout" in ctx["failed_capsules"][0].error

    def test_no_google_search_tool(self):
        _, mock_client = _run_step()
        from google.genai import types

        # Capsules step should NOT use Google Search
        for call in mock_client.models.generate_content.call_args_list:
            config = call.kwargs["config"]
            assert isinstance(config, types.GenerateContentConfig)
            # No tools or empty tools list
            assert config.tools is None or len(config.tools) == 0

    def test_prompt_includes_lab_context(self):
        _, mock_client = _run_step()

        # Find a call and check prompt content
        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]

        # Should include lab info
        assert "Creating Basic IAM Policies" in contents or "Debugging IAM Denials" in contents
        assert "guided" in contents or "challenge" in contents

    def test_preserves_existing_context(self):
        skeleton = _make_skeleton_with_labs()
        step = CapsulesSkeletonStep(max_workers=2)
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = _build_side_effect()

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.capsules_skeleton.genai.Client",
                return_value=mock_client,
            ),
            patch("pipeline.step4.capsules_skeleton.time.sleep"),
        ):
            ctx = PipelineContext(
                certification_name="AWS SAA-C03",
                course_skeleton=skeleton,
                some_key="preserved",
            )
            result = step.run(ctx)

        assert result["some_key"] == "preserved"
        assert "course_skeleton" in result

    def test_empty_labs_handled(self):
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
                                SubTopic(name="No Labs", labs=[]),
                            ],
                        ),
                    ],
                ),
            ],
        )
        ctx, _ = _run_step(skeleton=skeleton)
        assert "course_skeleton" in ctx
        assert len(ctx.get("failed_capsules", [])) == 0

    def test_target_capsule_count_in_prompt(self):
        _, mock_client = _run_step(target_capsule_count=6)

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]
        assert "6 capsules" in contents

    def test_question_types_in_prompt(self):
        _, mock_client = _run_step()

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]
        # Should include question types section
        assert "EXAM QUESTION TYPES" in contents
        assert "multiple choice" in contents

    def test_learning_objectives_in_prompt(self):
        _, mock_client = _run_step()

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]
        # Should include learning objectives
        assert "Learning Objectives" in contents
        assert "Configure IAM policies" in contents
