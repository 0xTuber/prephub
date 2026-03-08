import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    CourseModule,
    CourseOverview,
    CourseSkeleton,
    CourseTopic,
    Lab,
    LearningObjective,
    QuestionTypeGuide,
    SubTopic,
)
from pipeline.step4.labs_skeleton import LabsSkeletonStep


def _make_skeleton_with_subtopics() -> CourseSkeleton:
    """Create a minimal skeleton with subtopics for testing."""
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
                                key_concepts=["Principal", "Action", "Resource"],
                                practical_skills=["Write IAM policies"],
                                common_misconceptions=["Deny always overrides allow"],
                            ),
                            SubTopic(
                                name="Permission Boundaries",
                                description="Delegating permissions safely",
                                key_concepts=["Boundary policies"],
                                practical_skills=["Configure boundaries"],
                                common_misconceptions=[],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


SAMPLE_LABS_JSON = json.dumps(
    {
        "labs": [
            {
                "lab_id": "lab_01",
                "title": "Creating Basic IAM Policies",
                "description": "Learn to write IAM policies from scratch",
                "objective": "Create a custom IAM policy with least privilege",
                "lab_type": "guided",
                "estimated_duration_minutes": 30.0,
                "tools_required": ["AWS Console", "IAM Policy Simulator"],
                "prerequisites_within_subtopic": [],
                "success_criteria": [
                    "Policy validates successfully",
                    "Permissions are minimal",
                ],
                "real_world_application": "Secure resource access in production",
            },
            {
                "lab_id": "lab_02",
                "title": "Debugging IAM Denials",
                "description": "Troubleshoot permission issues",
                "objective": "Diagnose and fix IAM permission denials",
                "lab_type": "challenge",
                "estimated_duration_minutes": 45.0,
                "tools_required": ["CloudTrail", "IAM Access Analyzer"],
                "prerequisites_within_subtopic": ["lab_01"],
                "success_criteria": ["Identify root cause", "Fix the policy"],
                "real_world_application": "Production incident response",
            },
            {
                "lab_id": "lab_03",
                "title": "IAM Policy Best Practices",
                "description": "Apply security best practices",
                "objective": "Review and improve existing policies",
                "lab_type": "exploratory",
                "estimated_duration_minutes": 40.0,
                "tools_required": ["IAM Access Analyzer"],
                "prerequisites_within_subtopic": ["lab_01", "lab_02"],
                "success_criteria": ["All policies follow best practices"],
                "real_world_application": "Security audits",
            },
        ]
    }
)


def _make_mock_response(text: str):
    return SimpleNamespace(text=text)


def _build_side_effect(
    labs_responses=None,
    labs_errors=None,
):
    """Build a side_effect function for lab generation."""
    if labs_responses is None:
        labs_responses = {}
    if labs_errors is None:
        labs_errors = {}

    def side_effect(*, model, contents, config):
        # Check for specific subtopic names in the prompt
        for subtopic_name, error in labs_errors.items():
            if subtopic_name in contents:
                raise error

        for subtopic_name, resp in labs_responses.items():
            if subtopic_name in contents:
                return _make_mock_response(resp)

        # Default response for any subtopic
        return _make_mock_response(SAMPLE_LABS_JSON)

    return side_effect


def _run_step(
    skeleton=None,
    labs_responses=None,
    labs_errors=None,
    max_workers=2,
    target_lab_count=3,
) -> tuple[PipelineContext, MagicMock]:
    if skeleton is None:
        skeleton = _make_skeleton_with_subtopics()

    step = LabsSkeletonStep(
        max_workers=max_workers, target_lab_count=target_lab_count
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _build_side_effect(
        labs_responses=labs_responses,
        labs_errors=labs_errors,
    )

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch(
            "pipeline.step4.labs_skeleton.genai.Client",
            return_value=mock_client,
        ),
        patch("pipeline.step4.labs_skeleton.time.sleep"),
    ):
        ctx = PipelineContext(
            certification_name="AWS SAA-C03",
            course_skeleton=skeleton,
        )
        result = step.run(ctx)
        return result, mock_client


class TestLabsSkeletonStep:
    def test_generates_labs_for_all_subtopics(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        # Check that all subtopics have labs
        for module in skeleton.domain_modules:
            for topic in module.topics:
                for subtopic in topic.subtopics:
                    assert len(subtopic.labs) > 0, (
                        f"Subtopic '{subtopic.name}' should have labs"
                    )

    def test_lab_fields_populated_correctly(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        lab = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0]
        assert isinstance(lab, Lab)
        assert lab.lab_id == "lab_01"
        assert lab.title == "Creating Basic IAM Policies"
        assert lab.objective == "Create a custom IAM policy with least privilege"
        assert lab.lab_type == "guided"
        assert lab.estimated_duration_minutes == 30.0
        assert "AWS Console" in lab.tools_required
        assert len(lab.success_criteria) == 2
        assert lab.real_world_application == "Secure resource access in production"

    def test_lab_prerequisites_populated(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]

        labs = skeleton.domain_modules[0].topics[0].subtopics[0].labs
        # lab_02 should have lab_01 as prerequisite
        lab_02 = next((l for l in labs if l.lab_id == "lab_02"), None)
        assert lab_02 is not None
        assert "lab_01" in lab_02.prerequisites_within_subtopic

    def test_handles_code_fences_in_response(self):
        fenced = f"```json\n{SAMPLE_LABS_JSON}\n```"
        ctx, _ = _run_step(
            labs_responses={"Policy Structure": fenced, "Permission Boundaries": fenced}
        )
        skeleton = ctx["course_skeleton"]

        lab = skeleton.domain_modules[0].topics[0].subtopics[0].labs[0]
        assert isinstance(lab, Lab)
        assert lab.lab_id == "lab_01"

    def test_partial_failure_continues(self):
        ctx, _ = _run_step(
            labs_errors={
                "Permission Boundaries": RuntimeError("API timeout"),
            },
        )
        skeleton = ctx["course_skeleton"]

        # First subtopic should have labs
        first_subtopic = skeleton.domain_modules[0].topics[0].subtopics[0]
        assert len(first_subtopic.labs) > 0

        # Second subtopic should have no labs (failed)
        second_subtopic = skeleton.domain_modules[0].topics[0].subtopics[1]
        assert len(second_subtopic.labs) == 0

        # Failed results should be tracked
        assert len(ctx["failed_labs"]) == 1
        assert ctx["failed_labs"][0].subtopic_name == "Permission Boundaries"
        assert "API timeout" in ctx["failed_labs"][0].error

    def test_uses_google_search_tool(self):
        _, mock_client = _run_step()
        from google.genai import types

        # Check that at least one call used Google Search
        for call in mock_client.models.generate_content.call_args_list:
            config = call.kwargs["config"]
            assert isinstance(config, types.GenerateContentConfig)
            assert len(config.tools) == 1
            assert isinstance(config.tools[0].google_search, types.GoogleSearch)
            return

        pytest.fail("No calls found with Google Search tool")

    def test_prompt_includes_context(self):
        _, mock_client = _run_step()

        # Find a call and check prompt content
        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]

        # Should include certification, domain, topic, subtopic info
        assert "AWS Solutions Architect Associate" in contents
        assert "Design Secure Architectures" in contents
        assert "IAM Policies" in contents
        # Should include key concepts
        assert "Principal" in contents or "Action" in contents

    def test_preserves_existing_context(self):
        skeleton = _make_skeleton_with_subtopics()
        step = LabsSkeletonStep(max_workers=2)
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = _build_side_effect()

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.labs_skeleton.genai.Client",
                return_value=mock_client,
            ),
            patch("pipeline.step4.labs_skeleton.time.sleep"),
        ):
            ctx = PipelineContext(
                certification_name="AWS SAA-C03",
                course_skeleton=skeleton,
                some_key="preserved",
            )
            result = step.run(ctx)

        assert result["some_key"] == "preserved"
        assert "course_skeleton" in result

    def test_empty_subtopics_handled(self):
        skeleton = CourseSkeleton(
            certification_name="Test Cert",
            exam_code="TC-100",
            overview=CourseOverview(),
            domain_modules=[
                CourseModule(
                    domain_name="Test Domain",
                    topics=[
                        CourseTopic(name="Empty Topic", subtopics=[]),
                    ],
                ),
            ],
        )
        ctx, _ = _run_step(skeleton=skeleton)
        assert "course_skeleton" in ctx
        assert len(ctx.get("failed_labs", [])) == 0

    def test_target_lab_count_in_prompt(self):
        _, mock_client = _run_step(target_lab_count=5)

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]
        assert "5 labs" in contents

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
        assert "Apply" in contents
