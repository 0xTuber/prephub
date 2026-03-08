import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    CourseModule,
    CourseOverview,
    CourseSkeleton,
    ExamDomain,
    ExamFormat,
    QuestionType,
    QuestionTypeGuide,
)
from pipeline.step4.course_skeleton import CourseSkeletonStep


SAMPLE_EXAM_FORMAT = ExamFormat(
    certification_name="AWS Solutions Architect Associate",
    exam_code="SAA-C03",
    num_questions=65,
    time_limit_minutes=130,
    question_types=[
        QuestionType(
            name="multiple choice",
            description="Single correct answer from four options",
        ),
        QuestionType(
            name="multiple response",
            description="Two or more correct answers from five or more options",
        ),
    ],
    passing_score="720 out of 1000",
    domains=[
        ExamDomain(name="Design Secure Architectures", weight_pct=30.0),
        ExamDomain(name="Design Resilient Architectures", weight_pct=26.0),
    ],
    prerequisites=["Basic AWS knowledge"],
    raw_description="Sample raw description",
)

SAMPLE_SCAFFOLD_JSON = json.dumps(
    {
        "overview": {
            "target_audience": "Cloud architects and engineers",
            "course_description": "Comprehensive AWS SAA-C03 preparation",
            "total_estimated_study_hours": 80.0,
            "study_strategies": [
                {
                    "name": "Active Recall",
                    "description": "Test yourself regularly",
                    "when_to_use": "After each study session",
                }
            ],
            "exam_day_tips": ["Read questions carefully", "Manage your time"],
            "prerequisites_detail": ["Basic networking", "AWS free tier experience"],
        },
        "question_type_guides": [
            {
                "question_type_name": "multiple choice",
                "detailed_structure": "Single stem with 4 options, one correct",
                "reasoning_template": {
                    "approach_steps": [
                        "Read the scenario carefully",
                        "Identify the key requirement",
                        "Eliminate wrong answers",
                    ],
                    "time_allocation_advice": "90 seconds per question",
                    "common_traps": ["Distractor options that are partially correct"],
                },
                "explanation_template": {
                    "correct_answer_template": "This is correct because...",
                    "wrong_answer_template": "This is incorrect because...",
                    "partial_credit_template": None,
                },
                "difficulty_scaling_notes": "Harder questions have more nuanced scenarios",
                "answer_choice_design_notes": "Distractors should be plausible",
            },
            {
                "question_type_name": "multiple response",
                "detailed_structure": "Stem with 5+ options, 2+ correct",
                "reasoning_template": {
                    "approach_steps": [
                        "Count how many answers are needed",
                        "Evaluate each option independently",
                    ],
                    "time_allocation_advice": "120 seconds per question",
                    "common_traps": ["Selecting too many or too few options"],
                },
                "explanation_template": {
                    "correct_answer_template": "These options are correct because...",
                    "wrong_answer_template": "This option is wrong because...",
                    "partial_credit_template": None,
                },
                "difficulty_scaling_notes": "More options increase difficulty",
                "answer_choice_design_notes": "All options should be plausible",
            },
        ],
    }
)

SAMPLE_DOMAIN_SECURE_JSON = json.dumps(
    {
        "domain_name": "Design Secure Architectures",
        "domain_weight_pct": 30.0,
        "overview": "Covers security best practices for AWS architectures",
        "topics": [
            {
                "name": "IAM Policies",
                "description": "Identity and Access Management",
                "learning_objectives": [
                    {
                        "objective": "Configure IAM policies",
                        "bloom_level": "Apply",
                        "relevant_question_types": ["multiple choice"],
                    }
                ],
                "subtopics": [
                    {
                        "name": "Policy Structure",
                        "description": "JSON policy documents",
                        "key_concepts": ["Principal", "Action", "Resource"],
                        "practical_skills": ["Write IAM policies"],
                        "common_misconceptions": [
                            "Deny always overrides allow"
                        ],
                    }
                ],
                "estimated_study_hours": 4.0,
            }
        ],
        "prerequisites_for_domain": ["Basic AWS IAM"],
        "recommended_study_order": ["IAM Policies"],
        "official_references": ["https://docs.aws.amazon.com/IAM/"],
    }
)

SAMPLE_DOMAIN_RESILIENT_JSON = json.dumps(
    {
        "domain_name": "Design Resilient Architectures",
        "domain_weight_pct": 26.0,
        "overview": "Covers high-availability and fault-tolerant designs",
        "topics": [
            {
                "name": "Multi-AZ Deployments",
                "description": "Deploying across availability zones",
                "learning_objectives": [
                    {
                        "objective": "Design multi-AZ architectures",
                        "bloom_level": "Analyze",
                        "relevant_question_types": ["multiple choice"],
                    }
                ],
                "subtopics": [],
                "estimated_study_hours": 3.0,
            }
        ],
        "prerequisites_for_domain": [],
        "recommended_study_order": ["Multi-AZ Deployments"],
        "official_references": [],
    }
)


def _make_mock_response(text: str):
    return SimpleNamespace(text=text)


def _build_side_effect(
    scaffold_response=SAMPLE_SCAFFOLD_JSON,
    domain_responses=None,
    scaffold_error=None,
    domain_errors=None,
):
    """Build a side_effect function that dispatches by prompt content."""
    if domain_responses is None:
        domain_responses = {
            "Design Secure Architectures": SAMPLE_DOMAIN_SECURE_JSON,
            "Design Resilient Architectures": SAMPLE_DOMAIN_RESILIENT_JSON,
        }
    if domain_errors is None:
        domain_errors = {}

    def side_effect(*, model, contents, config):
        if "scaffold" in contents.lower() or "cross-cutting" in contents.lower():
            if scaffold_error:
                raise scaffold_error
            return _make_mock_response(scaffold_response)
        for domain_name, resp in domain_responses.items():
            if domain_name in contents:
                if domain_name in domain_errors:
                    raise domain_errors[domain_name]
                return _make_mock_response(resp)
        raise ValueError(f"Unexpected prompt content: {contents[:100]}")

    return side_effect


def _run_step(
    exam_format=SAMPLE_EXAM_FORMAT,
    scaffold_response=SAMPLE_SCAFFOLD_JSON,
    domain_responses=None,
    scaffold_error=None,
    domain_errors=None,
    max_workers=2,
) -> tuple[PipelineContext, MagicMock]:
    step = CourseSkeletonStep(max_workers=max_workers)
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _build_side_effect(
        scaffold_response=scaffold_response,
        domain_responses=domain_responses,
        scaffold_error=scaffold_error,
        domain_errors=domain_errors,
    )

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch(
            "pipeline.step4.course_skeleton.genai.Client",
            return_value=mock_client,
        ),
        patch("pipeline.step4.course_skeleton.time.sleep"),
    ):
        ctx = PipelineContext(
            certification_name="AWS SAA-C03",
            exam_format=exam_format,
        )
        result = step.run(ctx)
        return result, mock_client


class TestCourseSkeletonStep:
    def test_scaffold_produces_overview_and_guides(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]
        assert isinstance(skeleton.overview, CourseOverview)
        assert skeleton.overview.target_audience == "Cloud architects and engineers"
        assert skeleton.overview.total_estimated_study_hours == 80.0
        assert len(skeleton.overview.study_strategies) == 1
        assert len(skeleton.overview.exam_day_tips) == 2

    def test_scaffold_includes_all_question_types(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]
        guides = skeleton.question_type_guides
        assert len(guides) == 2
        guide_names = {g.question_type_name for g in guides}
        assert guide_names == {"multiple choice", "multiple response"}

    def test_scaffold_json_with_code_fences(self):
        fenced = f"```json\n{SAMPLE_SCAFFOLD_JSON}\n```"
        ctx, _ = _run_step(scaffold_response=fenced)
        skeleton = ctx["course_skeleton"]
        assert isinstance(skeleton.overview, CourseOverview)
        assert len(skeleton.question_type_guides) == 2

    def test_scaffold_failure_raises(self):
        with pytest.raises(RuntimeError, match="Scaffold generation failed"):
            _run_step(scaffold_error=RuntimeError("API error"))

    def test_scaffold_no_search_tool(self):
        _, mock_client = _run_step()
        from google.genai import types

        # Find the scaffold call (contains "scaffold" or "cross-cutting" in prompt)
        for call in mock_client.models.generate_content.call_args_list:
            contents = call.kwargs["contents"]
            if "scaffold" in contents.lower() or "cross-cutting" in contents.lower():
                config = call.kwargs["config"]
                assert isinstance(config, types.GenerateContentConfig)
                # No tools means no Google Search
                assert config.tools is None or len(config.tools) == 0
                return
        pytest.fail("Scaffold call not found")

    def test_domain_module_happy_path(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]
        assert len(skeleton.domain_modules) == 2
        secure_module = skeleton.domain_modules[0]
        assert isinstance(secure_module, CourseModule)
        assert secure_module.domain_name == "Design Secure Architectures"
        assert len(secure_module.topics) == 1
        assert secure_module.topics[0].name == "IAM Policies"

    def test_domain_module_uses_google_search(self):
        _, mock_client = _run_step()
        from google.genai import types

        # Find a domain call (contains "topic breakdown" — distinct from scaffold)
        for call in mock_client.models.generate_content.call_args_list:
            contents = call.kwargs["contents"]
            if "topic breakdown" in contents.lower() and "Design Secure Architectures" in contents:
                config = call.kwargs["config"]
                assert isinstance(config, types.GenerateContentConfig)
                assert len(config.tools) == 1
                assert isinstance(config.tools[0].google_search, types.GoogleSearch)
                return
        pytest.fail("Domain call not found")

    def test_domain_passes_cert_and_domain_in_prompt(self):
        _, mock_client = _run_step()
        found_secure = False
        found_resilient = False
        for call in mock_client.models.generate_content.call_args_list:
            contents = call.kwargs["contents"]
            if "Design Secure Architectures" in contents and "AWS Solutions Architect Associate" in contents:
                found_secure = True
            if "Design Resilient Architectures" in contents and "AWS Solutions Architect Associate" in contents:
                found_resilient = True
        assert found_secure, "Secure domain call not found with cert name"
        assert found_resilient, "Resilient domain call not found with cert name"

    def test_fan_out_all_succeed(self):
        ctx, _ = _run_step()
        skeleton = ctx["course_skeleton"]
        assert len(skeleton.domain_modules) == 2
        assert len(skeleton.failed_domains) == 0
        # Check order matches original domain order
        assert skeleton.domain_modules[0].domain_name == "Design Secure Architectures"
        assert skeleton.domain_modules[1].domain_name == "Design Resilient Architectures"

    def test_fan_out_partial_failure(self):
        ctx, _ = _run_step(
            domain_errors={
                "Design Resilient Architectures": RuntimeError("Search timeout"),
            },
        )
        skeleton = ctx["course_skeleton"]
        assert len(skeleton.domain_modules) == 1
        assert skeleton.domain_modules[0].domain_name == "Design Secure Architectures"
        assert len(skeleton.failed_domains) == 1
        assert skeleton.failed_domains[0].domain_name == "Design Resilient Architectures"
        assert skeleton.failed_domains[0].success is False
        assert "Search timeout" in skeleton.failed_domains[0].error

    def test_step_stores_skeleton_in_context(self):
        ctx, _ = _run_step()
        assert "course_skeleton" in ctx
        skeleton = ctx["course_skeleton"]
        assert isinstance(skeleton, CourseSkeleton)
        assert skeleton.certification_name == "AWS Solutions Architect Associate"
        assert skeleton.exam_code == "SAA-C03"

    def test_step_with_empty_domains(self):
        empty_format = ExamFormat(
            certification_name="Test Cert",
            exam_code="TC-100",
            question_types=[
                QuestionType(name="multiple choice", description="Basic MC"),
            ],
            domains=[],
            raw_description="Test",
        )
        ctx, _ = _run_step(
            exam_format=empty_format,
            domain_responses={},
        )
        skeleton = ctx["course_skeleton"]
        assert len(skeleton.domain_modules) == 0
        assert len(skeleton.failed_domains) == 0
        # Scaffold still works
        assert isinstance(skeleton.overview, CourseOverview)
        assert len(skeleton.question_type_guides) > 0

    def test_step_preserves_existing_context(self):
        step = CourseSkeletonStep(max_workers=2)
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = _build_side_effect()

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.course_skeleton.genai.Client",
                return_value=mock_client,
            ),
            patch("pipeline.step4.course_skeleton.time.sleep"),
        ):
            ctx = PipelineContext(
                certification_name="AWS SAA-C03",
                exam_format=SAMPLE_EXAM_FORMAT,
                books_requested=["book1"],
                some_key="preserved",
            )
            result = step.run(ctx)

        assert result["books_requested"] == ["book1"]
        assert result["some_key"] == "preserved"
        assert "course_skeleton" in result
