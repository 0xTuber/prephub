import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pipeline.base import PipelineContext
from pipeline.models import CourseSkeleton, ExamFormat
from pipeline.step4.pipeline import build_step4_pipeline


SAMPLE_RAW = "The AWS SAA-C03 exam has 65 questions."

SAMPLE_JSON = json.dumps(
    {
        "certification_name": "AWS Solutions Architect Associate",
        "exam_code": "SAA-C03",
        "num_questions": 65,
        "time_limit_minutes": 130,
        "question_types": [
            {
                "name": "multiple choice",
                "description": "Single correct answer from four options",
                "purpose": "Tests conceptual understanding",
                "skeleton": "Stem: <question>\nA) ...\nB) ...\nC) ...\nD) ...",
                "example": "Which service provides DNS? A) Route 53 B) CloudFront C) ELB D) S3",
                "grading_notes": "All-or-nothing",
            }
        ],
        "passing_score": "720",
        "domains": [{"name": "Security", "weight_pct": 30.0, "description": None}],
        "prerequisites": [],
        "cost_usd": "150",
        "validity_years": 3,
        "delivery_methods": ["Pearson VUE"],
        "languages": ["English"],
        "recertification_policy": None,
        "additional_notes": None,
    }
)

SAMPLE_SCAFFOLD_JSON = json.dumps(
    {
        "overview": {
            "target_audience": "Cloud architects",
            "course_description": "AWS SAA-C03 prep",
            "total_estimated_study_hours": 60.0,
            "study_strategies": [
                {
                    "name": "Practice Tests",
                    "description": "Take timed practice exams",
                    "when_to_use": "Final weeks before exam",
                }
            ],
            "exam_day_tips": ["Arrive early"],
            "prerequisites_detail": ["AWS basics"],
        },
        "question_type_guides": [
            {
                "question_type_name": "multiple choice",
                "detailed_structure": "4 options, 1 correct",
                "reasoning_template": {
                    "approach_steps": ["Read carefully", "Eliminate wrong answers"],
                    "time_allocation_advice": "90 seconds",
                    "common_traps": ["Partially correct options"],
                },
                "explanation_template": {
                    "correct_answer_template": "Correct because...",
                    "wrong_answer_template": "Wrong because...",
                    "partial_credit_template": None,
                },
                "difficulty_scaling_notes": "More complex scenarios",
                "answer_choice_design_notes": "Plausible distractors",
            }
        ],
    }
)

SAMPLE_DOMAIN_JSON = json.dumps(
    {
        "domain_name": "Security",
        "domain_weight_pct": 30.0,
        "overview": "Security fundamentals",
        "topics": [
            {
                "name": "IAM",
                "description": "Identity management",
                "learning_objectives": [
                    {
                        "objective": "Configure IAM",
                        "bloom_level": "Apply",
                        "relevant_question_types": ["multiple choice"],
                    }
                ],
                "subtopics": [],
                "estimated_study_hours": 3.0,
            }
        ],
        "prerequisites_for_domain": [],
        "recommended_study_order": ["IAM"],
        "official_references": [],
    }
)


def _make_mock_response(text: str):
    return SimpleNamespace(text=text)


def _build_integration_side_effect():
    """Side effect that handles both ExamFormatStep and CourseSkeletonStep calls."""
    call_count = {"value": 0}

    def side_effect(*, model, contents, config):
        # ExamFormatStep Phase 1: has GoogleSearch tool and asks about exam format
        # ExamFormatStep Phase 2: dict config, asks to extract JSON
        # CourseSkeletonStep scaffold: has "scaffold" or "cross-cutting" in prompt
        # CourseSkeletonStep domain: has domain name in prompt

        # Check if this is a CourseSkeletonStep call
        if "scaffold" in contents.lower() or "cross-cutting" in contents.lower():
            return _make_mock_response(SAMPLE_SCAFFOLD_JSON)
        if "Security" in contents and "topic breakdown" in contents.lower():
            return _make_mock_response(SAMPLE_DOMAIN_JSON)

        # ExamFormatStep calls are sequential: Phase 1 then Phase 2
        call_count["value"] += 1
        if call_count["value"] == 1:
            return _make_mock_response(SAMPLE_RAW)
        return _make_mock_response(SAMPLE_JSON)

    return side_effect


def _run_pipeline(context: PipelineContext | None = None) -> PipelineContext:
    pipeline = build_step4_pipeline(max_workers=2)
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _build_integration_side_effect()

    if context is None:
        context = PipelineContext(certification_name="AWS SAA-C03")

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch(
            "pipeline.step4.exam_format.genai.Client",
            return_value=mock_client,
        ),
        patch(
            "pipeline.step4.course_skeleton.genai.Client",
            return_value=mock_client,
        ),
        patch("pipeline.step4.course_skeleton.time.sleep"),
    ):
        return pipeline.run(context)


class TestStep4Integration:
    def test_full_pipeline(self):
        ctx = _run_pipeline()
        exam_format = ctx["exam_format"]
        assert isinstance(exam_format, ExamFormat)
        assert exam_format.exam_code == "SAA-C03"
        assert exam_format.num_questions == 65
        assert exam_format.time_limit_minutes == 130
        assert len(exam_format.domains) == 1
        assert len(exam_format.question_types) == 1
        assert exam_format.question_types[0].name == "multiple choice"
        assert exam_format.question_types[0].skeleton is not None

    def test_context_keys(self):
        ctx = _run_pipeline()
        assert "exam_format" in ctx
        assert "exam_format_raw" in ctx
        assert isinstance(ctx["exam_format"], ExamFormat)
        assert isinstance(ctx["exam_format_raw"], str)

    def test_pipeline_preserves_prior_context(self):
        context = PipelineContext(
            certification_name="AWS SAA-C03",
            books_requested=["book1", "book2"],
            some_other_key="preserved_value",
        )
        ctx = _run_pipeline(context)
        assert ctx["books_requested"] == ["book1", "book2"]
        assert ctx["some_other_key"] == "preserved_value"
        assert "exam_format" in ctx

    def test_full_pipeline_with_skeleton(self):
        ctx = _run_pipeline()
        skeleton = ctx["course_skeleton"]
        assert isinstance(skeleton, CourseSkeleton)
        assert skeleton.certification_name == "AWS Solutions Architect Associate"
        assert skeleton.exam_code == "SAA-C03"
        assert isinstance(skeleton.overview.target_audience, str)
        assert len(skeleton.question_type_guides) >= 1
        assert len(skeleton.domain_modules) >= 1
        assert skeleton.domain_modules[0].domain_name == "Security"

    def test_skeleton_context_keys(self):
        ctx = _run_pipeline()
        assert "exam_format" in ctx
        assert "course_skeleton" in ctx
        assert isinstance(ctx["exam_format"], ExamFormat)
        assert isinstance(ctx["course_skeleton"], CourseSkeleton)
