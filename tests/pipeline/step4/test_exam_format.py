import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import ExamFormat, QuestionType
from pipeline.step4.exam_format import ExamFormatStep


SAMPLE_RAW_DESCRIPTION = (
    "The AWS SAA-C03 exam has 65 questions and a 130-minute time limit. "
    "It covers 4 domains including Design Secure Architectures (30%)."
)

SAMPLE_STRUCTURED_JSON = json.dumps(
    {
        "certification_name": "AWS Solutions Architect Associate",
        "exam_code": "SAA-C03",
        "num_questions": 65,
        "time_limit_minutes": 130,
        "question_types": [
            {
                "name": "multiple choice",
                "description": "Single correct answer from four options",
                "purpose": "Tests recall and conceptual understanding of AWS services",
                "skeleton": "Stem: <scenario or question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>",
                "example": "A company needs a storage solution for infrequently accessed data. Which S3 storage class is MOST cost-effective?\nA) S3 Standard\nB) S3 Intelligent-Tiering\nC) S3 Glacier Instant Retrieval\nD) S3 One Zone-IA",
                "grading_notes": "All-or-nothing; one correct answer",
            },
            {
                "name": "multiple response",
                "description": "Two or more correct answers from five or more options",
                "purpose": "Tests ability to identify multiple relevant factors in a scenario",
                "skeleton": "Stem: <scenario> (Select TWO)\nA) <option>\nB) <option>\nC) <option>\nD) <option>\nE) <option>",
                "example": "Which TWO actions improve the security of an AWS account root user? (Select TWO)\nA) Enable MFA\nB) Create access keys\nC) Use a strong password\nD) Share credentials with admins\nE) Disable CloudTrail",
                "grading_notes": "All-or-nothing; must select all correct answers",
            },
        ],
        "passing_score": "720 out of 1000",
        "domains": [
            {
                "name": "Design Secure Architectures",
                "weight_pct": 30.0,
                "description": "Security controls and strategies",
            },
            {
                "name": "Design Resilient Architectures",
                "weight_pct": 26.0,
                "description": None,
            },
        ],
        "prerequisites": [],
        "cost_usd": "150",
        "validity_years": 3,
        "delivery_methods": ["Pearson VUE", "Online proctored"],
        "languages": ["English", "Japanese"],
        "recertification_policy": "Recertify before expiration",
        "additional_notes": None,
    }
)


def _make_mock_response(text: str):
    return SimpleNamespace(text=text)


class TestExamFormatStep:
    def _build_mock_client(self, phase1_text: str, phase2_text: str):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_mock_response(phase1_text),
            _make_mock_response(phase2_text),
        ]
        return mock_client

    def _run_step(
        self, phase1_text: str = SAMPLE_RAW_DESCRIPTION, phase2_text: str = SAMPLE_STRUCTURED_JSON
    ) -> PipelineContext:
        step = ExamFormatStep()
        mock_client = self._build_mock_client(phase1_text, phase2_text)

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.exam_format.genai.Client",
                return_value=mock_client,
            ),
        ):
            ctx = PipelineContext(certification_name="AWS SAA-C03")
            return step.run(ctx)

    def test_produces_exam_format(self):
        ctx = self._run_step()
        exam_format = ctx["exam_format"]
        assert isinstance(exam_format, ExamFormat)
        assert exam_format.certification_name == "AWS Solutions Architect Associate"
        assert exam_format.exam_code == "SAA-C03"
        assert exam_format.num_questions == 65
        assert exam_format.time_limit_minutes == 130
        assert len(exam_format.domains) == 2
        assert exam_format.domains[0].name == "Design Secure Architectures"
        assert exam_format.domains[0].weight_pct == 30.0
        assert len(exam_format.question_types) == 2

    def test_question_type_details(self):
        ctx = self._run_step()
        exam_format = ctx["exam_format"]
        qt = exam_format.question_types[0]
        assert isinstance(qt, QuestionType)
        assert qt.name == "multiple choice"
        assert qt.description is not None
        assert qt.purpose is not None
        assert qt.skeleton is not None
        assert "Stem" in qt.skeleton
        assert qt.example is not None
        assert qt.grading_notes is not None

    def test_question_type_string_fallback(self):
        """Phase 2 returns question_types as plain strings (old format)."""
        old_format = json.dumps({
            "certification_name": "Some Cert",
            "exam_code": "SC-100",
            "num_questions": 50,
            "time_limit_minutes": 90,
            "question_types": ["multiple choice", "drag-and-drop"],
            "passing_score": "700",
            "domains": [],
            "prerequisites": [],
            "cost_usd": None,
            "validity_years": None,
            "delivery_methods": [],
            "languages": [],
            "recertification_policy": None,
            "additional_notes": None,
        })
        ctx = self._run_step(phase2_text=old_format)
        exam_format = ctx["exam_format"]
        assert len(exam_format.question_types) == 2
        assert exam_format.question_types[0].name == "multiple choice"
        assert exam_format.question_types[1].name == "drag-and-drop"
        assert exam_format.question_types[0].skeleton is None

    def test_stores_raw_description(self):
        ctx = self._run_step()
        assert ctx["exam_format_raw"] == SAMPLE_RAW_DESCRIPTION
        assert ctx["exam_format"].raw_description == SAMPLE_RAW_DESCRIPTION

    def test_handles_json_with_code_fences(self):
        fenced = f"```json\n{SAMPLE_STRUCTURED_JSON}\n```"
        ctx = self._run_step(phase2_text=fenced)
        exam_format = ctx["exam_format"]
        assert exam_format.exam_code == "SAA-C03"
        assert exam_format.num_questions == 65

    def test_graceful_degradation_on_bad_json(self):
        ctx = self._run_step(phase2_text="not valid json at all")
        exam_format = ctx["exam_format"]
        assert isinstance(exam_format, ExamFormat)
        assert exam_format.certification_name == "AWS SAA-C03"
        assert exam_format.raw_description == SAMPLE_RAW_DESCRIPTION
        assert exam_format.exam_code is None
        assert exam_format.num_questions is None

    def test_phase1_failure_raises(self):
        step = ExamFormatStep()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.exam_format.genai.Client",
                return_value=mock_client,
            ),
        ):
            ctx = PipelineContext(certification_name="AWS SAA-C03")
            with pytest.raises(RuntimeError):
                step.run(ctx)

    def test_passes_certification_in_prompt(self):
        step = ExamFormatStep()
        mock_client = self._build_mock_client(
            SAMPLE_RAW_DESCRIPTION, SAMPLE_STRUCTURED_JSON
        )

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.exam_format.genai.Client",
                return_value=mock_client,
            ),
        ):
            ctx = PipelineContext(certification_name="CKA Kubernetes")
            step.run(ctx)

        # Phase 1 is the first call
        phase1_call = mock_client.models.generate_content.call_args_list[0]
        prompt_content = phase1_call.kwargs["contents"]
        assert "CKA Kubernetes" in prompt_content

    def test_uses_google_search_tool_in_phase1(self):
        step = ExamFormatStep()
        mock_client = self._build_mock_client(
            SAMPLE_RAW_DESCRIPTION, SAMPLE_STRUCTURED_JSON
        )

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.exam_format.genai.Client",
                return_value=mock_client,
            ),
        ):
            ctx = PipelineContext(certification_name="AWS SAA-C03")
            step.run(ctx)

        from google.genai import types

        phase1_call = mock_client.models.generate_content.call_args_list[0]
        config = phase1_call.kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert len(config.tools) == 1
        assert isinstance(config.tools[0].google_search, types.GoogleSearch)

    def test_phase2_no_search_tool(self):
        step = ExamFormatStep()
        mock_client = self._build_mock_client(
            SAMPLE_RAW_DESCRIPTION, SAMPLE_STRUCTURED_JSON
        )

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch(
                "pipeline.step4.exam_format.genai.Client",
                return_value=mock_client,
            ),
        ):
            ctx = PipelineContext(certification_name="AWS SAA-C03")
            step.run(ctx)

        phase2_call = mock_client.models.generate_content.call_args_list[1]
        config = phase2_call.kwargs["config"]
        assert isinstance(config, dict)
        assert "tools" not in config
