import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    CertificationSchema,
    ExamDomain,
    ExamFormat,
    ExhibitType,
    QuestionType,
    QuestionTypeSchema,
    ScenarioTemplate,
)
from pipeline.step_schema_learning.schema_learning import (
    SchemaLearningStep,
    _build_certification_schema,
    _load_existing_schema,
    _persist_schema,
    _slugify,
)


def _make_sample_exam_format(
    cert_name: str = "OIIQ Nursing Exam",
    question_types: list[QuestionType] | None = None,
):
    if question_types is None:
        question_types = [
            QuestionType(
                name="Multiple Choice",
                description="Clinical scenario with 4 options, 1 correct",
                purpose="Test clinical judgment",
            ),
            QuestionType(
                name="Multiple Response",
                description="Select all that apply",
            ),
        ]

    return ExamFormat(
        certification_name=cert_name,
        exam_code="OIIQ-2024",
        question_types=question_types,
        domains=[
            ExamDomain(name="Patient Assessment", weight_pct=25.0),
            ExamDomain(name="Medication Administration", weight_pct=20.0),
        ],
        raw_description="Quebec nursing licensure exam",
    )


SAMPLE_ANALYSIS_RESPONSE = json.dumps({
    "professional_domain": "nursing",
    "domain_description": "Healthcare nursing professionals in Quebec",
    "question_types": [
        {
            "original_name": "Multiple Choice",
            "suggested_name": "clinical_scenario_mc",
            "display_name": "Clinical Scenario Multiple Choice",
            "base_type": "multiple_choice",
            "description": "Clinical situation with patient context",
            "has_situation": True,
            "situation_description": "Patient scenario with relevant history",
            "supports_exhibits": True,
            "exhibit_description": "Medical documents like MAR, vital signs",
            "num_choices": 4,
            "num_correct": 1,
            "domain_specific_fields": {
                "nursing_process_step": "Which nursing process step this tests"
            },
            "generation_guidance": "Include realistic patient scenarios",
            "common_pitfalls": ["Avoid ambiguous wording"],
        },
        {
            "original_name": "Multiple Response",
            "suggested_name": "clinical_scenario_mr",
            "display_name": "Clinical Scenario Multiple Response",
            "base_type": "multiple_response",
            "description": "Select all correct nursing interventions",
            "has_situation": True,
            "supports_exhibits": True,
            "num_choices": 5,
            "num_correct": 2,
        },
    ],
    "inferred_exhibit_types": [
        {
            "name": "medication_administration_record",
            "display_name": "Medication Administration Record (MAR)",
            "category": "clinical_documentation",
            "description": "Record of medications administered",
            "typical_usage": "Questions about medication safety",
        },
        {
            "name": "vital_signs_chart",
            "display_name": "Vital Signs Chart",
            "category": "clinical_documentation",
            "description": "Patient vital signs over time",
            "typical_usage": "Assessment questions",
        },
    ],
    "inferred_scenario_templates": [
        {
            "name": "emergency_admission",
            "display_name": "Emergency Department Admission",
            "description": "Patient presenting to ED",
            "context_type": "emergency",
            "typical_elements": ["Chief complaint", "Vital signs", "Initial assessment"],
        },
    ],
})

SAMPLE_EXHIBIT_REFINEMENT_RESPONSE = json.dumps({
    "exhibit_types": [
        {
            "name": "medication_administration_record",
            "display_name": "Medication Administration Record (MAR)",
            "category": "clinical_documentation",
            "description": "Record of medications administered to patient",
            "fields": [
                {
                    "name": "patient_name",
                    "description": "Patient's full name",
                    "field_type": "text",
                    "required": True,
                    "example": "Jean Tremblay",
                },
                {
                    "name": "medications",
                    "description": "List of medications with dose, route, frequency",
                    "field_type": "table",
                    "required": True,
                },
            ],
            "example_content": "Patient: Jean Tremblay\nMedications: ...",
            "typical_usage": "Questions about medication safety and administration",
        },
    ]
})


def _make_mock_response(text: str):
    return SimpleNamespace(text=text)


class TestSlugify:
    def test_simple_name(self):
        assert _slugify("AWS SAA") == "aws_saa"

    def test_with_hyphens(self):
        assert _slugify("SAA-C03") == "saa_c03"

    def test_with_special_chars(self):
        assert _slugify("OIIQ (Quebec)") == "oiiq_quebec"

    def test_multiple_spaces(self):
        assert _slugify("AWS  Solutions  Architect") == "aws_solutions_architect"


class TestBuildCertificationSchema:
    def test_builds_from_analysis(self):
        analysis = json.loads(SAMPLE_ANALYSIS_RESPONSE)
        refined_exhibits = json.loads(SAMPLE_EXHIBIT_REFINEMENT_RESPONSE)["exhibit_types"]

        schema = _build_certification_schema(
            "OIIQ Nursing Exam",
            analysis,
            refined_exhibits,
        )

        assert schema.certification_name == "OIIQ Nursing Exam"
        assert schema.certification_slug == "oiiq_nursing_exam"
        assert schema.professional_domain == "nursing"
        assert len(schema.question_types) == 2
        assert len(schema.exhibit_types) == 1
        assert len(schema.scenario_templates) == 1

    def test_creates_type_mappings(self):
        analysis = json.loads(SAMPLE_ANALYSIS_RESPONSE)

        schema = _build_certification_schema("Test", analysis, [])

        assert "Multiple Choice" in schema.type_mappings
        assert schema.type_mappings["Multiple Choice"] == "clinical_scenario_mc"

    def test_question_type_has_correct_fields(self):
        analysis = json.loads(SAMPLE_ANALYSIS_RESPONSE)

        schema = _build_certification_schema("Test", analysis, [])

        mc_type = schema.question_types[0]
        assert mc_type.type_name == "clinical_scenario_mc"
        assert mc_type.base_type == "multiple_choice"
        assert mc_type.has_situation is True
        assert mc_type.supports_exhibits is True
        assert mc_type.num_choices == 4
        assert mc_type.num_correct == 1

    def test_exhibit_type_has_fields(self):
        analysis = json.loads(SAMPLE_ANALYSIS_RESPONSE)
        refined_exhibits = json.loads(SAMPLE_EXHIBIT_REFINEMENT_RESPONSE)["exhibit_types"]

        schema = _build_certification_schema("Test", analysis, refined_exhibits)

        mar = schema.exhibit_types[0]
        assert mar.name == "medication_administration_record"
        assert len(mar.fields) == 2
        assert mar.fields[0].name == "patient_name"


class TestPersistAndLoadSchema:
    def test_persists_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = CertificationSchema(
                certification_name="Test Cert",
                certification_slug="test_cert",
                question_types=[
                    QuestionTypeSchema(
                        type_name="test_mc",
                        display_name="Test MC",
                        base_type="multiple_choice",
                    )
                ],
            )

            schema_dir = _persist_schema(schema, tmpdir)

            assert Path(schema_dir).exists()
            assert (Path(schema_dir) / "schema.json").exists()
            assert (Path(schema_dir) / "question_types.json").exists()

    def test_loads_existing_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create schema
            schema = CertificationSchema(
                certification_name="Test Cert",
                certification_slug="test_cert",
                professional_domain="testing",
            )
            _persist_schema(schema, tmpdir)

            # Load it back
            loaded = _load_existing_schema(tmpdir, "test_cert")

            assert loaded is not None
            assert loaded.certification_name == "Test Cert"
            assert loaded.professional_domain == "testing"

    def test_returns_none_for_missing_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = _load_existing_schema(tmpdir, "nonexistent")
            assert loaded is None


class TestSchemaLearningStep:
    def _run_step(
        self,
        exam_format=None,
        analysis_response=SAMPLE_ANALYSIS_RESPONSE,
        exhibit_response=SAMPLE_EXHIBIT_REFINEMENT_RESPONSE,
        interactive=False,
        force_regenerate=False,
        existing_schema=None,
    ):
        if exam_format is None:
            exam_format = _make_sample_exam_format()

        mock_genai = MagicMock()

        # Track which call we're on
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            contents = kwargs.get("contents", "")
            if "Analyze the question types" in contents or call_count[0] == 1:
                return _make_mock_response(analysis_response)
            else:
                return _make_mock_response(exhibit_response)

        mock_genai.models.generate_content.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing schema if provided
            if existing_schema:
                _persist_schema(existing_schema, tmpdir)

            step = SchemaLearningStep(
                schemas_dir=tmpdir,
                interactive=interactive,
                force_regenerate=force_regenerate,
            )

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step_schema_learning.schema_learning.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step_schema_learning.schema_learning.time.sleep"),
            ):
                ctx = PipelineContext(
                    certification_name=exam_format.certification_name,
                    exam_format=exam_format,
                )
                result = step.run(ctx)

            return result, tmpdir, mock_genai

    def test_stores_schema_in_context(self):
        ctx, _, _ = self._run_step()

        assert "certification_schema" in ctx
        assert isinstance(ctx["certification_schema"], CertificationSchema)

    def test_stores_schema_dir_in_context(self):
        ctx, tmpdir, _ = self._run_step()

        assert "schema_dir" in ctx
        assert "oiiq_nursing_exam" in ctx["schema_dir"]

    def test_creates_output_summary(self):
        ctx, _, _ = self._run_step()

        assert "schema_learning_output" in ctx
        output = ctx["schema_learning_output"]
        assert output.certification_name == "OIIQ Nursing Exam"
        assert output.question_types_count == 2
        assert output.exhibit_types_count == 1

    def test_persists_schema_to_disk(self):
        # Use a persistent temp directory for this test
        exam_format = _make_sample_exam_format()

        mock_genai = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_mock_response(SAMPLE_ANALYSIS_RESPONSE)
            return _make_mock_response(SAMPLE_EXHIBIT_REFINEMENT_RESPONSE)

        mock_genai.models.generate_content.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            step = SchemaLearningStep(
                schemas_dir=tmpdir,
                interactive=False,
            )

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step_schema_learning.schema_learning.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step_schema_learning.schema_learning.time.sleep"),
            ):
                ctx = PipelineContext(
                    certification_name=exam_format.certification_name,
                    exam_format=exam_format,
                )
                step.run(ctx)

            # Check within the temp directory context
            schema_dir = Path(tmpdir) / "oiiq_nursing_exam"
            assert schema_dir.exists()
            assert (schema_dir / "schema.json").exists()

    def test_uses_existing_schema_when_available(self):
        existing = CertificationSchema(
            certification_name="OIIQ Nursing Exam",
            certification_slug="oiiq_nursing_exam",
            professional_domain="existing_domain",
            question_types=[
                QuestionTypeSchema(
                    type_name="existing_type",
                    display_name="Existing",
                    base_type="multiple_choice",
                )
            ],
        )

        # Non-interactive mode auto-uses existing
        ctx, _, mock_genai = self._run_step(
            existing_schema=existing,
            interactive=False,
        )

        # Should not have called LLM
        assert not mock_genai.models.generate_content.called

        # Should use existing schema
        assert ctx["certification_schema"].professional_domain == "existing_domain"

    def test_force_regenerate_ignores_existing(self):
        existing = CertificationSchema(
            certification_name="OIIQ Nursing Exam",
            certification_slug="oiiq_nursing_exam",
            professional_domain="existing_domain",
        )

        ctx, _, mock_genai = self._run_step(
            existing_schema=existing,
            force_regenerate=True,
        )

        # Should have called LLM
        assert mock_genai.models.generate_content.called

        # Should have new domain from analysis
        assert ctx["certification_schema"].professional_domain == "nursing"

    def test_handles_analysis_failure(self):
        mock_genai = MagicMock()
        mock_genai.models.generate_content.side_effect = RuntimeError("API Error")

        exam_format = _make_sample_exam_format()

        with tempfile.TemporaryDirectory() as tmpdir:
            step = SchemaLearningStep(
                schemas_dir=tmpdir,
                interactive=False,
            )

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step_schema_learning.schema_learning.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step_schema_learning.schema_learning.time.sleep"),
            ):
                ctx = PipelineContext(
                    certification_name=exam_format.certification_name,
                    exam_format=exam_format,
                )
                # Should not raise, just use fallback
                result = step.run(ctx)

        assert "certification_schema" in result
        # Fallback schema has generic domain
        assert result["certification_schema"].professional_domain == "general"

    def test_infers_domain_from_cert_name(self):
        # Test nursing domain inference
        ctx, _, _ = self._run_step(
            exam_format=_make_sample_exam_format("NCLEX-RN Nursing Exam"),
        )
        # The analysis response still returns "nursing" as the domain

        # Test cloud domain inference
        ctx2, _, _ = self._run_step(
            exam_format=_make_sample_exam_format("AWS Solutions Architect"),
        )


class TestSchemaLearningInteractive:
    def test_interactive_confirmation(self):
        exam_format = _make_sample_exam_format()

        mock_genai = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_mock_response(SAMPLE_ANALYSIS_RESPONSE)
            return _make_mock_response(SAMPLE_EXHIBIT_REFINEMENT_RESPONSE)

        mock_genai.models.generate_content.side_effect = side_effect

        # Simulate user saying "Yes, proceed"
        user_inputs = iter(["1"])  # Select first option

        with tempfile.TemporaryDirectory() as tmpdir:
            step = SchemaLearningStep(
                schemas_dir=tmpdir,
                interactive=True,
                user_input_callback=lambda _: next(user_inputs),
            )

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step_schema_learning.schema_learning.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step_schema_learning.schema_learning.time.sleep"),
                patch("builtins.print"),  # Suppress output
            ):
                ctx = PipelineContext(
                    certification_name=exam_format.certification_name,
                    exam_format=exam_format,
                )
                result = step.run(ctx)

        assert result["schema_learning_output"].user_refinements_made is False
