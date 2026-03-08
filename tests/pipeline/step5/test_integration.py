import json
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    Course,
    CourseModule,
    CourseOverview,
    CourseSkeleton,
    CourseTopic,
    ExamDomain,
    ExamFormat,
    LearningObjective,
    QuestionType,
    QuestionTypeGuide,
)
from pipeline.step5.pipeline import build_step5_pipeline


def _make_sample_context():
    skeleton = CourseSkeleton(
        certification_name="AWS SAA",
        exam_code="SAA-C03",
        overview=CourseOverview(
            target_audience="Cloud architects",
        ),
        question_type_guides=[
            QuestionTypeGuide(question_type_name="multiple choice"),
        ],
        domain_modules=[
            CourseModule(
                domain_name="Security",
                topics=[
                    CourseTopic(
                        name="IAM",
                        description="Identity management",
                        learning_objectives=[
                            LearningObjective(
                                objective="Configure IAM",
                                bloom_level="Apply",
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

    exam_format = ExamFormat(
        certification_name="AWS SAA",
        exam_code="SAA-C03",
        question_types=[QuestionType(name="multiple choice")],
        domains=[ExamDomain(name="Security", weight_pct=30.0)],
        raw_description="Test",
    )

    return PipelineContext(
        course_skeleton=skeleton,
        exam_format=exam_format,
        vectorstore_path="vectorstore",
        collection_name="aws_saa",
    )


SAMPLE_LLM_RESPONSE = json.dumps({
    "questions": [
        {
            "question_type": "multiple_choice",
            "stem": "What is IAM?",
            "choices": [
                {"label": "A", "text": "Identity and Access Management", "is_correct": True},
                {"label": "B", "text": "Internet Access Manager", "is_correct": False},
                {"label": "C", "text": "Internal Application Module", "is_correct": False},
                {"label": "D", "text": "Integrated Access Method", "is_correct": False},
            ],
            "correct_label": "A",
            "correct_answer_explanation": "IAM stands for Identity and Access Management.",
            "wrong_answer_explanations": {},
            "difficulty": "easy",
            "bloom_level": "Remember",
            "source_indices": [0],
        }
    ]
})


class TestStep5Integration:
    def test_build_pipeline(self):
        pipeline = build_step5_pipeline()

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].model == "gemini-flash-latest"
        assert pipeline.steps[0].max_workers == 4
        assert pipeline.steps[0].questions_per_type == 3
        assert pipeline.steps[0].top_k == 5

    def test_build_pipeline_custom_params(self):
        pipeline = build_step5_pipeline(
            model="gemini-pro",
            max_workers=8,
            questions_per_type=5,
            top_k=10,
            schemas_dir="custom_schemas",
        )

        step = pipeline.steps[0]
        assert step.model == "gemini-pro"
        assert step.max_workers == 8
        assert step.questions_per_type == 5
        assert step.top_k == 10
        assert step.schemas_dir == "custom_schemas"

    def test_full_pipeline_run(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["IAM is Identity and Access Management."]],
            "metadatas": [[{
                "book_title": "AWS Guide",
                "book_author": "Amazon",
                "section_heading": "IAM",
                "page_numbers": "15",
                "image_paths": "",
                "source_file": "aws.pdf",
            }]],
            "distances": [[0.3]],
        }

        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [[0.1] * 384]

        mock_genai = MagicMock()
        mock_genai.models.generate_content.return_value = SimpleNamespace(
            text=SAMPLE_LLM_RESPONSE
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = build_step5_pipeline(
                max_workers=1,
                questions_per_type=1,
                schemas_dir=tmpdir,
            )

            # Inject mocks
            pipeline.steps[0]._chroma_client = mock_chroma
            pipeline.steps[0]._embedding_model = mock_embedding

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step5.course_content.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step5.course_content.time.sleep"),
            ):
                ctx = _make_sample_context()
                result = pipeline.run(ctx)

        assert "course" in result
        course = result["course"]
        assert isinstance(course, Course)
        assert course.certification_name == "AWS SAA"
        assert len(course.topic_contents) == 1
        assert len(course.failed_topics) == 0

        # Check questions were generated
        topic_content = course.topic_contents[0]
        assert topic_content.topic_name == "IAM"
        assert len(topic_content.questions) == 1

        # Check question has source refs
        question = topic_content.questions[0]
        assert len(question.source_refs) == 1
        assert question.source_refs[0].book_title == "AWS Guide"

    def test_pipeline_creates_schemas_file(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [[0.1] * 384]

        mock_genai = MagicMock()
        mock_genai.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps({"questions": []})
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            pipeline = build_step5_pipeline(schemas_dir=tmpdir)
            pipeline.steps[0]._chroma_client = mock_chroma
            pipeline.steps[0]._embedding_model = mock_embedding

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step5.course_content.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step5.course_content.time.sleep"),
            ):
                ctx = _make_sample_context()
                pipeline.run(ctx)

            # Check schema file was created
            schema_path = Path(tmpdir) / "question_types.json"
            assert schema_path.exists()

            schema_data = json.loads(schema_path.read_text())
            assert "schemas" in schema_data
            assert "multiple_choice" in schema_data["schemas"]

    def test_pipeline_preserves_context(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [[0.1] * 384]

        mock_genai = MagicMock()
        mock_genai.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps({"questions": []})
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = build_step5_pipeline(schemas_dir=tmpdir)
            pipeline.steps[0]._chroma_client = mock_chroma
            pipeline.steps[0]._embedding_model = mock_embedding

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step5.course_content.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step5.course_content.time.sleep"),
            ):
                ctx = _make_sample_context()
                ctx["custom_key"] = "custom_value"
                result = pipeline.run(ctx)

        assert result["custom_key"] == "custom_value"
        assert "course_skeleton" in result
        assert "exam_format" in result
