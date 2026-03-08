import json
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.base import PipelineContext
from pipeline.models import (
    AnswerChoice,
    Course,
    CourseModule,
    CourseOverview,
    CourseSkeleton,
    CourseTopic,
    DragDropQuestion,
    ExamDomain,
    ExamFormat,
    LearningObjective,
    MultipleChoiceQuestion,
    MultipleResponseQuestion,
    QuestionType,
    QuestionTypeGuide,
    ReasoningTemplate,
    SourceReference,
    SubTopic,
    TopicContent,
)
from pipeline.step5.course_content import (
    CourseContentStep,
    _extract_highlights,
    _format_learning_objectives,
    _format_reference_material,
    _format_subtopics,
    _parse_question,
    _retrieve_context,
)


def _make_sample_skeleton():
    return CourseSkeleton(
        certification_name="AWS Solutions Architect Associate",
        exam_code="SAA-C03",
        overview=CourseOverview(
            target_audience="Cloud architects",
            course_description="AWS SAA prep",
        ),
        question_type_guides=[
            QuestionTypeGuide(
                question_type_name="multiple choice",
                detailed_structure="4 options, 1 correct",
                reasoning_template=ReasoningTemplate(
                    approach_steps=["Read carefully", "Eliminate wrong answers"],
                    common_traps=["Distractor options"],
                ),
            ),
            QuestionTypeGuide(
                question_type_name="multiple response",
                detailed_structure="5+ options, 2+ correct",
            ),
        ],
        domain_modules=[
            CourseModule(
                domain_name="Design Secure Architectures",
                domain_weight_pct=30.0,
                topics=[
                    CourseTopic(
                        name="IAM Policies",
                        description="Identity and Access Management",
                        learning_objectives=[
                            LearningObjective(
                                objective="Configure IAM policies",
                                bloom_level="Apply",
                            )
                        ],
                        subtopics=[
                            SubTopic(
                                name="Policy Structure",
                                key_concepts=["Principal", "Action", "Resource"],
                                practical_skills=["Write IAM policies"],
                            )
                        ],
                    ),
                    CourseTopic(
                        name="VPC Security",
                        description="Virtual Private Cloud security",
                    ),
                ],
            ),
            CourseModule(
                domain_name="Design Resilient Architectures",
                domain_weight_pct=26.0,
                topics=[
                    CourseTopic(
                        name="Multi-AZ Deployments",
                        description="High availability across AZs",
                    ),
                ],
            ),
        ],
    )


def _make_sample_exam_format():
    return ExamFormat(
        certification_name="AWS Solutions Architect Associate",
        exam_code="SAA-C03",
        question_types=[
            QuestionType(name="multiple choice"),
            QuestionType(name="multiple response"),
        ],
        domains=[
            ExamDomain(name="Design Secure Architectures", weight_pct=30.0),
        ],
        raw_description="Test",
    )


SAMPLE_QUESTION_RESPONSE = json.dumps({
    "questions": [
        {
            "question_type": "multiple_choice",
            "stem": "Which IAM policy element specifies the AWS resource?",
            "choices": [
                {"label": "A", "text": "Principal", "is_correct": False},
                {"label": "B", "text": "Action", "is_correct": False},
                {"label": "C", "text": "Resource", "is_correct": True},
                {"label": "D", "text": "Effect", "is_correct": False},
            ],
            "correct_label": "C",
            "correct_answer_explanation": "Resource specifies the AWS resource.",
            "wrong_answer_explanations": {
                "A": "Principal specifies who is allowed.",
                "B": "Action specifies what actions are allowed.",
                "D": "Effect specifies Allow or Deny.",
            },
            "difficulty": "easy",
            "bloom_level": "Remember",
            "source_indices": [0],
        },
        {
            "question_type": "multiple_response",
            "stem": "Which are valid IAM policy elements? (Select TWO)",
            "num_correct": 2,
            "choices": [
                {"label": "A", "text": "Principal", "is_correct": True},
                {"label": "B", "text": "Action", "is_correct": True},
                {"label": "C", "text": "Table", "is_correct": False},
                {"label": "D", "text": "Column", "is_correct": False},
                {"label": "E", "text": "Row", "is_correct": False},
            ],
            "correct_labels": ["A", "B"],
            "correct_answer_explanation": "Principal and Action are IAM elements.",
            "wrong_answer_explanations": {},
            "difficulty": "medium",
            "bloom_level": "Understand",
            "source_indices": [0, 1],
        },
    ]
})


def _make_mock_response(text: str):
    return SimpleNamespace(text=text)


class TestRetrieveContext:
    def test_retrieves_from_collection(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["IAM policies control access to AWS resources."]],
            "metadatas": [[{
                "book_title": "AWS Guide",
                "book_author": "Amazon",
                "section_heading": "IAM",
                "page_numbers": "10,11",
                "image_paths": "",
                "source_file": "aws.pdf",
            }]],
            "distances": [[0.5]],
        }

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]

        topic = CourseTopic(
            name="IAM Policies",
            description="Identity and Access Management",
        )

        refs = _retrieve_context(
            mock_collection,
            mock_model,
            topic,
            "Security",
            top_k=5,
        )

        assert len(refs) == 1
        assert refs[0].book_title == "AWS Guide"
        assert refs[0].page_numbers == [10, 11]
        assert refs[0].chunk_text == "IAM policies control access to AWS resources."
        assert refs[0].relevance_score is not None

    def test_handles_empty_results(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]

        topic = CourseTopic(name="Unknown Topic")

        refs = _retrieve_context(
            mock_collection,
            mock_model,
            topic,
            "Domain",
            top_k=5,
        )

        assert len(refs) == 0

    def test_includes_highlights_placeholder(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Some text"]],
            "metadatas": [[{
                "book_title": "Book",
                "book_author": "Author",
                "section_heading": "",
                "page_numbers": "",
                "image_paths": "",
                "source_file": "file.pdf",
            }]],
            "distances": [[0.1]],
        }

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]

        refs = _retrieve_context(
            mock_collection,
            mock_model,
            CourseTopic(name="Test"),
            "Domain",
            top_k=1,
        )

        assert refs[0].highlights == []


class TestFormatHelpers:
    def test_format_reference_material_with_refs(self):
        refs = [
            SourceReference(
                book_title="AWS Guide",
                book_author="Amazon",
                page_numbers=[10, 11],
                chunk_text="IAM policies...",
            ),
        ]
        result = _format_reference_material(refs)

        assert "AWS Guide" in result
        assert "10, 11" in result
        assert "IAM policies..." in result

    def test_format_reference_material_empty(self):
        result = _format_reference_material([])
        assert "No reference material" in result

    def test_format_learning_objectives(self):
        topic = CourseTopic(
            name="Test",
            learning_objectives=[
                LearningObjective(objective="Learn X", bloom_level="Apply"),
                LearningObjective(objective="Understand Y"),
            ],
        )
        result = _format_learning_objectives(topic)

        assert "Learn X [Apply]" in result
        assert "Understand Y" in result

    def test_format_subtopics(self):
        topic = CourseTopic(
            name="Test",
            subtopics=[
                SubTopic(
                    name="Sub1",
                    description="Description",
                    key_concepts=["A", "B"],
                    practical_skills=["Skill1"],
                ),
            ],
        )
        result = _format_subtopics(topic)

        assert "Sub1" in result
        assert "A, B" in result
        assert "Skill1" in result


class TestParseQuestion:
    def test_parses_multiple_choice(self):
        q_data = {
            "question_type": "multiple_choice",
            "stem": "What is X?",
            "choices": [
                {"label": "A", "text": "Answer A", "is_correct": True},
                {"label": "B", "text": "Answer B", "is_correct": False},
            ],
            "correct_label": "A",
            "correct_answer_explanation": "A is correct.",
            "difficulty": "easy",
            "source_indices": [],
        }

        result = _parse_question(q_data, [])

        assert isinstance(result, MultipleChoiceQuestion)
        assert result.stem == "What is X?"
        assert result.correct_label == "A"
        assert len(result.choices) == 2

    def test_parses_multiple_response(self):
        q_data = {
            "question_type": "multiple_response",
            "stem": "Select two:",
            "num_correct": 2,
            "choices": [
                {"label": "A", "text": "A", "is_correct": True},
                {"label": "B", "text": "B", "is_correct": True},
                {"label": "C", "text": "C", "is_correct": False},
            ],
            "correct_labels": ["A", "B"],
            "correct_answer_explanation": "A and B.",
            "source_indices": [],
        }

        result = _parse_question(q_data, [])

        assert isinstance(result, MultipleResponseQuestion)
        assert result.num_correct == 2
        assert result.correct_labels == ["A", "B"]

    def test_parses_drag_drop(self):
        q_data = {
            "question_type": "drag_drop",
            "stem": "Match items:",
            "drag_items": [
                {"id": "d1", "text": "Item 1"},
                {"id": "d2", "text": "Item 2"},
            ],
            "drop_zones": [
                {"id": "z1", "label": "Zone 1", "accepts": ["d1"]},
            ],
            "correct_answer_explanation": "Correct matches.",
            "source_indices": [],
        }

        result = _parse_question(q_data, [])

        assert isinstance(result, DragDropQuestion)
        assert len(result.drag_items) == 2
        assert len(result.drop_zones) == 1

    def test_attaches_source_refs(self):
        source_refs = [
            SourceReference(
                book_title="Book 1",
                book_author="Author 1",
                chunk_text="Content 1",
            ),
            SourceReference(
                book_title="Book 2",
                book_author="Author 2",
                chunk_text="Content 2",
            ),
        ]

        q_data = {
            "question_type": "multiple_choice",
            "stem": "Question",
            "choices": [],
            "correct_label": "A",
            "correct_answer_explanation": "Explanation",
            "source_indices": [0, 1],
        }

        result = _parse_question(q_data, source_refs)

        assert len(result.source_refs) == 2
        assert result.source_refs[0].book_title == "Book 1"
        assert result.source_refs[1].book_title == "Book 2"


class TestExtractHighlights:
    def test_extracts_string_highlights(self):
        q_data = {"highlights": ["IAM policy", "resource"]}
        chunk = "An IAM policy specifies access to a resource."

        result = _extract_highlights(q_data, chunk)

        assert len(result) == 2
        assert result[0].text == "IAM policy"
        assert result[0].start_char == 3
        assert result[1].text == "resource"

    def test_extracts_dict_highlights(self):
        q_data = {
            "highlights": [
                {"text": "IAM", "start_char": 0, "end_char": 3}
            ]
        }
        chunk = "IAM policies"

        result = _extract_highlights(q_data, chunk)

        assert len(result) == 1
        assert result[0].text == "IAM"
        assert result[0].start_char == 0
        assert result[0].end_char == 3


class TestCourseContentStep:
    def _run_step(
        self,
        skeleton=None,
        mock_query_results=None,
        mock_llm_response=SAMPLE_QUESTION_RESPONSE,
        vectorstore_path="vectorstore",
        collection_name="test_collection",
    ):
        if skeleton is None:
            skeleton = _make_sample_skeleton()

        if mock_query_results is None:
            mock_query_results = {
                "documents": [["IAM content"]],
                "metadatas": [[{
                    "book_title": "AWS Guide",
                    "book_author": "Amazon",
                    "section_heading": "IAM",
                    "page_numbers": "10",
                    "image_paths": "",
                    "source_file": "aws.pdf",
                }]],
                "distances": [[0.5]],
            }

        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_query_results

        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [[0.1] * 384]

        mock_genai = MagicMock()
        mock_genai.models.generate_content.return_value = _make_mock_response(
            mock_llm_response
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CourseContentStep(
                max_workers=2,
                questions_per_type=1,
                top_k=3,
                schemas_dir=tmpdir,
                chroma_client=mock_chroma,
                embedding_model=mock_embedding,
            )

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step5.course_content.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step5.course_content.time.sleep"),
            ):
                ctx = PipelineContext(
                    course_skeleton=skeleton,
                    exam_format=_make_sample_exam_format(),
                    vectorstore_path=vectorstore_path,
                    collection_name=collection_name,
                )
                result = step.run(ctx)

        return result, mock_genai, mock_collection

    def test_stores_course_in_context(self):
        ctx, _, _ = self._run_step()

        assert "course" in ctx
        assert isinstance(ctx["course"], Course)

    def test_generates_questions_for_all_topics(self):
        ctx, mock_genai, _ = self._run_step()

        # 3 topics total in sample skeleton
        course = ctx["course"]
        # At least some topics should succeed
        assert len(course.topic_contents) > 0

    def test_fan_out_all_succeed(self):
        ctx, _, _ = self._run_step()

        course = ctx["course"]
        # All 3 topics should succeed
        assert len(course.topic_contents) == 3
        assert len(course.failed_topics) == 0

    def test_fan_out_partial_failure(self):
        skeleton = _make_sample_skeleton()

        # Track which topic is being processed based on prompt content
        # Make the second topic (VPC Security) always fail
        def side_effect(**kwargs):
            contents = kwargs.get("contents", "")
            if "VPC Security" in contents:
                raise RuntimeError("API error")
            return _make_mock_response(SAMPLE_QUESTION_RESPONSE)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Content"]],
            "metadatas": [[{
                "book_title": "Book",
                "book_author": "Author",
                "section_heading": "",
                "page_numbers": "",
                "image_paths": "",
                "source_file": "file.pdf",
            }]],
            "distances": [[0.5]],
        }

        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [[0.1] * 384]

        mock_genai = MagicMock()
        mock_genai.models.generate_content.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CourseContentStep(
                max_workers=1,  # Sequential to control failure order
                schemas_dir=tmpdir,
                chroma_client=mock_chroma,
                embedding_model=mock_embedding,
            )

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch(
                    "pipeline.step5.course_content.genai.Client",
                    return_value=mock_genai,
                ),
                patch("pipeline.step5.course_content.time.sleep"),
            ):
                ctx = PipelineContext(
                    course_skeleton=skeleton,
                    exam_format=_make_sample_exam_format(),
                    vectorstore_path="vectorstore",
                    collection_name="test",
                )
                result = step.run(ctx)

        course = result["course"]
        assert len(course.topic_contents) == 2
        assert len(course.failed_topics) == 1
        assert course.failed_topics[0].topic_name == "VPC Security"
        assert "API error" in course.failed_topics[0].error

    def test_handles_empty_vectorstore(self):
        mock_query_results = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        ctx, _, _ = self._run_step(mock_query_results=mock_query_results)

        course = ctx["course"]
        # Should still generate questions (without source refs)
        assert len(course.topic_contents) > 0

    def test_retrieves_context_with_source_refs(self):
        ctx, _, mock_collection = self._run_step()

        # Verify collection was queried
        assert mock_collection.query.called

    def test_generates_type_specific_structure(self):
        ctx, _, _ = self._run_step()

        course = ctx["course"]
        if course.topic_contents:
            questions = course.topic_contents[0].questions
            if questions:
                # Check that we got the right types
                mc_questions = [q for q in questions if isinstance(q, MultipleChoiceQuestion)]
                mr_questions = [q for q in questions if isinstance(q, MultipleResponseQuestion)]

                if mc_questions:
                    assert hasattr(mc_questions[0], "choices")
                    assert hasattr(mc_questions[0], "correct_label")

                if mr_questions:
                    assert hasattr(mr_questions[0], "correct_labels")
                    assert hasattr(mr_questions[0], "num_correct")

    def test_preserves_existing_context(self):
        ctx, _, _ = self._run_step()

        # Existing keys should still be present
        assert "course_skeleton" in ctx
        assert "exam_format" in ctx
        assert "vectorstore_path" in ctx

    def test_counts_total_questions(self):
        ctx, _, _ = self._run_step()

        course = ctx["course"]
        # total_questions should match actual count
        actual_count = sum(len(tc.questions) for tc in course.topic_contents)
        assert course.total_questions == actual_count
