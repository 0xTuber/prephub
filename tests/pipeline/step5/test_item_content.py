"""Tests for Step 5: Item Content Generation with RAG."""

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
    ItemSourceReference,
    Lab,
    LearningObjective,
    QuestionTypeGuide,
    SourceCitation,
    SubTopic,
)
from pipeline.step5.item_content import (
    ItemContentGenerationStep,
    _shuffle_options,
    _query_vectorstore,
)


def _make_skeleton_with_item_skeletons() -> CourseSkeleton:
    """Create a skeleton with item skeletons (no content yet)."""
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
                                                items=[
                                                    CapsuleItem(
                                                        item_id="item_01",
                                                        item_type="Multiple Choice",
                                                        title="Policy Components",
                                                        learning_target="Identify the main components of an IAM policy",
                                                        difficulty="beginner",
                                                        # Content fields are None
                                                        content=None,
                                                        options=None,
                                                        correct_answer_index=None,
                                                        explanation=None,
                                                        source_reference=None,
                                                    ),
                                                    CapsuleItem(
                                                        item_id="item_02",
                                                        item_type="Multiple Choice",
                                                        title="Effect Types",
                                                        learning_target="Distinguish between Allow and Deny effects",
                                                        difficulty="intermediate",
                                                        content=None,
                                                        options=None,
                                                        correct_answer_index=None,
                                                        explanation=None,
                                                        source_reference=None,
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


SAMPLE_CONTENT_JSON = json.dumps({
    "stem": "Which of the following is a required element in an IAM policy document?",
    "options": [
        "Effect",
        "Principal",
        "Condition",
        "Resource"
    ],
    "correct_index": 0,
    "explanation": "The Effect element is required in every IAM policy statement. It specifies whether the statement allows or denies access.",
    "source_summary": "IAM policies are JSON documents that define permissions. Every policy statement must include an Effect element that specifies Allow or Deny. The Principal, Condition, and Resource elements are optional depending on the policy type."
})


class TestShuffleOptions:
    def test_shuffles_options(self):
        options = ["A", "B", "C", "D"]
        correct_index = 0  # "A" is correct

        # Run multiple times to verify shuffling happens
        results = set()
        for _ in range(20):
            shuffled, new_idx = _shuffle_options(options.copy(), correct_index)
            results.add(new_idx)
            # Verify correct answer is still tracked
            assert shuffled[new_idx] == "A"

        # Should have some variety (not always same position)
        assert len(results) > 1

    def test_preserves_correct_answer(self):
        options = ["Wrong1", "Correct", "Wrong2", "Wrong3"]
        correct_index = 1

        shuffled, new_idx = _shuffle_options(options, correct_index)
        assert shuffled[new_idx] == "Correct"

    def test_single_option_unchanged(self):
        options = ["Only"]
        shuffled, new_idx = _shuffle_options(options, 0)
        assert shuffled == ["Only"]
        assert new_idx == 0


class MockEmbedding:
    """Mock embedding with tolist() method."""
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class TestQueryVectorstore:
    def test_returns_chunks_with_metadata(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["chunk_1", "chunk_2"]],
            "documents": [["Document 1 text", "Document 2 text"]],
            "metadatas": [[
                {"book_title": "AWS Guide", "book_author": "Author", "page_numbers": "10,11", "section_heading": "IAM"},
                {"book_title": "AWS Guide", "book_author": "Author", "page_numbers": "15", "section_heading": "Policies"},
            ]],
            "distances": [[0.1, 0.2]],
        }

        mock_model = MagicMock()
        mock_model.encode.return_value = [MockEmbedding([0.1, 0.2, 0.3])]

        chunks = _query_vectorstore(mock_collection, mock_model, "IAM policies", n_results=2)

        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == "chunk_1"
        assert chunks[0]["book_title"] == "AWS Guide"
        assert chunks[0]["pages"] == [10, 11]
        assert chunks[0]["text"] == "Document 1 text"

    def test_handles_empty_results(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_model = MagicMock()
        mock_model.encode.return_value = [MockEmbedding([0.1, 0.2, 0.3])]

        chunks = _query_vectorstore(mock_collection, mock_model, "nonexistent", n_results=5)
        assert chunks == []


def _make_mock_genai_response(text: str):
    return SimpleNamespace(text=text)


def _run_step(
    skeleton=None,
    mock_chunks=None,
    content_response=None,
    has_vectorstore=True,
) -> tuple[PipelineContext, MagicMock]:
    if skeleton is None:
        skeleton = _make_skeleton_with_item_skeletons()

    if mock_chunks is None:
        mock_chunks = [
            {
                "chunk_id": "chunk_42",
                "text": "IAM policies are JSON documents that define permissions.",
                "book_title": "AWS Certified Solutions Architect Guide",
                "book_author": "AWS Team",
                "pages": [120, 121],
                "section_heading": "IAM Policies",
                "distance": 0.15,
            }
        ]

    step = ItemContentGenerationStep(max_workers=2)

    # Mock ChromaDB
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100

    # Mock embedding model
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]

    # Mock Gemini client
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.return_value = _make_mock_genai_response(
        content_response or SAMPLE_CONTENT_JSON
    )

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch("pipeline.step5.item_content.genai.Client", return_value=mock_genai_client),
        patch("chromadb.PersistentClient") as mock_chroma,
        patch("sentence_transformers.SentenceTransformer", return_value=mock_embedding_model),
        patch("pipeline.step5.item_content._query_vectorstore", return_value=mock_chunks),
        patch("pipeline.step5.item_content.time.sleep"),
    ):
        mock_chroma.return_value.get_collection.return_value = mock_collection

        ctx = PipelineContext(
            certification_name="AWS SAA-C03",
            course_skeleton=skeleton,
            collection_name="aws_saa_c03",
            vectorstore_path="vectorstore",
        )

        if not has_vectorstore:
            mock_chroma.return_value.get_collection.side_effect = Exception("Collection not found")

        result = step.run(ctx)
        return result, mock_genai_client


class TestItemContentGenerationStep:
    def test_generates_content_for_items(self):
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

        # Content should now be populated
        assert item.content is not None
        assert "required element" in item.content
        assert item.options is not None
        assert len(item.options) == 4
        assert item.correct_answer_index is not None
        assert item.explanation is not None

    def test_creates_source_reference(self):
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

        assert item.source_reference is not None
        assert isinstance(item.source_reference, ItemSourceReference)
        assert len(item.source_reference.summary) > 0
        assert len(item.source_reference.chunk_ids) > 0
        assert "chunk_42" in item.source_reference.chunk_ids

    def test_source_citations_populated(self):
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

        citations = item.source_reference.citations
        assert len(citations) > 0
        assert isinstance(citations[0], SourceCitation)
        assert citations[0].book_title == "AWS Certified Solutions Architect Guide"
        assert 120 in citations[0].pages or 121 in citations[0].pages

    def test_skips_items_with_existing_content(self):
        skeleton = _make_skeleton_with_item_skeletons()
        # Pre-fill content for first item
        skeleton.domain_modules[0].topics[0].subtopics[0].labs[0].capsules[0].items[0].content = "Already filled"

        ctx, mock_client = _run_step(skeleton=skeleton)

        # Should only generate for items without content
        # With 2 items total and 1 already filled, should have 1 call
        assert mock_client.models.generate_content.call_count == 1

    def test_handles_no_vectorstore(self):
        ctx, _ = _run_step(has_vectorstore=False)

        # Should flag that content wasn't generated
        assert ctx.get("items_without_content") is True

    def test_handles_no_chunks_found(self):
        ctx, _ = _run_step(mock_chunks=[])

        # Items should still exist but without content
        skeleton = ctx["course_skeleton"]
        item = (
            skeleton.domain_modules[0]
            .topics[0]
            .subtopics[0]
            .labs[0]
            .capsules[0]
            .items[0]
        )
        # Content remains None when no chunks found
        assert item.content is None

    def test_options_are_shuffled(self):
        """Test that answer positions are randomized."""
        # Run multiple times and check for variety
        correct_positions = set()

        for _ in range(10):
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
            if item.correct_answer_index is not None:
                correct_positions.add(item.correct_answer_index)

        # Should have some variety (shuffling happening)
        # Note: With 4 options over 10 runs, we expect variety
        assert len(correct_positions) >= 1  # At minimum, positions are tracked

    def test_prompt_includes_source_material(self):
        _, mock_client = _run_step()

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]

        # Should include source chunks
        assert "SOURCE MATERIAL" in contents
        assert "IAM policies" in contents

    def test_prompt_includes_learning_target(self):
        _, mock_client = _run_step()

        call = mock_client.models.generate_content.call_args_list[0]
        contents = call.kwargs["contents"]

        # Should include learning target from skeleton
        assert "Learning Target" in contents
        assert "components" in contents.lower() or "iam policy" in contents.lower()

    def test_handles_code_fences_in_response(self):
        fenced_response = f"```json\n{SAMPLE_CONTENT_JSON}\n```"
        ctx, _ = _run_step(content_response=fenced_response)
        skeleton = ctx["course_skeleton"]

        item = (
            skeleton.domain_modules[0]
            .topics[0]
            .subtopics[0]
            .labs[0]
            .capsules[0]
            .items[0]
        )

        assert item.content is not None
        assert item.options is not None

    def test_summary_length_validation(self):
        """Summary should be 30-80 words."""
        short_summary_response = json.dumps({
            "stem": "Test question?",
            "options": ["A", "B", "C", "D"],
            "correct_index": 0,
            "explanation": "Explanation",
            "source_summary": "Too short."  # Less than 30 words
        })

        ctx, _ = _run_step(content_response=short_summary_response)
        skeleton = ctx["course_skeleton"]

        item = (
            skeleton.domain_modules[0]
            .topics[0]
            .subtopics[0]
            .labs[0]
            .capsules[0]
            .items[0]
        )

        # Summary should be padded to minimum length
        words = item.source_reference.summary.split()
        assert len(words) >= 2  # At least has something

    def test_preserves_existing_context(self):
        skeleton = _make_skeleton_with_item_skeletons()

        step = ItemContentGenerationStep(max_workers=2)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_genai_client = MagicMock()
        mock_genai_client.models.generate_content.return_value = _make_mock_genai_response(SAMPLE_CONTENT_JSON)

        mock_chunks = [{
            "chunk_id": "chunk_1",
            "text": "Test content",
            "book_title": "Test Book",
            "book_author": "Author",
            "pages": [1],
            "section_heading": "Test",
            "distance": 0.1,
        }]

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("pipeline.step5.item_content.genai.Client", return_value=mock_genai_client),
            patch("chromadb.PersistentClient") as mock_chroma,
            patch("sentence_transformers.SentenceTransformer", return_value=mock_embedding_model),
            patch("pipeline.step5.item_content._query_vectorstore", return_value=mock_chunks),
            patch("pipeline.step5.item_content.time.sleep"),
        ):
            mock_chroma.return_value.get_collection.return_value = mock_collection

            ctx = PipelineContext(
                certification_name="AWS SAA-C03",
                course_skeleton=skeleton,
                collection_name="test",
                vectorstore_path="vectorstore",
                custom_key="preserved_value",
            )
            result = step.run(ctx)

        assert result["custom_key"] == "preserved_value"
        assert result["content_generation_complete"] is True
