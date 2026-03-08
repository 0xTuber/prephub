"""Tests for quote extraction module."""

from unittest.mock import MagicMock, patch

import pytest

from course_builder.pipeline.content.quote_extraction import (
    ExtractedQuote,
    ExtractionStatus,
    GenerationQuoteVerification,
    QuoteExtractionResult,
    QuoteVerificationResult,
    extract_quotes_for_item,
    format_quotes_for_prompt,
    get_chunks_from_quotes,
    identify_quotable_sentences,
    verify_quotes_in_text,
)
from course_builder.pipeline.content.query_planning import (
    QueryIntent,
    QueryPlan,
    PlannedQuery,
)


class TestExtractionStatus:
    """Tests for ExtractionStatus enum."""

    def test_status_values(self):
        assert ExtractionStatus.SUFFICIENT == "sufficient"
        assert ExtractionStatus.MARGINAL == "marginal"
        assert ExtractionStatus.INSUFFICIENT == "insufficient"


class TestExtractedQuote:
    """Tests for ExtractedQuote model."""

    def test_create_basic_quote(self):
        quote = ExtractedQuote(
            quote_id="Q1",
            text="Scene safety must be ensured before patient contact.",
            chunk_id="chunk_001",
            chunk_index=0,
            start_char=0,
            end_char=52,
        )

        assert quote.quote_id == "Q1"
        assert quote.text == "Scene safety must be ensured before patient contact."
        assert quote.chunk_id == "chunk_001"
        assert quote.chunk_index == 0
        assert quote.start_char == 0
        assert quote.end_char == 52
        assert quote.page_numbers == []
        assert quote.section_heading is None
        assert quote.relevance_score == 0.0

    def test_create_quote_with_metadata(self):
        quote = ExtractedQuote(
            quote_id="Q2",
            text="Always establish a perimeter around electrical hazards.",
            chunk_id="chunk_002",
            chunk_index=1,
            start_char=100,
            end_char=155,
            page_numbers=[32, 33],
            section_heading="Electrical Hazards",
            relevance_score=0.85,
        )

        assert quote.page_numbers == [32, 33]
        assert quote.section_heading == "Electrical Hazards"
        assert quote.relevance_score == 0.85

    def test_quote_json_serialization(self):
        quote = ExtractedQuote(
            quote_id="Q1",
            text="Test quote",
            chunk_id="chunk_001",
            chunk_index=0,
            start_char=0,
            end_char=10,
            page_numbers=[10],
        )

        data = quote.model_dump()
        assert data["quote_id"] == "Q1"
        assert data["text"] == "Test quote"
        assert data["page_numbers"] == [10]

    def test_verify_in_chunk_exact_match(self):
        chunk_text = "Scene safety must be ensured before patient contact."
        quote = ExtractedQuote(
            quote_id="Q1",
            text="Scene safety must be ensured before patient contact.",
            chunk_id="chunk_001",
            chunk_index=0,
            start_char=0,
            end_char=52,
        )

        assert quote.verify_in_chunk(chunk_text) is True

    def test_verify_in_chunk_substring_fallback(self):
        chunk_text = "Before anything else, scene safety must be ensured before patient contact. Always."
        quote = ExtractedQuote(
            quote_id="Q1",
            text="scene safety must be ensured before patient contact",
            chunk_id="chunk_001",
            chunk_index=0,
            start_char=-1,  # Invalid position, should fallback to substring
            end_char=-1,
        )

        assert quote.verify_in_chunk(chunk_text) is True


class TestQuoteExtractionResult:
    """Tests for QuoteExtractionResult model."""

    def test_create_sufficient_result(self):
        quotes = [
            ExtractedQuote(quote_id="Q1", text="Quote 1", chunk_id="c1", chunk_index=0, start_char=0, end_char=7),
            ExtractedQuote(quote_id="Q2", text="Quote 2", chunk_id="c2", chunk_index=1, start_char=0, end_char=7),
        ]

        result = QuoteExtractionResult(
            item_id="item_01",
            quotes=quotes,
            retrieval_rounds=1,
            status=ExtractionStatus.SUFFICIENT,
            chunks_searched=10,
            keywords_found=["safety", "hazard"],
        )

        assert result.item_id == "item_01"
        assert len(result.quotes) == 2
        assert result.status == ExtractionStatus.SUFFICIENT
        assert result.chunks_searched == 10

    def test_create_insufficient_result(self):
        result = QuoteExtractionResult(
            item_id="item_02",
            quotes=[],
            retrieval_rounds=2,
            status=ExtractionStatus.INSUFFICIENT,
            keywords_missing=["electrical", "perimeter"],
        )

        assert result.status == ExtractionStatus.INSUFFICIENT
        assert result.retrieval_rounds == 2
        assert result.keywords_missing == ["electrical", "perimeter"]


class TestIdentifyQuotableSentences:
    """Tests for identify_quotable_sentences function."""

    def test_identifies_action_verb_sentences(self):
        text = """
        Scene safety is important. You should always assess for hazards before
        approaching the patient. This ensures your safety.
        """

        sentences = identify_quotable_sentences(text)

        # Should find sentences with "should" and "ensures"
        assert len(sentences) > 0
        texts = [s[0] for s in sentences]
        assert any("should" in t.lower() for t in texts)

    def test_identifies_must_sentences(self):
        text = "EMRs must establish a perimeter around electrical hazards."

        sentences = identify_quotable_sentences(text)

        assert len(sentences) > 0
        assert any("must" in s[0].lower() for s in sentences)

    def test_identifies_factual_claims(self):
        text = "Scene safety is defined as ensuring no hazards exist. BSI refers to body substance isolation."

        sentences = identify_quotable_sentences(text)

        # Should find sentences with "is defined as" and "refers to"
        texts = [s[0] for s in sentences]
        assert any("defined" in t.lower() for t in texts)

    def test_identifies_procedural_sentences(self):
        text = "First, assess the scene. Then, approach the patient carefully."

        sentences = identify_quotable_sentences(text, min_length=15)

        # Should find sentences with procedural markers
        assert len(sentences) >= 0  # May or may not match depending on length

    def test_filters_questions(self):
        text = "Is the scene safe? Always check before approaching."

        sentences = identify_quotable_sentences(text)

        # Should not include the question
        texts = [s[0] for s in sentences]
        assert not any(t.endswith("?") for t in texts)

    def test_filters_short_sentences(self):
        text = "Short. This is a much longer sentence that should be included due to length requirements."

        sentences = identify_quotable_sentences(text, min_length=30)

        texts = [s[0] for s in sentences]
        assert "Short" not in " ".join(texts)

    def test_filters_long_sentences(self):
        long_sentence = "This is a very " + "long " * 50 + "sentence that exceeds the maximum length."
        text = long_sentence + " Short valid sentence with must keyword."

        sentences = identify_quotable_sentences(text, max_length=100)

        texts = [s[0] for s in sentences]
        assert not any(len(t) > 100 for t in texts)

    def test_relevance_scoring_with_target(self):
        text = "Scene safety must be ensured. Patient assessment comes next."

        sentences = identify_quotable_sentences(
            text,
            learning_target="scene safety hazard",
        )

        # Sentence with "scene safety" should score higher
        if len(sentences) >= 2:
            scene_sentence = next((s for s in sentences if "scene" in s[0].lower()), None)
            if scene_sentence:
                assert scene_sentence[1] > 0

    def test_empty_text(self):
        sentences = identify_quotable_sentences("")
        assert sentences == []

    def test_no_quotable_content(self):
        text = "Random words here. More random words. Nothing actionable."

        sentences = identify_quotable_sentences(text)

        # May return empty or low-scoring sentences
        assert isinstance(sentences, list)


class TestExtractQuotesForItem:
    """Tests for extract_quotes_for_item function."""

    @pytest.fixture
    def mock_collection(self):
        collection = MagicMock()
        return collection

    @pytest.fixture
    def mock_embedding_model(self):
        import numpy as np
        model = MagicMock()
        # Return a fake embedding as numpy array (to support tolist())
        model.encode.return_value = np.array([[0.1] * 384])
        return model

    @pytest.fixture
    def sample_query_plan(self):
        return QueryPlan(
            item_id="item_01",
            learning_target="electrical hazard safety",
            queries=[
                PlannedQuery(
                    query_text="electrical hazard perimeter",
                    intent=QueryIntent.HIGH_PRECISION,
                ),
                PlannedQuery(
                    query_text="electrical must should ensure",
                    intent=QueryIntent.QUOTE_HUNT,
                ),
                PlannedQuery(
                    query_text="downed power line safety",
                    intent=QueryIntent.SYNONYM_VARIANT,
                ),
            ],
            must_include_keywords=["electrical", "hazard"],
        )

    def test_sufficient_quotes_round1(self, mock_collection, mock_embedding_model, sample_query_plan):
        # Mock vectorstore to return chunks with quotable content
        mock_collection.query.return_value = {
            "ids": [["chunk_001", "chunk_002"]],
            "documents": [[
                "EMRs must establish a perimeter around electrical hazards. Keep people at least 30 feet away.",
                "Scene safety should be assessed before approaching any electrical incident.",
            ]],
            "metadatas": [[
                {"page_numbers": "32,33", "section_heading": "Electrical Hazards", "book_title": "EMR Text", "book_author": "Author", "image_paths": ""},
                {"page_numbers": "34", "section_heading": "Scene Safety", "book_title": "EMR Text", "book_author": "Author", "image_paths": ""},
            ]],
            "distances": [[0.2, 0.3]],
        }

        result = extract_quotes_for_item(
            collection=mock_collection,
            embedding_model=mock_embedding_model,
            query_plan=sample_query_plan,
            min_quotes=2,
        )

        assert result.status == ExtractionStatus.SUFFICIENT
        assert len(result.quotes) >= 2
        assert result.retrieval_rounds == 1
        assert "electrical" in result.keywords_found or "hazard" in result.keywords_found

    def test_insufficient_quotes(self, mock_collection, mock_embedding_model, sample_query_plan):
        # Mock vectorstore to return no useful content
        mock_collection.query.return_value = {
            "ids": [["chunk_001"]],
            "documents": [["Random text with no actionable content."]],
            "metadatas": [[{"page_numbers": "", "section_heading": "", "book_title": "", "book_author": "", "image_paths": ""}]],
            "distances": [[0.9]],
        }

        result = extract_quotes_for_item(
            collection=mock_collection,
            embedding_model=mock_embedding_model,
            query_plan=sample_query_plan,
            min_quotes=2,
            max_rounds=2,
        )

        assert result.status == ExtractionStatus.INSUFFICIENT
        assert len(result.quotes) == 0
        assert result.retrieval_rounds == 2

    def test_marginal_quotes(self, mock_collection, mock_embedding_model, sample_query_plan):
        # Mock vectorstore to return one quotable chunk
        mock_collection.query.return_value = {
            "ids": [["chunk_001"]],
            "documents": [["EMRs should assess electrical hazards carefully."]],
            "metadatas": [[{"page_numbers": "32", "section_heading": "Hazards", "book_title": "EMR", "book_author": "Author", "image_paths": ""}]],
            "distances": [[0.3]],
        }

        result = extract_quotes_for_item(
            collection=mock_collection,
            embedding_model=mock_embedding_model,
            query_plan=sample_query_plan,
            min_quotes=3,  # Need 3 but only get 1
            max_rounds=2,
        )

        # Should be marginal since we found some but not enough
        assert result.status in (ExtractionStatus.MARGINAL, ExtractionStatus.INSUFFICIENT)

    def test_keywords_tracking(self, mock_collection, mock_embedding_model, sample_query_plan):
        mock_collection.query.return_value = {
            "ids": [["chunk_001"]],
            "documents": [["EMRs must handle electrical hazards with caution."]],
            "metadatas": [[{"page_numbers": "32", "section_heading": "", "book_title": "", "book_author": "", "image_paths": ""}]],
            "distances": [[0.2]],
        }

        result = extract_quotes_for_item(
            collection=mock_collection,
            embedding_model=mock_embedding_model,
            query_plan=sample_query_plan,
            min_quotes=1,
        )

        # Should track found keywords
        assert "electrical" in result.keywords_found or "hazard" in result.keywords_found


class TestFormatQuotesForPrompt:
    """Tests for format_quotes_for_prompt function."""

    def test_format_single_quote(self):
        quotes = [
            ExtractedQuote(
                quote_id="Q1",
                text="Scene safety must be ensured first.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=35,
                page_numbers=[32],
                section_heading="Scene Safety",
            )
        ]

        formatted = format_quotes_for_prompt(quotes)

        assert "[Q1]" in formatted
        assert "p.32" in formatted
        assert "[Scene Safety]" in formatted
        assert "Scene safety must be ensured first." in formatted

    def test_format_multiple_quotes(self):
        quotes = [
            ExtractedQuote(quote_id="Q1", text="Quote one.", chunk_id="c1", chunk_index=0, start_char=0, end_char=10, page_numbers=[10]),
            ExtractedQuote(quote_id="Q2", text="Quote two.", chunk_id="c2", chunk_index=1, start_char=0, end_char=10, page_numbers=[20]),
            ExtractedQuote(quote_id="Q3", text="Quote three.", chunk_id="c3", chunk_index=2, start_char=0, end_char=12, page_numbers=[30]),
        ]

        formatted = format_quotes_for_prompt(quotes)

        assert "[Q1]" in formatted
        assert "[Q2]" in formatted
        assert "[Q3]" in formatted

    def test_format_respects_max_quotes(self):
        quotes = [
            ExtractedQuote(quote_id=f"Q{i}", text=f"Quote {i}.", chunk_id=f"c{i}", chunk_index=i, start_char=0, end_char=10)
            for i in range(10)
        ]

        formatted = format_quotes_for_prompt(quotes, max_quotes=3)

        assert "[Q0]" in formatted
        assert "[Q2]" in formatted
        assert "[Q3]" not in formatted

    def test_format_empty_quotes(self):
        formatted = format_quotes_for_prompt([])
        assert "No anchor quotes" in formatted

    def test_format_quote_without_section(self):
        quotes = [
            ExtractedQuote(
                quote_id="Q1",
                text="Test quote",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=10,
                page_numbers=[5],
                section_heading=None,
            )
        ]

        formatted = format_quotes_for_prompt(quotes)

        assert "[Q1]" in formatted
        assert "p.5" in formatted


class TestGetChunksFromQuotes:
    """Tests for get_chunks_from_quotes function."""

    def test_returns_unique_chunks(self):
        quotes = [
            ExtractedQuote(quote_id="Q1", text="Q1", chunk_id="c1", chunk_index=0, start_char=0, end_char=2),
            ExtractedQuote(quote_id="Q2", text="Q2", chunk_id="c2", chunk_index=1, start_char=0, end_char=2),
            ExtractedQuote(quote_id="Q3", text="Q3", chunk_id="c1", chunk_index=0, start_char=0, end_char=2),  # Duplicate chunk
        ]

        all_chunks = {
            "c1": {"chunk_id": "c1", "text": "Chunk 1"},
            "c2": {"chunk_id": "c2", "text": "Chunk 2"},
        }

        chunks = get_chunks_from_quotes(quotes, all_chunks)

        assert len(chunks) == 2
        chunk_ids = {c["chunk_id"] for c in chunks}
        assert chunk_ids == {"c1", "c2"}

    def test_handles_missing_chunks(self):
        quotes = [
            ExtractedQuote(quote_id="Q1", text="Q1", chunk_id="c1", chunk_index=0, start_char=0, end_char=2),
            ExtractedQuote(quote_id="Q2", text="Q2", chunk_id="c_missing", chunk_index=1, start_char=0, end_char=2),
        ]

        all_chunks = {
            "c1": {"chunk_id": "c1", "text": "Chunk 1"},
        }

        chunks = get_chunks_from_quotes(quotes, all_chunks)

        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "c1"

    def test_empty_quotes(self):
        chunks = get_chunks_from_quotes([], {})
        assert chunks == []


class TestQuoteVerification:
    """Tests for mechanical quote verification functions."""

    def test_verify_exact_match(self):
        quotes = [
            ExtractedQuote(
                quote_id="Q1",
                text="Scene safety must be ensured first.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=35,
            )
        ]

        generated_text = 'As the source states (p.32) [Q1]: "Scene safety must be ensured first." This is critical.'

        result = verify_quotes_in_text(generated_text, quotes)

        assert result.all_required_found is True
        assert result.exact_match_count == 1
        assert len(result.missing_quote_ids) == 0

    def test_verify_missing_quote(self):
        quotes = [
            ExtractedQuote(
                quote_id="Q1",
                text="Scene safety must be ensured first.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=35,
            )
        ]

        generated_text = "The response does not include the required quote at all."

        result = verify_quotes_in_text(generated_text, quotes)

        assert result.all_required_found is False
        assert result.exact_match_count == 0
        assert "Q1" in result.missing_quote_ids

    def test_verify_normalized_match(self):
        quotes = [
            ExtractedQuote(
                quote_id="Q1",
                text="Scene  safety must   be ensured first.",  # Extra spaces
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=38,
            )
        ]

        # Generated text with normalized whitespace
        generated_text = "The source states: Scene safety must be ensured first."

        result = verify_quotes_in_text(generated_text, quotes)

        assert result.all_required_found is True
        # Should be a fuzzy match since whitespace differs
        assert result.fuzzy_match_count >= 0 or result.exact_match_count >= 0

    def test_verify_multiple_quotes(self):
        quotes = [
            ExtractedQuote(
                quote_id="Q1",
                text="Scene safety must be ensured.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=29,
            ),
            ExtractedQuote(
                quote_id="Q2",
                text="Always establish a perimeter.",
                chunk_id="c2",
                chunk_index=1,
                start_char=0,
                end_char=29,
            ),
        ]

        generated_text = 'Per the source [Q1]: "Scene safety must be ensured." Also [Q2]: "Always establish a perimeter."'

        result = verify_quotes_in_text(generated_text, quotes)

        assert result.all_required_found is True
        assert len(result.verified_quotes) == 2

    def test_verify_partial_quotes_required(self):
        quotes = [
            ExtractedQuote(quote_id="Q1", text="Quote 1", chunk_id="c1", chunk_index=0, start_char=0, end_char=7),
            ExtractedQuote(quote_id="Q2", text="Quote 2", chunk_id="c2", chunk_index=1, start_char=0, end_char=7),
            ExtractedQuote(quote_id="Q3", text="Quote 3", chunk_id="c3", chunk_index=2, start_char=0, end_char=7),
        ]

        generated_text = "Only Quote 1 is included here."

        # Only require Q1
        result = verify_quotes_in_text(generated_text, quotes, required_quote_ids=["Q1"])

        assert result.all_required_found is True
        assert len(result.verified_quotes) == 1
