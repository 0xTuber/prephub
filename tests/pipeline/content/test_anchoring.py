"""Tests for anchor-driven drafting module."""

import pytest

from course_builder.domain.content import AnchorQuote
from course_builder.pipeline.content.anchoring import (
    AnchorSelectionResult,
    compute_relevance_score,
    extract_key_terms,
    extract_quotable_sentences,
    format_anchors_for_prompt,
    select_anchor_quotes,
    validate_anchor_usage,
)


class TestExtractKeyTerms:
    """Tests for extract_key_terms function."""

    def test_extracts_meaningful_terms(self):
        text = "Assess the patient for signs of respiratory distress"
        terms = extract_key_terms(text)

        assert "assess" in terms
        assert "patient" in terms
        assert "signs" in terms
        assert "respiratory" in terms
        assert "distress" in terms

    def test_filters_stop_words(self):
        text = "The patient is in the room with the doctor"
        terms = extract_key_terms(text)

        assert "the" not in terms
        assert "is" not in terms
        assert "in" not in terms
        assert "with" not in terms

    def test_filters_short_words(self):
        text = "Check for any red or blue skin"
        terms = extract_key_terms(text)

        assert "for" not in terms
        assert "any" not in terms
        assert "red" not in terms
        assert "skin" in terms

    def test_empty_text(self):
        terms = extract_key_terms("")
        assert terms == []


class TestComputeRelevanceScore:
    """Tests for compute_relevance_score function."""

    def test_exact_match_high_score(self):
        chunk = "Scene safety is the first priority for every EMR"
        target = "scene safety priority"
        score = compute_relevance_score(chunk, target, "beginner")

        assert score > 0.8

    def test_no_overlap_low_score(self):
        chunk = "Cardiac arrest requires immediate CPR"
        target = "scene safety hazard assessment"
        score = compute_relevance_score(chunk, target, "beginner")

        assert score < 0.3

    def test_difficulty_bonus_advanced(self):
        chunk = "First priority is to establish scene safety before patient contact"
        target = "scene safety"

        score_advanced = compute_relevance_score(chunk, target, "advanced")
        score_beginner = compute_relevance_score(chunk, target, "beginner")

        # Advanced gets bonus for priority language
        assert score_advanced >= score_beginner

    def test_action_verb_bonus(self):
        chunk = "You should ensure scene safety before approaching"
        target = "scene safety"
        score = compute_relevance_score(chunk, target, "intermediate")

        # Should get action verb bonus
        assert score > 0.5


class TestExtractQuotableSentences:
    """Tests for extract_quotable_sentences function."""

    def test_extracts_substantive_sentences(self):
        text = """
        Scene safety is critical. Always assess for hazards before
        approaching the patient. This ensures EMR safety.
        """
        sentences = extract_quotable_sentences(text, min_length=20, max_length=100)

        assert len(sentences) > 0
        assert any("safety" in s.lower() for s in sentences)

    def test_filters_questions(self):
        text = "Is the scene safe? Always check before approaching."
        sentences = extract_quotable_sentences(text)

        # Should not include the question
        assert not any(s.endswith("?") for s in sentences)

    def test_filters_list_items(self):
        text = "- First item\n- Second item\nThis is a complete sentence."
        sentences = extract_quotable_sentences(text)

        assert not any(s.startswith("-") for s in sentences)

    def test_filters_short_sentences(self):
        text = "Short. This is a much longer sentence that should be included."
        sentences = extract_quotable_sentences(text, min_length=30)

        assert "Short" not in " ".join(sentences)

    def test_prefers_sentences_with_action_verbs(self):
        text = """
        Scene safety is important. You should always ensure scene safety
        before patient contact. EMR stands for Emergency Medical Responder.
        """
        sentences = extract_quotable_sentences(text)

        # Should prefer "should ensure" sentence
        action_sentences = [s for s in sentences if "should" in s.lower()]
        assert len(action_sentences) > 0


class TestSelectAnchorQuotes:
    """Tests for select_anchor_quotes function."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                "chunk_id": "chunk_1",
                "text": "Scene safety must be ensured before any patient contact. Always scan for hazards including traffic, electrical, and environmental dangers.",
                "pages": [32, 33],
            },
            {
                "chunk_id": "chunk_2",
                "text": "The EMR should position the vehicle to protect the scene. Warning lights should be activated to alert other drivers.",
                "pages": [75, 76],
            },
            {
                "chunk_id": "chunk_3",
                "text": "Assessment findings guide treatment decisions. Vital signs include pulse, respiration, and blood pressure.",
                "pages": [102],
            },
        ]

    def test_selects_relevant_anchors(self, sample_chunks):
        result = select_anchor_quotes(
            chunks=sample_chunks,
            learning_target="scene safety hazard assessment",
            difficulty="intermediate",
        )

        assert isinstance(result, AnchorSelectionResult)
        assert len(result.anchors) > 0
        assert result.coverage_score > 0

    def test_respects_max_anchors(self, sample_chunks):
        result = select_anchor_quotes(
            chunks=sample_chunks,
            learning_target="scene safety",
            difficulty="intermediate",
            max_anchors=1,
        )

        assert len(result.anchors) <= 1

    def test_handles_empty_chunks(self):
        result = select_anchor_quotes(
            chunks=[],
            learning_target="scene safety",
            difficulty="intermediate",
        )

        assert len(result.anchors) == 0
        assert result.coverage_score == 0.0

    def test_includes_page_numbers(self, sample_chunks):
        result = select_anchor_quotes(
            chunks=sample_chunks,
            learning_target="scene safety",
            difficulty="intermediate",
        )

        if result.anchors:
            assert any(len(a.page_numbers) > 0 for a in result.anchors)

    def test_includes_rationale(self, sample_chunks):
        result = select_anchor_quotes(
            chunks=sample_chunks,
            learning_target="scene safety",
            difficulty="intermediate",
        )

        assert result.selection_rationale is not None
        assert len(result.selection_rationale) > 0


class TestFormatAnchorsForPrompt:
    """Tests for format_anchors_for_prompt function."""

    def test_formats_single_anchor(self):
        anchors = [
            AnchorQuote(
                text="Scene safety is the first priority",
                chunk_id="chunk_1",
                chunk_index=0,
                page_numbers=[32],
                relevance_score=0.9,
            )
        ]

        formatted = format_anchors_for_prompt(anchors)

        assert "ANCHOR 1" in formatted
        assert "p.32" in formatted
        assert "Scene safety is the first priority" in formatted

    def test_formats_multiple_anchors(self):
        anchors = [
            AnchorQuote(
                text="First quote",
                chunk_id="chunk_1",
                chunk_index=0,
                page_numbers=[32],
                relevance_score=0.9,
            ),
            AnchorQuote(
                text="Second quote",
                chunk_id="chunk_2",
                chunk_index=1,
                page_numbers=[45, 46],
                relevance_score=0.8,
            ),
        ]

        formatted = format_anchors_for_prompt(anchors)

        assert "ANCHOR 1" in formatted
        assert "ANCHOR 2" in formatted
        assert "First quote" in formatted
        assert "Second quote" in formatted


class TestValidateAnchorUsage:
    """Tests for validate_anchor_usage function."""

    def test_validates_used_anchors(self):
        question_data = {
            "options": ["Scene safety must be ensured first", "Other option"],
            "correct_index": 0,
            "explanation": "As the source states, scene safety must be ensured before patient contact.",
            "anchor_usage": ["scene safety", "ensured"],
        }
        anchors = [
            AnchorQuote(
                text="Scene safety must be ensured before any patient contact",
                chunk_id="chunk_1",
                chunk_index=0,
                page_numbers=[32],
                relevance_score=0.9,
            )
        ]

        is_valid, matched, missing = validate_anchor_usage(question_data, anchors)

        assert is_valid
        assert len(matched) > 0

    def test_detects_missing_anchor_usage(self):
        question_data = {
            "options": ["Completely different answer", "Other option"],
            "correct_index": 0,
            "explanation": "Because it's the right thing to do.",
            "anchor_usage": [],
        }
        anchors = [
            AnchorQuote(
                text="Scene safety must be ensured before any patient contact",
                chunk_id="chunk_1",
                chunk_index=0,
                page_numbers=[32],
                relevance_score=0.9,
            )
        ]

        is_valid, matched, missing = validate_anchor_usage(question_data, anchors)

        assert len(missing) > 0
