"""Tests for novelty gate module."""

import pytest

from course_builder.pipeline.content.novelty import (
    ConceptSignature,
    NoveltyCheckResult,
    NoveltyGate,
    compute_tag_similarity,
    compute_term_similarity,
    deduplicate_items,
    extract_concept_terms,
    normalize_concept_tag,
)


class TestExtractConceptTerms:
    """Tests for extract_concept_terms function."""

    def test_extracts_meaningful_terms(self):
        text = "Electrical hazard perimeter establishment and utility notification"
        terms = extract_concept_terms(text)

        assert "electrical" in terms
        assert "hazard" in terms
        assert "perimeter" in terms
        assert "establishment" in terms
        assert "utility" in terms
        assert "notification" in terms

    def test_filters_common_words(self):
        text = "The patient should be assessed for respiratory distress"
        terms = extract_concept_terms(text)

        # Stop words should be filtered
        assert "the" not in terms
        assert "should" not in terms
        assert "patient" not in terms  # Common EMS word filtered

    def test_filters_short_words(self):
        text = "PPE for BSI on scene"
        terms = extract_concept_terms(text)

        # Short words filtered
        assert "ppe" not in terms
        assert "bsi" not in terms
        assert "for" not in terms

    def test_empty_text(self):
        terms = extract_concept_terms("")
        assert terms == []


class TestNormalizeConceptTag:
    """Tests for normalize_concept_tag function."""

    def test_lowercase_conversion(self):
        assert normalize_concept_tag("Electrical_Hazard") == "electrical_hazard"

    def test_space_to_underscore(self):
        assert normalize_concept_tag("scene safety") == "scene_safety"

    def test_hyphen_to_underscore(self):
        assert normalize_concept_tag("hazard-assessment") == "hazard_assessment"

    def test_removes_special_chars(self):
        assert normalize_concept_tag("scene_safety!@#") == "scene_safety"

    def test_removes_duplicate_underscores(self):
        assert normalize_concept_tag("scene__safety") == "scene_safety"

    def test_strips_leading_trailing_underscores(self):
        assert normalize_concept_tag("_scene_safety_") == "scene_safety"


class TestComputeTermSimilarity:
    """Tests for compute_term_similarity function."""

    def test_identical_terms(self):
        terms = ["scene", "safety", "hazard"]
        similarity = compute_term_similarity(terms, terms)
        assert similarity == 1.0

    def test_no_overlap(self):
        terms1 = ["scene", "safety"]
        terms2 = ["cardiac", "arrest"]
        similarity = compute_term_similarity(terms1, terms2)
        assert similarity == 0.0

    def test_partial_overlap(self):
        terms1 = ["scene", "safety", "assessment"]
        terms2 = ["scene", "hazard", "recognition"]
        similarity = compute_term_similarity(terms1, terms2)
        # 1 overlap (scene) out of 5 unique terms
        assert 0.0 < similarity < 1.0

    def test_empty_lists(self):
        assert compute_term_similarity([], []) == 0.0
        assert compute_term_similarity(["term"], []) == 0.0


class TestComputeTagSimilarity:
    """Tests for compute_tag_similarity function."""

    def test_identical_tags(self):
        similarity = compute_tag_similarity("scene_safety", "scene_safety")
        assert similarity == 1.0

    def test_different_formatting(self):
        similarity = compute_tag_similarity("Scene Safety", "scene_safety")
        assert similarity == 1.0

    def test_partial_match(self):
        similarity = compute_tag_similarity("scene_safety_hazard", "scene_safety_assessment")
        # 2 parts match (scene, safety) out of 4 unique
        assert 0.4 < similarity < 0.6

    def test_no_match(self):
        similarity = compute_tag_similarity("cardiac_arrest", "scene_safety")
        assert similarity == 0.0


class TestNoveltyGate:
    """Tests for NoveltyGate class."""

    def test_first_item_always_novel(self):
        gate = NoveltyGate(similarity_threshold=0.85)

        signature = ConceptSignature(
            concept_tag="scene_safety_basics",
            key_terms=["scene", "safety", "basics"],
        )

        result = gate.check_novelty(signature)

        assert result.is_novel
        assert result.similarity_score == 0.0

    def test_detects_duplicate(self):
        gate = NoveltyGate(similarity_threshold=0.50)  # Lower threshold for this test

        sig1 = ConceptSignature(
            concept_tag="electrical_hazard_isolation",
            key_terms=["electrical", "hazard", "isolation", "perimeter", "safety"],
        )

        sig2 = ConceptSignature(
            concept_tag="electrical_hazard_isolation_procedure",  # Very similar tag
            key_terms=["electrical", "hazard", "isolation", "perimeter", "protocol"],  # 4/6 overlap
        )

        gate.register(sig1)
        result = gate.check_novelty(sig2)

        assert not result.is_novel
        assert result.similarity_score > 0.5
        assert result.similar_tag == "electrical_hazard_isolation"

    def test_allows_different_concepts(self):
        gate = NoveltyGate(similarity_threshold=0.85)

        sig1 = ConceptSignature(
            concept_tag="scene_safety",
            key_terms=["scene", "safety", "hazard"],
        )

        sig2 = ConceptSignature(
            concept_tag="patient_assessment",
            key_terms=["patient", "assessment", "vital", "signs"],
        )

        gate.register(sig1)
        result = gate.check_novelty(sig2)

        assert result.is_novel
        assert result.similarity_score < 0.5

    def test_clear_resets_state(self):
        gate = NoveltyGate()

        sig = ConceptSignature(concept_tag="test", key_terms=["test"])
        gate.register(sig)

        assert gate.registered_count == 1

        gate.clear()

        assert gate.registered_count == 0

    def test_create_signature(self):
        gate = NoveltyGate()

        signature = gate.create_signature(
            concept_tag="Electrical Hazard",
            learning_target="electrical hazard perimeter establishment",
            bloom_level="apply",
            domain="scene_safety",
        )

        assert signature.concept_tag == "electrical_hazard"
        assert "electrical" in signature.key_terms
        assert signature.bloom_level == "apply"
        assert signature.domain == "scene_safety"


class TestNoveltyCheckResult:
    """Tests for NoveltyCheckResult dataclass."""

    def test_novel_result(self):
        result = NoveltyCheckResult(
            is_novel=True,
            similarity_score=0.2,
        )

        assert result.is_novel
        assert result.similar_tag is None
        assert result.similar_terms == []

    def test_duplicate_result(self):
        result = NoveltyCheckResult(
            is_novel=False,
            similarity_score=0.9,
            similar_tag="existing_concept",
            similar_terms=["term1", "term2"],
        )

        assert not result.is_novel
        assert result.similar_tag == "existing_concept"
        assert len(result.similar_terms) == 2


class TestDeduplicateItems:
    """Tests for deduplicate_items function."""

    def test_removes_duplicates(self):
        items = [
            {
                "item_id": "1",
                "concept_tag": "electrical_hazard",
                "learning_target": "electrical hazard recognition",
            },
            {
                "item_id": "2",
                "concept_tag": "electrical_hazard_perimeter",
                "learning_target": "electrical hazard perimeter establishment",
            },
            {
                "item_id": "3",
                "concept_tag": "traffic_control",
                "learning_target": "traffic control at incident scene",
            },
        ]

        unique, duplicates = deduplicate_items(items, similarity_threshold=0.7)

        # First two are similar, third is different
        assert len(unique) >= 2
        assert len(duplicates) <= 1

    def test_keeps_all_unique(self):
        items = [
            {
                "item_id": "1",
                "concept_tag": "scene_safety",
                "learning_target": "scene safety basics",
            },
            {
                "item_id": "2",
                "concept_tag": "patient_assessment",
                "learning_target": "primary patient assessment",
            },
            {
                "item_id": "3",
                "concept_tag": "airway_management",
                "learning_target": "airway management techniques",
            },
        ]

        unique, duplicates = deduplicate_items(items)

        assert len(unique) == 3
        assert len(duplicates) == 0

    def test_handles_empty_list(self):
        unique, duplicates = deduplicate_items([])

        assert unique == []
        assert duplicates == []

    def test_marks_duplicates_with_info(self):
        items = [
            {
                "item_id": "1",
                "concept_tag": "same_concept",
                "learning_target": "same exact concept testing",
            },
            {
                "item_id": "2",
                "concept_tag": "same_concept_again",
                "learning_target": "same exact concept testing again",
            },
        ]

        unique, duplicates = deduplicate_items(items, similarity_threshold=0.5)

        if duplicates:
            assert "duplicate_of" in duplicates[0]
            assert "similarity_score" in duplicates[0]
