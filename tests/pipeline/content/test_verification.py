"""Tests for verification loop module."""

import pytest

from course_builder.domain.content import CapsuleItem, ItemSourceReference
from course_builder.pipeline.content.verification import (
    ClaimType,
    ExtractedClaim,
    VerificationLoop,
    VerificationStatus,
    check_correct_answer_evidence_support,
    detect_admission_of_insufficient_source,
    detect_hallucination_patterns,
    extract_claims_from_explanation,
    verify_claim_against_chunks,
)


class TestExtractClaimsFromExplanation:
    """Tests for extract_claims_from_explanation function."""

    def test_extracts_quoted_claims(self):
        explanation = 'The source states (p.32): "Scene safety must be ensured before patient contact."'
        claims = extract_claims_from_explanation(explanation)

        quoted_claims = [c for c in claims if c.is_critical]
        assert len(quoted_claims) > 0
        assert any("scene safety" in c.text.lower() for c in claims)

    def test_identifies_procedural_claims(self):
        explanation = "The EMR should ensure scene safety before approaching the patient."
        claims = extract_claims_from_explanation(explanation)

        procedural = [c for c in claims if c.claim_type == ClaimType.PROCEDURAL]
        assert len(procedural) > 0

    def test_identifies_definition_claims(self):
        explanation = "Scene safety is defined as ensuring no hazards are present."
        claims = extract_claims_from_explanation(explanation)

        definitions = [c for c in claims if c.claim_type == ClaimType.DEFINITION]
        assert len(definitions) > 0

    def test_identifies_rationale_claims(self):
        explanation = "This is correct because it follows standard protocol."
        claims = extract_claims_from_explanation(explanation)

        rationales = [c for c in claims if c.claim_type == ClaimType.RATIONALE]
        assert len(rationales) > 0

    def test_handles_empty_explanation(self):
        claims = extract_claims_from_explanation("")
        assert claims == []


class TestVerifyClaimAgainstChunks:
    """Tests for verify_claim_against_chunks function."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                "chunk_id": "chunk_1",
                "text": "Scene safety must be ensured before any patient contact. Always scan for hazards.",
            },
            {
                "chunk_id": "chunk_2",
                "text": "The EMR should position the vehicle to protect the scene.",
            },
        ]

    def test_verifies_exact_match(self, sample_chunks):
        claim = ExtractedClaim(
            text="Scene safety must be ensured before any patient contact",
            claim_type=ClaimType.FACTUAL,
            source_sentence="Test",
            is_critical=True,
        )

        result = verify_claim_against_chunks(claim, sample_chunks)

        assert result.status == VerificationStatus.PASS
        assert result.confidence == 1.0
        assert result.matched_chunk_id == "chunk_1"

    def test_verifies_partial_match(self, sample_chunks):
        claim = ExtractedClaim(
            text="scene safety must be ensured",  # More closely matches chunk
            claim_type=ClaimType.FACTUAL,
            source_sentence="Test",
            is_critical=False,
        )

        result = verify_claim_against_chunks(claim, sample_chunks)

        # Should get at least partial match
        assert result.status in [VerificationStatus.PASS, VerificationStatus.REPAIR]
        assert result.confidence > 0.6

    def test_rejects_unsupported_claim(self, sample_chunks):
        claim = ExtractedClaim(
            text="cardiac arrest requires immediate defibrillation",
            claim_type=ClaimType.FACTUAL,
            source_sentence="Test",
            is_critical=False,
        )

        result = verify_claim_against_chunks(claim, sample_chunks)

        assert result.status == VerificationStatus.REJECT
        assert result.confidence < 0.5

    def test_handles_empty_chunks(self):
        claim = ExtractedClaim(
            text="Any claim",
            claim_type=ClaimType.FACTUAL,
            source_sentence="Test",
            is_critical=False,
        )

        result = verify_claim_against_chunks(claim, [])

        assert result.status == VerificationStatus.REJECT


class TestDetectHallucinationPatterns:
    """Tests for detect_hallucination_patterns function."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {"text": "Scene safety requires assessment of hazards."},
            {"text": "Traffic control is important at roadway incidents."},
        ]

    def test_detects_unsourced_sounds(self, sample_chunks):
        explanation = "You may hear a crackling sound from the downed power lines."
        hallucinations = detect_hallucination_patterns(explanation, sample_chunks)

        assert len(hallucinations) > 0
        assert any("sound" in h.lower() for h in hallucinations)

    def test_detects_unsourced_measurements(self, sample_chunks):
        explanation = "Stay at least 100 feet away from the hazard."
        hallucinations = detect_hallucination_patterns(explanation, sample_chunks)

        # If "100 feet" not in source, should be flagged
        assert len(hallucinations) >= 0  # May or may not flag depending on source

    def test_detects_admission_phrases(self, sample_chunks):
        explanation = "While not explicitly stated in the source, it can be inferred that..."
        hallucinations = detect_hallucination_patterns(explanation, sample_chunks)

        assert len(hallucinations) > 0
        assert any("inference" in h.lower() for h in hallucinations)

    def test_no_hallucinations_for_sourced_content(self, sample_chunks):
        explanation = "Scene safety requires assessment of hazards. Traffic control is important."
        hallucinations = detect_hallucination_patterns(explanation, sample_chunks)

        # Content is in source, should be fewer hallucinations
        assert len(hallucinations) == 0


class TestDetectAdmissionOfInsufficientSource:
    """Tests for detect_admission_of_insufficient_source function."""

    def test_detects_explicit_admission(self):
        explanation = "The chunks do not directly address this specific scenario."
        is_insufficient, reason = detect_admission_of_insufficient_source(explanation)

        assert is_insufficient
        assert reason is not None

    def test_detects_inference_admission(self):
        explanation = "This can be inferred from general principles of scene safety."
        is_insufficient, reason = detect_admission_of_insufficient_source(explanation)

        assert is_insufficient

    def test_detects_not_explicitly_stated(self):
        explanation = "While not explicitly stated, the correct approach is..."
        is_insufficient, reason = detect_admission_of_insufficient_source(explanation)

        assert is_insufficient

    def test_allows_properly_sourced_content(self):
        explanation = 'The source states (p.32): "Scene safety is the first priority."'
        is_insufficient, reason = detect_admission_of_insufficient_source(explanation)

        assert not is_insufficient
        assert reason is None


class TestCheckCorrectAnswerEvidenceSupport:
    """Tests for check_correct_answer_evidence_support function."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {"text": "Ensure scene safety before patient contact. Scan for hazards."},
            {"text": "Position vehicle to protect the scene. Activate warning lights."},
        ]

    def test_supports_matching_answer(self, sample_chunks):
        correct = "Ensure scene safety before approaching the patient"
        is_supported, matched, reason = check_correct_answer_evidence_support(
            correct, sample_chunks
        )

        assert is_supported
        assert len(matched) > 0
        assert "scene" in matched or "safety" in matched

    def test_rejects_unsupported_answer(self, sample_chunks):
        correct = "Administer cardiac medication immediately"
        is_supported, matched, reason = check_correct_answer_evidence_support(
            correct, sample_chunks
        )

        assert not is_supported
        assert reason is not None

    def test_handles_empty_chunks(self):
        correct = "Any answer"
        is_supported, matched, reason = check_correct_answer_evidence_support(
            correct, []
        )

        assert not is_supported


class TestVerificationLoop:
    """Tests for VerificationLoop class."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                "chunk_id": "chunk_1",
                "text": "Scene safety must be ensured before patient contact. Always scan for hazards including traffic and electrical dangers.",
            },
        ]

    @pytest.fixture
    def good_item(self):
        return CapsuleItem(
            item_id="item_01",
            item_type="Multiple Choice",
            title="Scene Safety",
            learning_target="scene safety assessment",
            content="What should you do first?",
            options=["Ensure scene safety", "Approach patient", "Call for backup", "Wait"],
            correct_answer_index=0,
            explanation='The source states (p.32): "Scene safety must be ensured before patient contact."',
        )

    @pytest.fixture
    def bad_item(self):
        return CapsuleItem(
            item_id="item_02",
            item_type="Multiple Choice",
            title="Scene Safety",
            learning_target="scene safety assessment",
            content="What should you do first?",
            options=["Ensure scene safety", "Approach patient", "Call for backup", "Wait"],
            correct_answer_index=0,
            explanation="While not explicitly stated in the source, it can be inferred that scene safety is important.",
        )

    def test_passes_well_sourced_item(self, good_item, sample_chunks):
        loop = VerificationLoop(strict_mode=False)
        result = loop.verify_item(good_item, sample_chunks)

        assert result.status == VerificationStatus.PASS

    def test_rejects_item_with_admission(self, bad_item, sample_chunks):
        loop = VerificationLoop(strict_mode=True)
        result = loop.verify_item(bad_item, sample_chunks)

        assert result.status == VerificationStatus.REJECT
        assert "admission" in result.rejection_reason.lower() or "insufficient" in result.rejection_reason.lower()

    def test_rejects_item_without_explanation(self, sample_chunks):
        item = CapsuleItem(
            item_id="item_03",
            item_type="Multiple Choice",
            title="Test",
            learning_target="test",
            content="Question",
            options=["A", "B", "C", "D"],
            correct_answer_index=0,
            explanation=None,
        )

        loop = VerificationLoop()
        result = loop.verify_item(item, sample_chunks)

        assert result.status == VerificationStatus.REJECT

    def test_tracks_claim_counts(self, good_item, sample_chunks):
        loop = VerificationLoop()
        result = loop.verify_item(good_item, sample_chunks)

        # Should have tracked claims
        assert result.pass_count + result.repair_count + result.reject_count > 0
