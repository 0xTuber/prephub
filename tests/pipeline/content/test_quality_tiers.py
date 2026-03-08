"""Tests for quality tiers module."""

import pytest

from course_builder.pipeline.content.quality_tiers import (
    GateTier,
    GateResult,
    GateCheckResult,
    QualityGateReport,
    check_verbatim_quote_presence,
    check_stem_answer_consistency,
    check_scope_legality,
    check_single_best_answer,
    check_internal_consistency,
    check_learning_target_alignment,
    check_distractor_quality,
    check_difficulty_alignment,
    check_decision_grounding,
    check_novelty,
    check_verbosity,
    check_style,
    run_all_gates,
    get_repair_hints,
)
from course_builder.pipeline.content.quote_extraction import (
    ExtractedQuote,
    GenerationQuoteVerification,
    QuoteVerificationResult,
)


class TestGateEnums:
    """Tests for gate tier and result enums."""

    def test_gate_tier_values(self):
        assert GateTier.TIER_1_HARD == "tier_1_hard"
        assert GateTier.TIER_2_REPAIR == "tier_2_repair"
        assert GateTier.TIER_3_SOFT == "tier_3_soft"

    def test_gate_result_values(self):
        assert GateResult.PASS == "pass"
        assert GateResult.FAIL == "fail"
        assert GateResult.WARN == "warn"


class TestTier1VerbatimQuotePresence:
    """Tests for Tier 1: Verbatim quote presence check."""

    def test_all_quotes_found(self):
        verification = GenerationQuoteVerification(
            item_id="item_01",
            required_quotes=["Q1", "Q2"],
            verified_quotes=[
                QuoteVerificationResult(quote_id="Q1", found=True, exact_match=True),
                QuoteVerificationResult(quote_id="Q2", found=True, exact_match=True),
            ],
            all_required_found=True,
            exact_match_count=2,
            fuzzy_match_count=0,
            missing_quote_ids=[],
        )

        result = check_verbatim_quote_presence("generated text", verification)

        assert result.result == GateResult.PASS
        assert result.tier == GateTier.TIER_1_HARD

    def test_missing_quotes(self):
        verification = GenerationQuoteVerification(
            item_id="item_01",
            required_quotes=["Q1", "Q2"],
            verified_quotes=[
                QuoteVerificationResult(quote_id="Q1", found=True, exact_match=True),
                QuoteVerificationResult(quote_id="Q2", found=False, exact_match=False),
            ],
            all_required_found=False,
            exact_match_count=1,
            fuzzy_match_count=0,
            missing_quote_ids=["Q2"],
        )

        result = check_verbatim_quote_presence("generated text", verification)

        assert result.result == GateResult.FAIL
        assert "Q2" in result.message


class TestTier1StemAnswerConsistency:
    """Tests for Tier 1: Stem-answer consistency check."""

    def test_no_leakage(self):
        result = check_stem_answer_consistency(
            stem="What is the first priority at an accident scene?",
            correct_answer="Ensure scene safety",
            distractors=["Call for backup", "Begin CPR", "Transport patient"],
        )

        assert result.result == GateResult.PASS

    def test_article_leakage(self):
        # "an" only matches correct answer starting with vowel
        result = check_stem_answer_consistency(
            stem="The device used is an",
            correct_answer="AED",  # Starts with vowel
            distractors=["CPR mask", "Bandage", "Tourniquet"],  # All start with consonants
        )

        assert result.result == GateResult.FAIL
        assert "article" in result.message.lower()

    def test_multiple_negations(self):
        result = check_stem_answer_consistency(
            stem="Which is NOT something you should never do?",
            correct_answer="Check the scene",
            distractors=["Ignore hazards", "Rush in", "Forget PPE"],
        )

        assert result.result == GateResult.FAIL
        assert "negation" in result.message.lower()


class TestTier1ScopeLegality:
    """Tests for Tier 1: Scope legality check."""

    def test_within_scope(self):
        item_content = {
            "item_id": "item_01",
            "scope_tags": ["scene_safety", "hazard_assessment"],
        }
        allowed_scope = ["scene_safety", "hazard_assessment", "patient_assessment"]

        result = check_scope_legality(item_content, allowed_scope)

        assert result.result == GateResult.PASS

    def test_out_of_scope(self):
        item_content = {
            "item_id": "item_01",
            "scope_tags": ["advanced_cardiac", "medication_admin"],
        }
        allowed_scope = ["scene_safety", "hazard_assessment"]

        result = check_scope_legality(item_content, allowed_scope)

        assert result.result == GateResult.FAIL
        assert "out-of-scope" in result.message.lower()

    def test_no_scope_tags(self):
        item_content = {"item_id": "item_01"}
        allowed_scope = ["scene_safety"]

        result = check_scope_legality(item_content, allowed_scope)

        assert result.result == GateResult.FAIL


class TestTier1SingleBestAnswer:
    """Tests for Tier 1: Single best answer check."""

    def test_valid_single_answer(self):
        result = check_single_best_answer(
            correct_answer="Ensure scene safety",
            distractors=["Begin CPR", "Call dispatch", "Transport patient"],
            explanation="Scene safety must be ensured before any other action.",
        )

        assert result.result == GateResult.PASS

    def test_duplicate_options(self):
        result = check_single_best_answer(
            correct_answer="Check the scene",
            distractors=["Check the scene", "Call for help", "Begin assessment"],
            explanation="Always check the scene first.",
        )

        assert result.result == GateResult.FAIL
        assert "duplicate" in result.message.lower()

    def test_also_correct_language(self):
        result = check_single_best_answer(
            correct_answer="Check the scene",
            distractors=["Call for help", "Begin assessment", "Document findings"],
            explanation="Check the scene first, though calling for help is also correct.",
        )

        assert result.result == GateResult.FAIL
        assert "multiple correct" in result.message.lower()


class TestTier1InternalConsistency:
    """Tests for Tier 1: Internal consistency check."""

    def test_consistent_item(self):
        result = check_internal_consistency(
            stem="What is the first step in patient assessment?",
            correct_answer="Assess scene safety",
            explanation="Scene safety should always be assessed first to protect responders.",
        )

        assert result.result == GateResult.PASS

    def test_contradiction_should_vs_should_not(self):
        result = check_internal_consistency(
            stem="What should you do first?",
            correct_answer="You should assess the scene",
            explanation="You should not assess the scene immediately. Instead wait for backup.",
        )

        assert result.result == GateResult.FAIL
        assert len(result.details.get("contradictions", [])) > 0

    def test_contradiction_always_vs_never(self):
        result = check_internal_consistency(
            stem="When should you always call for help?",
            correct_answer="Always call for help immediately",
            explanation="You should never call for help immediately. Wait to assess first.",
        )

        assert result.result == GateResult.FAIL

    def test_however_contradiction(self):
        result = check_internal_consistency(
            stem="What is the correct procedure?",
            correct_answer="Assess the scene first",
            explanation="You should assess the scene first. However, this is incorrect in emergencies.",
        )

        assert result.result == GateResult.FAIL


class TestTier2DistractorQuality:
    """Tests for Tier 2: Distractor quality check."""

    def test_good_distractors(self):
        result = check_distractor_quality(
            distractors=[
                "Begin chest compressions immediately",
                "Contact medical control for guidance",
                "Document patient's vital signs",
            ],
            correct_answer="Ensure scene safety first",
            explanation="Scene safety is the first priority.",
        )

        assert result.result == GateResult.PASS

    def test_too_short_distractor(self):
        result = check_distractor_quality(
            distractors=["Yes", "No", "Maybe"],
            correct_answer="Ensure scene safety first",
            explanation="Scene safety is the first priority.",
        )

        assert result.result == GateResult.FAIL
        assert result.repair_hint is not None

    def test_meta_option_distractor(self):
        result = check_distractor_quality(
            distractors=[
                "Begin assessment",
                "Call for help",
                "All of the above",
            ],
            correct_answer="Check scene safety",
            explanation="Scene safety first.",
        )

        assert result.result == GateResult.FAIL
        assert any("meta-option" in issue.lower() for issue in result.details.get("issues", []))


class TestTier2DifficultyAlignment:
    """Tests for Tier 2: Difficulty alignment check."""

    def test_matching_difficulty(self):
        item_content = {"difficulty": "medium"}
        result = check_difficulty_alignment(item_content, "medium")

        assert result.result == GateResult.PASS

    def test_one_level_off(self):
        item_content = {"difficulty": "easy"}
        result = check_difficulty_alignment(item_content, "medium")

        assert result.result == GateResult.FAIL
        assert result.repair_hint is not None

    def test_two_levels_off(self):
        item_content = {"difficulty": "easy"}
        result = check_difficulty_alignment(item_content, "hard")

        assert result.result == GateResult.FAIL
        assert "major" in result.repair_hint.lower()


class TestTier2DecisionGrounding:
    """Tests for Tier 2: Decision grounding check."""

    def test_well_grounded(self):
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

        result = check_decision_grounding(
            explanation='As stated in the source [Q1]: "Scene safety must be ensured first."',
            correct_answer="Ensure scene safety",
            quotes_used=quotes,
        )

        assert result.result == GateResult.PASS

    def test_missing_quote_references(self):
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

        result = check_decision_grounding(
            explanation="Scene safety is important because it protects everyone.",
            correct_answer="Ensure scene safety",
            quotes_used=quotes,
        )

        assert result.result == GateResult.FAIL
        assert result.repair_hint is not None


class TestTier3Novelty:
    """Tests for Tier 3: Novelty check."""

    def test_novel_item(self):
        item_content = {
            "stem": "What is the first step in scene assessment?",
            "correct_answer": "Check for hazards",
        }
        existing_items = [
            {
                "item_id": "existing_01",
                "stem": "What protective equipment should be worn?",
                "correct_answer": "Gloves and eye protection",
            }
        ]

        result = check_novelty(item_content, existing_items)

        assert result.result == GateResult.PASS

    def test_duplicate_item(self):
        item_content = {
            "stem": "What is the first step in scene assessment?",
            "correct_answer": "Check for hazards",
        }
        existing_items = [
            {
                "item_id": "existing_01",
                "stem": "What is the first step in scene assessment?",
                "correct_answer": "Check for hazards",
            }
        ]

        result = check_novelty(item_content, existing_items)

        assert result.result == GateResult.WARN


class TestTier3Verbosity:
    """Tests for Tier 3: Verbosity check."""

    def test_appropriate_length(self):
        explanation = " ".join(["word"] * 50)  # 50 words
        result = check_verbosity(explanation)

        assert result.result == GateResult.PASS

    def test_too_short(self):
        explanation = "Too short."
        result = check_verbosity(explanation, min_words=20)

        assert result.result == GateResult.WARN
        assert "brief" in result.message.lower()

    def test_too_long(self):
        explanation = " ".join(["word"] * 200)
        result = check_verbosity(explanation, max_words=150)

        assert result.result == GateResult.WARN
        assert "verbose" in result.message.lower()


class TestTier3Style:
    """Tests for Tier 3: Style check."""

    def test_good_style(self):
        result = check_style(
            stem="What is the correct procedure?",
            explanation="The procedure requires careful assessment of the situation.",
        )

        assert result.result == GateResult.PASS

    def test_informal_language(self):
        result = check_style(
            stem="What is the correct procedure?",
            explanation="Yeah, you gotta check the scene first.",
        )

        assert result.result == GateResult.WARN
        assert "informal" in result.message.lower()


class TestRunAllGates:
    """Tests for run_all_gates function."""

    @pytest.fixture
    def mock_verification_pass(self):
        return GenerationQuoteVerification(
            item_id="item_01",
            required_quotes=["Q1"],
            verified_quotes=[
                QuoteVerificationResult(quote_id="Q1", found=True, exact_match=True),
            ],
            all_required_found=True,
            exact_match_count=1,
            fuzzy_match_count=0,
            missing_quote_ids=[],
        )

    @pytest.fixture
    def mock_quotes(self):
        return [
            ExtractedQuote(
                quote_id="Q1",
                text="Scene safety must be ensured.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=29,
            )
        ]

    def test_all_gates_pass(self, mock_verification_pass, mock_quotes):
        item_content = {
            "item_id": "item_01",
            "stem": "What is the first priority at an accident scene?",
            "correct_answer": "Ensure scene safety",
            "distractors": ["Begin CPR immediately", "Call dispatch first", "Transport patient"],
            "explanation": 'According to the source [Q1]: "Scene safety must be ensured." This protects everyone.',
            "scope_tags": ["scene_safety"],
            "difficulty": "medium",
        }

        report = run_all_gates(
            item_content=item_content,
            quote_verification=mock_verification_pass,
            quotes_used=mock_quotes,
            allowed_scope_tags=["scene_safety", "hazard_assessment"],
            target_difficulty="medium",
        )

        assert report.tier_1_passed is True
        assert report.overall_pass is True
        assert report.requires_repair is False

    def test_tier_1_failure(self, mock_quotes):
        # Missing quotes
        mock_verification_fail = GenerationQuoteVerification(
            item_id="item_01",
            required_quotes=["Q1"],
            verified_quotes=[
                QuoteVerificationResult(quote_id="Q1", found=False, exact_match=False),
            ],
            all_required_found=False,
            exact_match_count=0,
            fuzzy_match_count=0,
            missing_quote_ids=["Q1"],
        )

        item_content = {
            "item_id": "item_01",
            "stem": "What is the first priority?",
            "correct_answer": "Ensure scene safety",
            "distractors": ["Begin CPR", "Call dispatch", "Transport patient"],
            "explanation": "Scene safety is important.",
            "scope_tags": ["scene_safety"],
        }

        report = run_all_gates(
            item_content=item_content,
            quote_verification=mock_verification_fail,
            quotes_used=mock_quotes,
            allowed_scope_tags=["scene_safety"],
        )

        assert report.tier_1_passed is False
        assert report.overall_pass is False

    def test_tier_2_failure_requires_repair(self, mock_verification_pass, mock_quotes):
        item_content = {
            "item_id": "item_01",
            "stem": "What is the first priority?",
            "correct_answer": "Ensure scene safety",
            "distractors": ["Yes", "No", "Maybe"],  # Bad distractors
            "explanation": 'As stated [Q1]: "Scene safety must be ensured." This is critical.',
            "scope_tags": ["scene_safety"],
            "difficulty": "easy",  # Wrong difficulty
        }

        report = run_all_gates(
            item_content=item_content,
            quote_verification=mock_verification_pass,
            quotes_used=mock_quotes,
            allowed_scope_tags=["scene_safety"],
            target_difficulty="hard",
        )

        assert report.tier_1_passed is True
        assert report.tier_2_passed is False
        assert report.requires_repair is True
        assert len(report.repair_targets) > 0


class TestGetRepairHints:
    """Tests for get_repair_hints function."""

    def test_extracts_repair_hints(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=False,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="distractor_quality",
                    tier=GateTier.TIER_2_REPAIR,
                    result=GateResult.FAIL,
                    message="Distractors too short",
                    repair_hint="Expand distractors",
                ),
                GateCheckResult(
                    gate_name="difficulty_alignment",
                    tier=GateTier.TIER_2_REPAIR,
                    result=GateResult.FAIL,
                    message="Wrong difficulty",
                    repair_hint="Increase complexity",
                ),
            ],
            overall_pass=True,
            requires_repair=True,
            repair_targets=["distractor_quality", "difficulty_alignment"],
        )

        hints = get_repair_hints(report)

        assert "distractor_quality" in hints
        assert "difficulty_alignment" in hints
        assert hints["distractor_quality"] == "Expand distractors"


class TestTier1LearningTargetAlignment:
    """Tests for Tier 1: Learning target alignment check."""

    def test_well_aligned_item(self):
        """Question that properly tests the stated learning target."""
        result = check_learning_target_alignment(
            learning_target="identify hazards at an emergency scene",
            stem="What should you look for when assessing scene hazards?",
            correct_answer="Identify potential dangers like fire, electrical, or structural hazards",
            explanation="Scene hazard identification is critical. You must identify fire, electrical, and structural hazards before approaching.",
        )

        assert result.result == GateResult.PASS
        assert result.tier == GateTier.TIER_1_HARD

    def test_learning_target_drift_different_topic(self):
        """Question drifts to a different topic than the learning target."""
        result = check_learning_target_alignment(
            learning_target="identify hazards at an emergency scene",
            stem="What medication should you administer for cardiac arrest?",
            correct_answer="Epinephrine 1mg IV",
            explanation="In cardiac arrest, epinephrine is the first-line medication.",
        )

        assert result.result == GateResult.FAIL
        assert "drift" in result.message.lower() or "issues" in result.message.lower()
        assert len(result.details.get("issues", [])) > 0

    def test_missing_key_terms_from_learning_target(self):
        """Stem and answer don't contain key terms from learning target."""
        result = check_learning_target_alignment(
            learning_target="properly position an unconscious patient for airway management",
            stem="What color is the ambulance?",
            correct_answer="White with red stripes",
            explanation="Ambulances are typically white with red stripes.",
        )

        assert result.result == GateResult.FAIL
        # Should flag that key terms are missing
        assert result.details.get("overlap_ratio", 1.0) < 0.25

    def test_first_action_without_priority_in_learning_target(self):
        """FIRST action question but learning target doesn't mention prioritization."""
        result = check_learning_target_alignment(
            learning_target="understand basic anatomy of the respiratory system",
            stem="What is the FIRST action you should take?",
            correct_answer="Check the airway",
            explanation="Always check the airway first.",
        )

        assert result.result == GateResult.FAIL
        assert any("FIRST action" in issue for issue in result.details.get("issues", []))

    def test_short_learning_target_passes(self):
        """Very short learning targets should pass to avoid false positives."""
        result = check_learning_target_alignment(
            learning_target="CPR",
            stem="When should you perform CPR?",
            correct_answer="When the patient is unresponsive and not breathing",
            explanation="CPR is performed when the patient is unresponsive and not breathing.",
        )

        assert result.result == GateResult.PASS

    def test_partial_alignment_passes(self):
        """Question with partial but sufficient overlap should pass."""
        result = check_learning_target_alignment(
            learning_target="assess and manage scene safety at motor vehicle accidents",
            stem="At a motor vehicle accident, how should you assess the scene?",
            correct_answer="Look for hazards like fire, traffic, and unstable vehicles",
            explanation="Scene assessment at motor vehicle accidents involves identifying hazards like fire, ongoing traffic, and vehicle instability.",
        )

        assert result.result == GateResult.PASS
        # Should have good overlap
        assert result.details.get("overlap_ratio", 0) >= 0.25

    def test_explanation_must_reference_learning_target(self):
        """Explanation should reference learning target concepts."""
        result = check_learning_target_alignment(
            learning_target="properly apply a tourniquet for severe bleeding control",
            stem="How do you apply a tourniquet?",
            correct_answer="Apply 2-3 inches above the wound",
            explanation="The weather today is nice and sunny.",  # Completely unrelated
        )

        assert result.result == GateResult.FAIL
        assert any("Explanation" in issue for issue in result.details.get("issues", []))
