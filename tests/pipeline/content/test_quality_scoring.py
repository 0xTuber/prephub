"""Tests for quality scoring module."""

import pytest

from course_builder.pipeline.content.quality_scoring import (
    QualityDecision,
    DimensionScore,
    QualityScore,
    ACCEPT_THRESHOLD,
    FLAG_THRESHOLD,
    score_grounding,
    score_consistency,
    score_scope,
    score_distractors,
    score_novelty,
    compute_quality_score,
    score_after_repair,
    format_score_report,
    get_improvement_suggestions,
)
from course_builder.pipeline.content.quote_extraction import (
    GenerationQuoteVerification,
    QuoteVerificationResult,
)
from course_builder.pipeline.content.quality_tiers import (
    GateTier,
    GateResult,
    GateCheckResult,
    QualityGateReport,
)


class TestQualityDecision:
    """Tests for QualityDecision enum."""

    def test_decision_values(self):
        assert QualityDecision.ACCEPT == "accept"
        assert QualityDecision.ACCEPT_WITH_FLAGS == "accept_with_flags"
        assert QualityDecision.REPAIR == "repair"
        assert QualityDecision.REJECT == "reject"


class TestThresholds:
    """Tests for threshold constants."""

    def test_accept_threshold(self):
        assert ACCEPT_THRESHOLD == 0.80

    def test_flag_threshold(self):
        assert FLAG_THRESHOLD == 0.65


class TestScoreGrounding:
    """Tests for score_grounding function."""

    def test_all_quotes_exact_match(self):
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

        score = score_grounding(verification)

        assert score.dimension == "grounding"
        assert score.score >= 0.9  # High score for exact matches

    def test_missing_quotes(self):
        verification = GenerationQuoteVerification(
            item_id="item_01",
            required_quotes=["Q1", "Q2", "Q3"],
            verified_quotes=[
                QuoteVerificationResult(quote_id="Q1", found=True, exact_match=True),
                QuoteVerificationResult(quote_id="Q2", found=False, exact_match=False),
                QuoteVerificationResult(quote_id="Q3", found=False, exact_match=False),
            ],
            all_required_found=False,
            exact_match_count=1,
            fuzzy_match_count=0,
            missing_quote_ids=["Q2", "Q3"],  # Two missing
        )

        score = score_grounding(verification, min_quotes_required=2)

        # With 2 missing quotes, penalty should reduce score
        assert score.score < 0.9  # Penalty for missing quotes
        assert "Q2" in score.details["missing"]

    def test_no_quotes_required(self):
        verification = GenerationQuoteVerification(
            item_id="item_01",
            required_quotes=[],
            verified_quotes=[],
            all_required_found=True,
            exact_match_count=0,
            fuzzy_match_count=0,
            missing_quote_ids=[],
        )

        score = score_grounding(verification)

        assert score.score == 0.5  # Neutral score


class TestScoreConsistency:
    """Tests for score_consistency function."""

    def test_all_consistency_checks_pass(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="stem_answer_consistency",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
                GateCheckResult(
                    gate_name="single_best_answer",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
            ],
            overall_pass=True,
            requires_repair=False,
            repair_targets=[],
        )

        score = score_consistency(report)

        assert score.dimension == "consistency"
        assert score.score == 1.0

    def test_some_consistency_checks_fail(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=False,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="stem_answer_consistency",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.FAIL,
                    message="Stem leaks answer",
                ),
                GateCheckResult(
                    gate_name="single_best_answer",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
            ],
            overall_pass=False,
            requires_repair=False,
            repair_targets=[],
        )

        score = score_consistency(report)

        assert score.score == 0.5


class TestScoreScope:
    """Tests for score_scope function."""

    def test_within_scope(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="scope_legality",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
            ],
            overall_pass=True,
            requires_repair=False,
            repair_targets=[],
        )

        score = score_scope(report)

        assert score.score == 1.0

    def test_out_of_scope(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=False,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="scope_legality",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.FAIL,
                    message="Out of scope",
                    details={"out_of_scope_tags": ["advanced"]},
                ),
            ],
            overall_pass=False,
            requires_repair=False,
            repair_targets=[],
        )

        score = score_scope(report)

        assert score.score == 0.0


class TestScoreDistractors:
    """Tests for score_distractors function."""

    def test_good_distractors(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="distractor_quality",
                    tier=GateTier.TIER_2_REPAIR,
                    result=GateResult.PASS,
                    message="OK",
                ),
            ],
            overall_pass=True,
            requires_repair=False,
            repair_targets=[],
        )

        score = score_distractors(report)

        assert score.score == 1.0

    def test_distractor_issues(self):
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
                    message="Issues",
                    details={"issues": ["too short", "meta option"]},
                ),
            ],
            overall_pass=True,
            requires_repair=True,
            repair_targets=["distractor_quality"],
        )

        score = score_distractors(report)

        assert score.score < 1.0
        assert score.score >= 0.25


class TestScoreNovelty:
    """Tests for score_novelty function."""

    def test_novel_item(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="novelty",
                    tier=GateTier.TIER_3_SOFT,
                    result=GateResult.PASS,
                    message="OK",
                ),
            ],
            overall_pass=True,
            requires_repair=False,
            repair_targets=[],
        )

        score = score_novelty(report)

        assert score.score == 1.0

    def test_similar_item_found(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_flags=["novelty"],
            all_checks=[
                GateCheckResult(
                    gate_name="novelty",
                    tier=GateTier.TIER_3_SOFT,
                    result=GateResult.WARN,
                    message="Similar item found",
                    details={"similarity": 0.85, "similar_item_id": "item_02"},
                ),
            ],
            overall_pass=True,
            requires_repair=False,
            repair_targets=[],
        )

        score = score_novelty(report)

        assert score.score < 1.0


class TestComputeQualityScore:
    """Tests for compute_quality_score function."""

    @pytest.fixture
    def good_verification(self):
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
    def good_gate_report(self):
        return QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="stem_answer_consistency",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
                GateCheckResult(
                    gate_name="single_best_answer",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
                GateCheckResult(
                    gate_name="scope_legality",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
                GateCheckResult(
                    gate_name="distractor_quality",
                    tier=GateTier.TIER_2_REPAIR,
                    result=GateResult.PASS,
                    message="OK",
                ),
                GateCheckResult(
                    gate_name="novelty",
                    tier=GateTier.TIER_3_SOFT,
                    result=GateResult.PASS,
                    message="OK",
                ),
            ],
            overall_pass=True,
            requires_repair=False,
            repair_targets=[],
        )

    def test_high_quality_item_accepted(self, good_verification, good_gate_report):
        score = compute_quality_score(
            item_id="item_01",
            gate_report=good_gate_report,
            quote_verification=good_verification,
        )

        assert score.composite_score >= ACCEPT_THRESHOLD
        assert score.decision == QualityDecision.ACCEPT

    def test_tier1_failure_rejected(self, good_verification):
        bad_report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=False,  # Hard blocker failed
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[],
            overall_pass=False,
            requires_repair=False,
            repair_targets=[],
        )

        score = compute_quality_score(
            item_id="item_01",
            gate_report=bad_report,
            quote_verification=good_verification,
        )

        assert score.decision == QualityDecision.REJECT

    def test_borderline_accepted_with_flags(self):
        # Verification with missing quotes to lower grounding score
        borderline_verification = GenerationQuoteVerification(
            item_id="item_01",
            required_quotes=["Q1", "Q2"],
            verified_quotes=[
                QuoteVerificationResult(quote_id="Q1", found=True, exact_match=False, similarity=0.7),
                QuoteVerificationResult(quote_id="Q2", found=False, exact_match=False),
            ],
            all_required_found=False,
            exact_match_count=0,  # No exact matches
            fuzzy_match_count=1,
            missing_quote_ids=["Q2"],
        )

        borderline_report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=False,
            tier_3_flags=["verbosity", "novelty"],
            all_checks=[
                GateCheckResult(
                    gate_name="stem_answer_consistency",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
                GateCheckResult(
                    gate_name="single_best_answer",
                    tier=GateTier.TIER_1_HARD,
                    result=GateResult.PASS,
                    message="OK",
                ),
                GateCheckResult(
                    gate_name="distractor_quality",
                    tier=GateTier.TIER_2_REPAIR,
                    result=GateResult.FAIL,
                    message="Issues",
                    details={"issues": ["too short", "meta option", "identical prefix"]},
                ),
                GateCheckResult(
                    gate_name="novelty",
                    tier=GateTier.TIER_3_SOFT,
                    result=GateResult.WARN,
                    message="Similar item",
                    details={"similarity": 0.75},
                ),
            ],
            overall_pass=True,
            requires_repair=True,
            repair_targets=["distractor_quality"],
        )

        score = compute_quality_score(
            item_id="item_01",
            gate_report=borderline_report,
            quote_verification=borderline_verification,
        )

        # Score should reflect the issues
        assert score.composite_score < ACCEPT_THRESHOLD
        # Should be flagged or need repair
        assert score.decision in (
            QualityDecision.ACCEPT_WITH_FLAGS,
            QualityDecision.REPAIR,
        )


class TestScoreAfterRepair:
    """Tests for score_after_repair function."""

    def test_improved_score_accepted(self):
        original = QualityScore(
            item_id="item_01",
            composite_score=0.55,
            decision=QualityDecision.REPAIR,
            dimension_scores=[],
            flags=[],
            recommendation="Repair needed",
        )

        improved_report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[],
            overall_pass=True,
            requires_repair=False,
            repair_targets=[],
        )

        improved_verification = GenerationQuoteVerification(
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

        new_score = score_after_repair(original, improved_report, improved_verification)

        # Should be accepted or at least flagged
        assert new_score.decision in (
            QualityDecision.ACCEPT,
            QualityDecision.ACCEPT_WITH_FLAGS,
        )

    def test_still_low_after_repair_rejected(self):
        original = QualityScore(
            item_id="item_01",
            composite_score=0.55,
            decision=QualityDecision.REPAIR,
            dimension_scores=[],
            flags=[],
            recommendation="Repair needed",
        )

        # Report still has issues
        still_bad_report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=False,
            tier_3_flags=[],
            all_checks=[
                GateCheckResult(
                    gate_name="distractor_quality",
                    tier=GateTier.TIER_2_REPAIR,
                    result=GateResult.FAIL,
                    message="Still bad",
                    details={"issues": ["issue1", "issue2", "issue3"]},
                ),
            ],
            overall_pass=True,
            requires_repair=True,
            repair_targets=["distractor_quality"],
        )

        bad_verification = GenerationQuoteVerification(
            item_id="item_01",
            required_quotes=["Q1", "Q2"],
            verified_quotes=[],
            all_required_found=False,
            exact_match_count=0,
            fuzzy_match_count=0,
            missing_quote_ids=["Q1", "Q2"],
        )

        new_score = score_after_repair(original, still_bad_report, bad_verification)

        assert new_score.decision == QualityDecision.REJECT
        assert "failed_after_repair" in new_score.flags


class TestFormatScoreReport:
    """Tests for format_score_report function."""

    def test_formats_complete_report(self):
        score = QualityScore(
            item_id="item_01",
            composite_score=0.85,
            decision=QualityDecision.ACCEPT,
            dimension_scores=[
                DimensionScore(dimension="grounding", score=0.9, weight=0.3),
                DimensionScore(dimension="consistency", score=0.8, weight=0.25),
            ],
            flags=["borderline_consistency"],
            recommendation="Quality meets threshold",
        )

        report = format_score_report(score)

        assert "item_01" in report
        assert "0.85" in report
        assert "ACCEPT" in report
        assert "grounding" in report
        assert "borderline_consistency" in report


class TestGetImprovementSuggestions:
    """Tests for get_improvement_suggestions function."""

    def test_suggestions_for_low_grounding(self):
        score = QualityScore(
            item_id="item_01",
            composite_score=0.6,
            decision=QualityDecision.REPAIR,
            dimension_scores=[
                DimensionScore(dimension="grounding", score=0.4, weight=0.3),
            ],
            flags=[],
            recommendation="Repair needed",
        )

        suggestions = get_improvement_suggestions(score)

        assert any("quote" in s.lower() for s in suggestions)

    def test_no_suggestions_for_high_quality(self):
        score = QualityScore(
            item_id="item_01",
            composite_score=0.95,
            decision=QualityDecision.ACCEPT,
            dimension_scores=[
                DimensionScore(dimension="grounding", score=0.95, weight=0.3),
                DimensionScore(dimension="consistency", score=0.95, weight=0.25),
            ],
            flags=[],
            recommendation="Quality meets threshold",
        )

        suggestions = get_improvement_suggestions(score)

        assert len(suggestions) == 0
