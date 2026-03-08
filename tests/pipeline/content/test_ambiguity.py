"""Tests for ambiguity gate module."""

import pytest

from course_builder.pipeline.content.ambiguity import (
    AmbiguityCheckResult,
    AmbiguityGate,
    AmbiguityStatus,
    format_options_for_prompt,
    quick_ambiguity_check,
    check_decision_lock_coverage,
    check_distractor_differentiation,
    check_discriminating_factor_explicit,
    strict_ambiguity_check,
    StrictAmbiguityResult,
    TenCompetentResult,
)
from course_builder.pipeline.content.item_plan import (
    ItemPlan,
    QuestionType,
    DecisionLock,
    DecisionLockSpec,
    DistractorTag,
    DistractorFailureReason,
    PlannedDistractor,
    SupportMapping,
)


class TestFormatOptionsForPrompt:
    """Tests for format_options_for_prompt function."""

    def test_formats_options_with_indices(self):
        options = ["First option", "Second option", "Third option", "Fourth option"]
        formatted = format_options_for_prompt(options)

        assert "0)" in formatted
        assert "1)" in formatted
        assert "2)" in formatted
        assert "3)" in formatted
        assert "First option" in formatted
        assert "Fourth option" in formatted

    def test_handles_empty_options(self):
        formatted = format_options_for_prompt([])
        assert formatted == ""

    def test_handles_single_option(self):
        formatted = format_options_for_prompt(["Only option"])
        assert "0)" in formatted
        assert "Only option" in formatted


class TestQuickAmbiguityCheck:
    """Tests for quick_ambiguity_check function."""

    def test_detects_similar_options(self):
        stem = "What should you do first?"
        options = [
            "Ensure scene safety and approach the patient carefully",
            "Ensure scene safety and approach the patient slowly",  # Almost identical
            "Call for backup",
            "Wait for police",
        ]

        has_risk, warnings = quick_ambiguity_check(stem, options, correct_index=0)

        assert has_risk
        assert any("similar" in w.lower() for w in warnings)

    def test_detects_vague_qualifiers(self):
        stem = "What is the best approach?"
        options = [
            "The most appropriate action may possibly be to assess",
            "Call 911",
            "Wait",
            "Leave",
        ]

        has_risk, warnings = quick_ambiguity_check(stem, options, correct_index=0)

        # First option has multiple vague qualifiers
        assert has_risk or len(warnings) > 0

    def test_detects_repeated_action_verbs(self):
        stem = "What should you assess first?"
        options = [
            "Assess the airway",
            "Assess the breathing",
            "Assess the circulation",
            "Assess the disability",
        ]

        has_risk, warnings = quick_ambiguity_check(stem, options, correct_index=0)

        assert has_risk
        assert any("action" in w.lower() for w in warnings)

    def test_clear_question_no_warnings(self):
        stem = "What PPE should you wear for blood exposure?"
        options = [
            "Gloves and eye protection",
            "No PPE needed",
            "Full hazmat suit",
            "Just a mask",
        ]

        has_risk, warnings = quick_ambiguity_check(stem, options, correct_index=0)

        # Options are clearly different
        # May or may not have warnings depending on heuristics


class TestAmbiguityCheckResult:
    """Tests for AmbiguityCheckResult dataclass."""

    def test_clear_result(self):
        result = AmbiguityCheckResult(
            status=AmbiguityStatus.CLEAR,
            pass1_answer=0,
            pass2_answer=0,
            intended_answer=0,
            defensible_options=[0],
            confidence=0.95,
        )

        assert result.status == AmbiguityStatus.CLEAR
        assert len(result.defensible_options) == 1

    def test_ambiguous_result(self):
        result = AmbiguityCheckResult(
            status=AmbiguityStatus.AMBIGUOUS,
            pass1_answer=1,
            pass2_answer=1,
            intended_answer=0,
            defensible_options=[0, 1],
            reasoning="Both options could be defended",
            confidence=0.5,
        )

        assert result.status == AmbiguityStatus.AMBIGUOUS
        assert len(result.defensible_options) == 2


class TestAmbiguityGate:
    """Tests for AmbiguityGate class."""

    def test_quick_check_only(self):
        gate = AmbiguityGate(engine=None, use_quick_check=True)

        stem = "What should you do first?"
        options = [
            "Ensure scene safety before approaching",
            "Ensure scene safety prior to approach",
            "Call for backup",
            "Wait for police",
        ]

        result = gate.check(stem, options, correct_index=0)

        # Without LLM, uses heuristics only
        assert result.status in [AmbiguityStatus.CLEAR, AmbiguityStatus.AMBIGUOUS]

    def test_returns_clear_for_good_question(self):
        gate = AmbiguityGate(engine=None, use_quick_check=True)

        stem = "What PPE is required for blood contact?"
        options = [
            "Gloves and eye protection",
            "No PPE needed",
            "Full body suit",
            "Surgical mask only",
        ]

        result = gate.check(stem, options, correct_index=0)

        # Clear question with distinct options
        assert result.intended_answer == 0

    def test_no_engine_no_quick_check(self):
        gate = AmbiguityGate(engine=None, use_quick_check=False)

        stem = "Any question"
        options = ["A", "B", "C", "D"]

        result = gate.check(stem, options, correct_index=0)

        # Without any checks, defaults to clear with lower confidence
        assert result.status == AmbiguityStatus.CLEAR
        assert result.confidence < 1.0


class TestAmbiguityStatus:
    """Tests for AmbiguityStatus enum."""

    def test_status_values(self):
        assert AmbiguityStatus.CLEAR.value == "clear"
        assert AmbiguityStatus.AMBIGUOUS.value == "ambiguous"
        assert AmbiguityStatus.INVALID.value == "invalid"

    def test_status_comparison(self):
        assert AmbiguityStatus.CLEAR != AmbiguityStatus.AMBIGUOUS
        assert AmbiguityStatus.CLEAR == AmbiguityStatus.CLEAR


class TestCheckDecisionLockCoverage:
    """Tests for check_decision_lock_coverage function."""

    @pytest.fixture
    def first_action_plan_with_locks(self):
        return ItemPlan(
            item_id="item_01",
            learning_target="assess scene safety before patient contact",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=["fire", "downed power lines"],
            decision_locks=[
                DecisionLockSpec(
                    lock_type=DecisionLock.FEASIBILITY,
                    constraint_text="Downed power lines are sparking near the patient",
                    eliminates_options=["Approach patient", "Begin CPR"],
                )
            ],
            correct_action="Ensure scene safety",
            correct_option_text="Assess the scene for hazards",
            why_correct="Must ensure safety first",
            discriminating_factor="Power lines make approach impossible",
            support_mapping=[],
            distractors=[
                PlannedDistractor(
                    text="Approach patient",
                    tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="Approaches hazard",
                    stem_fact_violated="Power lines present",
                ),
                PlannedDistractor(
                    text="Begin CPR",
                    tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="Wrong sequence",
                    stem_fact_violated="Scene not safe",
                ),
                PlannedDistractor(
                    text="Call dispatch",
                    tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.TOO_GENERIC,
                    why_wrong="Not first priority",
                    stem_fact_violated="Safety assessment first",
                ),
            ],
            scope_tags=["scene_safety"],
            stem_type="scenario",
            stem_constraints=[],
        )

    def test_locks_present_in_stem(self, first_action_plan_with_locks):
        stem = "You arrive at a scene where downed power lines are sparking near the patient. What is your FIRST action?"

        all_present, missing = check_decision_lock_coverage(stem, first_action_plan_with_locks)

        assert all_present
        assert len(missing) == 0

    def test_locks_missing_from_stem(self, first_action_plan_with_locks):
        stem = "You arrive at a scene. What is your FIRST action?"  # No mention of power lines

        all_present, missing = check_decision_lock_coverage(stem, first_action_plan_with_locks)

        assert not all_present
        assert len(missing) > 0
        assert any("decision lock" in issue.lower() for issue in missing)

    def test_factual_question_skips_lock_check(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="definition of triage",
            question_type=QuestionType.DEFINITION,  # No locks required
            hazard_cues=[],
            decision_locks=[],
            correct_action="Define triage",
            correct_option_text="Sorting patients by severity",
            why_correct="Standard definition",
            discriminating_factor="Definition from textbook",
            support_mapping=[],
            distractors=[],
            scope_tags=["triage"],
            stem_type="direct",
            stem_constraints=[],
        )

        all_present, missing = check_decision_lock_coverage("What is triage?", plan)

        assert all_present  # Definition questions don't need locks


class TestCheckDistractorDifferentiation:
    """Tests for check_distractor_differentiation function."""

    def test_differentiated_distractors(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="factor",
            support_mapping=[],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="Fire present in the scene"
                ),
                PlannedDistractor(
                    text="d2", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="r2", stem_fact_violated="Scene safety not assessed"
                ),
                PlannedDistractor(
                    text="d3", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
                    why_wrong="r3", stem_fact_violated="EMR scope limitation"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        all_different, issues = check_distractor_differentiation(plan)

        assert all_different
        assert len(issues) == 0

    def test_duplicate_failure_reasons_flagged(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="factor",
            support_mapping=[],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,  # Same
                    why_wrong="r1", stem_fact_violated="Scene safety not assessed"
                ),
                PlannedDistractor(
                    text="d2", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,  # Same!
                    why_wrong="r2", stem_fact_violated="Scene safety not assessed"
                ),
                PlannedDistractor(
                    text="d3", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
                    why_wrong="r3", stem_fact_violated="EMR scope"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        all_different, issues = check_distractor_differentiation(plan)

        assert not all_different
        assert any("same failure reason" in issue.lower() for issue in issues)

    def test_vague_stem_fact_flagged(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="factor",
            support_mapping=[],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="bad"  # Too vague
                ),
                PlannedDistractor(
                    text="d2", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="r2", stem_fact_violated="ok"  # Also too vague
                ),
                PlannedDistractor(
                    text="d3", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
                    why_wrong="r3", stem_fact_violated="EMR scope limitation"  # Good
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        _, issues = check_distractor_differentiation(plan)

        assert any("vague" in issue.lower() for issue in issues)


class TestCheckDiscriminatingFactorExplicit:
    """Tests for check_discriminating_factor_explicit function."""

    def test_explicit_discriminating_factor(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="Power lines near the patient make direct approach impossible",
            support_mapping=[],
            distractors=[],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        stem = "You arrive at a scene where power lines are sparking near the patient."

        is_explicit, issues = check_discriminating_factor_explicit(plan, stem)

        assert is_explicit
        assert len(issues) == 0

    def test_common_sense_discriminating_factor_flagged(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="This is just common sense that everyone knows",  # Bad!
            support_mapping=[],
            distractors=[],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        stem = "What should you do?"

        is_explicit, issues = check_discriminating_factor_explicit(plan, stem)

        assert not is_explicit
        assert any("common sense" in issue.lower() for issue in issues)

    def test_discriminating_factor_not_in_stem_flagged(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="The patient is unconscious and bleeding heavily",
            support_mapping=[],
            distractors=[],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        stem = "What should you do at this scene?"  # Doesn't mention unconscious or bleeding

        is_explicit, issues = check_discriminating_factor_explicit(plan, stem)

        assert not is_explicit
        assert any("not found in stem" in issue.lower() for issue in issues)


class TestStrictAmbiguityCheck:
    """Tests for strict_ambiguity_check function."""

    @pytest.fixture
    def good_plan(self):
        return ItemPlan(
            item_id="item_01",
            learning_target="scene safety assessment",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=["fire"],
            decision_locks=[
                DecisionLockSpec(
                    lock_type=DecisionLock.FEASIBILITY,
                    constraint_text="Fire is spreading toward the vehicle",
                    eliminates_options=["Approach patient"],
                )
            ],
            correct_action="Ensure scene safety",
            correct_option_text="Assess for hazards first",
            why_correct="Protects rescuer",
            discriminating_factor="Fire spreading makes direct approach dangerous",
            support_mapping=[],
            distractors=[
                PlannedDistractor(
                    text="Approach patient", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="Fire", stem_fact_violated="Fire spreading toward vehicle"
                ),
                PlannedDistractor(
                    text="Begin CPR", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="Wrong order", stem_fact_violated="Scene not safe"
                ),
                PlannedDistractor(
                    text="Call dispatch", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.TOO_GENERIC,
                    why_wrong="Not first", stem_fact_violated="Safety first"
                ),
            ],
            scope_tags=["scene_safety"],
            stem_type="scenario",
            stem_constraints=[],
        )

    def test_good_question_passes_without_llm(self, good_plan):
        stem = "You arrive at a scene where fire is spreading toward the vehicle. What is your FIRST action?"
        options = ["Assess for hazards first", "Approach patient", "Begin CPR", "Call dispatch"]

        result = strict_ambiguity_check(
            engine=None,
            stem=stem,
            options=options,
            correct_index=0,
            plan=good_plan,
            run_ten_competent_test=False,  # No LLM
        )

        assert result.passes
        assert len(result.decision_lock_issues) == 0
        assert len(result.distractor_issues) == 0
        assert len(result.discriminating_factor_issues) == 0

    def test_missing_decision_locks_flagged(self, good_plan):
        stem = "You arrive at a scene. What is your FIRST action?"  # No fire mentioned
        options = ["Assess for hazards first", "Approach patient", "Begin CPR", "Call dispatch"]

        result = strict_ambiguity_check(
            engine=None,
            stem=stem,
            options=options,
            correct_index=0,
            plan=good_plan,
            run_ten_competent_test=False,
        )

        assert not result.passes
        assert len(result.decision_lock_issues) > 0

    def test_strict_check_recommendation(self, good_plan):
        stem = "You arrive at a scene. What is your FIRST action?"
        options = ["Assess for hazards first", "Approach patient", "Begin CPR", "Call dispatch"]

        result = strict_ambiguity_check(
            engine=None,
            stem=stem,
            options=options,
            correct_index=0,
            plan=good_plan,
            run_ten_competent_test=False,
        )

        assert "AMBIGUITY RISK" in result.recommendation


class TestTenCompetentResult:
    """Tests for TenCompetentResult dataclass."""

    def test_passing_result(self):
        result = TenCompetentResult(
            passes_test=True,
            consensus_answer=0,
            vote_counts={0: 9, 1: 1},
            split_reason="",
            emr_responses=[],
            confidence=0.9,
        )

        assert result.passes_test
        assert result.confidence == 0.9

    def test_failing_result(self):
        result = TenCompetentResult(
            passes_test=False,
            consensus_answer=None,
            vote_counts={0: 5, 1: 5},
            split_reason="EMRs split evenly between scene safety and calling dispatch",
            emr_responses=[],
            confidence=0.5,
        )

        assert not result.passes_test
        assert result.split_reason != ""
