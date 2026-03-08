"""Tests for item planning module."""

from unittest.mock import MagicMock

import pytest

from course_builder.domain.content import CapsuleItem
from course_builder.pipeline.content.item_plan import (
    DecisionLock,
    DecisionLockSpec,
    DistractorFailureReason,
    DistractorTag,
    QuestionType,
    REQUIRES_DECISION_LOCK,
    SupportMapping,
    PlannedDistractor,
    ItemPlan,
    plan_item,
    validate_plan,
    validate_plan_ambiguity,
    format_plan_for_generation,
    extract_scope_tags_from_plan,
    get_required_quote_ids,
    _generate_fallback_plan,
)
from course_builder.pipeline.content.quote_extraction import ExtractedQuote


class TestDistractorTag:
    """Tests for DistractorTag enum."""

    def test_tag_values(self):
        assert DistractorTag.UNSAFE == "unsafe"
        assert DistractorTag.WRONG_SEQUENCE == "wrong_sequence"
        assert DistractorTag.OUT_OF_SCOPE == "out_of_scope"
        assert DistractorTag.PARTIAL == "partial"
        assert DistractorTag.COMMON_MISCONCEPTION == "common_misconception"


class TestSupportMapping:
    """Tests for SupportMapping model."""

    def test_create_mapping(self):
        mapping = SupportMapping(
            quote_id="Q1",
            role="primary_support",
            supports_claim="scene safety is first priority",
        )

        assert mapping.quote_id == "Q1"
        assert mapping.role == "primary_support"
        assert mapping.supports_claim == "scene safety is first priority"


class TestDecisionLock:
    """Tests for DecisionLock enum and DecisionLockSpec model."""

    def test_lock_values(self):
        assert DecisionLock.FEASIBILITY == "feasibility"
        assert DecisionLock.AUTHORITY == "authority"
        assert DecisionLock.DISTANCE == "distance"
        assert DecisionLock.RESOURCE == "resource"
        assert DecisionLock.SEQUENCE == "sequence"
        assert DecisionLock.TIME == "time"

    def test_create_lock_spec(self):
        lock = DecisionLockSpec(
            lock_type=DecisionLock.RESOURCE,
            constraint_text="You are alone with no other units on scene",
            eliminates_options=["Call for backup", "Wait for ALS"],
        )

        assert lock.lock_type == DecisionLock.RESOURCE
        assert "alone" in lock.constraint_text
        assert len(lock.eliminates_options) == 2


class TestDistractorFailureReason:
    """Tests for DistractorFailureReason enum."""

    def test_failure_reason_values(self):
        assert DistractorFailureReason.APPROACHES_HAZARD == "approaches_hazard"
        assert DistractorFailureReason.WRONG_SEQUENCE == "wrong_sequence"
        assert DistractorFailureReason.EXCEEDS_SCOPE == "exceeds_scope"
        assert DistractorFailureReason.CONTRADICTS_STEM == "contradicts_stem"


class TestQuestionType:
    """Tests for QuestionType enum."""

    def test_question_type_values(self):
        assert QuestionType.FIRST_ACTION == "first_action"
        assert QuestionType.BEST_ACTION == "best_action"
        assert QuestionType.DEFINITION == "definition"

    def test_requires_decision_lock(self):
        assert QuestionType.FIRST_ACTION in REQUIRES_DECISION_LOCK
        assert QuestionType.BEST_ACTION in REQUIRES_DECISION_LOCK
        assert QuestionType.DEFINITION not in REQUIRES_DECISION_LOCK
        assert QuestionType.FACTUAL not in REQUIRES_DECISION_LOCK


class TestPlannedDistractor:
    """Tests for PlannedDistractor model."""

    def test_create_distractor(self):
        distractor = PlannedDistractor(
            text="Begin CPR immediately",
            tag=DistractorTag.WRONG_SEQUENCE,
            failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
            why_wrong="CPR comes after scene safety assessment",
            stem_fact_violated="Scene safety has not been assessed",
            plausibility_source="CPR is often associated with emergencies",
        )

        assert distractor.text == "Begin CPR immediately"
        assert distractor.tag == DistractorTag.WRONG_SEQUENCE
        assert distractor.failure_reason == DistractorFailureReason.WRONG_SEQUENCE
        assert distractor.why_wrong == "CPR comes after scene safety assessment"
        assert distractor.stem_fact_violated == "Scene safety has not been assessed"


class TestItemPlan:
    """Tests for ItemPlan model."""

    def test_create_complete_plan(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="scene safety assessment",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=["traffic accident", "downed power lines"],
            scenario_context="You arrive at a car accident scene",
            decision_locks=[
                DecisionLockSpec(
                    lock_type=DecisionLock.FEASIBILITY,
                    constraint_text="Downed power lines are sparking near the vehicle",
                    eliminates_options=["Approach the vehicle", "Move the patient"],
                )
            ],
            correct_action="Ensure scene safety before approaching",
            correct_option_text="Assess for hazards before approaching the patient",
            why_correct="Scene safety prevents rescuer injury",
            discriminating_factor="Sparking power lines make direct approach impossible",
            support_mapping=[
                SupportMapping(
                    quote_id="Q1",
                    role="primary_support",
                    supports_claim="scene safety first",
                )
            ],
            distractors=[
                PlannedDistractor(
                    text="Begin patient assessment",
                    tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="Assessment comes after scene safety",
                    stem_fact_violated="Downed power lines are sparking",
                ),
                PlannedDistractor(
                    text="Move the patient immediately",
                    tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.CREATES_NEW_HAZARD,
                    why_wrong="Moving without assessment is dangerous",
                    stem_fact_violated="Power lines near the vehicle",
                ),
                PlannedDistractor(
                    text="Call for ALS backup",
                    tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="Correct but not the first priority",
                    stem_fact_violated="Scene safety must come first",
                ),
            ],
            scope_tags=["scene_safety", "hazard_assessment"],
            out_of_scope_warning="Do not mention advanced procedures",
            stem_type="scenario",
            stem_constraints=["Include traffic scene context"],
        )

        assert plan.item_id == "item_01"
        assert plan.question_type == QuestionType.FIRST_ACTION
        assert len(plan.decision_locks) == 1
        assert len(plan.distractors) == 3
        assert len(plan.support_mapping) == 1

    def test_plan_json_serialization(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test target",
            question_type=QuestionType.FACTUAL,
            hazard_cues=["cue1"],
            decision_locks=[],
            correct_action="test action",
            correct_option_text="test option",
            why_correct="test reason",
            discriminating_factor="Explicit source citation",
            support_mapping=[],
            distractors=[],
            scope_tags=["test"],
            stem_type="direct",
            stem_constraints=[],
        )

        data = plan.model_dump()
        assert data["item_id"] == "item_01"
        assert data["stem_type"] == "direct"
        assert data["question_type"] == "factual"


class TestPlanItem:
    """Tests for plan_item function."""

    @pytest.fixture
    def sample_item(self):
        return CapsuleItem(
            item_id="item_01",
            item_type="Multiple Choice",
            title="Scene Safety",
            learning_target="assess scene for hazards before approaching",
        )

    @pytest.fixture
    def sample_quotes(self):
        return [
            ExtractedQuote(
                quote_id="Q1",
                text="Scene safety must be ensured before patient contact.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=50,
            ),
            ExtractedQuote(
                quote_id="Q2",
                text="Always identify potential hazards at the scene.",
                chunk_id="c2",
                chunk_index=1,
                start_char=0,
                end_char=46,
            ),
        ]

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        return engine

    def test_successful_planning(self, mock_engine, sample_item, sample_quotes):
        mock_response = MagicMock()
        mock_response.text = '''```json
{
  "question_type": "first_action",
  "hazard_cues": ["traffic accident", "electrical hazard"],
  "scenario_context": "You respond to a car accident",
  "decision_locks": [
    {
      "lock_type": "feasibility",
      "constraint_text": "Downed power lines are blocking direct access",
      "eliminates_options": ["Begin CPR", "Move patient"]
    }
  ],
  "correct_action": "Assess scene for hazards",
  "correct_option_text": "Survey the scene for potential hazards",
  "why_correct": "Ensures rescuer safety as stated in Q1",
  "discriminating_factor": "Power lines blocking access makes direct patient contact impossible",
  "support_mapping": [
    {"quote_id": "Q1", "role": "primary_support", "supports_claim": "scene safety first"}
  ],
  "distractors": [
    {"text": "Begin CPR", "tag": "wrong_sequence", "failure_reason": "approaches_hazard", "why_wrong": "Scene safety comes first", "stem_fact_violated": "Power lines blocking access", "plausibility_source": "CPR is life-saving"},
    {"text": "Move patient", "tag": "unsafe", "failure_reason": "creates_new_hazard", "why_wrong": "No assessment done", "stem_fact_violated": "Electrical hazard present", "plausibility_source": "Moving patient seems helpful"},
    {"text": "Call dispatch", "tag": "partial", "failure_reason": "wrong_sequence", "why_wrong": "Correct but not first", "stem_fact_violated": "Scene assessment precedes notification", "plausibility_source": "Dispatch coordination is important"}
  ],
  "scope_tags": ["scene_safety", "hazard_assessment"],
  "out_of_scope_warning": "Avoid advanced procedures",
  "stem_type": "scenario",
  "stem_constraints": ["Include scene description"]
}
```'''
        mock_engine.generate.return_value = mock_response

        plan = plan_item(
            engine=mock_engine,
            item=sample_item,
            quotes=sample_quotes,
            topic_name="Scene Safety",
            subtopic_name="Hazard Assessment",
            allowed_scope_tags=["scene_safety", "hazard_assessment", "patient_assessment"],
        )

        assert plan.item_id == "item_01"
        assert plan.question_type == QuestionType.FIRST_ACTION
        assert len(plan.decision_locks) == 1
        assert len(plan.distractors) == 3
        assert len(plan.support_mapping) == 1
        assert plan.support_mapping[0].quote_id == "Q1"
        assert plan.discriminating_factor != ""

    def test_fallback_on_json_error(self, mock_engine, sample_item, sample_quotes):
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        mock_engine.generate.return_value = mock_response

        plan = plan_item(
            engine=mock_engine,
            item=sample_item,
            quotes=sample_quotes,
            topic_name="Scene Safety",
            subtopic_name="Hazard Assessment",
            allowed_scope_tags=["scene_safety"],
        )

        # Should use fallback plan
        assert plan.item_id == "item_01"
        assert len(plan.distractors) == 3


class TestGenerateFallbackPlan:
    """Tests for _generate_fallback_plan function."""

    @pytest.fixture
    def sample_item(self):
        return CapsuleItem(
            item_id="item_01",
            item_type="Multiple Choice",
            title="Test",
            learning_target="test learning target",
        )

    @pytest.fixture
    def sample_quotes(self):
        return [
            ExtractedQuote(
                quote_id="Q1",
                text="Test quote text.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=16,
            )
        ]

    def test_fallback_with_quotes(self, sample_item, sample_quotes):
        plan = _generate_fallback_plan(
            sample_item,
            sample_quotes,
            ["scope1", "scope2"],
        )

        assert plan.item_id == "item_01"
        assert len(plan.support_mapping) == 1
        assert plan.support_mapping[0].quote_id == "Q1"
        assert len(plan.distractors) == 3

    def test_fallback_without_quotes(self, sample_item):
        plan = _generate_fallback_plan(
            sample_item,
            [],  # No quotes
            ["scope1"],
        )

        assert plan.item_id == "item_01"
        assert len(plan.support_mapping) == 0


class TestValidatePlan:
    """Tests for validate_plan function."""

    @pytest.fixture
    def sample_quotes(self):
        return [
            ExtractedQuote(
                quote_id="Q1",
                text="Test quote.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=11,
            )
        ]

    def test_valid_plan(self, sample_quotes):
        """Test a fully valid plan with all required fields."""
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,  # No decision locks required
            hazard_cues=["cue1"],
            decision_locks=[],
            correct_action="action",
            correct_option_text="Valid option text",
            why_correct="reason",
            discriminating_factor="Source explicitly states this",
            support_mapping=[
                SupportMapping(quote_id="Q1", role="primary_support", supports_claim="test")
            ],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="hazard present"
                ),
                PlannedDistractor(
                    text="d2", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="r2", stem_fact_violated="sequence constraint"
                ),
                PlannedDistractor(
                    text="d3", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
                    why_wrong="r3", stem_fact_violated="scope constraint"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan(plan, sample_quotes)
        assert len(issues) == 0

    def test_missing_support_mapping(self, sample_quotes):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="Source states this",
            support_mapping=[],  # Missing
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="constraint"
                ),
                PlannedDistractor(
                    text="d2", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="r2", stem_fact_violated="constraint"
                ),
                PlannedDistractor(
                    text="d3", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
                    why_wrong="r3", stem_fact_violated="constraint"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan(plan, sample_quotes)
        assert "No support mapping" in issues[0]

    def test_invalid_quote_reference(self, sample_quotes):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="Source states this",
            support_mapping=[
                SupportMapping(quote_id="Q99", role="primary_support", supports_claim="test")  # Invalid
            ],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="constraint"
                ),
                PlannedDistractor(
                    text="d2", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="r2", stem_fact_violated="constraint"
                ),
                PlannedDistractor(
                    text="d3", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
                    why_wrong="r3", stem_fact_violated="constraint"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan(plan, sample_quotes)
        assert any("Q99" in issue for issue in issues)

    def test_wrong_distractor_count(self, sample_quotes):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="Source states this",
            support_mapping=[
                SupportMapping(quote_id="Q1", role="primary_support", supports_claim="test")
            ],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="constraint"
                ),
            ],  # Only 1
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan(plan, sample_quotes)
        assert any("3 distractors" in issue for issue in issues)

    def test_first_action_requires_decision_lock(self, sample_quotes):
        """FIRST_ACTION questions require decision locks to avoid ambiguity."""
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,  # Requires decision locks
            hazard_cues=["cue1"],
            decision_locks=[],  # Missing!
            correct_action="action",
            correct_option_text="Valid option",
            why_correct="reason",
            discriminating_factor="Explicit stem fact",
            support_mapping=[
                SupportMapping(quote_id="Q1", role="primary_support", supports_claim="test")
            ],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="constraint"
                ),
                PlannedDistractor(
                    text="d2", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="r2", stem_fact_violated="constraint"
                ),
                PlannedDistractor(
                    text="d3", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
                    why_wrong="r3", stem_fact_violated="constraint"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan(plan, sample_quotes)
        assert any("AMBIGUITY RISK" in issue and "decision lock" in issue for issue in issues)

    def test_duplicate_failure_reasons_flagged(self, sample_quotes):
        """Each distractor must have a DIFFERENT failure reason."""
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="Valid option",
            why_correct="reason",
            discriminating_factor="Source states this",
            support_mapping=[
                SupportMapping(quote_id="Q1", role="primary_support", supports_claim="test")
            ],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,  # Same
                    why_wrong="r1", stem_fact_violated="constraint"
                ),
                PlannedDistractor(
                    text="d2", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,  # Same!
                    why_wrong="r2", stem_fact_violated="constraint"
                ),
                PlannedDistractor(
                    text="d3", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
                    why_wrong="r3", stem_fact_violated="constraint"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan(plan, sample_quotes)
        assert any("DISTRACTOR ASYMMETRY" in issue for issue in issues)


class TestFormatPlanForGeneration:
    """Tests for format_plan_for_generation function."""

    def test_formats_complete_plan(self):
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

        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=["traffic", "fire"],
            scenario_context="Car accident scene",
            decision_locks=[
                DecisionLockSpec(
                    lock_type=DecisionLock.FEASIBILITY,
                    constraint_text="Fire is spreading toward the vehicle",
                    eliminates_options=["Begin CPR", "Move patient"],
                )
            ],
            correct_action="Ensure scene safety",
            correct_option_text="Assess for hazards first",
            why_correct="Protects rescuer",
            discriminating_factor="Fire spreading makes direct approach impossible",
            support_mapping=[
                SupportMapping(quote_id="Q1", role="primary_support", supports_claim="safety first")
            ],
            distractors=[
                PlannedDistractor(
                    text="Begin CPR", tag=DistractorTag.WRONG_SEQUENCE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="Order wrong", stem_fact_violated="Fire present"
                ),
                PlannedDistractor(
                    text="Move patient", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.CREATES_NEW_HAZARD,
                    why_wrong="Dangerous", stem_fact_violated="Fire spreading"
                ),
                PlannedDistractor(
                    text="Call dispatch", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="Not first", stem_fact_violated="Safety assessment first"
                ),
            ],
            scope_tags=["scene_safety"],
            out_of_scope_warning="Avoid advanced procedures",
            stem_type="scenario",
            stem_constraints=["Include context"],
        )

        formatted = format_plan_for_generation(plan, quotes)

        assert "ITEM PLAN" in formatted
        assert "traffic" in formatted
        assert "[Q1]" in formatted
        assert "DECISION LOCKS" in formatted
        assert "feasibility" in formatted.lower()
        assert "DISCRIMINATING FACTOR" in formatted
        assert "failure reason" in formatted.lower()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_scope_tags(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="Source states this",
            support_mapping=[],
            distractors=[],
            scope_tags=["scene_safety", "hazard_assessment"],
            stem_type="direct",
            stem_constraints=[],
        )

        tags = extract_scope_tags_from_plan(plan)
        assert tags == ["scene_safety", "hazard_assessment"]

    def test_get_required_quote_ids(self):
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="Source states this",
            support_mapping=[
                SupportMapping(quote_id="Q1", role="primary_support", supports_claim="main"),
                SupportMapping(quote_id="Q2", role="secondary_support", supports_claim="extra"),
                SupportMapping(quote_id="Q3", role="primary_support", supports_claim="also main"),
            ],
            distractors=[],
            scope_tags=[],
            stem_type="direct",
            stem_constraints=[],
        )

        required = get_required_quote_ids(plan)
        assert required == ["Q1", "Q3"]


class TestValidatePlanAmbiguity:
    """Tests for validate_plan_ambiguity function."""

    def test_distractor_not_covered_by_lock(self):
        """Each distractor should be eliminated by a decision lock."""
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FIRST_ACTION,
            hazard_cues=["fire"],
            decision_locks=[
                DecisionLockSpec(
                    lock_type=DecisionLock.FEASIBILITY,
                    constraint_text="Fire is present",
                    eliminates_options=["Begin CPR"],  # Only covers one distractor
                )
            ],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="Fire makes approach impossible",
            support_mapping=[],
            distractors=[
                PlannedDistractor(
                    text="Begin CPR", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="Fire"
                ),
                PlannedDistractor(
                    text="Move patient", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.CREATES_NEW_HAZARD,
                    why_wrong="r2", stem_fact_violated="Fire"
                ),
                PlannedDistractor(
                    text="Call dispatch", tag=DistractorTag.PARTIAL,
                    failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
                    why_wrong="r3", stem_fact_violated="constraint"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan_ambiguity(plan)
        # Should flag distractors not covered by decision locks
        assert any("not explicitly eliminated" in issue for issue in issues)

    def test_common_sense_discriminating_factor(self):
        """Discriminating factor should not rely on 'common sense'."""
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="This is common sense that everyone knows",  # Bad!
            support_mapping=[],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="constraint"
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan_ambiguity(plan)
        assert any("common sense" in issue for issue in issues)

    def test_missing_plausibility_source(self):
        """Each distractor should explain why someone might choose it."""
        plan = ItemPlan(
            item_id="item_01",
            learning_target="test",
            question_type=QuestionType.FACTUAL,
            hazard_cues=[],
            decision_locks=[],
            correct_action="action",
            correct_option_text="option",
            why_correct="reason",
            discriminating_factor="Source explicitly states this",
            support_mapping=[],
            distractors=[
                PlannedDistractor(
                    text="d1", tag=DistractorTag.UNSAFE,
                    failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
                    why_wrong="r1", stem_fact_violated="constraint",
                    plausibility_source=None,  # Missing!
                ),
            ],
            scope_tags=["scope1"],
            stem_type="direct",
            stem_constraints=[],
        )

        issues = validate_plan_ambiguity(plan)
        assert any("plausibility_source" in issue for issue in issues)
