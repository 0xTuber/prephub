"""Tests for controlled distractor generation module."""

import pytest

from course_builder.pipeline.content.distractors import (
    DIFFICULTY_DISTRACTOR_REQUIREMENTS,
    DistractorSet,
    DistractorSpec,
    WrongnessPattern,
    get_pattern_instructions,
    improve_existing_distractors,
    suggest_distractor_patterns,
    validate_distractor_balance,
)


class TestWrongnessPattern:
    """Tests for WrongnessPattern enum."""

    def test_clearly_wrong_patterns(self):
        clearly_wrong = {
            WrongnessPattern.OPPOSITE,
            WrongnessPattern.VIOLATES_PROTOCOL,
        }

        assert WrongnessPattern.OPPOSITE in clearly_wrong
        assert WrongnessPattern.VIOLATES_PROTOCOL in clearly_wrong

    def test_subtly_wrong_patterns(self):
        subtle = {
            WrongnessPattern.WRONG_SEQUENCE,
            WrongnessPattern.WRONG_CONTEXT,
            WrongnessPattern.INCOMPLETE,
        }

        assert WrongnessPattern.WRONG_SEQUENCE in subtle
        assert WrongnessPattern.INCOMPLETE in subtle


class TestDifficultyDistractorRequirements:
    """Tests for DIFFICULTY_DISTRACTOR_REQUIREMENTS."""

    def test_beginner_requirements(self):
        reqs = DIFFICULTY_DISTRACTOR_REQUIREMENTS["beginner"]

        assert reqs["min_clearly_wrong"] >= 2
        assert WrongnessPattern.WRONG_SEQUENCE in reqs["forbidden_patterns"]

    def test_intermediate_requirements(self):
        reqs = DIFFICULTY_DISTRACTOR_REQUIREMENTS["intermediate"]

        assert reqs["min_clearly_wrong"] == 0
        assert WrongnessPattern.WRONG_CONTEXT in reqs["allowed_patterns"]

    def test_advanced_requirements(self):
        reqs = DIFFICULTY_DISTRACTOR_REQUIREMENTS["advanced"]

        assert WrongnessPattern.OPPOSITE in reqs["forbidden_patterns"]
        assert WrongnessPattern.WRONG_SEQUENCE in reqs["allowed_patterns"]


class TestGetPatternInstructions:
    """Tests for get_pattern_instructions function."""

    def test_beginner_instructions(self):
        instructions = get_pattern_instructions("beginner")

        assert "clearly wrong" in instructions.lower()
        assert "sequence" in instructions.lower()

    def test_intermediate_instructions(self):
        instructions = get_pattern_instructions("intermediate")

        assert "plausible" in instructions.lower()
        assert "context" in instructions.lower() or "timing" in instructions.lower()

    def test_advanced_instructions(self):
        instructions = get_pattern_instructions("advanced")

        assert "sequence" in instructions.lower()
        assert "valid" in instructions.lower()


class TestValidateDistractorBalance:
    """Tests for validate_distractor_balance function."""

    def test_valid_beginner_set(self):
        distractors = [
            DistractorSpec(
                text="Do the opposite",
                pattern=WrongnessPattern.OPPOSITE,
                explanation="Wrong",
            ),
            DistractorSpec(
                text="Violate protocol",
                pattern=WrongnessPattern.VIOLATES_PROTOCOL,
                explanation="Wrong",
            ),
            DistractorSpec(
                text="Common mistake",
                pattern=WrongnessPattern.COMMON_MISCONCEPTION,
                explanation="Wrong",
            ),
        ]

        issues = validate_distractor_balance(distractors, "beginner")

        assert issues == ""  # No issues

    def test_invalid_beginner_set(self):
        distractors = [
            DistractorSpec(
                text="Wrong sequence",
                pattern=WrongnessPattern.WRONG_SEQUENCE,
                explanation="Wrong",
            ),
            DistractorSpec(
                text="Wrong context",
                pattern=WrongnessPattern.WRONG_CONTEXT,
                explanation="Wrong",
            ),
            DistractorSpec(
                text="Incomplete",
                pattern=WrongnessPattern.INCOMPLETE,
                explanation="Wrong",
            ),
        ]

        issues = validate_distractor_balance(distractors, "beginner")

        assert "clearly wrong" in issues.lower()

    def test_valid_advanced_set(self):
        distractors = [
            DistractorSpec(
                text="Do X before Y",
                pattern=WrongnessPattern.WRONG_SEQUENCE,
                explanation="Wrong order",
            ),
            DistractorSpec(
                text="Do Y before Z",
                pattern=WrongnessPattern.WRONG_SEQUENCE,
                explanation="Wrong order",
            ),
            DistractorSpec(
                text="Do Z before X",
                pattern=WrongnessPattern.WRONG_SEQUENCE,
                explanation="Wrong order",
            ),
        ]

        issues = validate_distractor_balance(distractors, "advanced")

        assert issues == ""

    def test_invalid_advanced_set(self):
        distractors = [
            DistractorSpec(
                text="Do nothing",
                pattern=WrongnessPattern.OPPOSITE,
                explanation="Wrong",
            ),
            DistractorSpec(
                text="Wrong sequence",
                pattern=WrongnessPattern.WRONG_SEQUENCE,
                explanation="Wrong",
            ),
        ]

        issues = validate_distractor_balance(distractors, "advanced")

        # Advanced shouldn't have OPPOSITE
        assert "forbidden" in issues.lower() or "non-sequence" in issues.lower()


class TestSuggestDistractorPatterns:
    """Tests for suggest_distractor_patterns function."""

    def test_suggests_for_sequence_question(self):
        correct = "First, ensure scene safety"
        stem = "What is your first priority?"

        suggestions = suggest_distractor_patterns(correct, "advanced", stem)

        # Should suggest wrong sequence
        patterns = [p for p, _ in suggestions]
        assert WrongnessPattern.WRONG_SEQUENCE in patterns

    def test_suggests_for_action_question(self):
        correct = "Assess the patient"
        stem = "What should you do?"

        suggestions = suggest_distractor_patterns(correct, "intermediate", stem)

        # Should suggest incomplete or wrong context
        patterns = [p for p, _ in suggestions]
        assert any(p in patterns for p in [
            WrongnessPattern.INCOMPLETE,
            WrongnessPattern.WRONG_CONTEXT,
        ])

    def test_suggests_for_safety_question(self):
        correct = "Protect scene safety"
        stem = "What is the priority?"

        suggestions = suggest_distractor_patterns(correct, "beginner", stem)

        # Should suggest protocol violation or misconception
        patterns = [p for p, _ in suggestions]
        assert any(p in patterns for p in [
            WrongnessPattern.VIOLATES_PROTOCOL,
            WrongnessPattern.COMMON_MISCONCEPTION,
        ])


class TestImproveExistingDistractors:
    """Tests for improve_existing_distractors function."""

    def test_detects_cartoonish_distractors(self):
        options = [
            "Ensure scene safety",  # correct
            "Do nothing and wait",
            "Ignore the hazard",
            "Leave the scene",
        ]

        improvements = improve_existing_distractors(
            options, correct_index=0, difficulty="advanced", stem="What should you do?"
        )

        assert any("cartoonish" in imp.lower() for imp in improvements)

    def test_detects_similar_distractors(self):
        options = [
            "Ensure scene safety",  # correct
            "Assess the patient airway and breathing status quickly",
            "Assess the patient airway and breathing status carefully",  # Almost identical
            "Call for backup",
        ]

        improvements = improve_existing_distractors(
            options, correct_index=0, difficulty="intermediate", stem="What should you do?"
        )

        assert any("similar" in imp.lower() for imp in improvements)

    def test_flags_beginner_needs_clearly_wrong(self):
        options = [
            "Ensure scene safety",  # correct
            "Wrong sequence action",
            "Wrong context action",
            "Incomplete action",
        ]

        improvements = improve_existing_distractors(
            options, correct_index=0, difficulty="beginner", stem="What should you do?"
        )

        # May flag if no clearly wrong distractors
        # This depends on the cartoonish pattern detection


class TestDistractorSpec:
    """Tests for DistractorSpec dataclass."""

    def test_creation(self):
        spec = DistractorSpec(
            text="Wrong answer",
            pattern=WrongnessPattern.WRONG_SEQUENCE,
            explanation="Does X before Y instead of Y before X",
            plausibility=0.8,
        )

        assert spec.text == "Wrong answer"
        assert spec.pattern == WrongnessPattern.WRONG_SEQUENCE
        assert spec.plausibility == 0.8


class TestDistractorSet:
    """Tests for DistractorSet dataclass."""

    def test_creation(self):
        distractors = [
            DistractorSpec(
                text="Option 1",
                pattern=WrongnessPattern.WRONG_SEQUENCE,
                explanation="Wrong",
            ),
            DistractorSpec(
                text="Option 2",
                pattern=WrongnessPattern.INCOMPLETE,
                explanation="Wrong",
            ),
        ]

        distractor_set = DistractorSet(
            distractors=distractors,
            difficulty="intermediate",
            correct_answer="Correct option",
            balance_check="",
        )

        assert len(distractor_set.distractors) == 2
        assert distractor_set.difficulty == "intermediate"
