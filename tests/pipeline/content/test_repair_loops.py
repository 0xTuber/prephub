"""Tests for repair loops module."""

import json
from unittest.mock import MagicMock

import pytest

from course_builder.pipeline.content.repair_loops import (
    RepairType,
    RepairAttempt,
    RepairResult,
    repair_explanation,
    repair_stem,
    repair_distractors,
    repair_correct_option,
    run_repair_loop,
    can_repair,
    get_repair_summary,
    MAX_REPAIR_ATTEMPTS,
)
from course_builder.pipeline.content.quote_extraction import ExtractedQuote
from course_builder.pipeline.content.quality_tiers import (
    GateTier,
    GateResult,
    GateCheckResult,
    QualityGateReport,
)


class TestRepairType:
    """Tests for RepairType enum."""

    def test_repair_type_values(self):
        assert RepairType.EXPLANATION == "explanation"
        assert RepairType.STEM == "stem"
        assert RepairType.DISTRACTORS == "distractors"
        assert RepairType.CORRECT_OPTION == "correct_option"


class TestRepairAttempt:
    """Tests for RepairAttempt model."""

    def test_create_successful_attempt(self):
        attempt = RepairAttempt(
            repair_type=RepairType.EXPLANATION,
            attempt_number=1,
            original_text="Original explanation.",
            repaired_text="Repaired explanation with [Q1] citation.",
            success=True,
        )

        assert attempt.success is True
        assert attempt.failure_reason is None

    def test_create_failed_attempt(self):
        attempt = RepairAttempt(
            repair_type=RepairType.DISTRACTORS,
            attempt_number=2,
            original_text='["d1", "d2"]',
            repaired_text='["d1", "d2"]',
            success=False,
            failure_reason="Could not generate valid distractors",
        )

        assert attempt.success is False
        assert "valid distractors" in attempt.failure_reason


class TestRepairExplanation:
    """Tests for repair_explanation function."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        return engine

    @pytest.fixture
    def sample_quotes(self):
        return [
            ExtractedQuote(
                quote_id="Q1",
                text="Scene safety must be ensured first.",
                chunk_id="c1",
                chunk_index=0,
                start_char=0,
                end_char=35,
            ),
            ExtractedQuote(
                quote_id="Q2",
                text="Always assess for hazards.",
                chunk_id="c2",
                chunk_index=1,
                start_char=0,
                end_char=26,
            ),
        ]

    def test_successful_repair(self, mock_engine, sample_quotes):
        mock_response = MagicMock()
        mock_response.text = 'Scene safety is critical [Q1]. As the source states: "Scene safety must be ensured first."'
        mock_engine.generate.return_value = mock_response

        result = repair_explanation(
            engine=mock_engine,
            current_explanation="Scene safety is critical.",
            required_quotes=sample_quotes,
            missing_quote_ids=["Q1"],
        )

        assert result.success is True
        assert "[Q1]" in result.repaired_text

    def test_failed_repair_missing_citation(self, mock_engine, sample_quotes):
        mock_response = MagicMock()
        mock_response.text = "Scene safety is critical."  # No citation
        mock_engine.generate.return_value = mock_response

        result = repair_explanation(
            engine=mock_engine,
            current_explanation="Scene safety is critical.",
            required_quotes=sample_quotes,
            missing_quote_ids=["Q1"],
        )

        assert result.success is False
        assert result.failure_reason is not None


class TestRepairStem:
    """Tests for repair_stem function."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        return engine

    def test_successful_repair(self, mock_engine):
        mock_response = MagicMock()
        mock_response.text = "What is the first priority when arriving at an accident scene?"
        mock_engine.generate.return_value = mock_response

        result = repair_stem(
            engine=mock_engine,
            current_stem="What safety action should you take first?",
            issues=["Stem hints at 'safety' answer"],
            correct_answer="Ensure scene safety",
        )

        assert result.success is True
        assert "safety" not in result.repaired_text.lower() or "scene" in result.repaired_text.lower()

    def test_failed_repair_still_leaks(self, mock_engine):
        mock_response = MagicMock()
        # The repaired stem still contains the exact correct answer
        mock_response.text = "What should you do to ensure scene safety first?"
        mock_engine.generate.return_value = mock_response

        result = repair_stem(
            engine=mock_engine,
            current_stem="What safety action is needed?",
            issues=["Leaks answer"],
            correct_answer="ensure scene safety",  # This appears verbatim in repaired text
        )

        assert result.success is False


class TestRepairDistractors:
    """Tests for repair_distractors function."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        return engine

    def test_successful_repair(self, mock_engine):
        mock_response = MagicMock()
        mock_response.text = '["Begin chest compressions immediately", "Contact medical control", "Document vital signs"]'
        mock_engine.generate.return_value = mock_response

        result = repair_distractors(
            engine=mock_engine,
            current_distractors=["Yes", "No", "Maybe"],
            correct_answer="Ensure scene safety first",
            issues=["Distractors too short"],
            topic_context="Scene safety assessment",
        )

        assert result.success is True
        repaired = json.loads(result.repaired_text)
        assert len(repaired) == 3
        assert all(len(d) > 5 for d in repaired)

    def test_failed_repair_invalid_json(self, mock_engine):
        mock_response = MagicMock()
        mock_response.text = "Not valid JSON"
        mock_engine.generate.return_value = mock_response

        result = repair_distractors(
            engine=mock_engine,
            current_distractors=["d1", "d2", "d3"],
            correct_answer="correct",
            issues=["Issue"],
            topic_context="context",
        )

        assert result.success is False
        assert "parse" in result.failure_reason.lower()


class TestRepairCorrectOption:
    """Tests for repair_correct_option function."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        return engine

    @pytest.fixture
    def sample_quote(self):
        return ExtractedQuote(
            quote_id="Q1",
            text="Scene safety must be ensured before patient contact.",
            chunk_id="c1",
            chunk_index=0,
            start_char=0,
            end_char=50,
        )

    def test_successful_repair(self, mock_engine, sample_quote):
        mock_response = MagicMock()
        mock_response.text = "Assess the scene for hazards before approaching"
        mock_engine.generate.return_value = mock_response

        result = repair_correct_option(
            engine=mock_engine,
            current_option="Check safety",
            source_quote=sample_quote,
        )

        assert result.success is True
        assert result.repaired_text != "Check safety"

    def test_failed_repair_verbatim_copy(self, mock_engine, sample_quote):
        mock_response = MagicMock()
        mock_response.text = "Scene safety must be ensured before patient contact."  # Verbatim
        mock_engine.generate.return_value = mock_response

        result = repair_correct_option(
            engine=mock_engine,
            current_option="Check safety",
            source_quote=sample_quote,
        )

        assert result.success is False
        assert "Verbatim" in result.failure_reason


class TestRunRepairLoop:
    """Tests for run_repair_loop function."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        return engine

    @pytest.fixture
    def sample_quotes(self):
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

    @pytest.fixture
    def sample_item_content(self):
        return {
            "item_id": "item_01",
            "stem": "What is the first priority?",
            "correct_answer": "Ensure scene safety",
            "distractors": ["Yes", "No", "Maybe"],
            "explanation": "Scene safety is important.",
        }

    def test_repairs_distractors(self, mock_engine, sample_quotes, sample_item_content):
        # Setup mock to return valid distractors
        mock_response = MagicMock()
        mock_response.text = '["Begin patient assessment", "Call for backup", "Document findings"]'
        mock_engine.generate.return_value = mock_response

        gate_report = QualityGateReport(
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
                    details={"issues": ["Too short"]},
                    repair_hint="Expand distractors",
                ),
            ],
            overall_pass=True,
            requires_repair=True,
            repair_targets=["distractor_quality"],
        )

        result = run_repair_loop(
            engine=mock_engine,
            item_content=sample_item_content,
            gate_report=gate_report,
            quotes=sample_quotes,
            topic_context="Scene safety",
        )

        assert len(result.repairs_attempted) == 1
        assert result.repairs_attempted[0].repair_type == RepairType.DISTRACTORS


class TestCanRepair:
    """Tests for can_repair function."""

    def test_can_repair_tier2_failure(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=False,
            tier_3_flags=[],
            all_checks=[],
            overall_pass=True,
            requires_repair=True,
            repair_targets=["distractor_quality"],
        )

        assert can_repair(report) is True

    def test_cannot_repair_tier1_failure(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=False,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[],
            overall_pass=False,
            requires_repair=False,
            repair_targets=[],
        )

        assert can_repair(report) is False

    def test_no_repair_needed(self):
        report = QualityGateReport(
            item_id="item_01",
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_flags=[],
            all_checks=[],
            overall_pass=True,
            requires_repair=False,
            repair_targets=[],
        )

        assert can_repair(report) is False


class TestGetRepairSummary:
    """Tests for get_repair_summary function."""

    def test_summary_fully_repaired(self):
        result = RepairResult(
            item_id="item_01",
            repairs_attempted=[
                RepairAttempt(
                    repair_type=RepairType.DISTRACTORS,
                    attempt_number=1,
                    original_text="[]",
                    repaired_text="[]",
                    success=True,
                )
            ],
            fully_repaired=True,
            remaining_issues=[],
            final_content={},
        )

        summary = get_repair_summary(result)

        assert "FULLY REPAIRED" in summary
        assert "item_01" in summary

    def test_summary_partially_repaired(self):
        result = RepairResult(
            item_id="item_01",
            repairs_attempted=[
                RepairAttempt(
                    repair_type=RepairType.EXPLANATION,
                    attempt_number=1,
                    original_text="text",
                    repaired_text="text",
                    success=False,
                    failure_reason="Quote not cited",
                )
            ],
            fully_repaired=False,
            remaining_issues=["Explanation still missing quote"],
            final_content={},
        )

        summary = get_repair_summary(result)

        assert "PARTIALLY REPAIRED" in summary
        assert "FAILED" in summary
        assert "Quote not cited" in summary


class TestMaxRepairAttempts:
    """Tests for MAX_REPAIR_ATTEMPTS constant."""

    def test_max_attempts_value(self):
        assert MAX_REPAIR_ATTEMPTS == 2
