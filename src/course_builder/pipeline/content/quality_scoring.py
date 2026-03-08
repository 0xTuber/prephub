"""Quality scoring system for generated question items.

This module computes composite quality scores from multiple dimensions:
- grounding_score: How well the item is grounded in source quotes
- consistency_score: Internal consistency (no contradictions)
- scope_score: Stays within allowed scope
- distractor_score: Quality of distractors
- novelty_score: Uniqueness compared to existing items

Thresholds:
- >= 0.80: Accept
- 0.65 - 0.80: Accept with flags
- < 0.65: Repair, then reject if still low
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from course_builder.pipeline.content.quality_tiers import QualityGateReport
    from course_builder.pipeline.content.quote_extraction import (
        ExtractedQuote,
        GenerationQuoteVerification,
    )


class QualityDecision(str, Enum):
    """Decision based on quality score."""

    ACCEPT = "accept"  # >= 0.80
    ACCEPT_WITH_FLAGS = "accept_with_flags"  # 0.65 - 0.80
    REPAIR = "repair"  # < 0.65 but Tier 1 passed
    REJECT = "reject"  # < 0.65 after repair OR Tier 1 failed


# Thresholds
ACCEPT_THRESHOLD = 0.80
FLAG_THRESHOLD = 0.65


class DimensionScore(BaseModel):
    """Score for a single quality dimension."""

    dimension: str
    score: float  # 0.0 to 1.0
    weight: float  # Weight in composite score
    details: dict[str, Any] = {}


class QualityScore(BaseModel):
    """Complete quality score with all dimensions."""

    item_id: str
    composite_score: float  # Weighted average
    decision: QualityDecision
    dimension_scores: list[DimensionScore]
    flags: list[str]  # Any quality flags
    recommendation: str  # Human-readable recommendation


# ============================================================================
# Dimension Scoring Functions
# ============================================================================


def score_grounding(
    quote_verification: "GenerationQuoteVerification",
    min_quotes_required: int = 1,
) -> DimensionScore:
    """Score how well the item is grounded in source quotes.

    Factors:
    - Number of quotes cited
    - Exact match vs fuzzy match
    - All required quotes present
    """
    total_required = len(quote_verification.required_quotes)
    if total_required == 0:
        # No quotes required - give neutral score
        return DimensionScore(
            dimension="grounding",
            score=0.5,
            weight=0.30,
            details={"reason": "No quotes required"},
        )

    # Base score from quote presence
    found_count = quote_verification.exact_match_count + quote_verification.fuzzy_match_count
    presence_score = min(found_count / max(min_quotes_required, 1), 1.0)

    # Bonus for exact matches
    if found_count > 0:
        exact_ratio = quote_verification.exact_match_count / found_count
        exactness_bonus = exact_ratio * 0.2
    else:
        exactness_bonus = 0.0

    # Penalty for missing required quotes
    missing_penalty = len(quote_verification.missing_quote_ids) * 0.2

    score = min(max(presence_score + exactness_bonus - missing_penalty, 0.0), 1.0)

    return DimensionScore(
        dimension="grounding",
        score=score,
        weight=0.30,
        details={
            "required": total_required,
            "found": found_count,
            "exact_matches": quote_verification.exact_match_count,
            "missing": quote_verification.missing_quote_ids,
        },
    )


def score_consistency(
    gate_report: "QualityGateReport",
) -> DimensionScore:
    """Score internal consistency of the item.

    Factors:
    - Stem-answer consistency
    - Single-best-answer logic
    - No contradictions in explanation
    """
    from course_builder.pipeline.content.quality_tiers import GateTier, GateResult

    consistency_checks = [
        "stem_answer_consistency",
        "single_best_answer",
    ]

    passed = 0
    total = 0
    issues = []

    for check in gate_report.all_checks:
        if check.gate_name in consistency_checks:
            total += 1
            if check.result == GateResult.PASS:
                passed += 1
            else:
                issues.append(check.message)

    if total == 0:
        score = 1.0
    else:
        score = passed / total

    return DimensionScore(
        dimension="consistency",
        score=score,
        weight=0.25,
        details={
            "passed": passed,
            "total": total,
            "issues": issues,
        },
    )


def score_scope(
    gate_report: "QualityGateReport",
) -> DimensionScore:
    """Score scope compliance.

    Factors:
    - Stays within allowed scope
    - No out-of-scope content
    """
    from course_builder.pipeline.content.quality_tiers import GateResult

    for check in gate_report.all_checks:
        if check.gate_name == "scope_legality":
            if check.result == GateResult.PASS:
                return DimensionScore(
                    dimension="scope",
                    score=1.0,
                    weight=0.15,
                    details={"status": "within_scope"},
                )
            else:
                # Out of scope - severe penalty
                out_of_scope = check.details.get("out_of_scope_tags", [])
                return DimensionScore(
                    dimension="scope",
                    score=0.0,
                    weight=0.15,
                    details={
                        "status": "out_of_scope",
                        "out_of_scope_tags": out_of_scope,
                    },
                )

    # No scope check found - assume compliant
    return DimensionScore(
        dimension="scope",
        score=1.0,
        weight=0.15,
        details={"status": "not_checked"},
    )


def score_distractors(
    gate_report: "QualityGateReport",
) -> DimensionScore:
    """Score distractor quality.

    Factors:
    - Plausibility
    - Distinctness
    - No meta-options
    """
    from course_builder.pipeline.content.quality_tiers import GateTier, GateResult

    for check in gate_report.all_checks:
        if check.gate_name == "distractor_quality":
            if check.result == GateResult.PASS:
                return DimensionScore(
                    dimension="distractors",
                    score=1.0,
                    weight=0.20,
                    details={"status": "good_distractors"},
                )
            else:
                # Count issues for partial scoring
                issues = check.details.get("issues", [])
                # Each issue reduces score by 0.25
                penalty = min(len(issues) * 0.25, 0.75)
                score = max(1.0 - penalty, 0.25)
                return DimensionScore(
                    dimension="distractors",
                    score=score,
                    weight=0.20,
                    details={
                        "status": "issues_found",
                        "issue_count": len(issues),
                        "issues": issues,
                    },
                )

    # No distractor check found
    return DimensionScore(
        dimension="distractors",
        score=0.8,
        weight=0.20,
        details={"status": "not_checked"},
    )


def score_novelty(
    gate_report: "QualityGateReport",
) -> DimensionScore:
    """Score novelty/uniqueness.

    Factors:
    - Not duplicate of existing items
    """
    from course_builder.pipeline.content.quality_tiers import GateResult

    for check in gate_report.all_checks:
        if check.gate_name == "novelty":
            if check.result == GateResult.PASS:
                return DimensionScore(
                    dimension="novelty",
                    score=1.0,
                    weight=0.10,
                    details={"status": "novel"},
                )
            elif check.result == GateResult.WARN:
                # Warning - moderate penalty
                similarity = check.details.get("similarity", 0.0)
                score = max(1.0 - similarity, 0.3)
                return DimensionScore(
                    dimension="novelty",
                    score=score,
                    weight=0.10,
                    details={
                        "status": "similar_found",
                        "similarity": similarity,
                        "similar_to": check.details.get("similar_item_id"),
                    },
                )
            else:
                return DimensionScore(
                    dimension="novelty",
                    score=0.5,
                    weight=0.10,
                    details={"status": "possibly_duplicate"},
                )

    # No novelty check found
    return DimensionScore(
        dimension="novelty",
        score=1.0,
        weight=0.10,
        details={"status": "not_checked"},
    )


# ============================================================================
# Composite Scoring
# ============================================================================


def compute_quality_score(
    item_id: str,
    gate_report: "QualityGateReport",
    quote_verification: "GenerationQuoteVerification",
    min_quotes_required: int = 1,
) -> QualityScore:
    """Compute composite quality score from all dimensions.

    Args:
        item_id: Item identifier
        gate_report: Quality gate report
        quote_verification: Quote verification result
        min_quotes_required: Minimum quotes required for grounding

    Returns:
        QualityScore with composite score and decision
    """
    # Compute dimension scores
    dimensions = [
        score_grounding(quote_verification, min_quotes_required),
        score_consistency(gate_report),
        score_scope(gate_report),
        score_distractors(gate_report),
        score_novelty(gate_report),
    ]

    # Compute weighted average
    total_weight = sum(d.weight for d in dimensions)
    weighted_sum = sum(d.score * d.weight for d in dimensions)
    composite = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Determine decision
    if not gate_report.tier_1_passed:
        decision = QualityDecision.REJECT
        recommendation = "Tier 1 hard blockers failed - item cannot be used"
    elif composite >= ACCEPT_THRESHOLD:
        decision = QualityDecision.ACCEPT
        recommendation = "Quality meets threshold - item ready for use"
    elif composite >= FLAG_THRESHOLD:
        decision = QualityDecision.ACCEPT_WITH_FLAGS
        recommendation = "Quality acceptable but flagged for review"
    elif gate_report.requires_repair:
        decision = QualityDecision.REPAIR
        recommendation = "Quality below threshold - attempt repair"
    else:
        decision = QualityDecision.REJECT
        recommendation = "Quality too low and not repairable"

    # Collect flags
    flags = gate_report.tier_3_flags.copy()

    # Add flags for borderline dimensions
    for dim in dimensions:
        if dim.score < 0.5:
            flags.append(f"low_{dim.dimension}")
        elif dim.score < 0.7:
            flags.append(f"borderline_{dim.dimension}")

    return QualityScore(
        item_id=item_id,
        composite_score=round(composite, 3),
        decision=decision,
        dimension_scores=dimensions,
        flags=flags,
        recommendation=recommendation,
    )


def score_after_repair(
    original_score: QualityScore,
    new_gate_report: "QualityGateReport",
    new_quote_verification: "GenerationQuoteVerification",
) -> QualityScore:
    """Recompute score after repair and determine final decision.

    Args:
        original_score: Score before repair
        new_gate_report: Gate report after repair
        new_quote_verification: Quote verification after repair

    Returns:
        Updated QualityScore with final decision
    """
    new_score = compute_quality_score(
        item_id=original_score.item_id,
        gate_report=new_gate_report,
        quote_verification=new_quote_verification,
    )

    # If score improved above threshold, accept
    if new_score.composite_score >= FLAG_THRESHOLD:
        return new_score

    # If still below after repair, reject
    return QualityScore(
        item_id=new_score.item_id,
        composite_score=new_score.composite_score,
        decision=QualityDecision.REJECT,
        dimension_scores=new_score.dimension_scores,
        flags=new_score.flags + ["failed_after_repair"],
        recommendation="Quality still too low after repair - item rejected",
    )


def format_score_report(score: QualityScore) -> str:
    """Format quality score as human-readable report."""
    lines = [
        f"Quality Score Report: {score.item_id}",
        "=" * 50,
        f"Composite Score: {score.composite_score:.2f}",
        f"Decision: {score.decision.value.upper()}",
        "",
        "Dimension Breakdown:",
    ]

    for dim in score.dimension_scores:
        bar = "█" * int(dim.score * 10) + "░" * (10 - int(dim.score * 10))
        lines.append(f"  {dim.dimension:15} [{bar}] {dim.score:.2f} (weight: {dim.weight:.2f})")

    if score.flags:
        lines.append("")
        lines.append("Flags:")
        for flag in score.flags:
            lines.append(f"  - {flag}")

    lines.append("")
    lines.append(f"Recommendation: {score.recommendation}")

    return "\n".join(lines)


def get_improvement_suggestions(score: QualityScore) -> list[str]:
    """Get suggestions for improving quality score."""
    suggestions = []

    for dim in score.dimension_scores:
        if dim.score < 0.7:
            if dim.dimension == "grounding":
                suggestions.append("Add more explicit quote citations in explanation")
            elif dim.dimension == "consistency":
                suggestions.append("Review stem and answer for contradictions or leakage")
            elif dim.dimension == "scope":
                suggestions.append("Ensure content stays within allowed scope")
            elif dim.dimension == "distractors":
                suggestions.append("Improve distractor plausibility and distinctness")
            elif dim.dimension == "novelty":
                suggestions.append("Differentiate from similar existing items")

    return suggestions
