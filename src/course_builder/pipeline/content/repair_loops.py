"""Targeted repair loops for fixing Tier 2 quality failures.

This module implements micro-rewrites that fix specific issues without
requiring full regeneration:

1. repair_explanation: Insert missing quote citations
2. repair_stem: Remove contradictions or answer-leaking language
3. repair_distractors: Replace problematic distractors
4. repair_correct_option: Force paraphrase of support quote

Each repair type has a maximum of 2 attempts before escalating.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from course_builder.engine import GenerationEngine
    from course_builder.pipeline.content.quote_extraction import ExtractedQuote
    from course_builder.pipeline.content.quality_tiers import QualityGateReport


class RepairType(str, Enum):
    """Types of targeted repairs."""

    EXPLANATION = "explanation"
    STEM = "stem"
    DISTRACTORS = "distractors"
    CORRECT_OPTION = "correct_option"


class RepairAttempt(BaseModel):
    """Record of a single repair attempt."""

    repair_type: RepairType
    attempt_number: int
    original_text: str
    repaired_text: str
    success: bool
    failure_reason: str | None = None


class RepairResult(BaseModel):
    """Result of a repair loop."""

    item_id: str
    repairs_attempted: list[RepairAttempt]
    fully_repaired: bool
    remaining_issues: list[str]
    final_content: dict[str, Any]


MAX_REPAIR_ATTEMPTS = 2


# ============================================================================
# Repair: Explanation (insert missing quotes)
# ============================================================================

REPAIR_EXPLANATION_PROMPT = """Fix this explanation by inserting the required quote citations.

CURRENT EXPLANATION:
{current_explanation}

REQUIRED QUOTES (must cite ALL):
{required_quotes}

INSTRUCTIONS:
1. Insert quote citations in format: [Q1], [Q2], etc.
2. Place citations where the quote text is referenced
3. Use exact quote text where possible
4. Keep the explanation concise

Return ONLY the repaired explanation text, nothing else."""


def repair_explanation(
    engine: "GenerationEngine",
    current_explanation: str,
    required_quotes: list["ExtractedQuote"],
    missing_quote_ids: list[str],
) -> RepairAttempt:
    """Repair explanation by inserting missing quote citations.

    Args:
        engine: Generation engine for LLM calls
        current_explanation: Current explanation text
        required_quotes: All required quotes
        missing_quote_ids: Quote IDs that are missing

    Returns:
        RepairAttempt with result
    """
    from course_builder.engine import GenerationConfig

    # Format required quotes
    quote_strs = []
    for q in required_quotes:
        if q.quote_id in missing_quote_ids:
            quote_strs.append(f"[{q.quote_id}]: \"{q.text}\"")

    prompt = REPAIR_EXPLANATION_PROMPT.format(
        current_explanation=current_explanation,
        required_quotes="\n".join(quote_strs),
    )

    config = GenerationConfig(
        system_prompt="You are an expert at fixing exam question explanations.",
        max_tokens=500,
    )
    result = engine.generate(prompt, config=config)

    repaired_text = result.text.strip()

    # Verify repair success
    all_cited = all(
        f"[{qid}]" in repaired_text
        for qid in missing_quote_ids
    )

    return RepairAttempt(
        repair_type=RepairType.EXPLANATION,
        attempt_number=1,
        original_text=current_explanation,
        repaired_text=repaired_text,
        success=all_cited,
        failure_reason=None if all_cited else "Not all quotes cited",
    )


# ============================================================================
# Repair: Stem (remove contradictions/leakage)
# ============================================================================

REPAIR_STEM_PROMPT = """Fix this question stem to remove the identified issues.

CURRENT STEM:
{current_stem}

ISSUES TO FIX:
{issues}

CORRECT ANSWER (do NOT leak this):
{correct_answer}

INSTRUCTIONS:
1. Remove any language that hints at the correct answer
2. Fix grammatical cues (a/an mismatches)
3. Remove double negatives
4. Keep the clinical scenario intact

Return ONLY the repaired stem text, nothing else."""


def repair_stem(
    engine: "GenerationEngine",
    current_stem: str,
    issues: list[str],
    correct_answer: str,
) -> RepairAttempt:
    """Repair stem by removing contradictions and answer leakage.

    Args:
        engine: Generation engine for LLM calls
        current_stem: Current stem text
        issues: List of identified issues
        correct_answer: The correct answer (to avoid leaking)

    Returns:
        RepairAttempt with result
    """
    from course_builder.engine import GenerationConfig

    prompt = REPAIR_STEM_PROMPT.format(
        current_stem=current_stem,
        issues="\n".join(f"- {issue}" for issue in issues),
        correct_answer=correct_answer,
    )

    config = GenerationConfig(
        system_prompt="You are an expert at fixing exam question stems.",
        max_tokens=300,
    )
    result = engine.generate(prompt, config=config)

    repaired_text = result.text.strip()

    # Basic validation - stem should not contain correct answer verbatim
    correct_lower = correct_answer.lower()
    stem_lower = repaired_text.lower()
    has_leakage = correct_lower in stem_lower

    return RepairAttempt(
        repair_type=RepairType.STEM,
        attempt_number=1,
        original_text=current_stem,
        repaired_text=repaired_text,
        success=not has_leakage,
        failure_reason="Still leaks answer" if has_leakage else None,
    )


# ============================================================================
# Repair: Distractors (replace problematic ones)
# ============================================================================

REPAIR_DISTRACTORS_PROMPT = """Replace the problematic distractors with better alternatives.

CORRECT ANSWER:
{correct_answer}

CURRENT DISTRACTORS (indices to replace: {indices_to_replace}):
{current_distractors}

ISSUES:
{issues}

TOPIC CONTEXT:
{topic_context}

INSTRUCTIONS:
1. Generate replacement distractors that are:
   - Plausible but clearly incorrect
   - Not too short (at least 5 words)
   - Not identical to correct answer
   - Not "All of the above" style
2. Return a JSON array with exactly {num_distractors} distractors

Return ONLY a JSON array like: ["distractor 1", "distractor 2", "distractor 3"]"""


def repair_distractors(
    engine: "GenerationEngine",
    current_distractors: list[str],
    correct_answer: str,
    issues: list[str],
    topic_context: str,
    indices_to_replace: list[int] | None = None,
) -> RepairAttempt:
    """Repair distractors by replacing problematic ones.

    Args:
        engine: Generation engine for LLM calls
        current_distractors: Current distractor list
        correct_answer: The correct answer
        issues: List of identified issues
        topic_context: Topic context for generating alternatives
        indices_to_replace: Specific indices to replace (None = all)

    Returns:
        RepairAttempt with result
    """
    from course_builder.engine import GenerationConfig

    if indices_to_replace is None:
        indices_to_replace = list(range(len(current_distractors)))

    # Format current distractors with indices
    dist_strs = [
        f"{i+1}. \"{d}\"" + (" [REPLACE]" if i in indices_to_replace else "")
        for i, d in enumerate(current_distractors)
    ]

    prompt = REPAIR_DISTRACTORS_PROMPT.format(
        correct_answer=correct_answer,
        indices_to_replace=indices_to_replace,
        current_distractors="\n".join(dist_strs),
        issues="\n".join(f"- {issue}" for issue in issues),
        topic_context=topic_context,
        num_distractors=len(current_distractors),
    )

    config = GenerationConfig(
        system_prompt="You are an expert at writing plausible exam distractors.",
        max_tokens=300,
    )
    result = engine.generate(prompt, config=config)

    # Parse JSON response
    try:
        # Try to extract JSON array
        text = result.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        new_distractors = json.loads(text)

        if not isinstance(new_distractors, list):
            raise ValueError("Expected list")

        # Validate new distractors
        all_valid = True
        for d in new_distractors:
            if len(d.strip()) < 5:
                all_valid = False
            if d.lower().strip() == correct_answer.lower().strip():
                all_valid = False

        return RepairAttempt(
            repair_type=RepairType.DISTRACTORS,
            attempt_number=1,
            original_text=json.dumps(current_distractors),
            repaired_text=json.dumps(new_distractors),
            success=all_valid and len(new_distractors) == len(current_distractors),
            failure_reason=None if all_valid else "Invalid distractor generated",
        )

    except (json.JSONDecodeError, ValueError) as e:
        return RepairAttempt(
            repair_type=RepairType.DISTRACTORS,
            attempt_number=1,
            original_text=json.dumps(current_distractors),
            repaired_text=json.dumps(current_distractors),  # Keep original
            success=False,
            failure_reason=f"Failed to parse response: {e}",
        )


# ============================================================================
# Repair: Correct Option (paraphrase from quote)
# ============================================================================

REPAIR_CORRECT_OPTION_PROMPT = """Rewrite the correct answer option to better reflect the source quote.

CURRENT CORRECT OPTION:
{current_option}

SOURCE QUOTE (base answer on this):
{source_quote}

INSTRUCTIONS:
1. Paraphrase the key concept from the quote
2. Keep it concise (5-15 words)
3. Make it a complete statement
4. Do NOT copy the quote verbatim

Return ONLY the rewritten option text, nothing else."""


def repair_correct_option(
    engine: "GenerationEngine",
    current_option: str,
    source_quote: "ExtractedQuote",
) -> RepairAttempt:
    """Repair correct option by paraphrasing from source quote.

    Args:
        engine: Generation engine for LLM calls
        current_option: Current correct option text
        source_quote: The source quote to paraphrase

    Returns:
        RepairAttempt with result
    """
    from course_builder.engine import GenerationConfig

    prompt = REPAIR_CORRECT_OPTION_PROMPT.format(
        current_option=current_option,
        source_quote=f"[{source_quote.quote_id}]: \"{source_quote.text}\"",
    )

    config = GenerationConfig(
        system_prompt="You are an expert at writing clear exam answer options.",
        max_tokens=100,
    )
    result = engine.generate(prompt, config=config)

    repaired_text = result.text.strip()

    # Validate - should be different from original and not verbatim quote
    is_different = repaired_text.lower() != current_option.lower()
    is_not_verbatim = source_quote.text.lower() not in repaired_text.lower()
    is_reasonable_length = 5 <= len(repaired_text.split()) <= 20

    success = is_different and is_not_verbatim and is_reasonable_length

    failure_reasons = []
    if not is_different:
        failure_reasons.append("Same as original")
    if not is_not_verbatim:
        failure_reasons.append("Verbatim quote copy")
    if not is_reasonable_length:
        failure_reasons.append("Inappropriate length")

    return RepairAttempt(
        repair_type=RepairType.CORRECT_OPTION,
        attempt_number=1,
        original_text=current_option,
        repaired_text=repaired_text,
        success=success,
        failure_reason="; ".join(failure_reasons) if failure_reasons else None,
    )


# ============================================================================
# Main Repair Loop
# ============================================================================


def run_repair_loop(
    engine: "GenerationEngine",
    item_content: dict[str, Any],
    gate_report: "QualityGateReport",
    quotes: list["ExtractedQuote"],
    topic_context: str = "",
) -> RepairResult:
    """Run targeted repairs based on quality gate report.

    Args:
        engine: Generation engine for LLM calls
        item_content: Current item content
        gate_report: Quality gate report with failures
        quotes: Available quotes
        topic_context: Topic context for distractor generation

    Returns:
        RepairResult with all repairs and final content
    """
    from course_builder.pipeline.content.quality_tiers import GateTier, GateResult

    repairs: list[RepairAttempt] = []
    current_content = item_content.copy()
    remaining_issues: list[str] = []

    # Get repair hints from gate report
    repair_targets = gate_report.repair_targets

    # Track attempts per repair type
    attempts: dict[RepairType, int] = {rt: 0 for rt in RepairType}

    for check in gate_report.all_checks:
        if check.tier != GateTier.TIER_2_REPAIR:
            continue
        if check.result != GateResult.FAIL:
            continue

        # Determine repair type from gate name
        if check.gate_name == "distractor_quality":
            repair_type = RepairType.DISTRACTORS
        elif check.gate_name == "difficulty_alignment":
            # Difficulty is harder to repair automatically
            remaining_issues.append(f"Difficulty mismatch: {check.repair_hint}")
            continue
        elif check.gate_name == "decision_grounding":
            repair_type = RepairType.EXPLANATION
        else:
            remaining_issues.append(f"Unknown repair target: {check.gate_name}")
            continue

        # Check attempt limit
        if attempts[repair_type] >= MAX_REPAIR_ATTEMPTS:
            remaining_issues.append(f"Max attempts reached for {repair_type.value}")
            continue

        attempts[repair_type] += 1

        # Execute repair
        if repair_type == RepairType.EXPLANATION:
            missing_ids = check.details.get("missing_quote_ids", [])
            if not missing_ids:
                # Try to determine from verification
                required_ids = [q.quote_id for q in quotes[:2]]  # Use first 2
                missing_ids = required_ids

            attempt = repair_explanation(
                engine=engine,
                current_explanation=current_content.get("explanation", ""),
                required_quotes=quotes,
                missing_quote_ids=missing_ids,
            )
            attempt.attempt_number = attempts[repair_type]

            if attempt.success:
                current_content["explanation"] = attempt.repaired_text
            else:
                remaining_issues.append(f"Explanation repair failed: {attempt.failure_reason}")

            repairs.append(attempt)

        elif repair_type == RepairType.DISTRACTORS:
            issues = check.details.get("issues", [])
            attempt = repair_distractors(
                engine=engine,
                current_distractors=current_content.get("distractors", []),
                correct_answer=current_content.get("correct_answer", ""),
                issues=issues,
                topic_context=topic_context,
            )
            attempt.attempt_number = attempts[repair_type]

            if attempt.success:
                current_content["distractors"] = json.loads(attempt.repaired_text)
            else:
                remaining_issues.append(f"Distractor repair failed: {attempt.failure_reason}")

            repairs.append(attempt)

    return RepairResult(
        item_id=current_content.get("item_id", ""),
        repairs_attempted=repairs,
        fully_repaired=len(remaining_issues) == 0,
        remaining_issues=remaining_issues,
        final_content=current_content,
    )


def can_repair(gate_report: "QualityGateReport") -> bool:
    """Check if the gate report indicates repairable issues.

    Returns True if:
    - Tier 1 passed (hard blockers ok)
    - Tier 2 has failures (repairable issues exist)
    """
    return gate_report.tier_1_passed and not gate_report.tier_2_passed


def get_repair_summary(result: RepairResult) -> str:
    """Get human-readable summary of repair result."""
    lines = [f"Repair Result for {result.item_id}:"]

    if result.fully_repaired:
        lines.append("  Status: FULLY REPAIRED")
    else:
        lines.append("  Status: PARTIALLY REPAIRED")

    lines.append(f"  Repairs attempted: {len(result.repairs_attempted)}")

    for repair in result.repairs_attempted:
        status = "SUCCESS" if repair.success else "FAILED"
        lines.append(f"    - {repair.repair_type.value} (attempt {repair.attempt_number}): {status}")
        if repair.failure_reason:
            lines.append(f"      Reason: {repair.failure_reason}")

    if result.remaining_issues:
        lines.append("  Remaining issues:")
        for issue in result.remaining_issues:
            lines.append(f"    - {issue}")

    return "\n".join(lines)
