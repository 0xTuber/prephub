"""Ambiguity gate for answer defensibility checking.

This module implements the two-pass answerability test:
1. Pass 1: Answer without seeing marked correct answer
2. Pass 2: Answer with different framing
3. If passes disagree → question is ambiguous

Quality invariant enforced:
- Single best answer: If 2 answers are defensible → item is invalid
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from course_builder.engine import GenerationConfig, GenerationEngine
    from course_builder.pipeline.content.item_plan import ItemPlan


class AmbiguityStatus(str, Enum):
    """Status of ambiguity check."""

    CLEAR = "clear"  # Single best answer
    AMBIGUOUS = "ambiguous"  # Multiple defensible answers
    INVALID = "invalid"  # Question has issues


@dataclass
class AmbiguityCheckResult:
    """Result of ambiguity check."""

    status: AmbiguityStatus
    pass1_answer: int | None = None  # Index selected in pass 1
    pass2_answer: int | None = None  # Index selected in pass 2
    intended_answer: int | None = None  # Originally marked correct
    defensible_options: list[int] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0


AMBIGUITY_CHECK_PROMPT_PASS1 = """You are an expert EMS instructor evaluating this exam question.

QUESTION:
{stem}

OPTIONS:
{options}

Based ONLY on EMS best practices and the information in the question, which option is the SINGLE BEST answer?

Return a JSON object:
{{
  "selected_index": 0,
  "confidence": 0.9,
  "reasoning": "Brief explanation of why this is best"
}}

Return ONLY the JSON object."""


AMBIGUITY_CHECK_PROMPT_PASS2 = """You are a skeptical exam reviewer looking for ambiguity.

QUESTION:
{stem}

OPTIONS:
{options}

A student has argued that option {alternative_index} ("{alternative_text}") could also be correct.
Could a reasonable expert defend this answer? If so, explain how.

Return a JSON object:
{{
  "is_defensible": true,
  "defense": "How this option could be defended",
  "weakness": "Why it's not the best answer (if any)"
}}

Return ONLY the JSON object."""


DEFENSIBILITY_CHECK_PROMPT = """You are an expert EMS examiner checking for answer ambiguity.

QUESTION:
{stem}

OPTIONS:
{options}

INTENDED CORRECT ANSWER: Option {correct_index} - "{correct_text}"

For each OTHER option, determine if it could reasonably be defended as correct:

Return a JSON object:
{{
  "defensible_options": [
    {{"index": 0, "is_defensible": false, "reason": "Clearly wrong because..."}},
    {{"index": 1, "is_defensible": true, "reason": "Could be defended because..."}},
    ...
  ],
  "overall_clarity": "clear" or "ambiguous",
  "ambiguity_reason": "Why there might be ambiguity (if any)"
}}

Return ONLY the JSON object."""


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def format_options_for_prompt(options: list[str]) -> str:
    """Format options for inclusion in prompt."""
    return "\n".join(f"{i}) {opt}" for i, opt in enumerate(options))


def two_pass_answerability_test(
    engine: "GenerationEngine",
    stem: str,
    options: list[str],
    intended_correct_index: int,
) -> AmbiguityCheckResult:
    """Run two-pass answerability test to detect ambiguous questions.

    Pass 1: Have model answer without knowing intended correct
    Pass 2: Check if alternative answers are defensible

    Args:
        engine: Generation engine for LLM calls
        stem: Question stem
        options: Answer options
        intended_correct_index: The intended correct answer index

    Returns:
        AmbiguityCheckResult with ambiguity status
    """
    from course_builder.engine import GenerationConfig

    # Pass 1: Answer without seeing marked correct
    prompt1 = AMBIGUITY_CHECK_PROMPT_PASS1.format(
        stem=stem,
        options=format_options_for_prompt(options),
    )

    config = GenerationConfig(temperature=0.0)  # Deterministic
    result1 = engine.generate(prompt1, config=config)

    try:
        raw_json = _strip_code_fences(result1.text)
        data1 = json.loads(raw_json)
        pass1_answer = data1.get("selected_index")
        pass1_confidence = data1.get("confidence", 0.5)
        pass1_reasoning = data1.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        pass1_answer = None
        pass1_confidence = 0.0
        pass1_reasoning = ""

    # Pass 2: Check if model's answer differs from intended
    if pass1_answer is not None and pass1_answer != intended_correct_index:
        # Model selected different answer - check defensibility
        alternative_index = pass1_answer
        alternative_text = options[alternative_index] if 0 <= alternative_index < len(options) else ""

        prompt2 = AMBIGUITY_CHECK_PROMPT_PASS2.format(
            stem=stem,
            options=format_options_for_prompt(options),
            alternative_index=alternative_index,
            alternative_text=alternative_text,
        )

        result2 = engine.generate(prompt2, config=config)

        try:
            raw_json = _strip_code_fences(result2.text)
            data2 = json.loads(raw_json)
            is_defensible = data2.get("is_defensible", False)
            defense = data2.get("defense", "")
        except (json.JSONDecodeError, KeyError):
            is_defensible = False
            defense = ""

        if is_defensible:
            return AmbiguityCheckResult(
                status=AmbiguityStatus.AMBIGUOUS,
                pass1_answer=pass1_answer,
                pass2_answer=pass1_answer,
                intended_answer=intended_correct_index,
                defensible_options=[intended_correct_index, pass1_answer],
                reasoning=f"Model selected {pass1_answer} instead of {intended_correct_index}. Defense: {defense}",
                confidence=1.0 - pass1_confidence,  # Lower confidence = more ambiguity
            )

    # Pass 1 agrees with intended - question is clear
    return AmbiguityCheckResult(
        status=AmbiguityStatus.CLEAR,
        pass1_answer=pass1_answer,
        pass2_answer=pass1_answer,
        intended_answer=intended_correct_index,
        defensible_options=[intended_correct_index],
        reasoning=pass1_reasoning,
        confidence=pass1_confidence,
    )


def check_all_options_defensibility(
    engine: "GenerationEngine",
    stem: str,
    options: list[str],
    correct_index: int,
) -> AmbiguityCheckResult:
    """Check defensibility of all options in one call.

    More efficient than two-pass for comprehensive check.

    Args:
        engine: Generation engine for LLM calls
        stem: Question stem
        options: Answer options
        correct_index: The correct answer index

    Returns:
        AmbiguityCheckResult with detailed defensibility info
    """
    from course_builder.engine import GenerationConfig

    prompt = DEFENSIBILITY_CHECK_PROMPT.format(
        stem=stem,
        options=format_options_for_prompt(options),
        correct_index=correct_index,
        correct_text=options[correct_index] if 0 <= correct_index < len(options) else "",
    )

    config = GenerationConfig(temperature=0.0)
    result = engine.generate(prompt, config=config)

    try:
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)

        defensible = [correct_index]  # Correct is always defensible
        for opt_check in data.get("defensible_options", []):
            if opt_check.get("is_defensible") and opt_check.get("index") != correct_index:
                defensible.append(opt_check.get("index"))

        overall_clarity = data.get("overall_clarity", "clear")
        ambiguity_reason = data.get("ambiguity_reason", "")

        if len(defensible) > 1:
            status = AmbiguityStatus.AMBIGUOUS
        elif overall_clarity == "ambiguous":
            status = AmbiguityStatus.AMBIGUOUS
        else:
            status = AmbiguityStatus.CLEAR

        return AmbiguityCheckResult(
            status=status,
            intended_answer=correct_index,
            defensible_options=defensible,
            reasoning=ambiguity_reason,
            confidence=0.9 if status == AmbiguityStatus.CLEAR else 0.5,
        )

    except (json.JSONDecodeError, KeyError):
        return AmbiguityCheckResult(
            status=AmbiguityStatus.INVALID,
            intended_answer=correct_index,
            reasoning="Failed to parse defensibility check response",
            confidence=0.0,
        )


def quick_ambiguity_check(
    stem: str,
    options: list[str],
    correct_index: int,
) -> tuple[bool, list[str]]:
    """Quick heuristic check for common ambiguity patterns.

    Does not require LLM call. Checks for:
    - Options that are too similar
    - Multiple "correct-sounding" options
    - Vague qualifying language

    Args:
        stem: Question stem
        options: Answer options
        correct_index: Correct answer index

    Returns:
        (has_ambiguity_risk, warnings)
    """
    warnings = []

    # Check for similar options
    for i, opt1 in enumerate(options):
        for j, opt2 in enumerate(options):
            if i >= j:
                continue
            # Simple word overlap check
            words1 = set(opt1.lower().split())
            words2 = set(opt2.lower().split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            if overlap > 0.7:
                warnings.append(f"Options {i} and {j} are very similar ({overlap:.0%} overlap)")

    # Check for vague qualifying language in correct answer
    correct = options[correct_index] if 0 <= correct_index < len(options) else ""
    vague_patterns = [
        r"\bmost\b", r"\bbest\b", r"\bappropriate\b", r"\bpossibly\b",
        r"\bmay\b", r"\bcould\b", r"\bmight\b",
    ]
    vague_in_correct = sum(1 for p in vague_patterns if re.search(p, correct.lower()))
    if vague_in_correct >= 2:
        warnings.append(f"Correct answer uses multiple vague qualifiers ({vague_in_correct})")

    # Check for multiple options with similar action verbs
    action_verbs = [
        "assess", "evaluate", "check", "monitor", "administer", "provide",
        "establish", "maintain", "position", "secure", "protect",
    ]
    action_counts = {}
    for verb in action_verbs:
        count = sum(1 for opt in options if verb in opt.lower())
        if count >= 3:
            action_counts[verb] = count

    if action_counts:
        warnings.append(f"Multiple options use same actions: {action_counts}")

    has_risk = len(warnings) > 0
    return has_risk, warnings


class AmbiguityGate:
    """Gate for checking question ambiguity.

    Usage:
        gate = AmbiguityGate(engine=engine, strict=True)

        result = gate.check(stem, options, correct_index)

        if result.status == AmbiguityStatus.AMBIGUOUS:
            # Reject or repair the question
    """

    def __init__(
        self,
        engine: "GenerationEngine | None" = None,
        strict: bool = False,
        use_quick_check: bool = True,
    ):
        """Initialize ambiguity gate.

        Args:
            engine: Generation engine for LLM-based checks
            strict: If True, use two-pass test; else use single-call check
            use_quick_check: Run quick heuristic check first
        """
        self.engine = engine
        self.strict = strict
        self.use_quick_check = use_quick_check

    def check(
        self,
        stem: str,
        options: list[str],
        correct_index: int,
    ) -> AmbiguityCheckResult:
        """Check question for ambiguity.

        Args:
            stem: Question stem
            options: Answer options
            correct_index: Correct answer index

        Returns:
            AmbiguityCheckResult with status
        """
        # Quick heuristic check first
        if self.use_quick_check:
            has_risk, warnings = quick_ambiguity_check(stem, options, correct_index)
            if has_risk and not self.engine:
                # No LLM available, return based on heuristics
                return AmbiguityCheckResult(
                    status=AmbiguityStatus.AMBIGUOUS,
                    intended_answer=correct_index,
                    reasoning="; ".join(warnings),
                    confidence=0.6,
                )

        # LLM-based check
        if self.engine:
            if self.strict:
                return two_pass_answerability_test(
                    self.engine, stem, options, correct_index
                )
            else:
                return check_all_options_defensibility(
                    self.engine, stem, options, correct_index
                )

        # No LLM, no heuristic issues
        return AmbiguityCheckResult(
            status=AmbiguityStatus.CLEAR,
            intended_answer=correct_index,
            defensible_options=[correct_index],
            confidence=0.7,  # Lower confidence without LLM check
        )


# ============================================================================
# "10 Competent People" Test
# ============================================================================


TEN_COMPETENT_TEST_PROMPT = """You are simulating a panel of 10 competent EMRs answering this exam question.

QUESTION:
{stem}

OPTIONS:
{options}

TASK: Act as 10 different competent EMRs with varying experience levels (2-15 years).
For each EMR, determine which answer they would select and briefly note why.

A question PASSES the test if ≥9 EMRs select the same answer.
A question FAILS if EMRs split their votes (ambiguity).

Return JSON:
{{
  "emr_responses": [
    {{"emr_id": 1, "years_exp": 5, "selected": 0, "reasoning": "Brief reason"}},
    ...
  ],
  "vote_counts": {{"0": 8, "1": 1, "2": 1, "3": 0}},
  "passes_test": false,
  "consensus_answer": 0,
  "split_reason": "Why EMRs disagreed (if any)"
}}

Return ONLY the JSON object."""


@dataclass
class TenCompetentResult:
    """Result of the 10 competent people test."""

    passes_test: bool
    consensus_answer: int | None
    vote_counts: dict[int, int]
    split_reason: str
    emr_responses: list[dict]
    confidence: float


def ten_competent_people_test(
    engine: "GenerationEngine",
    stem: str,
    options: list[str],
) -> TenCompetentResult:
    """Run the "10 competent people" test.

    This simulates 10 EMRs with varying experience answering the question.
    If ≥9 select the same answer, the question passes.
    If votes are split, the question is ambiguous.

    Args:
        engine: Generation engine for LLM calls
        stem: Question stem
        options: Answer options

    Returns:
        TenCompetentResult with vote distribution
    """
    from course_builder.engine import GenerationConfig

    prompt = TEN_COMPETENT_TEST_PROMPT.format(
        stem=stem,
        options=format_options_for_prompt(options),
    )

    config = GenerationConfig(temperature=0.3)  # Slight variation for realistic responses
    result = engine.generate(prompt, config=config)

    try:
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)

        passes = data.get("passes_test", False)
        consensus = data.get("consensus_answer")
        vote_counts = {int(k): v for k, v in data.get("vote_counts", {}).items()}
        split_reason = data.get("split_reason", "")
        emr_responses = data.get("emr_responses", [])

        # Calculate confidence based on vote distribution
        if vote_counts:
            max_votes = max(vote_counts.values())
            confidence = max_votes / 10.0
        else:
            confidence = 0.5

        return TenCompetentResult(
            passes_test=passes,
            consensus_answer=consensus,
            vote_counts=vote_counts,
            split_reason=split_reason,
            emr_responses=emr_responses,
            confidence=confidence,
        )

    except (json.JSONDecodeError, KeyError, ValueError):
        return TenCompetentResult(
            passes_test=False,
            consensus_answer=None,
            vote_counts={},
            split_reason="Failed to parse LLM response",
            emr_responses=[],
            confidence=0.0,
        )


def check_decision_lock_coverage(
    stem: str,
    plan: "ItemPlan",
) -> tuple[bool, list[str]]:
    """Check that decision locks from plan are present in the stem.

    For FIRST/BEST questions to be unambiguous, their decision locks
    must actually appear in the generated stem.

    Args:
        stem: Generated question stem
        plan: Item plan with decision locks

    Returns:
        (all_locks_present, missing_locks)
    """
    from course_builder.pipeline.content.item_plan import REQUIRES_DECISION_LOCK

    # Only check for question types that require decision locks
    if plan.question_type not in REQUIRES_DECISION_LOCK:
        return True, []

    if not plan.decision_locks:
        return False, ["No decision locks defined for FIRST/BEST question"]

    missing = []
    stem_lower = stem.lower()

    for lock in plan.decision_locks:
        # Check if key terms from the constraint appear in stem
        constraint_words = set(re.findall(r"\b\w{4,}\b", lock.constraint_text.lower()))
        # Remove common words
        constraint_words -= {
            "that", "this", "with", "from", "have", "been", "will", "would",
            "could", "should", "about", "their", "when", "what", "which",
        }

        if len(constraint_words) < 2:
            continue

        stem_words = set(re.findall(r"\b\w{4,}\b", stem_lower))
        overlap = constraint_words & stem_words

        # At least 40% of constraint terms should appear in stem
        if len(overlap) / len(constraint_words) < 0.4:
            missing.append(
                f"Decision lock '{lock.lock_type.value}' not adequately reflected in stem. "
                f"Expected terms: {constraint_words}, found: {overlap}"
            )

    return len(missing) == 0, missing


def check_distractor_differentiation(
    plan: "ItemPlan",
) -> tuple[bool, list[str]]:
    """Check that distractors fail for DIFFERENT reasons.

    Each distractor should have a unique failure_reason to avoid
    "distractor asymmetry" where some are obviously wrong.

    Args:
        plan: Item plan with distractors

    Returns:
        (all_different, issues)
    """
    issues = []

    if len(plan.distractors) < 2:
        return True, []

    # Collect failure reasons
    reasons = [d.failure_reason for d in plan.distractors]
    reason_counts: dict[str, int] = {}
    for r in reasons:
        reason_counts[r.value] = reason_counts.get(r.value, 0) + 1

    # Check for duplicates
    duplicates = [r for r, count in reason_counts.items() if count > 1]
    if duplicates:
        issues.append(
            f"Multiple distractors share same failure reason: {duplicates}. "
            "Each distractor should fail for a DIFFERENT reason."
        )

    # Check that stem_fact_violated is specific
    for i, d in enumerate(plan.distractors):
        if not d.stem_fact_violated or len(d.stem_fact_violated) < 10:
            issues.append(
                f"Distractor {i+1} has vague stem_fact_violated: '{d.stem_fact_violated}'. "
                "Should reference a specific fact from the stem."
            )

    return len(issues) == 0, issues


def check_discriminating_factor_explicit(
    plan: "ItemPlan",
    stem: str,
) -> tuple[bool, list[str]]:
    """Check that discriminating factor is explicit in stem, not common sense.

    The discriminating factor should be a specific fact stated in the stem,
    not something that "everyone knows" or "common sense."

    Args:
        plan: Item plan with discriminating_factor
        stem: Generated question stem

    Returns:
        (is_explicit, issues)
    """
    issues = []
    df = plan.discriminating_factor

    if not df:
        return False, ["No discriminating factor defined"]

    # Check for "common sense" language
    common_sense_phrases = [
        "common sense", "everyone knows", "obviously", "general knowledge",
        "standard practice", "always", "typically", "usually", "normally",
        "in general", "as a rule",
    ]

    df_lower = df.lower()
    for phrase in common_sense_phrases:
        if phrase in df_lower:
            issues.append(
                f"Discriminating factor relies on '{phrase}'. "
                "Should reference explicit stem fact instead."
            )

    # Check that key discriminating terms appear in stem
    df_words = set(re.findall(r"\b\w{4,}\b", df_lower))
    df_words -= {
        "that", "this", "with", "from", "have", "been", "will", "would",
        "makes", "answer", "correct", "option", "because",
    }

    if len(df_words) >= 2:
        stem_lower = stem.lower()
        stem_words = set(re.findall(r"\b\w{4,}\b", stem_lower))
        overlap = df_words & stem_words

        if len(overlap) / len(df_words) < 0.3:
            issues.append(
                f"Discriminating factor terms not found in stem. "
                f"Expected: {df_words}, found in stem: {overlap}"
            )

    return len(issues) == 0, issues


@dataclass
class StrictAmbiguityResult:
    """Result of strict ambiguity check."""

    passes: bool
    decision_lock_issues: list[str]
    distractor_issues: list[str]
    discriminating_factor_issues: list[str]
    ten_competent_result: TenCompetentResult | None
    overall_confidence: float
    recommendation: str


def strict_ambiguity_check(
    engine: "GenerationEngine | None",
    stem: str,
    options: list[str],
    correct_index: int,
    plan: "ItemPlan | None" = None,
    run_ten_competent_test: bool = True,
) -> StrictAmbiguityResult:
    """Run comprehensive strict ambiguity check.

    This combines:
    1. Decision lock coverage check
    2. Distractor differentiation check
    3. Discriminating factor explicitness check
    4. "10 competent people" test (optional, requires LLM)

    Args:
        engine: Generation engine for LLM calls (optional)
        stem: Question stem
        options: Answer options
        correct_index: Correct answer index
        plan: Item plan (optional, but recommended)
        run_ten_competent_test: Whether to run the 10 competent test

    Returns:
        StrictAmbiguityResult with comprehensive analysis
    """
    decision_lock_issues: list[str] = []
    distractor_issues: list[str] = []
    discriminating_factor_issues: list[str] = []
    ten_competent_result: TenCompetentResult | None = None

    # Check plan-based constraints if plan provided
    if plan:
        _, dl_issues = check_decision_lock_coverage(stem, plan)
        decision_lock_issues = dl_issues

        _, dist_issues = check_distractor_differentiation(plan)
        distractor_issues = dist_issues

        _, df_issues = check_discriminating_factor_explicit(plan, stem)
        discriminating_factor_issues = df_issues

    # Run 10 competent people test if engine available
    if engine and run_ten_competent_test:
        ten_competent_result = ten_competent_people_test(engine, stem, options)

    # Calculate overall pass/fail
    structural_issues = (
        len(decision_lock_issues) +
        len(distractor_issues) +
        len(discriminating_factor_issues)
    )

    ten_competent_passes = (
        ten_competent_result is None or
        ten_competent_result.passes_test
    )

    passes = structural_issues == 0 and ten_competent_passes

    # Calculate confidence
    if ten_competent_result:
        confidence = ten_competent_result.confidence
    elif structural_issues == 0:
        confidence = 0.8
    else:
        confidence = max(0.3, 1.0 - (structural_issues * 0.2))

    # Generate recommendation
    if passes:
        recommendation = "Question passes strict ambiguity check"
    else:
        issues_summary = []
        if decision_lock_issues:
            issues_summary.append("missing decision locks in stem")
        if distractor_issues:
            issues_summary.append("distractors not differentiated")
        if discriminating_factor_issues:
            issues_summary.append("discriminating factor not explicit")
        if ten_competent_result and not ten_competent_result.passes_test:
            issues_summary.append(f"10 competent test failed: {ten_competent_result.split_reason}")

        recommendation = f"AMBIGUITY RISK: {', '.join(issues_summary)}"

    return StrictAmbiguityResult(
        passes=passes,
        decision_lock_issues=decision_lock_issues,
        distractor_issues=distractor_issues,
        discriminating_factor_issues=discriminating_factor_issues,
        ten_competent_result=ten_competent_result,
        overall_confidence=confidence,
        recommendation=recommendation,
    )
