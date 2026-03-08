"""Item planning phase for structured question generation.

This module implements Phase B.75: a structured planning phase that occurs
AFTER quote extraction but BEFORE generation. The plan provides:

1. Explicit grounding decisions
2. Decision locks to eliminate ambiguity
3. Distractor strategies with clear failure reasons
4. Scope boundaries
5. Stem constraints

The plan JSON guides generation, making it more consistent and verifiable.

Key concepts:
- Decision Locks: Explicit constraints in the stem that make exactly one answer correct
- Failure Reasons: Each distractor must fail for exactly one clear, different reason
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from course_builder.domain.content import CapsuleItem
    from course_builder.engine import GenerationEngine
    from course_builder.pipeline.content.quote_extraction import ExtractedQuote


# ============================================================================
# Decision Locks - Eliminate Ambiguity
# ============================================================================


class DecisionLock(str, Enum):
    """Explicit constraints that eliminate answer ambiguity.

    A decision lock is a fact stated in the stem that makes exactly one
    answer defensible. Without decision locks, "FIRST action" questions
    often have 2+ defensible answers.

    Examples:
    - FEASIBILITY: "The main breaker is inside the hazard area."
    - AUTHORITY: "Only the utility company can disconnect power."
    - DISTANCE: "Patient is 30 ft away; bystanders are within 5 ft."
    - RESOURCE: "You are alone; no other units are on scene."
    - SEQUENCE: "You have already called dispatch."
    - TIME: "The patient has been down for 10 minutes."
    """

    FEASIBILITY = "feasibility"    # Something is/isn't physically possible
    AUTHORITY = "authority"        # Defines who can/cannot do something
    DISTANCE = "distance"          # Spatial constraint that affects priority
    RESOURCE = "resource"          # Available/unavailable resources
    SEQUENCE = "sequence"          # What has already happened
    TIME = "time"                  # Time-based constraint


class DecisionLockSpec(BaseModel):
    """A specific decision lock with its constraint text."""

    lock_type: DecisionLock
    constraint_text: str  # The exact text to include in stem
    eliminates_options: list[str]  # Which distractors this makes clearly wrong


# ============================================================================
# Distractor Failure Taxonomy
# ============================================================================


class DistractorFailureReason(str, Enum):
    """Why a distractor is wrong - each distractor needs a DIFFERENT reason.

    The "10 competent people" test: If you gave this question to 10 competent
    EMRs, ≥9 should pick the same answer. Each distractor must fail for a
    clear reason that's obvious from the stem.
    """

    # Safety failures
    APPROACHES_HAZARD = "approaches_hazard"        # Enters danger zone
    CREATES_NEW_HAZARD = "creates_new_hazard"      # Makes situation worse

    # Sequence/timing failures
    WRONG_SEQUENCE = "wrong_sequence"              # Right action, wrong order
    SKIPS_REQUIRED_STEP = "skips_required_step"    # Misses prerequisite
    PREMATURE_ACTION = "premature_action"          # Too early given constraints

    # Authority/scope failures
    EXCEEDS_SCOPE = "exceeds_scope"                # Beyond EMR authority
    REQUIRES_UNAVAILABLE = "requires_unavailable"  # Needs resource not present

    # Logic failures
    CONTRADICTS_STEM = "contradicts_stem"          # Ignores stated constraint
    TOO_GENERIC = "too_generic"                    # Delays specific required action
    ADDRESSES_WRONG_PROBLEM = "addresses_wrong_problem"  # Misidentifies issue


class DistractorTag(str, Enum):
    """High-level tags for distractor types (legacy, kept for compatibility)."""

    UNSAFE = "unsafe"
    WRONG_SEQUENCE = "wrong_sequence"
    OUT_OF_SCOPE = "out_of_scope"
    PARTIAL = "partial"
    COMMON_MISCONCEPTION = "common_misconception"


class SupportMapping(BaseModel):
    """Maps a quote to its support role."""

    quote_id: str
    role: str  # "primary_support", "secondary_support", "context"
    supports_claim: str  # What claim this quote supports


class PlannedDistractor(BaseModel):
    """A planned distractor with clear failure reason."""

    text: str
    tag: DistractorTag  # Legacy tag
    failure_reason: DistractorFailureReason  # Specific reason this is wrong
    why_wrong: str  # Human-readable explanation
    stem_fact_violated: str  # Which stem fact makes this wrong
    plausibility_source: str | None = None  # Why someone might choose this


class QuestionType(str, Enum):
    """Question types that may require decision locks."""

    FIRST_ACTION = "first_action"      # "What is the FIRST thing you should do?"
    BEST_ACTION = "best_action"        # "What is the BEST action?"
    MOST_IMPORTANT = "most_important"  # "What is the MOST important?"
    PRIORITY = "priority"              # "What should be your priority?"
    DEFINITION = "definition"          # "What is the definition of X?"
    FACTUAL = "factual"                # "Which statement is correct?"


# Question types that REQUIRE decision locks to avoid ambiguity
REQUIRES_DECISION_LOCK = {
    QuestionType.FIRST_ACTION,
    QuestionType.BEST_ACTION,
    QuestionType.MOST_IMPORTANT,
    QuestionType.PRIORITY,
}


class ItemPlan(BaseModel):
    """Complete plan for generating a question item."""

    item_id: str
    learning_target: str

    # Question classification
    question_type: QuestionType  # Type of question being asked

    # Scenario framing
    hazard_cues: list[str]  # Situational cues that set up the question
    scenario_context: str | None = None  # Optional scenario description

    # CRITICAL: Decision locks to eliminate ambiguity
    decision_locks: list[DecisionLockSpec]  # 1-2 locks for FIRST/BEST questions

    # Answer planning
    correct_action: str  # The conceptual correct action
    correct_option_text: str  # The exact text for the correct option
    why_correct: str  # Brief justification
    discriminating_factor: str  # What makes this THE answer (not just A answer)

    # Evidence mapping
    support_mapping: list[SupportMapping]  # Links quotes to claims

    # Distractor planning - each must fail for DIFFERENT reason
    distractors: list[PlannedDistractor]

    # Scope control
    scope_tags: list[str]  # What this item tests (e.g., "scene_safety", "hazard_assessment")
    out_of_scope_warning: str | None = None  # Topics to avoid

    # Stem constraints
    stem_type: str  # "scenario", "direct", "best_action", "definition"
    stem_constraints: list[str]  # E.g., "no double negatives", "avoid 'always'"


ITEM_PLANNING_SYSTEM_PROMPT = """You are an expert exam item writer who creates UNAMBIGUOUS questions.

Your job is to create a detailed PLAN for a multiple-choice question that passes the "10 competent people" test:
- If you gave this question to 10 competent EMRs, ≥9 would pick the same answer
- The correct answer is correct because of explicit stem facts, not "common sense"
- Every distractor is wrong for a DIFFERENT, clear reason obvious from the stem

KEY CONCEPT - DECISION LOCKS:
For FIRST/BEST/MOST questions, you MUST include 1-2 "decision locks" - explicit constraints
in the stem that eliminate ambiguity. Without these, multiple answers become defensible.

Decision lock types:
- FEASIBILITY: "The main breaker is inside the hazard area" (can't do X)
- AUTHORITY: "Only the utility company can disconnect power" (scope limit)
- DISTANCE: "Patient is 30 ft away; bystanders within 5 ft" (priority shift)
- RESOURCE: "You are alone; no other units on scene" (limits options)
- SEQUENCE: "You have already called dispatch" (eliminates that option)
- TIME: "The patient has been down for 10 minutes" (urgency factor)"""


ITEM_PLANNING_USER_PROMPT = """Create a detailed plan for this exam question.

LEARNING TARGET: {learning_target}
TOPIC: {topic_name} > {subtopic_name}

AVAILABLE ANCHOR QUOTES (must use at least one):
{formatted_quotes}

ALLOWED SCOPE TAGS: {scope_tags}

Generate a JSON plan with these fields:

{{
  "question_type": "first_action|best_action|most_important|priority|definition|factual",
  "hazard_cues": ["cue1", "cue2"],
  "scenario_context": "Brief scenario description",

  "decision_locks": [
    {{
      "lock_type": "feasibility|authority|distance|resource|sequence|time",
      "constraint_text": "Exact text to include in stem that eliminates ambiguity",
      "eliminates_options": ["which distractors this makes clearly wrong"]
    }}
  ],

  "correct_action": "The conceptual correct action",
  "correct_option_text": "Exact text for correct answer option",
  "why_correct": "1-2 sentences justifying based on quotes",
  "discriminating_factor": "The ONE stem fact that makes this THE answer, not just A answer",

  "support_mapping": [
    {{"quote_id": "Q1", "role": "primary_support", "supports_claim": "what it supports"}}
  ],

  "distractors": [
    {{
      "text": "Distractor option text",
      "tag": "unsafe|wrong_sequence|out_of_scope|partial|common_misconception",
      "failure_reason": "approaches_hazard|creates_new_hazard|wrong_sequence|skips_required_step|premature_action|exceeds_scope|requires_unavailable|contradicts_stem|too_generic|addresses_wrong_problem",
      "why_wrong": "Brief reason this is incorrect",
      "stem_fact_violated": "Which specific stem fact makes this wrong",
      "plausibility_source": "Why someone might choose this"
    }}
  ],

  "scope_tags": ["tag1", "tag2"],
  "out_of_scope_warning": "Topics to avoid",
  "stem_type": "scenario|direct|best_action|definition",
  "stem_constraints": ["constraint1", "constraint2"]
}}

CRITICAL REQUIREMENTS:
1. For FIRST/BEST/MOST/PRIORITY questions, you MUST include 1-2 decision_locks
2. Each distractor MUST have a DIFFERENT failure_reason (no duplicates!)
3. Each distractor MUST reference a specific stem_fact_violated
4. The discriminating_factor MUST be something explicitly stated in the stem
5. Correct answer MUST be directly supported by at least one quote

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


def plan_item(
    engine: "GenerationEngine",
    item: "CapsuleItem",
    quotes: list["ExtractedQuote"],
    topic_name: str,
    subtopic_name: str,
    allowed_scope_tags: list[str],
) -> ItemPlan:
    """Generate a structured plan for an item before generation.

    This is Phase B.75 - after quote extraction, before generation.

    Args:
        engine: Generation engine for LLM calls
        item: The CapsuleItem to plan
        quotes: Extracted quotes to use as anchors
        topic_name: Parent topic name
        subtopic_name: Parent subtopic name
        allowed_scope_tags: List of allowed scope tags

    Returns:
        ItemPlan with structured planning data
    """
    from course_builder.engine import GenerationConfig
    from course_builder.pipeline.content.quote_extraction import format_quotes_for_prompt

    formatted_quotes = format_quotes_for_prompt(quotes)
    scope_tags_str = ", ".join(allowed_scope_tags)

    prompt = ITEM_PLANNING_USER_PROMPT.format(
        learning_target=item.learning_target,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        formatted_quotes=formatted_quotes,
        scope_tags=scope_tags_str,
    )

    config = GenerationConfig(system_prompt=ITEM_PLANNING_SYSTEM_PROMPT)
    result = engine.generate(prompt, config=config)

    try:
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)

        # Parse question type
        try:
            question_type = QuestionType(data.get("question_type", "factual"))
        except ValueError:
            question_type = QuestionType.FACTUAL

        # Parse decision locks
        decision_locks = []
        for dl in data.get("decision_locks", []):
            try:
                lock_type = DecisionLock(dl.get("lock_type", "feasibility"))
            except ValueError:
                lock_type = DecisionLock.FEASIBILITY

            decision_locks.append(DecisionLockSpec(
                lock_type=lock_type,
                constraint_text=dl.get("constraint_text", ""),
                eliminates_options=dl.get("eliminates_options", []),
            ))

        # Parse support mappings
        support_mappings = []
        for sm in data.get("support_mapping", []):
            support_mappings.append(SupportMapping(
                quote_id=sm.get("quote_id", ""),
                role=sm.get("role", "primary_support"),
                supports_claim=sm.get("supports_claim", ""),
            ))

        # Parse distractors with failure reasons
        distractors = []
        for d in data.get("distractors", []):
            try:
                tag = DistractorTag(d.get("tag", "partial"))
            except ValueError:
                tag = DistractorTag.PARTIAL

            try:
                failure_reason = DistractorFailureReason(d.get("failure_reason", "too_generic"))
            except ValueError:
                failure_reason = DistractorFailureReason.TOO_GENERIC

            distractors.append(PlannedDistractor(
                text=d.get("text", ""),
                tag=tag,
                failure_reason=failure_reason,
                why_wrong=d.get("why_wrong", ""),
                stem_fact_violated=d.get("stem_fact_violated", ""),
                plausibility_source=d.get("plausibility_source"),
            ))

        return ItemPlan(
            item_id=item.item_id,
            learning_target=item.learning_target,
            question_type=question_type,
            hazard_cues=data.get("hazard_cues", []),
            scenario_context=data.get("scenario_context"),
            decision_locks=decision_locks,
            correct_action=data.get("correct_action", ""),
            correct_option_text=data.get("correct_option_text", ""),
            why_correct=data.get("why_correct", ""),
            discriminating_factor=data.get("discriminating_factor", ""),
            support_mapping=support_mappings,
            distractors=distractors,
            scope_tags=data.get("scope_tags", []),
            out_of_scope_warning=data.get("out_of_scope_warning"),
            stem_type=data.get("stem_type", "direct"),
            stem_constraints=data.get("stem_constraints", []),
        )

    except (json.JSONDecodeError, KeyError):
        # Fallback to basic plan
        return _generate_fallback_plan(item, quotes, allowed_scope_tags)


def _generate_fallback_plan(
    item: "CapsuleItem",
    quotes: list["ExtractedQuote"],
    allowed_scope_tags: list[str],
) -> ItemPlan:
    """Generate a basic plan without LLM (fallback).

    Used when LLM planning fails.
    """
    # Use first quote as primary support
    primary_quote = quotes[0] if quotes else None

    support_mappings = []
    if primary_quote:
        support_mappings.append(SupportMapping(
            quote_id=primary_quote.quote_id,
            role="primary_support",
            supports_claim="correct answer",
        ))

    # Generate generic distractors with different failure reasons
    distractors = [
        PlannedDistractor(
            text="[Distractor 1 - to be generated]",
            tag=DistractorTag.UNSAFE,
            failure_reason=DistractorFailureReason.APPROACHES_HAZARD,
            why_wrong="Approaches the hazard zone",
            stem_fact_violated="[hazard constraint]",
        ),
        PlannedDistractor(
            text="[Distractor 2 - to be generated]",
            tag=DistractorTag.WRONG_SEQUENCE,
            failure_reason=DistractorFailureReason.WRONG_SEQUENCE,
            why_wrong="Correct action but wrong order",
            stem_fact_violated="[sequence constraint]",
        ),
        PlannedDistractor(
            text="[Distractor 3 - to be generated]",
            tag=DistractorTag.OUT_OF_SCOPE,
            failure_reason=DistractorFailureReason.EXCEEDS_SCOPE,
            why_wrong="Beyond EMR scope of practice",
            stem_fact_violated="[scope constraint]",
        ),
    ]

    return ItemPlan(
        item_id=item.item_id,
        learning_target=item.learning_target,
        question_type=QuestionType.FACTUAL,
        hazard_cues=[],
        scenario_context=None,
        decision_locks=[],  # Fallback has no locks - will fail validation for FIRST/BEST
        correct_action=item.learning_target,
        correct_option_text="[To be generated based on quotes]",
        why_correct="Based on source material",
        discriminating_factor="[To be determined]",
        support_mapping=support_mappings,
        distractors=distractors,
        scope_tags=allowed_scope_tags[:2] if allowed_scope_tags else [],
        out_of_scope_warning=None,
        stem_type="direct",
        stem_constraints=["No double negatives", "Clear and concise"],
    )


def validate_plan(plan: ItemPlan, quotes: list["ExtractedQuote"]) -> list[str]:
    """Validate a plan against requirements.

    Returns list of validation issues (empty if valid).
    """
    issues = []

    # Check 1: At least one support mapping
    if not plan.support_mapping:
        issues.append("No support mapping defined")

    # Check 2: Support mappings reference valid quotes
    quote_ids = {q.quote_id for q in quotes}
    for sm in plan.support_mapping:
        if sm.quote_id not in quote_ids:
            issues.append(f"Support mapping references invalid quote_id: {sm.quote_id}")

    # Check 3: Exactly 3 distractors
    if len(plan.distractors) != 3:
        issues.append(f"Expected 3 distractors, got {len(plan.distractors)}")

    # Check 4: CRITICAL - Decision locks required for FIRST/BEST/MOST questions
    if plan.question_type in REQUIRES_DECISION_LOCK:
        if not plan.decision_locks:
            issues.append(
                f"AMBIGUITY RISK: {plan.question_type.value} question requires at least "
                f"1 decision lock to eliminate ambiguity"
            )
        elif len(plan.decision_locks) < 1:
            issues.append(
                f"AMBIGUITY RISK: {plan.question_type.value} question should have "
                f"1-2 decision locks"
            )

    # Check 5: CRITICAL - Distractors must have DIFFERENT failure reasons
    failure_reasons = [d.failure_reason for d in plan.distractors]
    unique_reasons = set(failure_reasons)
    if len(unique_reasons) < len(failure_reasons):
        duplicate_reasons = [r for r in failure_reasons if failure_reasons.count(r) > 1]
        issues.append(
            f"DISTRACTOR ASYMMETRY: Multiple distractors have same failure reason: "
            f"{set(duplicate_reasons)}. Each distractor must fail for a DIFFERENT reason."
        )

    # Check 6: Each distractor must reference a stem fact
    for i, d in enumerate(plan.distractors):
        if not d.stem_fact_violated or d.stem_fact_violated.startswith("["):
            issues.append(
                f"Distractor {i+1} missing stem_fact_violated - "
                f"each distractor must be wrong due to an explicit stem fact"
            )

    # Check 7: Discriminating factor must be specified
    if not plan.discriminating_factor or plan.discriminating_factor.startswith("["):
        issues.append(
            "Missing discriminating_factor - must specify what makes this THE answer"
        )

    # Check 8: Correct option text is not empty
    if not plan.correct_option_text or plan.correct_option_text.startswith("["):
        issues.append("Correct option text is missing or placeholder")

    # Check 9: Scope tags are defined
    if not plan.scope_tags:
        issues.append("No scope tags defined")

    return issues


def validate_plan_ambiguity(plan: ItemPlan) -> list[str]:
    """Additional validation focused specifically on ambiguity issues.

    This is a stricter check that should be run before generation.
    """
    issues = []

    # Check 1: Decision lock coverage
    if plan.question_type in REQUIRES_DECISION_LOCK:
        # Each distractor should be eliminated by at least one decision lock
        distractors_covered = set()
        for lock in plan.decision_locks:
            for eliminated in lock.eliminates_options:
                distractors_covered.add(eliminated.lower())

        for i, d in enumerate(plan.distractors):
            if d.text.lower() not in distractors_covered:
                # Check if the distractor text partially matches
                partially_covered = any(
                    d.text.lower() in covered or covered in d.text.lower()
                    for covered in distractors_covered
                )
                if not partially_covered:
                    issues.append(
                        f"Distractor {i+1} not explicitly eliminated by any decision lock"
                    )

    # Check 2: Plausibility-wrongness balance
    # Each distractor should have both why it's tempting AND why it's wrong
    for i, d in enumerate(plan.distractors):
        if not d.plausibility_source:
            issues.append(
                f"Distractor {i+1} missing plausibility_source - "
                f"should explain why someone might choose this"
            )

    # Check 3: Discriminating factor should not be "common sense"
    common_sense_phrases = [
        "common sense", "obviously", "everyone knows", "general knowledge",
        "standard practice", "always", "never", "typical"
    ]
    if plan.discriminating_factor:
        df_lower = plan.discriminating_factor.lower()
        for phrase in common_sense_phrases:
            if phrase in df_lower:
                issues.append(
                    f"Discriminating factor relies on '{phrase}' - "
                    f"should be an explicit stem fact, not common sense"
                )
                break

    return issues


def format_plan_for_generation(plan: ItemPlan, quotes: list["ExtractedQuote"]) -> str:
    """Format a plan for inclusion in generation prompt.

    This provides structured guidance to the generation LLM.
    """
    lines = ["=== ITEM PLAN (follow this structure) ===", ""]

    # Question type
    lines.append(f"QUESTION TYPE: {plan.question_type.value}")
    lines.append("")

    # Scenario
    if plan.hazard_cues:
        lines.append(f"HAZARD CUES: {', '.join(plan.hazard_cues)}")
    if plan.scenario_context:
        lines.append(f"SCENARIO: {plan.scenario_context}")
    lines.append("")

    # CRITICAL: Decision locks
    if plan.decision_locks:
        lines.append("=== DECISION LOCKS (MUST include in stem) ===")
        for i, lock in enumerate(plan.decision_locks, 1):
            lines.append(f"  Lock {i} ({lock.lock_type.value}):")
            lines.append(f"    Text: \"{lock.constraint_text}\"")
            lines.append(f"    Eliminates: {', '.join(lock.eliminates_options)}")
        lines.append("")

    # Correct answer
    lines.append(f"CORRECT ACTION: {plan.correct_action}")
    lines.append(f"CORRECT OPTION TEXT: \"{plan.correct_option_text}\"")
    lines.append(f"JUSTIFICATION: {plan.why_correct}")
    lines.append(f"DISCRIMINATING FACTOR: {plan.discriminating_factor}")
    lines.append("")

    # Evidence mapping
    lines.append("EVIDENCE MAPPING:")
    for sm in plan.support_mapping:
        quote = next((q for q in quotes if q.quote_id == sm.quote_id), None)
        if quote:
            lines.append(f"  [{sm.quote_id}] ({sm.role}): \"{quote.text}\"")
            lines.append(f"    -> Supports: {sm.supports_claim}")
    lines.append("")

    # Distractors with failure reasons
    lines.append("PLANNED DISTRACTORS (each fails for DIFFERENT reason):")
    for i, d in enumerate(plan.distractors, 1):
        lines.append(f"  {i}. \"{d.text}\"")
        lines.append(f"     Failure reason: {d.failure_reason.value}")
        lines.append(f"     Stem fact violated: {d.stem_fact_violated}")
        lines.append(f"     Why tempting: {d.plausibility_source or 'N/A'}")
    lines.append("")

    # Constraints
    lines.append(f"STEM TYPE: {plan.stem_type}")
    if plan.stem_constraints:
        lines.append(f"STEM CONSTRAINTS: {', '.join(plan.stem_constraints)}")
    if plan.out_of_scope_warning:
        lines.append(f"AVOID: {plan.out_of_scope_warning}")

    return "\n".join(lines)


def extract_scope_tags_from_plan(plan: ItemPlan) -> list[str]:
    """Extract scope tags for quality gate checking."""
    return plan.scope_tags


def get_required_quote_ids(plan: ItemPlan) -> list[str]:
    """Get quote IDs that must appear in the generated content."""
    return [sm.quote_id for sm in plan.support_mapping if sm.role == "primary_support"]
