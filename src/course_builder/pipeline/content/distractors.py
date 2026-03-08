"""Controlled distractor generation for difficulty-appropriate distractors.

This module generates distractors (wrong answers) that are:
1. Appropriate for the difficulty level
2. Based on specific wrongness patterns
3. Plausible but clearly distinguishable from correct answer

Wrongness Patterns:
- WRONG_SEQUENCE: Right action, wrong order
- WRONG_CONTEXT: Right action, wrong situation
- INCOMPLETE: Partially correct
- COMMON_MISCONCEPTION: Plausible but wrong

Difficulty Mapping:
- Beginner: 2+ clearly wrong (OPPOSITE, MISCONCEPTION)
- Intermediate: All plausible, wrong context
- Advanced: All valid actions, wrong sequence only
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from course_builder.engine import GenerationConfig, GenerationEngine


class WrongnessPattern(str, Enum):
    """Pattern describing why a distractor is wrong."""

    # Makes the distractor clearly wrong
    OPPOSITE = "opposite"  # Does the opposite of what's needed
    VIOLATES_PROTOCOL = "violates_protocol"  # Breaks a basic safety rule
    COMMON_MISCONCEPTION = "common_misconception"  # Common but wrong belief

    # Makes the distractor subtly wrong
    WRONG_SEQUENCE = "wrong_sequence"  # Right action, wrong order
    WRONG_CONTEXT = "wrong_context"  # Right action, wrong situation
    INCOMPLETE = "incomplete"  # Partially correct but missing key element
    EXCESSIVE = "excessive"  # Correct direction but overdone
    DELAYED = "delayed"  # Right action but should be done later


@dataclass
class DistractorSpec:
    """Specification for a distractor."""

    text: str  # The distractor text
    pattern: WrongnessPattern  # Why it's wrong
    explanation: str  # Why it's wrong (for explanation)
    plausibility: float = 0.5  # 0-1, how plausible (higher = more tempting)


@dataclass
class DistractorSet:
    """Complete set of distractors for a question."""

    distractors: list[DistractorSpec]
    difficulty: str  # Difficulty these were generated for
    correct_answer: str  # The correct answer
    balance_check: str = ""  # Notes on distractor balance


# Difficulty-specific requirements
DIFFICULTY_DISTRACTOR_REQUIREMENTS = {
    "beginner": {
        "min_clearly_wrong": 2,  # At least 2 obviously wrong
        "allowed_patterns": [
            WrongnessPattern.OPPOSITE,
            WrongnessPattern.VIOLATES_PROTOCOL,
            WrongnessPattern.COMMON_MISCONCEPTION,
            WrongnessPattern.INCOMPLETE,
        ],
        "forbidden_patterns": [
            WrongnessPattern.WRONG_SEQUENCE,  # Too subtle for beginner
        ],
        "description": "Beginner: 2+ clearly wrong distractors, no sequence-only differences",
    },
    "intermediate": {
        "min_clearly_wrong": 0,  # All can be plausible
        "allowed_patterns": [
            WrongnessPattern.WRONG_CONTEXT,
            WrongnessPattern.WRONG_SEQUENCE,
            WrongnessPattern.INCOMPLETE,
            WrongnessPattern.EXCESSIVE,
            WrongnessPattern.DELAYED,
        ],
        "forbidden_patterns": [],
        "description": "Intermediate: All plausible actions, wrong timing/context",
    },
    "advanced": {
        "min_clearly_wrong": 0,  # All must be plausible
        "allowed_patterns": [
            WrongnessPattern.WRONG_SEQUENCE,
        ],
        "forbidden_patterns": [
            WrongnessPattern.OPPOSITE,
            WrongnessPattern.VIOLATES_PROTOCOL,
        ],
        "description": "Advanced: All 4 valid actions, differ in sequence only",
    },
}


DISTRACTOR_GENERATION_PROMPT = """Generate distractors (wrong answers) for this EMS exam question.

QUESTION STEM:
{stem}

CORRECT ANSWER:
{correct_answer}

DIFFICULTY: {difficulty}
{difficulty_requirements}

SOURCE CONTEXT:
{source_context}

Generate exactly {num_distractors} distractors following these patterns:
{pattern_instructions}

Each distractor must:
1. Be a plausible EMS action (not cartoonish)
2. Be clearly distinguishable from the correct answer
3. Have a specific reason for being wrong

Return a JSON object:
{{
  "distractors": [
    {{
      "text": "The distractor text",
      "pattern": "wrong_sequence|wrong_context|incomplete|...",
      "explanation": "Why this is wrong",
      "plausibility": 0.7
    }}
  ]
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


def get_pattern_instructions(difficulty: str) -> str:
    """Get pattern instructions for a difficulty level."""
    reqs = DIFFICULTY_DISTRACTOR_REQUIREMENTS.get(difficulty, DIFFICULTY_DISTRACTOR_REQUIREMENTS["intermediate"])

    instructions = []

    if difficulty == "beginner":
        instructions.append("- At least 2 distractors should be CLEARLY wrong (violate basic protocol or common misconception)")
        instructions.append("- Do NOT use sequence-only differences (too subtle)")
        instructions.append("- Patterns: OPPOSITE, VIOLATES_PROTOCOL, COMMON_MISCONCEPTION, INCOMPLETE")

    elif difficulty == "intermediate":
        instructions.append("- All distractors should be plausible EMS actions")
        instructions.append("- Wrong due to: wrong timing, wrong context, or incomplete")
        instructions.append("- Patterns: WRONG_CONTEXT, WRONG_SEQUENCE, INCOMPLETE, EXCESSIVE, DELAYED")

    elif difficulty == "advanced":
        instructions.append("- ALL 4 options (including correct) must be valid EMS actions")
        instructions.append("- Distractors differ from correct answer in SEQUENCE only")
        instructions.append("- Do NOT use clearly wrong options")
        instructions.append("- Pattern: WRONG_SEQUENCE only")

    return "\n".join(instructions)


def generate_controlled_distractors(
    engine: "GenerationEngine",
    stem: str,
    correct_answer: str,
    difficulty: str,
    source_context: str,
    num_distractors: int = 3,
) -> DistractorSet:
    """Generate difficulty-appropriate distractors.

    Args:
        engine: Generation engine for LLM calls
        stem: Question stem
        correct_answer: The correct answer
        difficulty: "beginner", "intermediate", or "advanced"
        source_context: Source material context
        num_distractors: Number of distractors to generate

    Returns:
        DistractorSet with generated distractors
    """
    from course_builder.engine import GenerationConfig

    reqs = DIFFICULTY_DISTRACTOR_REQUIREMENTS.get(difficulty, DIFFICULTY_DISTRACTOR_REQUIREMENTS["intermediate"])

    prompt = DISTRACTOR_GENERATION_PROMPT.format(
        stem=stem,
        correct_answer=correct_answer,
        difficulty=difficulty,
        difficulty_requirements=reqs["description"],
        source_context=source_context[:1000],  # Limit context
        num_distractors=num_distractors,
        pattern_instructions=get_pattern_instructions(difficulty),
    )

    config = GenerationConfig(temperature=0.7)  # Some creativity
    result = engine.generate(prompt, config=config)

    try:
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)

        distractors = []
        for d in data.get("distractors", []):
            pattern_str = d.get("pattern", "incomplete")
            try:
                pattern = WrongnessPattern(pattern_str)
            except ValueError:
                pattern = WrongnessPattern.INCOMPLETE

            distractors.append(DistractorSpec(
                text=d.get("text", ""),
                pattern=pattern,
                explanation=d.get("explanation", ""),
                plausibility=d.get("plausibility", 0.5),
            ))

        # Validate against difficulty requirements
        balance_check = validate_distractor_balance(distractors, difficulty)

        return DistractorSet(
            distractors=distractors,
            difficulty=difficulty,
            correct_answer=correct_answer,
            balance_check=balance_check,
        )

    except (json.JSONDecodeError, KeyError):
        # Return empty set on failure
        return DistractorSet(
            distractors=[],
            difficulty=difficulty,
            correct_answer=correct_answer,
            balance_check="Generation failed",
        )


def validate_distractor_balance(
    distractors: list[DistractorSpec],
    difficulty: str,
) -> str:
    """Validate that distractors meet difficulty requirements.

    Args:
        distractors: Generated distractors
        difficulty: Difficulty level

    Returns:
        Validation message (empty if valid)
    """
    reqs = DIFFICULTY_DISTRACTOR_REQUIREMENTS.get(difficulty, DIFFICULTY_DISTRACTOR_REQUIREMENTS["intermediate"])

    issues = []

    # Check minimum clearly wrong count
    clearly_wrong_patterns = {
        WrongnessPattern.OPPOSITE,
        WrongnessPattern.VIOLATES_PROTOCOL,
    }
    clearly_wrong_count = sum(
        1 for d in distractors if d.pattern in clearly_wrong_patterns
    )

    min_required = reqs.get("min_clearly_wrong", 0)
    if clearly_wrong_count < min_required:
        issues.append(f"Need {min_required} clearly wrong, have {clearly_wrong_count}")

    # Check for forbidden patterns
    forbidden = set(reqs.get("forbidden_patterns", []))
    for d in distractors:
        if d.pattern in forbidden:
            issues.append(f"Forbidden pattern used: {d.pattern.value}")

    # Check for advanced: all should be WRONG_SEQUENCE
    if difficulty == "advanced":
        non_sequence = sum(
            1 for d in distractors if d.pattern != WrongnessPattern.WRONG_SEQUENCE
        )
        if non_sequence > 0:
            issues.append(f"Advanced has {non_sequence} non-sequence distractors")

    return "; ".join(issues) if issues else ""


def suggest_distractor_patterns(
    correct_answer: str,
    difficulty: str,
    stem: str,
) -> list[tuple[WrongnessPattern, str]]:
    """Suggest distractor patterns based on correct answer and difficulty.

    Returns pattern suggestions without generating full distractors.

    Args:
        correct_answer: The correct answer
        difficulty: Difficulty level
        stem: Question stem for context

    Returns:
        List of (pattern, hint) tuples
    """
    suggestions = []
    reqs = DIFFICULTY_DISTRACTOR_REQUIREMENTS.get(difficulty, DIFFICULTY_DISTRACTOR_REQUIREMENTS["intermediate"])
    allowed = set(reqs.get("allowed_patterns", []))

    # Analyze correct answer for suggestion opportunities
    correct_lower = correct_answer.lower()
    stem_lower = stem.lower()

    # Sequence-related correct answers
    sequence_words = ["first", "before", "then", "after", "priority", "initial"]
    has_sequence = any(w in correct_lower or w in stem_lower for w in sequence_words)

    if has_sequence and WrongnessPattern.WRONG_SEQUENCE in allowed:
        suggestions.append((
            WrongnessPattern.WRONG_SEQUENCE,
            "Suggest doing a different valid action first",
        ))

    # Action-based correct answers
    action_verbs = ["assess", "establish", "position", "secure", "administer"]
    for verb in action_verbs:
        if verb in correct_lower:
            if WrongnessPattern.INCOMPLETE in allowed:
                suggestions.append((
                    WrongnessPattern.INCOMPLETE,
                    f"Do '{verb}' but miss a key component",
                ))
            if WrongnessPattern.WRONG_CONTEXT in allowed:
                suggestions.append((
                    WrongnessPattern.WRONG_CONTEXT,
                    f"'{verb}' but in wrong situation",
                ))

    # Safety-related correct answers
    safety_words = ["protect", "safety", "hazard", "danger", "secure"]
    has_safety = any(w in correct_lower or w in stem_lower for w in safety_words)

    if has_safety:
        if WrongnessPattern.VIOLATES_PROTOCOL in allowed:
            suggestions.append((
                WrongnessPattern.VIOLATES_PROTOCOL,
                "Skip safety step to provide faster care",
            ))
        if WrongnessPattern.COMMON_MISCONCEPTION in allowed:
            suggestions.append((
                WrongnessPattern.COMMON_MISCONCEPTION,
                "Common but wrong approach to this safety scenario",
            ))

    return suggestions[:4]  # Limit to 4 suggestions


def improve_existing_distractors(
    existing_options: list[str],
    correct_index: int,
    difficulty: str,
    stem: str,
) -> list[str]:
    """Analyze and suggest improvements for existing distractors.

    Does not require LLM call. Uses heuristics to identify issues.

    Args:
        existing_options: Current answer options
        correct_index: Index of correct answer
        difficulty: Difficulty level
        stem: Question stem

    Returns:
        List of improvement suggestions
    """
    improvements = []
    reqs = DIFFICULTY_DISTRACTOR_REQUIREMENTS.get(difficulty, DIFFICULTY_DISTRACTOR_REQUIREMENTS["intermediate"])

    correct = existing_options[correct_index] if 0 <= correct_index < len(existing_options) else ""
    distractors = [opt for i, opt in enumerate(existing_options) if i != correct_index]

    # Check for cartoonish distractors
    cartoonish_patterns = [
        r"do nothing",
        r"ignore",
        r"refuse",
        r"leave.*alone",
        r"walk away",
        r"cover.*with.*blanket",
        r"move.*wires?.*with.*stick",
    ]

    clearly_wrong = 0
    for d in distractors:
        d_lower = d.lower()
        for pattern in cartoonish_patterns:
            if re.search(pattern, d_lower):
                clearly_wrong += 1
                if difficulty == "advanced":
                    improvements.append(f"Remove cartoonish distractor: '{d[:50]}...'")
                break

    # For beginner, need at least 2 clearly wrong
    if difficulty == "beginner" and clearly_wrong < 2:
        improvements.append("Beginner needs 2+ clearly wrong distractors")

    # For advanced, should have no clearly wrong
    if difficulty == "advanced" and clearly_wrong > 0:
        improvements.append("Advanced should not have clearly wrong distractors")

    # Check for similar distractors
    for i, d1 in enumerate(distractors):
        for j, d2 in enumerate(distractors):
            if i >= j:
                continue
            words1 = set(d1.lower().split())
            words2 = set(d2.lower().split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            if overlap > 0.7:
                improvements.append(f"Distractors too similar: '{d1[:30]}...' and '{d2[:30]}...'")

    return improvements
