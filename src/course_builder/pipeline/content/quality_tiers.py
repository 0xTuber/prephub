"""3-Tier Quality Gate System for question generation.

This module implements a structured approach to quality validation:

Tier 1 (Hard Blockers): Must pass or item is rejected
- Verbatim quote presence (mechanical verification)
- Stem-answer internal consistency
- Scope legality (tests correct content)
- Single-best-answer logic

Tier 2 (Repair-First): Can fail but triggers repair loop
- Distractor quality (plausible, no correct alternatives)
- Difficulty alignment
- Decision grounding (explanation supports answer)

Tier 3 (Soft Flags): Logged but don't block
- Novelty (doesn't duplicate existing)
- Verbosity (explanation length)
- Style (tone, formatting)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

if TYPE_CHECKING:
    from course_builder.pipeline.content.quote_extraction import (
        ExtractedQuote,
        GenerationQuoteVerification,
    )


class GateTier(str, Enum):
    """Quality gate tiers."""

    TIER_1_HARD = "tier_1_hard"
    TIER_2_REPAIR = "tier_2_repair"
    TIER_3_SOFT = "tier_3_soft"


class GateResult(str, Enum):
    """Result of a quality gate check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"  # For soft flags


class GateCheckResult(BaseModel):
    """Result of a single gate check."""

    gate_name: str
    tier: GateTier
    result: GateResult
    message: str
    details: dict[str, Any] = {}
    repair_hint: str | None = None  # For Tier 2 failures


class QualityGateReport(BaseModel):
    """Complete quality gate report for an item."""

    item_id: str
    tier_1_passed: bool
    tier_2_passed: bool
    tier_3_flags: list[str]
    all_checks: list[GateCheckResult]
    overall_pass: bool
    requires_repair: bool
    repair_targets: list[str]  # Which components need repair


# ============================================================================
# Tier 1: Hard Blockers
# ============================================================================


def check_verbatim_quote_presence(
    generated_text: str,
    quote_verification: "GenerationQuoteVerification",
    min_required: int = 1,
) -> GateCheckResult:
    """Tier 1: Check that required quotes appear verbatim in generated text.

    This is a MECHANICAL check - no LLM involved.
    """
    if quote_verification.all_required_found:
        return GateCheckResult(
            gate_name="verbatim_quote_presence",
            tier=GateTier.TIER_1_HARD,
            result=GateResult.PASS,
            message=f"All {len(quote_verification.required_quotes)} required quotes found",
            details={
                "exact_matches": quote_verification.exact_match_count,
                "fuzzy_matches": quote_verification.fuzzy_match_count,
            },
        )

    missing = quote_verification.missing_quote_ids
    return GateCheckResult(
        gate_name="verbatim_quote_presence",
        tier=GateTier.TIER_1_HARD,
        result=GateResult.FAIL,
        message=f"Missing {len(missing)} required quotes: {missing}",
        details={
            "missing_quote_ids": missing,
            "exact_matches": quote_verification.exact_match_count,
        },
    )


def check_stem_answer_consistency(
    stem: str,
    correct_answer: str,
    distractors: list[str],
) -> GateCheckResult:
    """Tier 1: Check that stem doesn't leak the correct answer.

    Detects:
    - Grammatical cues (a/an mismatch)
    - Word overlap between stem and only the correct answer
    - Negation inconsistency
    """
    issues = []

    # Check 1: Grammatical article cues
    stem_lower = stem.lower()
    if stem_lower.rstrip().endswith(" a") or stem_lower.rstrip().endswith(" an"):
        article = "an" if stem_lower.rstrip().endswith(" an") else "a"
        correct_starts_vowel = correct_answer.strip()[0].lower() in "aeiou"

        # Check if article matches correct but not distractors
        correct_match = (article == "an") == correct_starts_vowel
        distractor_matches = [
            (article == "an") == (d.strip()[0].lower() in "aeiou")
            for d in distractors if d.strip()
        ]

        if correct_match and not any(distractor_matches):
            issues.append(f"Article '{article}' only matches correct answer")

    # Check 2: Unique word overlap with correct answer
    stem_words = set(re.findall(r"\b\w{4,}\b", stem_lower))
    correct_words = set(re.findall(r"\b\w{4,}\b", correct_answer.lower()))

    for distractor in distractors:
        dist_words = set(re.findall(r"\b\w{4,}\b", distractor.lower()))
        correct_only = correct_words - dist_words
        overlap = stem_words & correct_only
        if overlap and len(overlap) > 1:
            issues.append(f"Stem shares unique words with correct: {overlap}")

    # Check 3: Double negatives or negation traps
    negation_patterns = [r"\bnot\b", r"\bexcept\b", r"\bnever\b", r"\bwithout\b"]
    stem_negations = sum(1 for p in negation_patterns if re.search(p, stem_lower))
    if stem_negations > 1:
        issues.append("Multiple negations in stem (confusing)")

    if issues:
        return GateCheckResult(
            gate_name="stem_answer_consistency",
            tier=GateTier.TIER_1_HARD,
            result=GateResult.FAIL,
            message=f"Stem leaks answer: {'; '.join(issues)}",
            details={"issues": issues},
        )

    return GateCheckResult(
        gate_name="stem_answer_consistency",
        tier=GateTier.TIER_1_HARD,
        result=GateResult.PASS,
        message="Stem does not leak the correct answer",
    )


def check_scope_legality(
    item_content: dict[str, Any],
    allowed_scope_tags: list[str],
) -> GateCheckResult:
    """Tier 1: Check that item tests content within allowed scope.

    The item_content should have 'scope_tags' field from the PLAN phase.
    """
    item_tags = item_content.get("scope_tags", [])

    if not item_tags:
        return GateCheckResult(
            gate_name="scope_legality",
            tier=GateTier.TIER_1_HARD,
            result=GateResult.FAIL,
            message="No scope tags defined for item",
            details={"allowed_scope": allowed_scope_tags},
        )

    # Check if all item tags are within allowed scope
    out_of_scope = [tag for tag in item_tags if tag not in allowed_scope_tags]

    if out_of_scope:
        return GateCheckResult(
            gate_name="scope_legality",
            tier=GateTier.TIER_1_HARD,
            result=GateResult.FAIL,
            message=f"Item tests out-of-scope content: {out_of_scope}",
            details={
                "out_of_scope_tags": out_of_scope,
                "item_tags": item_tags,
                "allowed_scope": allowed_scope_tags,
            },
        )

    return GateCheckResult(
        gate_name="scope_legality",
        tier=GateTier.TIER_1_HARD,
        result=GateResult.PASS,
        message="Item is within allowed scope",
        details={"scope_tags": item_tags},
    )


def check_single_best_answer(
    correct_answer: str,
    distractors: list[str],
    explanation: str,
) -> GateCheckResult:
    """Tier 1: Check that exactly one answer is defensibly correct.

    This is a structural check looking for:
    - Identical or near-identical options
    - Contradictory claims in explanation
    - "Also correct" language in explanation
    """
    issues = []
    explanation_lower = explanation.lower()

    # Check 1: Near-duplicate options
    all_options = [correct_answer] + distractors
    for i, opt1 in enumerate(all_options):
        for opt2 in all_options[i + 1:]:
            # Normalize and compare
            norm1 = re.sub(r"[^\w\s]", "", opt1.lower()).strip()
            norm2 = re.sub(r"[^\w\s]", "", opt2.lower()).strip()
            if norm1 == norm2:
                issues.append(f"Duplicate options: '{opt1}' and '{opt2}'")

    # Check 2: "Also correct" language in explanation
    also_correct_patterns = [
        r"\balso (?:correct|valid|acceptable)\b",
        r"\bcould also be\b",
        r"\banother valid\b",
        r"\beither .+ or .+ (?:is|are) correct\b",
    ]
    for pattern in also_correct_patterns:
        if re.search(pattern, explanation_lower):
            issues.append(f"Explanation suggests multiple correct answers")
            break

    # Check 3: Contradictions in explanation
    contradiction_pairs = [
        (r"\bshould\b", r"\bshould not\b"),
        (r"\bmust\b", r"\bmust not\b"),
        (r"\balways\b", r"\bnever\b"),
    ]
    for affirm, negate in contradiction_pairs:
        if re.search(affirm, explanation_lower) and re.search(negate, explanation_lower):
            # Both present - might be contradiction
            issues.append("Explanation contains potentially contradictory statements")
            break

    if issues:
        return GateCheckResult(
            gate_name="single_best_answer",
            tier=GateTier.TIER_1_HARD,
            result=GateResult.FAIL,
            message=f"Single-best-answer violated: {'; '.join(issues)}",
            details={"issues": issues},
        )

    return GateCheckResult(
        gate_name="single_best_answer",
        tier=GateTier.TIER_1_HARD,
        result=GateResult.PASS,
        message="Exactly one defensible correct answer",
    )


def check_learning_target_alignment(
    learning_target: str,
    stem: str,
    correct_answer: str,
    explanation: str,
) -> GateCheckResult:
    """Tier 1: Check that the item actually tests the stated learning target.

    Detects "learning target drift" where:
    - The question asks about a different concept
    - Key learning target terms are absent from stem/answer
    - The explanation justifies a different topic

    Args:
        learning_target: The learning target this item should test
        stem: The question stem
        correct_answer: The correct answer text
        explanation: The explanation text

    Returns:
        GateCheckResult indicating alignment status
    """
    issues = []
    lt_lower = learning_target.lower()
    stem_lower = stem.lower()
    answer_lower = correct_answer.lower()
    explanation_lower = explanation.lower()

    # Extract key concepts from learning target (words ≥4 chars, excluding stopwords)
    stopwords = {
        "that", "this", "with", "from", "have", "been", "were", "they", "will",
        "would", "could", "should", "about", "their", "when", "what", "which",
        "more", "some", "such", "into", "than", "most", "also", "just", "only",
        "each", "other", "after", "before", "during", "between", "through",
        "being", "both", "same", "very", "must", "can", "able", "need", "does",
    }
    lt_words = set(re.findall(r"\b\w{4,}\b", lt_lower)) - stopwords

    if len(lt_words) < 2:
        # Learning target too short to analyze
        return GateCheckResult(
            gate_name="learning_target_alignment",
            tier=GateTier.TIER_1_HARD,
            result=GateResult.PASS,
            message="Learning target too brief to analyze alignment",
            details={"learning_target": learning_target},
        )

    # Check 1: Key terms from learning target should appear in stem OR correct answer
    combined_text = stem_lower + " " + answer_lower
    stem_answer_words = set(re.findall(r"\b\w{4,}\b", combined_text)) - stopwords

    overlap = lt_words & stem_answer_words
    overlap_ratio = len(overlap) / len(lt_words) if lt_words else 0

    if overlap_ratio < 0.25:
        # Less than 25% of learning target concepts appear in stem/answer
        issues.append(
            f"Only {len(overlap)}/{len(lt_words)} learning target terms found in stem/answer "
            f"(expected ≥25%): missing {lt_words - stem_answer_words}"
        )

    # Check 2: Explanation should reference learning target concepts
    explanation_words = set(re.findall(r"\b\w{4,}\b", explanation_lower)) - stopwords
    explanation_overlap = lt_words & explanation_words
    explanation_ratio = len(explanation_overlap) / len(lt_words) if lt_words else 0

    if explanation_ratio < 0.2:
        issues.append(
            f"Explanation doesn't reference learning target concepts "
            f"({len(explanation_overlap)}/{len(lt_words)} terms)"
        )

    # Check 3: Detect topic shift - stem asks about something entirely different
    # Common drift patterns: stem mentions different EMR topics
    other_topics = {
        "cpr": "cardiac",
        "aed": "cardiac",
        "bleeding": "hemorrhage",
        "fracture": "musculoskeletal",
        "airway": "respiratory",
        "breathing": "respiratory",
        "shock": "circulatory",
        "poison": "toxicology",
        "medication": "pharmacology",
        "transport": "logistics",
    }

    # Find topics mentioned in stem but not in learning target
    for topic_word, topic_category in other_topics.items():
        if topic_word in stem_lower and topic_word not in lt_lower:
            # Stem mentions a topic not in the learning target
            if topic_category not in lt_lower:
                issues.append(
                    f"Stem mentions '{topic_word}' ({topic_category}) "
                    f"but learning target is about: {learning_target[:50]}"
                )

    # Check 4: "FIRST action" questions should have learning target about prioritization
    first_action_patterns = [
        r"\bfirst\b.*\b(?:action|step|thing)\b",
        r"\bpriority\b",
        r"\binitial\b.*\b(?:action|response)\b",
    ]
    is_first_action_question = any(
        re.search(p, stem_lower) for p in first_action_patterns
    )
    if is_first_action_question:
        priority_in_lt = any(
            p in lt_lower for p in ["first", "priority", "initial", "before", "sequence"]
        )
        if not priority_in_lt:
            issues.append(
                "Question asks about 'FIRST action' but learning target "
                "doesn't mention prioritization or sequence"
            )

    if issues:
        return GateCheckResult(
            gate_name="learning_target_alignment",
            tier=GateTier.TIER_1_HARD,
            result=GateResult.FAIL,
            message=f"Learning target drift detected: {len(issues)} issues",
            details={
                "learning_target": learning_target,
                "issues": issues,
                "overlap_ratio": round(overlap_ratio, 2),
                "key_terms_found": list(overlap),
                "key_terms_missing": list(lt_words - stem_answer_words),
            },
        )

    return GateCheckResult(
        gate_name="learning_target_alignment",
        tier=GateTier.TIER_1_HARD,
        result=GateResult.PASS,
        message=f"Item aligns with learning target ({len(overlap)}/{len(lt_words)} key terms)",
        details={
            "learning_target": learning_target,
            "overlap_ratio": round(overlap_ratio, 2),
            "key_terms_found": list(overlap),
        },
    )


def check_internal_consistency(
    stem: str,
    correct_answer: str,
    explanation: str,
) -> GateCheckResult:
    """Tier 1: Check for internal contradictions in the item.

    Detects:
    - Explanation contradicts the correct answer
    - Stem and explanation conflict
    - Contradictory temporal/logical statements
    """
    issues = []
    explanation_lower = explanation.lower()
    stem_lower = stem.lower()
    correct_lower = correct_answer.lower()

    # Define contradiction pairs
    contradiction_pairs = [
        (r"\bshould\b", r"\bshould not\b"),
        (r"\bshould not\b", r"\bshould\b"),
        (r"\bmust\b", r"\bmust not\b"),
        (r"\bmust not\b", r"\bmust\b"),
        (r"\balways\b", r"\bnever\b"),
        (r"\bnever\b", r"\balways\b"),
        (r"\bbefore\b", r"\bafter\b"),
        (r"\bfirst\b", r"\blast\b"),
        (r"\bimmediately\b", r"\bdelay\b"),
        (r"\brequired\b", r"\boptional\b"),
        (r"\bsafe\b", r"\bdangerous\b"),
        (r"\bcorrect\b", r"\bincorrect\b"),
    ]

    # Check 1: Explanation contradicts correct answer
    correct_words = set(re.findall(r"\b\w{4,}\b", correct_lower))
    for affirm_pattern, negate_pattern in contradiction_pairs:
        # If correct answer implies one thing and explanation the opposite
        if re.search(affirm_pattern, correct_lower):
            if re.search(negate_pattern, explanation_lower):
                # Check if it's about the same topic
                for word in correct_words:
                    if word in explanation_lower:
                        issues.append(
                            f"Explanation may contradict correct answer "
                            f"('{affirm_pattern}' vs '{negate_pattern}')"
                        )
                        break

    # Check 2: Stem creates contradictory expectation
    stem_has_never = bool(re.search(r"\bnever\b", stem_lower))
    stem_has_always = bool(re.search(r"\balways\b", stem_lower))

    if stem_has_never and re.search(r"\balways\b", explanation_lower):
        issues.append("Stem uses 'never' but explanation uses 'always'")
    if stem_has_always and re.search(r"\bnever\b", explanation_lower):
        issues.append("Stem uses 'always' but explanation uses 'never'")

    # Check 3: Contradictory procedural statements
    procedural_contradictions = [
        (r"first.{0,50}assess", r"first.{0,50}treat"),
        (r"before.{0,50}(calling|contact)", r"immediately.{0,50}(call|contact)"),
    ]

    for pattern1, pattern2 in procedural_contradictions:
        if re.search(pattern1, explanation_lower) and re.search(pattern2, explanation_lower):
            issues.append("Explanation contains contradictory procedural statements")
            break

    # Check 4: "However" or "but" followed by opposite claim
    however_pattern = r"(?:however|but|although).{0,100}(?:should not|must not|never|incorrect)"
    if re.search(however_pattern, explanation_lower):
        # Check if this contradicts the main claim
        if re.search(r"\bshould\b|\bmust\b|\balways\b", explanation_lower[:100]):
            issues.append("Explanation may contain self-contradicting 'however/but' clause")

    if issues:
        return GateCheckResult(
            gate_name="internal_consistency",
            tier=GateTier.TIER_1_HARD,
            result=GateResult.FAIL,
            message=f"Internal contradictions detected: {len(issues)}",
            details={"contradictions": issues},
        )

    return GateCheckResult(
        gate_name="internal_consistency",
        tier=GateTier.TIER_1_HARD,
        result=GateResult.PASS,
        message="No internal contradictions detected",
    )


# ============================================================================
# Tier 2: Repair-First Gates
# ============================================================================


def check_distractor_quality(
    distractors: list[str],
    correct_answer: str,
    explanation: str,
) -> GateCheckResult:
    """Tier 2: Check distractor quality (plausible, not also-correct).

    Repair hint provided if failed.
    """
    issues = []
    repair_hints = []

    for i, distractor in enumerate(distractors):
        # Check 1: Too short (likely not plausible)
        if len(distractor.strip()) < 5:
            issues.append(f"Distractor {i+1} too short: '{distractor}'")
            repair_hints.append(f"Expand distractor {i+1} to be more substantive")

        # Check 2: Contains "all of the above" or similar
        dist_lower = distractor.lower()
        if any(p in dist_lower for p in ["all of the above", "none of the above", "both a and b"]):
            issues.append(f"Distractor {i+1} uses meta-option: '{distractor}'")
            repair_hints.append(f"Replace distractor {i+1} with specific content")

        # Check 3: Identical prefix/suffix with correct (lazy generation)
        if len(distractor) > 10 and len(correct_answer) > 10:
            # Check if first 50% of words are identical
            dist_words = distractor.lower().split()[:3]
            correct_words = correct_answer.lower().split()[:3]
            if dist_words == correct_words and len(dist_words) >= 2:
                issues.append(f"Distractor {i+1} has identical prefix to correct answer")
                repair_hints.append(f"Vary the beginning of distractor {i+1}")

    if issues:
        return GateCheckResult(
            gate_name="distractor_quality",
            tier=GateTier.TIER_2_REPAIR,
            result=GateResult.FAIL,
            message=f"Distractor quality issues: {len(issues)}",
            details={"issues": issues},
            repair_hint="; ".join(repair_hints),
        )

    return GateCheckResult(
        gate_name="distractor_quality",
        tier=GateTier.TIER_2_REPAIR,
        result=GateResult.PASS,
        message="All distractors are plausible",
    )


def check_difficulty_alignment(
    item_content: dict[str, Any],
    target_difficulty: str,
) -> GateCheckResult:
    """Tier 2: Check that item difficulty matches target.

    Difficulty levels: easy, medium, hard
    """
    actual_difficulty = item_content.get("difficulty", "medium")

    if actual_difficulty == target_difficulty:
        return GateCheckResult(
            gate_name="difficulty_alignment",
            tier=GateTier.TIER_2_REPAIR,
            result=GateResult.PASS,
            message=f"Difficulty matches target: {target_difficulty}",
        )

    # Calculate how far off we are
    difficulty_order = {"easy": 1, "medium": 2, "hard": 3}
    target_level = difficulty_order.get(target_difficulty, 2)
    actual_level = difficulty_order.get(actual_difficulty, 2)
    diff = abs(target_level - actual_level)

    if diff == 1:
        # One level off - repairable
        return GateCheckResult(
            gate_name="difficulty_alignment",
            tier=GateTier.TIER_2_REPAIR,
            result=GateResult.FAIL,
            message=f"Difficulty mismatch: target={target_difficulty}, actual={actual_difficulty}",
            details={"target": target_difficulty, "actual": actual_difficulty},
            repair_hint=f"Adjust complexity to match {target_difficulty} level",
        )
    else:
        # Two levels off - still repairable but note severity
        return GateCheckResult(
            gate_name="difficulty_alignment",
            tier=GateTier.TIER_2_REPAIR,
            result=GateResult.FAIL,
            message=f"Significant difficulty mismatch: target={target_difficulty}, actual={actual_difficulty}",
            details={"target": target_difficulty, "actual": actual_difficulty, "levels_off": diff},
            repair_hint=f"Major adjustment needed to reach {target_difficulty} level",
        )


def check_decision_grounding(
    explanation: str,
    correct_answer: str,
    quotes_used: list["ExtractedQuote"],
) -> GateCheckResult:
    """Tier 2: Check that explanation grounds the correct answer decision.

    Looks for:
    - Quote references in explanation
    - Logical connection between quote and answer
    """
    explanation_lower = explanation.lower()
    correct_lower = correct_answer.lower()

    # Check 1: Does explanation reference quotes?
    quote_refs = re.findall(r"\[Q\d+\]", explanation)
    if not quote_refs and quotes_used:
        return GateCheckResult(
            gate_name="decision_grounding",
            tier=GateTier.TIER_2_REPAIR,
            result=GateResult.FAIL,
            message="Explanation does not reference source quotes",
            details={"quotes_available": len(quotes_used)},
            repair_hint="Add explicit quote references: [Q1], [Q2], etc.",
        )

    # Check 2: Does explanation mention key terms from correct answer?
    correct_key_words = set(re.findall(r"\b\w{5,}\b", correct_lower))
    explanation_words = set(re.findall(r"\b\w{5,}\b", explanation_lower))
    overlap = correct_key_words & explanation_words

    if len(correct_key_words) > 0 and len(overlap) / len(correct_key_words) < 0.3:
        return GateCheckResult(
            gate_name="decision_grounding",
            tier=GateTier.TIER_2_REPAIR,
            result=GateResult.FAIL,
            message="Explanation does not connect to correct answer terminology",
            details={
                "correct_terms": list(correct_key_words),
                "terms_mentioned": list(overlap),
            },
            repair_hint="Explain why the correct answer is supported by the source",
        )

    return GateCheckResult(
        gate_name="decision_grounding",
        tier=GateTier.TIER_2_REPAIR,
        result=GateResult.PASS,
        message="Explanation adequately grounds the decision",
        details={"quote_refs": quote_refs},
    )


# ============================================================================
# Tier 3: Soft Flags
# ============================================================================


def check_novelty(
    item_content: dict[str, Any],
    existing_items: list[dict[str, Any]],
    similarity_threshold: float = 0.8,
) -> GateCheckResult:
    """Tier 3: Check that item is sufficiently novel.

    Compares stem and correct answer to existing items.
    """
    current_stem = item_content.get("stem", "").lower()
    current_answer = item_content.get("correct_answer", "").lower()

    for existing in existing_items:
        existing_stem = existing.get("stem", "").lower()
        existing_answer = existing.get("correct_answer", "").lower()

        # Simple word overlap similarity
        current_words = set(current_stem.split() + current_answer.split())
        existing_words = set(existing_stem.split() + existing_answer.split())

        if current_words and existing_words:
            overlap = len(current_words & existing_words)
            union = len(current_words | existing_words)
            similarity = overlap / union if union > 0 else 0

            if similarity > similarity_threshold:
                return GateCheckResult(
                    gate_name="novelty",
                    tier=GateTier.TIER_3_SOFT,
                    result=GateResult.WARN,
                    message=f"Item may be too similar to existing item (similarity={similarity:.2f})",
                    details={
                        "similar_item_id": existing.get("item_id"),
                        "similarity": similarity,
                    },
                )

    return GateCheckResult(
        gate_name="novelty",
        tier=GateTier.TIER_3_SOFT,
        result=GateResult.PASS,
        message="Item is sufficiently novel",
    )


def check_verbosity(
    explanation: str,
    min_words: int = 20,
    max_words: int = 150,
) -> GateCheckResult:
    """Tier 3: Check explanation verbosity.

    Too short = insufficient justification
    Too long = likely rambling
    """
    word_count = len(explanation.split())

    if word_count < min_words:
        return GateCheckResult(
            gate_name="verbosity",
            tier=GateTier.TIER_3_SOFT,
            result=GateResult.WARN,
            message=f"Explanation too brief ({word_count} words, min={min_words})",
            details={"word_count": word_count, "min": min_words},
        )

    if word_count > max_words:
        return GateCheckResult(
            gate_name="verbosity",
            tier=GateTier.TIER_3_SOFT,
            result=GateResult.WARN,
            message=f"Explanation may be too verbose ({word_count} words, max={max_words})",
            details={"word_count": word_count, "max": max_words},
        )

    return GateCheckResult(
        gate_name="verbosity",
        tier=GateTier.TIER_3_SOFT,
        result=GateResult.PASS,
        message=f"Explanation length appropriate ({word_count} words)",
        details={"word_count": word_count},
    )


def check_style(
    stem: str,
    explanation: str,
) -> GateCheckResult:
    """Tier 3: Check style conventions.

    - Stem should end with question mark or colon
    - No emojis
    - No informal language
    """
    issues = []

    # Check 1: Stem ending
    stem_stripped = stem.strip()
    if not stem_stripped.endswith(("?", ":", ".")):
        issues.append("Stem should end with punctuation")

    # Check 2: Informal language
    informal_patterns = [
        r"\b(?:gonna|wanna|gotta|kinda|sorta)\b",
        r"\b(?:yeah|yep|nope|ok|okay)\b",
        r"\blol\b",
    ]
    for pattern in informal_patterns:
        if re.search(pattern, explanation.lower()):
            issues.append("Explanation contains informal language")
            break

    # Check 3: Emojis (basic check)
    if re.search(r"[\U0001F300-\U0001F9FF]", explanation):
        issues.append("Explanation contains emojis")

    if issues:
        return GateCheckResult(
            gate_name="style",
            tier=GateTier.TIER_3_SOFT,
            result=GateResult.WARN,
            message=f"Style issues: {'; '.join(issues)}",
            details={"issues": issues},
        )

    return GateCheckResult(
        gate_name="style",
        tier=GateTier.TIER_3_SOFT,
        result=GateResult.PASS,
        message="Style conventions followed",
    )


# ============================================================================
# Main Gate Runner
# ============================================================================


def run_all_gates(
    item_content: dict[str, Any],
    quote_verification: "GenerationQuoteVerification",
    quotes_used: list["ExtractedQuote"],
    allowed_scope_tags: list[str],
    target_difficulty: str = "medium",
    existing_items: list[dict[str, Any]] | None = None,
    learning_target: str = "",
) -> QualityGateReport:
    """Run all quality gates and return comprehensive report.

    Args:
        item_content: Generated item with stem, correct_answer, distractors, explanation
        quote_verification: Result from verify_quotes_in_text()
        quotes_used: List of ExtractedQuote objects
        allowed_scope_tags: List of allowed scope tags for this item
        target_difficulty: Target difficulty level
        existing_items: Existing items for novelty check
        learning_target: The learning target this item should test

    Returns:
        QualityGateReport with all results
    """
    stem = item_content.get("stem", "")
    correct_answer = item_content.get("correct_answer", "")
    distractors = item_content.get("distractors", [])
    explanation = item_content.get("explanation", "")

    # Use learning_target from argument or fall back to item_content
    lt = learning_target or item_content.get("learning_target", "")

    all_checks: list[GateCheckResult] = []

    # Tier 1: Hard Blockers
    all_checks.append(check_verbatim_quote_presence(
        explanation, quote_verification
    ))
    all_checks.append(check_stem_answer_consistency(
        stem, correct_answer, distractors
    ))
    all_checks.append(check_scope_legality(
        item_content, allowed_scope_tags
    ))
    all_checks.append(check_single_best_answer(
        correct_answer, distractors, explanation
    ))
    all_checks.append(check_internal_consistency(
        stem, correct_answer, explanation
    ))
    # Learning target alignment check
    if lt:
        all_checks.append(check_learning_target_alignment(
            lt, stem, correct_answer, explanation
        ))

    # Tier 2: Repair-First
    all_checks.append(check_distractor_quality(
        distractors, correct_answer, explanation
    ))
    all_checks.append(check_difficulty_alignment(
        item_content, target_difficulty
    ))
    all_checks.append(check_decision_grounding(
        explanation, correct_answer, quotes_used
    ))

    # Tier 3: Soft Flags
    all_checks.append(check_novelty(
        item_content, existing_items or []
    ))
    all_checks.append(check_verbosity(explanation))
    all_checks.append(check_style(stem, explanation))

    # Aggregate results
    tier_1_checks = [c for c in all_checks if c.tier == GateTier.TIER_1_HARD]
    tier_2_checks = [c for c in all_checks if c.tier == GateTier.TIER_2_REPAIR]
    tier_3_checks = [c for c in all_checks if c.tier == GateTier.TIER_3_SOFT]

    tier_1_passed = all(c.result == GateResult.PASS for c in tier_1_checks)
    tier_2_passed = all(c.result == GateResult.PASS for c in tier_2_checks)
    tier_3_flags = [c.gate_name for c in tier_3_checks if c.result == GateResult.WARN]

    # Determine repair targets from Tier 2 failures
    repair_targets = [
        c.gate_name for c in tier_2_checks if c.result == GateResult.FAIL
    ]

    return QualityGateReport(
        item_id=item_content.get("item_id", ""),
        tier_1_passed=tier_1_passed,
        tier_2_passed=tier_2_passed,
        tier_3_flags=tier_3_flags,
        all_checks=all_checks,
        overall_pass=tier_1_passed,  # Tier 1 must pass; Tier 2 can be repaired
        requires_repair=not tier_2_passed and tier_1_passed,
        repair_targets=repair_targets,
    )


def get_repair_hints(report: QualityGateReport) -> dict[str, str]:
    """Extract repair hints from a quality gate report.

    Returns mapping of gate_name -> repair_hint for failed Tier 2 checks.
    """
    hints = {}
    for check in report.all_checks:
        if check.tier == GateTier.TIER_2_REPAIR and check.result == GateResult.FAIL:
            if check.repair_hint:
                hints[check.gate_name] = check.repair_hint
    return hints
