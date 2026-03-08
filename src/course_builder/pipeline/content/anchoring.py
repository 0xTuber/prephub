"""Anchor-driven drafting for evidence-first question generation.

This module implements the anchor-first approach:
1. Select anchor quotes FIRST from evidence chunks
2. Derive question stem constrained by anchors
3. Validate correct answer uses anchor language

This ensures questions are grounded in evidence, not invented freely.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from course_builder.domain.content import AnchorQuote

if TYPE_CHECKING:
    from course_builder.engine import GenerationConfig, GenerationEngine


class AnchorSelectionResult(BaseModel):
    """Result of selecting anchor quotes for an item."""

    anchors: list[AnchorQuote]
    coverage_score: float  # 0-1, how well anchors cover the learning target
    selection_rationale: str  # Why these anchors were selected


def extract_key_terms(text: str) -> list[str]:
    """Extract key terms from text for matching."""
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "and", "or", "in", "on", "at", "for", "with", "by",
        "from", "as", "this", "that", "it", "its", "their", "your", "my",
        "should", "would", "could", "must", "will", "can", "may",
    }

    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w) > 3 and w not in stop_words]


def compute_relevance_score(
    chunk_text: str,
    learning_target: str,
    difficulty: str,
) -> float:
    """Compute how relevant a chunk is to the learning target and difficulty.

    Args:
        chunk_text: The text content of the chunk
        learning_target: The learning target to score against
        difficulty: "beginner", "intermediate", or "advanced"

    Returns:
        Relevance score from 0.0 to 1.0
    """
    target_terms = set(extract_key_terms(learning_target))
    chunk_terms = set(extract_key_terms(chunk_text))

    if not target_terms:
        return 0.5  # Can't score without target terms

    # Base score: term overlap
    overlap = len(target_terms & chunk_terms)
    base_score = min(overlap / len(target_terms), 1.0)

    # Bonus for containing exact phrases from target
    phrase_bonus = 0.0
    target_lower = learning_target.lower()
    chunk_lower = chunk_text.lower()
    if target_lower in chunk_lower:
        phrase_bonus = 0.2

    # Bonus for containing action verbs (procedural content)
    action_verbs = [
        "should", "must", "ensure", "verify", "check", "assess", "perform",
        "establish", "maintain", "monitor", "provide", "administer",
    ]
    action_bonus = 0.1 if any(v in chunk_lower for v in action_verbs) else 0.0

    # Difficulty-specific bonuses
    difficulty_bonus = 0.0
    if difficulty == "advanced":
        # Advanced prefers content with prioritization language
        priority_words = ["first", "priority", "before", "critical", "immediate"]
        if any(w in chunk_lower for w in priority_words):
            difficulty_bonus = 0.1
    elif difficulty == "beginner":
        # Beginner prefers simple definitions and single concepts
        definition_patterns = [r"\bis\s+(?:a|an|the)\b", r"\bare\s+(?:a|an|the)\b"]
        if any(re.search(p, chunk_lower) for p in definition_patterns):
            difficulty_bonus = 0.1

    return min(base_score + phrase_bonus + action_bonus + difficulty_bonus, 1.0)


def extract_quotable_sentences(chunk_text: str, min_length: int = 30, max_length: int = 200) -> list[str]:
    """Extract sentences from chunk that could serve as anchor quotes.

    Args:
        chunk_text: Full text of the chunk
        min_length: Minimum sentence length to consider
        max_length: Maximum sentence length to consider

    Returns:
        List of candidate sentences for anchoring
    """
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text)

    candidates = []
    for sentence in sentences:
        sentence = sentence.strip()
        length = len(sentence)

        if length < min_length or length > max_length:
            continue

        # Skip questions (they don't make good anchors)
        if sentence.endswith("?"):
            continue

        # Skip list-like content
        if sentence.startswith(("-", "•", "*", "–")):
            continue

        # Prefer sentences with action verbs or factual claims
        has_action = any(
            word in sentence.lower()
            for word in ["should", "must", "ensure", "always", "never", "important"]
        )
        has_factual = re.search(r"\b(is|are|was|were)\s+(a|an|the|considered|defined)", sentence.lower())

        if has_action or has_factual:
            candidates.append(sentence)

    return candidates


def select_anchor_quotes(
    chunks: list[dict],
    learning_target: str,
    difficulty: str,
    max_anchors: int = 2,
    min_anchors: int = 1,
) -> AnchorSelectionResult:
    """Select anchor quotes from chunks to ground question generation.

    This is the key function in anchor-driven drafting:
    - Finds the best quotes to anchor the correct answer
    - Ensures quotes are verbatim from source material
    - Scores by relevance to learning target

    Args:
        chunks: Retrieved source chunks
        learning_target: The learning target for this item
        difficulty: Difficulty level
        max_anchors: Maximum number of anchors to select
        min_anchors: Minimum required anchors

    Returns:
        AnchorSelectionResult with selected anchors
    """
    if not chunks:
        return AnchorSelectionResult(
            anchors=[],
            coverage_score=0.0,
            selection_rationale="No chunks provided",
        )

    # Score all chunks for relevance
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        score = compute_relevance_score(text, learning_target, difficulty)
        scored_chunks.append((score, i, chunk))

    # Sort by relevance (descending)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Select anchors from top chunks
    anchors = []
    used_chunk_ids = set()
    coverage_terms = set()
    target_terms = set(extract_key_terms(learning_target))

    for score, chunk_index, chunk in scored_chunks:
        if len(anchors) >= max_anchors:
            break

        chunk_id = chunk.get("chunk_id", f"chunk_{chunk_index}")
        text = chunk.get("text", "")
        pages = chunk.get("pages", [])

        # Skip if we've already used this chunk
        if chunk_id in used_chunk_ids:
            continue

        # Extract quotable sentences
        candidates = extract_quotable_sentences(text)
        if not candidates:
            continue

        # Score candidates by term coverage
        best_candidate = None
        best_candidate_score = 0.0

        for sentence in candidates:
            sentence_terms = set(extract_key_terms(sentence))
            new_terms = sentence_terms & target_terms - coverage_terms
            candidate_score = len(new_terms) / max(len(target_terms), 1)

            if candidate_score > best_candidate_score:
                best_candidate = sentence
                best_candidate_score = candidate_score

        if best_candidate:
            anchor = AnchorQuote(
                text=best_candidate,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                page_numbers=pages,
                relevance_score=score,
            )
            anchors.append(anchor)
            used_chunk_ids.add(chunk_id)
            coverage_terms.update(extract_key_terms(best_candidate))

    # Compute overall coverage
    coverage_score = len(coverage_terms & target_terms) / max(len(target_terms), 1)

    # Build rationale
    if not anchors:
        rationale = "No suitable anchor quotes found in chunks"
    elif len(anchors) < min_anchors:
        rationale = f"Found {len(anchors)} anchors (below minimum {min_anchors})"
    else:
        pages_cited = sorted(set(p for a in anchors for p in a.page_numbers))
        rationale = f"Selected {len(anchors)} anchors from pages {pages_cited}"

    return AnchorSelectionResult(
        anchors=anchors,
        coverage_score=coverage_score,
        selection_rationale=rationale,
    )


ANCHOR_DRIVEN_SYSTEM_PROMPT = """You are an expert exam question writer for certification preparation.
You will generate questions GROUNDED in specific anchor quotes from source material.
Your questions must:
1. Use language from the anchor quotes in the correct answer
2. Explain why the answer is correct by citing the anchor quotes
3. Never introduce facts not present in the provided source material"""


ANCHOR_DRIVEN_USER_PROMPT = """Generate a practice question ANCHORED in these specific quotes.

CERTIFICATION: {certification} ({exam_code})
DOMAIN: {domain_name}
TOPIC: {topic_name}
LEARNING TARGET: {learning_target}
DIFFICULTY: {difficulty}

=== ANCHOR QUOTES (correct answer MUST use these) ===
{anchor_quotes}

=== SUPPORTING EVIDENCE (for distractors and context) ===
{source_chunks}

=== REQUIREMENTS ===

1. CORRECT ANSWER must directly reflect the anchor quote language
   - Use key phrases from the anchors in the correct option
   - The explanation must cite the anchor quotes with page numbers

2. QUESTION STYLE:
   - Start with a realistic scenario: "You arrive at..." or "You respond to..."
   - Make the clinical picture clear enough to select ONE correct answer

3. DISTRACTORS:
   - Must be plausible EMS actions (not cartoonish)
   - Wrong due to: wrong sequence, wrong context, incomplete, or common misconception

4. EXPLANATION:
   - Quote the anchor: "The source states (p.XX): '...exact quote...'"
   - Explain why distractors are wrong

Return a JSON object:
{{
  "stem": "You arrive at... [realistic scenario question]",
  "options": ["option 1", "option 2", "option 3", "option 4"],
  "correct_index": 0,
  "explanation": "The source states (p.XX): '...' [quote anchors and explain]",
  "source_summary": "30-50 word summary of the key concept",
  "anchor_usage": ["quoted phrase 1 from anchor", "quoted phrase 2 from anchor"]
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


def format_anchors_for_prompt(anchors: list[AnchorQuote]) -> str:
    """Format anchor quotes for inclusion in prompt."""
    lines = []
    for i, anchor in enumerate(anchors, 1):
        pages = ", ".join(str(p) for p in anchor.page_numbers) if anchor.page_numbers else "N/A"
        lines.append(f"ANCHOR {i} (p.{pages}):\n\"{anchor.text}\"")
    return "\n\n".join(lines)


def format_chunks_for_prompt(chunks: list[dict], max_chunks: int = 5) -> str:
    """Format source chunks for inclusion in prompt."""
    lines = []
    for chunk in chunks[:max_chunks]:
        pages = ", ".join(str(p) for p in chunk.get("pages", [])) or "N/A"
        text = chunk.get("text", "")[:500]  # Truncate long chunks
        lines.append(f"[Source: p.{pages}]\n{text}")
    return "\n\n---\n\n".join(lines)


def derive_stem_from_anchors(
    engine: "GenerationEngine",
    anchors: list[AnchorQuote],
    chunks: list[dict],
    certification: str,
    exam_code: str | None,
    domain_name: str,
    topic_name: str,
    learning_target: str,
    difficulty: str,
) -> dict | None:
    """Generate question stem and options constrained by anchor quotes.

    This is the second step in anchor-driven drafting:
    - Uses anchor quotes to constrain the correct answer
    - Validates the generated content uses anchor language

    Args:
        engine: Generation engine for LLM calls
        anchors: Selected anchor quotes
        chunks: Full source chunks for context
        certification: Certification name
        exam_code: Exam code
        domain_name: Domain name
        topic_name: Topic name
        learning_target: Learning target
        difficulty: Difficulty level

    Returns:
        Dict with stem, options, correct_index, explanation, source_summary, anchor_usage
        or None if generation fails
    """
    from course_builder.engine import GenerationConfig

    if not anchors:
        return None

    prompt = ANCHOR_DRIVEN_USER_PROMPT.format(
        certification=certification,
        exam_code=exam_code or "N/A",
        domain_name=domain_name,
        topic_name=topic_name,
        learning_target=learning_target,
        difficulty=difficulty,
        anchor_quotes=format_anchors_for_prompt(anchors),
        source_chunks=format_chunks_for_prompt(chunks),
    )

    config = GenerationConfig(system_prompt=ANCHOR_DRIVEN_SYSTEM_PROMPT)
    result = engine.generate(prompt, config=config)

    try:
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def validate_anchor_usage(
    question_data: dict,
    anchors: list[AnchorQuote],
    min_phrase_match_length: int = 15,
) -> tuple[bool, list[str], list[str]]:
    """Validate that the generated question uses anchor language.

    Args:
        question_data: Generated question dict
        anchors: Anchor quotes that should be used
        min_phrase_match_length: Minimum phrase length for matching

    Returns:
        (is_valid, matched_phrases, missing_phrases)
    """
    correct_answer = question_data.get("options", [])[question_data.get("correct_index", 0)]
    explanation = question_data.get("explanation", "")
    anchor_usage = question_data.get("anchor_usage", [])

    combined_text = f"{correct_answer} {explanation}".lower()

    matched = []
    missing = []

    for anchor in anchors:
        anchor_text = anchor.text.lower()

        # Check if any substantial phrase from anchor appears
        found = False
        words = anchor_text.split()

        # Try to find phrases of increasing length
        for phrase_len in range(3, len(words) + 1):
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i : i + phrase_len])
                if len(phrase) >= min_phrase_match_length and phrase in combined_text:
                    matched.append(phrase)
                    found = True
                    break
            if found:
                break

        if not found:
            missing.append(anchor.text[:50] + "...")

    is_valid = len(matched) >= 1 and len(missing) <= len(anchors) // 2

    return is_valid, matched, missing
