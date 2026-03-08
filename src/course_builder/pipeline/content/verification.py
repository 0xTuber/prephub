"""Verification loop for post-generation quality validation.

This module implements the PASS/REPAIR/REJECT verification flow:
1. Extract claims from generated explanation
2. Verify each claim against source chunks
3. PASS: claim found verbatim or paraphrased
4. REPAIR: claim close but needs wording adjustment
5. REJECT: no support found, regenerate with different anchor

Key quality invariants enforced:
- Evidence-first: CORRECT ANSWER must be traceable to source chunks
- Single best answer: Correct answer must be unambiguously supported
- Distractor reasoning: Allowed to use standard EMS inference patterns

Validation Modes:
- STRICT: Quote required for correct answer AND all distractor reasoning
- BALANCED: Quote required for correct answer, allow inference for distractors
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from course_builder.domain.content import CapsuleItem, QuoteVerification

if TYPE_CHECKING:
    from course_builder.engine import GenerationEngine


# =============================================================================
# INFERENCE WHITELIST FOR DISTRACTOR REASONING
# =============================================================================
# These patterns are allowed in distractor explanations without quote support.
# They represent standard EMS reasoning that doesn't introduce new facts.

ALLOWED_INFERENCE_PATTERNS = [
    # Priority/sequence reasoning
    r"is secondary",
    r"should come after",
    r"is not the first",
    r"is not the priority",
    r"is less urgent",
    r"is less immediate",
    r"should be done later",
    r"comes after",
    r"follows",
    r"is premature",
    r"is delayed",
    # Safety reasoning
    r"introduces risk",
    r"puts .* at risk",
    r"is unsafe",
    r"is dangerous",
    r"doesn't protect",
    r"doesn't address .* hazard",
    r"ignores .* safety",
    r"compromises safety",
    # Incompleteness reasoning
    r"doesn't address",
    r"doesn't adequately",
    r"is incomplete",
    r"is insufficient",
    r"only partially",
    r"fails to address",
    # Scope/appropriateness
    r"is not appropriate",
    r"is not indicated",
    r"is outside .* scope",
    r"is not within",
    r"exceeds",
    r"is excessive",
    # Context mismatch
    r"wrong context",
    r"wrong situation",
    r"different scenario",
    r"not applicable here",
]


class VerificationMode(str, Enum):
    """Verification strictness mode."""

    STRICT = "strict"  # Quote required for correct answer AND distractors
    BALANCED = "balanced"  # Quote for correct, allow inference for distractors


class VerificationStatus(str, Enum):
    """Status of a verification check."""

    PASS = "pass"
    REPAIR = "repair"
    REJECT = "reject"


def is_allowed_inference(claim_text: str) -> bool:
    """Check if a claim matches allowed inference patterns.

    These are standard EMS reasoning patterns that don't introduce
    new facts and are acceptable without quote support.

    Args:
        claim_text: The claim text to check

    Returns:
        True if claim matches an allowed inference pattern
    """
    claim_lower = claim_text.lower()

    for pattern in ALLOWED_INFERENCE_PATTERNS:
        if re.search(pattern, claim_lower):
            return True

    return False


class ClaimType(str, Enum):
    """Type of claim extracted from explanation."""

    FACTUAL = "factual"  # A fact that must be sourced
    PROCEDURAL = "procedural"  # A procedure or sequence claim
    DEFINITION = "definition"  # A definition or description
    RATIONALE = "rationale"  # Reasoning for why answer is correct
    DISTRACTOR_EXPLANATION = "distractor"  # Why distractor is wrong


@dataclass
class ExtractedClaim:
    """A claim extracted from explanation text."""

    text: str  # The claim text
    claim_type: ClaimType  # Type of claim
    source_sentence: str  # Original sentence containing claim
    is_critical: bool = False  # True if claim is critical for correct answer


@dataclass
class ClaimVerificationResult:
    """Result of verifying a single claim."""

    claim: ExtractedClaim
    status: VerificationStatus
    matched_chunk_id: str | None = None
    matched_text: str | None = None
    confidence: float = 0.0
    repair_suggestion: str | None = None


@dataclass
class VerificationResult:
    """Overall result of verifying an item."""

    status: VerificationStatus
    claim_results: list[ClaimVerificationResult] = field(default_factory=list)
    pass_count: int = 0
    repair_count: int = 0
    reject_count: int = 0
    critical_failures: list[str] = field(default_factory=list)
    repaired_explanation: str | None = None
    rejection_reason: str | None = None


def extract_claims_from_explanation(explanation: str) -> list[ExtractedClaim]:
    """Extract verifiable claims from an explanation.

    Identifies factual assertions that need source verification.

    Args:
        explanation: The explanation text from a generated item

    Returns:
        List of extracted claims
    """
    claims = []

    # Split into sentences
    sentences = re.split(r"(?<=[.!])\s+", explanation)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Skip meta-sentences (references to "the source states")
        if re.match(r"^the source states?", sentence.lower()):
            # Extract the quoted part as a claim
            quoted = re.findall(r'"([^"]+)"', sentence)
            for quote in quoted:
                claims.append(ExtractedClaim(
                    text=quote,
                    claim_type=ClaimType.FACTUAL,
                    source_sentence=sentence,
                    is_critical=True,  # Quoted claims are critical
                ))
            continue

        # Detect claim types
        claim_type = ClaimType.FACTUAL  # Default

        # Procedural claims (should/must/requires)
        if re.search(r"\b(should|must|requires?|ensures?)\b", sentence.lower()):
            claim_type = ClaimType.PROCEDURAL

        # Definition claims (is defined as, is a, means)
        if re.search(r"\b(is defined as|is a|is an|means|refers to)\b", sentence.lower()):
            claim_type = ClaimType.DEFINITION

        # Rationale claims (because, therefore, since)
        if re.search(r"\b(because|therefore|thus|since|this is why)\b", sentence.lower()):
            claim_type = ClaimType.RATIONALE

        # Distractor explanations (option X is wrong, incorrect)
        if re.search(r"\b(is wrong|is incorrect|incorrect because|wrong because)\b", sentence.lower()):
            claim_type = ClaimType.DISTRACTOR_EXPLANATION

        # Check if critical (supports correct answer)
        is_critical = bool(re.search(
            r"\b(correct|right|best|appropriate|should)\b",
            sentence.lower(),
        ))

        # Only add substantive claims (skip very short sentences)
        if len(sentence) > 30:
            claims.append(ExtractedClaim(
                text=sentence,
                claim_type=claim_type,
                source_sentence=sentence,
                is_critical=is_critical,
            ))

    return claims


def verify_claim_against_chunks(
    claim: ExtractedClaim,
    chunks: list[dict],
    min_match_threshold: float = 0.6,
) -> ClaimVerificationResult:
    """Verify a single claim against source chunks.

    Args:
        claim: The claim to verify
        chunks: Source chunks to check against
        min_match_threshold: Minimum similarity for partial match

    Returns:
        ClaimVerificationResult with status and details
    """
    claim_text = claim.text.lower()
    claim_words = set(claim_text.split())

    best_match = None
    best_score = 0.0
    best_chunk_id = None

    for chunk in chunks:
        chunk_text = chunk.get("text", "").lower()
        chunk_id = chunk.get("chunk_id", "unknown")

        # Check for exact substring match
        if claim_text in chunk_text:
            return ClaimVerificationResult(
                claim=claim,
                status=VerificationStatus.PASS,
                matched_chunk_id=chunk_id,
                matched_text=claim.text,
                confidence=1.0,
            )

        # Check for high word overlap (partial match)
        chunk_words = set(chunk_text.split())
        overlap = len(claim_words & chunk_words)
        similarity = overlap / max(len(claim_words), 1)

        if similarity > best_score:
            best_score = similarity
            best_chunk_id = chunk_id
            # Find the most relevant sentence in chunk
            for sentence in chunk_text.split("."):
                sentence_words = set(sentence.split())
                sent_overlap = len(claim_words & sentence_words) / max(len(claim_words), 1)
                if sent_overlap > 0.5:
                    best_match = sentence.strip()
                    break

    # Determine status based on best match
    if best_score >= 0.8:
        return ClaimVerificationResult(
            claim=claim,
            status=VerificationStatus.PASS,
            matched_chunk_id=best_chunk_id,
            matched_text=best_match,
            confidence=best_score,
        )
    elif best_score >= min_match_threshold:
        return ClaimVerificationResult(
            claim=claim,
            status=VerificationStatus.REPAIR,
            matched_chunk_id=best_chunk_id,
            matched_text=best_match,
            confidence=best_score,
            repair_suggestion=f"Adjust wording to match: '{best_match[:100]}...'" if best_match else None,
        )
    else:
        return ClaimVerificationResult(
            claim=claim,
            status=VerificationStatus.REJECT,
            matched_chunk_id=None,
            matched_text=None,
            confidence=best_score,
        )


def detect_hallucination_patterns(explanation: str, chunks: list[dict]) -> list[str]:
    """Detect common hallucination patterns in explanations.

    These are facts that are likely invented rather than sourced.

    Args:
        explanation: The explanation text
        chunks: Source chunks

    Returns:
        List of potential hallucinations
    """
    hallucinations = []

    # Combine all chunk text for searching
    all_source_text = " ".join(c.get("text", "").lower() for c in chunks)

    # Pattern 1: Specific sensory details (sounds, colors)
    sensory_patterns = [
        (r"(\w+ing sound)", "sound"),
        (r"(crackling|popping|sizzling)", "sound"),
        (r"(smell(?:s)? (?:of |like )?\w+)", "smell"),
        (r"(blue|yellow|orange|green) (?:color|flash|glow)", "visual"),
    ]

    for pattern, pattern_type in sensory_patterns:
        matches = re.findall(pattern, explanation.lower())
        for match in matches:
            if match not in all_source_text:
                hallucinations.append(f"Unsourced {pattern_type}: '{match}'")

    # Pattern 2: Specific numbers/measurements not in source
    number_patterns = [
        r"(\d+)\s*(?:feet|meters|yards|inches)",
        r"(\d+)\s*(?:minutes?|seconds?|hours?)",
        r"(\d+)\s*(?:percent|%)",
    ]

    for pattern in number_patterns:
        matches = re.findall(pattern, explanation.lower())
        for match in matches:
            # Check if this specific number appears in source
            if match not in all_source_text:
                hallucinations.append(f"Unsourced measurement: '{match}'")

    # Pattern 3: Admission of insufficient source
    admission_phrases = [
        "while not explicitly stated",
        "although not mentioned",
        "can be inferred",
        "extrapolating from",
        "generally accepted",
        "common practice",
        "based on general principles",
    ]

    for phrase in admission_phrases:
        if phrase in explanation.lower():
            hallucinations.append(f"Admission of inference: '{phrase}'")

    return hallucinations[:5]  # Limit to top 5


def detect_admission_of_insufficient_source(explanation: str) -> tuple[bool, str | None]:
    """Detect if explanation admits the source doesn't support the claim.

    HARD GATE: If model admits lack of evidence, reject immediately.

    Args:
        explanation: The explanation text

    Returns:
        (is_insufficient, reason)
    """
    explanation_lower = explanation.lower()

    admission_phrases = [
        "chunks don't directly address",
        "chunks do not directly address",
        "sources don't directly address",
        "sources do not directly address",
        "not explicitly stated",
        "not explicitly mentioned",
        "while not explicitly",
        "although not explicitly",
        "inferred from general",
        "inferred from the",
        "no direct evidence",
        "no explicit evidence",
        "doesn't directly mention",
        "does not directly mention",
        "not directly covered",
        "not directly discussed",
        "extrapolating from",
        "based on general principles",
        "common practice suggests",
        "generally accepted that",
    ]

    for phrase in admission_phrases:
        if phrase in explanation_lower:
            return True, f"Explanation admits: '{phrase}'"

    return False, None


def check_correct_answer_evidence_support(
    correct_answer: str,
    chunks: list[dict],
    min_keyword_matches: int = 2,
) -> tuple[bool, list[str], str | None]:
    """Verify the correct answer has direct textual support.

    Args:
        correct_answer: The correct answer option text
        chunks: Source chunks
        min_keyword_matches: Minimum keyword matches required

    Returns:
        (is_supported, matched_keywords, failure_reason)
    """
    if not correct_answer or not chunks:
        return False, [], "No correct answer or chunks provided"

    # Extract meaningful keywords (skip stop words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "and", "or", "in", "on", "at", "for", "with", "by",
        "from", "as", "this", "that", "it", "its", "their", "your", "my",
        "should", "would", "could", "must", "will", "can", "may", "first",
        "then", "next", "before", "after", "when", "while", "during",
    }

    words = correct_answer.lower().split()
    keywords = [
        w.strip(".,!?()[]\"'")
        for w in words
        if len(w) > 3 and w.lower() not in stop_words
    ]

    if not keywords:
        return True, [], None

    # Combine all chunk text
    all_chunk_text = " ".join(c.get("text", "").lower() for c in chunks)

    # Check keyword matches
    matched = [kw for kw in keywords if kw in all_chunk_text]

    if len(matched) >= min(min_keyword_matches, len(keywords)):
        return True, matched, None

    missing = [kw for kw in keywords if kw not in matched]
    return False, matched, f"Keywords not in evidence: {missing[:5]}"


class VerificationLoop:
    """Verification loop for post-generation quality validation.

    Implements the PASS/REPAIR/REJECT flow:
    - PASS: Claim verified, keep as-is
    - REPAIR: Claim close, adjust wording
    - REJECT: No support, regenerate

    Verification Modes:
    - STRICT: Quote required for correct answer AND all distractor reasoning
    - BALANCED: Quote required for correct answer, allow standard EMS inference
              for distractor reasoning (priority, safety, scope patterns)

    Usage:
        loop = VerificationLoop(mode=VerificationMode.BALANCED)
        result = loop.verify_item(item, chunks)

        if result.status == VerificationStatus.PASS:
            # Item is good
        elif result.status == VerificationStatus.REPAIR:
            # Use result.repaired_explanation
        else:
            # Regenerate with different anchors
    """

    def __init__(
        self,
        engine: "GenerationEngine | None" = None,
        strict_mode: bool = False,
        mode: VerificationMode = VerificationMode.BALANCED,
        max_repair_attempts: int = 2,
    ):
        """Initialize verification loop.

        Args:
            engine: Generation engine for repairs (optional)
            strict_mode: DEPRECATED - use mode=VerificationMode.STRICT instead
            mode: Verification mode (STRICT or BALANCED)
            max_repair_attempts: Maximum repair attempts before rejection
        """
        self.engine = engine
        # Handle legacy strict_mode parameter
        if strict_mode:
            self.mode = VerificationMode.STRICT
        else:
            self.mode = mode
        self.max_repair_attempts = max_repair_attempts

    def verify_item(
        self,
        item: CapsuleItem,
        chunks: list[dict],
    ) -> VerificationResult:
        """Verify a generated item against source chunks.

        Args:
            item: The generated CapsuleItem
            chunks: Source chunks used for generation

        Returns:
            VerificationResult with status and details
        """
        if not item.explanation:
            return VerificationResult(
                status=VerificationStatus.REJECT,
                rejection_reason="No explanation to verify",
            )

        # Check for admission of insufficient source (HARD GATE)
        is_insufficient, reason = detect_admission_of_insufficient_source(item.explanation)
        if is_insufficient:
            return VerificationResult(
                status=VerificationStatus.REJECT,
                rejection_reason=f"Admission of insufficient source: {reason}",
            )

        # Check correct answer evidence support (HARD GATE)
        if item.options and item.correct_answer_index is not None:
            correct_answer = item.options[item.correct_answer_index]
            is_supported, matched_kw, failure_reason = check_correct_answer_evidence_support(
                correct_answer, chunks
            )
            if not is_supported:
                return VerificationResult(
                    status=VerificationStatus.REJECT,
                    rejection_reason=f"Correct answer not supported: {failure_reason}",
                )

        # Detect hallucinations
        hallucinations = detect_hallucination_patterns(item.explanation, chunks)
        if hallucinations and self.mode == VerificationMode.STRICT:
            return VerificationResult(
                status=VerificationStatus.REJECT,
                rejection_reason=f"Hallucinations detected: {hallucinations}",
            )

        # Extract and verify claims
        claims = extract_claims_from_explanation(item.explanation)
        claim_results = []
        pass_count = 0
        repair_count = 0
        reject_count = 0
        inference_allowed_count = 0
        critical_failures = []

        for claim in claims:
            result = verify_claim_against_chunks(claim, chunks)

            # In BALANCED mode, allow inference for distractor explanations
            if (
                self.mode == VerificationMode.BALANCED
                and result.status == VerificationStatus.REJECT
                and claim.claim_type == ClaimType.DISTRACTOR_EXPLANATION
            ):
                # Check if it matches allowed inference patterns
                if is_allowed_inference(claim.text):
                    # Upgrade to PASS - this is acceptable inference
                    result.status = VerificationStatus.PASS
                    result.matched_text = "[allowed inference]"
                    inference_allowed_count += 1

            # Also allow inference for rationale claims about priority/sequence
            if (
                self.mode == VerificationMode.BALANCED
                and result.status == VerificationStatus.REJECT
                and claim.claim_type == ClaimType.RATIONALE
                and not claim.is_critical
            ):
                if is_allowed_inference(claim.text):
                    result.status = VerificationStatus.PASS
                    result.matched_text = "[allowed inference]"
                    inference_allowed_count += 1

            claim_results.append(result)

            if result.status == VerificationStatus.PASS:
                pass_count += 1
            elif result.status == VerificationStatus.REPAIR:
                repair_count += 1
            else:
                reject_count += 1
                if claim.is_critical:
                    critical_failures.append(claim.text[:50])

        # Determine overall status
        # In BALANCED mode, only fail if critical claims (correct answer support) fail
        if self.mode == VerificationMode.STRICT:
            if critical_failures:
                status = VerificationStatus.REJECT
                rejection_reason = f"Critical claims unverified: {critical_failures}"
            elif reject_count > pass_count:
                status = VerificationStatus.REJECT
                rejection_reason = f"Too many unverified claims ({reject_count}/{len(claims)})"
            elif repair_count > 0:
                status = VerificationStatus.REPAIR
                rejection_reason = None
            else:
                status = VerificationStatus.PASS
                rejection_reason = None
        else:  # BALANCED mode
            if critical_failures:
                # Only reject if CRITICAL claims (correct answer support) fail
                status = VerificationStatus.REJECT
                rejection_reason = f"Correct answer not supported: {critical_failures}"
            elif repair_count > pass_count and reject_count > inference_allowed_count:
                status = VerificationStatus.REPAIR
                rejection_reason = None
            else:
                status = VerificationStatus.PASS
                rejection_reason = None

        # Build quote verification records
        quote_verifications = []
        for result in claim_results:
            quote_verifications.append(QuoteVerification(
                quote_text=result.claim.text[:100],
                found=result.status == VerificationStatus.PASS,
                chunk_id=result.matched_chunk_id,
                match_type="exact" if result.confidence == 1.0 else "partial",
                match_confidence=result.confidence,
            ))

        return VerificationResult(
            status=status,
            claim_results=claim_results,
            pass_count=pass_count,
            repair_count=repair_count,
            reject_count=reject_count,
            critical_failures=critical_failures,
            rejection_reason=rejection_reason,
        )

    def repair_item(
        self,
        item: CapsuleItem,
        verification_result: VerificationResult,
        chunks: list[dict],
    ) -> CapsuleItem | None:
        """Attempt to repair an item based on verification results.

        Args:
            item: The item to repair
            verification_result: Results from verify_item
            chunks: Source chunks

        Returns:
            Repaired CapsuleItem or None if repair fails
        """
        if not self.engine:
            return None

        if verification_result.status != VerificationStatus.REPAIR:
            return None

        # Collect repair suggestions
        repairs_needed = []
        for result in verification_result.claim_results:
            if result.status == VerificationStatus.REPAIR and result.repair_suggestion:
                repairs_needed.append({
                    "claim": result.claim.text[:100],
                    "suggestion": result.repair_suggestion,
                    "source_text": result.matched_text,
                })

        if not repairs_needed:
            return None

        # TODO: Implement LLM-based repair
        # For now, return None to indicate repair not implemented
        return None
