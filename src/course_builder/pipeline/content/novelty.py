"""Novelty gate for concept deduplication.

This module prevents duplicate questions by:
1. Computing concept signatures for each item
2. Checking new items against previously seen signatures
3. Rejecting items that are too similar to existing ones

Quality invariant enforced:
- No duplicate concepts: Same question cannot be asked multiple ways
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class ConceptSignature(BaseModel):
    """Signature for a question concept for deduplication.

    A concept signature captures the essential "what is being tested"
    independent of surface-level wording.
    """

    concept_tag: str  # Unique identifier (e.g., "electrical_hazard_isolation")
    key_terms: list[str]  # Core terms that define this concept
    bloom_level: str | None = None  # Bloom's taxonomy level
    domain: str | None = None  # Domain area (e.g., "scene_safety")
    embedding: list[float] | None = None  # Optional vector embedding


@dataclass
class NoveltyCheckResult:
    """Result of checking an item's novelty."""

    is_novel: bool  # True if item is sufficiently novel
    similarity_score: float  # 0-1, how similar to most similar existing item
    similar_tag: str | None = None  # Tag of most similar item if not novel
    similar_terms: list[str] = field(default_factory=list)  # Overlapping terms


def extract_concept_terms(text: str) -> list[str]:
    """Extract concept-defining terms from text.

    Filters out common words to get the core concept terms.

    Args:
        text: Learning target or question text

    Returns:
        List of concept-defining terms
    """
    # Stop words that don't define concepts
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "and", "or", "in", "on", "at", "for", "with", "by",
        "from", "as", "this", "that", "it", "its", "their", "your", "my",
        "should", "would", "could", "must", "will", "can", "may", "have",
        "has", "had", "do", "does", "did", "what", "which", "who", "when",
        "where", "why", "how", "first", "then", "next", "before", "after",
        "during", "while", "patient", "scene", "ems", "emr", "emt",
        "appropriate", "correct", "proper", "identify", "describe",
        "explain", "understand", "recognize", "select", "determine",
    }

    # Extract words, lowercase, filter
    words = re.findall(r"\b\w+\b", text.lower())
    terms = [
        w for w in words
        if len(w) > 3 and w not in stop_words and not w.isdigit()
    ]

    return terms


def normalize_concept_tag(tag: str) -> str:
    """Normalize a concept tag for comparison.

    Args:
        tag: Raw concept tag

    Returns:
        Normalized tag (lowercase, underscores, no special chars)
    """
    # Convert to lowercase
    tag = tag.lower()

    # Replace spaces and hyphens with underscores
    tag = re.sub(r"[\s\-]+", "_", tag)

    # Remove special characters
    tag = re.sub(r"[^a-z0-9_]", "", tag)

    # Remove duplicate underscores
    tag = re.sub(r"_+", "_", tag)

    return tag.strip("_")


def compute_term_similarity(terms1: list[str], terms2: list[str]) -> float:
    """Compute Jaccard similarity between two term lists.

    Args:
        terms1: First term list
        terms2: Second term list

    Returns:
        Similarity score from 0.0 to 1.0
    """
    if not terms1 or not terms2:
        return 0.0

    set1 = set(terms1)
    set2 = set(terms2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def compute_tag_similarity(tag1: str, tag2: str) -> float:
    """Compute similarity between two concept tags.

    Uses normalized Levenshtein-like comparison.

    Args:
        tag1: First concept tag
        tag2: Second concept tag

    Returns:
        Similarity score from 0.0 to 1.0
    """
    norm1 = normalize_concept_tag(tag1)
    norm2 = normalize_concept_tag(tag2)

    if norm1 == norm2:
        return 1.0

    # Split into parts and compare
    parts1 = set(norm1.split("_"))
    parts2 = set(norm2.split("_"))

    if not parts1 or not parts2:
        return 0.0

    intersection = len(parts1 & parts2)
    union = len(parts1 | parts2)

    return intersection / union if union > 0 else 0.0


class NoveltyGate:
    """Gate for checking concept novelty and preventing duplicates.

    Maintains a registry of seen concepts and checks new items
    against it.

    Usage:
        gate = NoveltyGate(similarity_threshold=0.85)

        # Check if a new concept is novel
        result = gate.check_novelty(signature)
        if result.is_novel:
            gate.register(signature)
        else:
            # Reject duplicate

        # Clear for new capsule/session
        gate.clear()
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_embeddings: bool = False,
    ):
        """Initialize novelty gate.

        Args:
            similarity_threshold: Max similarity for an item to be considered novel
            use_embeddings: Whether to use vector embeddings for comparison
        """
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self._registered: list[ConceptSignature] = []

    def clear(self) -> None:
        """Clear all registered signatures."""
        self._registered = []

    def register(self, signature: ConceptSignature) -> None:
        """Register a concept signature as seen.

        Args:
            signature: The concept signature to register
        """
        self._registered.append(signature)

    def check_novelty(self, signature: ConceptSignature) -> NoveltyCheckResult:
        """Check if a concept signature is sufficiently novel.

        Args:
            signature: The concept signature to check

        Returns:
            NoveltyCheckResult with novelty status and details
        """
        if not self._registered:
            return NoveltyCheckResult(
                is_novel=True,
                similarity_score=0.0,
            )

        best_similarity = 0.0
        most_similar_tag = None
        overlapping_terms: list[str] = []

        for existing in self._registered:
            # Compute tag similarity
            tag_sim = compute_tag_similarity(signature.concept_tag, existing.concept_tag)

            # Compute term similarity
            term_sim = compute_term_similarity(signature.key_terms, existing.key_terms)

            # Combined similarity (weighted)
            combined_sim = (tag_sim * 0.4) + (term_sim * 0.6)

            # Use embedding similarity if available
            if self.use_embeddings and signature.embedding and existing.embedding:
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(signature.embedding, existing.embedding))
                norm1 = sum(a * a for a in signature.embedding) ** 0.5
                norm2 = sum(b * b for b in existing.embedding) ** 0.5
                if norm1 > 0 and norm2 > 0:
                    embed_sim = dot_product / (norm1 * norm2)
                    combined_sim = (combined_sim * 0.5) + (embed_sim * 0.5)

            if combined_sim > best_similarity:
                best_similarity = combined_sim
                most_similar_tag = existing.concept_tag
                overlapping_terms = list(
                    set(signature.key_terms) & set(existing.key_terms)
                )

        is_novel = best_similarity < self.similarity_threshold

        return NoveltyCheckResult(
            is_novel=is_novel,
            similarity_score=best_similarity,
            similar_tag=most_similar_tag if not is_novel else None,
            similar_terms=overlapping_terms if not is_novel else [],
        )

    def create_signature(
        self,
        concept_tag: str,
        learning_target: str,
        bloom_level: str | None = None,
        domain: str | None = None,
    ) -> ConceptSignature:
        """Create a concept signature from item data.

        Args:
            concept_tag: Unique concept identifier
            learning_target: The learning target text
            bloom_level: Bloom's taxonomy level
            domain: Domain area

        Returns:
            ConceptSignature for the item
        """
        key_terms = extract_concept_terms(learning_target)

        return ConceptSignature(
            concept_tag=normalize_concept_tag(concept_tag),
            key_terms=key_terms,
            bloom_level=bloom_level,
            domain=domain,
        )

    @property
    def registered_count(self) -> int:
        """Return count of registered signatures."""
        return len(self._registered)

    @property
    def registered_tags(self) -> list[str]:
        """Return list of registered concept tags."""
        return [sig.concept_tag for sig in self._registered]


def deduplicate_items(
    items: list[dict],
    similarity_threshold: float = 0.85,
) -> tuple[list[dict], list[dict]]:
    """Deduplicate a list of item dictionaries by concept.

    Args:
        items: List of item dicts with 'concept_tag' and 'learning_target'
        similarity_threshold: Max similarity for items to be considered unique

    Returns:
        (unique_items, duplicate_items)
    """
    gate = NoveltyGate(similarity_threshold=similarity_threshold)

    unique = []
    duplicates = []

    for item in items:
        concept_tag = item.get("concept_tag", "")
        learning_target = item.get("learning_target", "")

        if not concept_tag:
            # Generate tag from learning target
            concept_tag = normalize_concept_tag(learning_target[:50])

        signature = gate.create_signature(
            concept_tag=concept_tag,
            learning_target=learning_target,
            bloom_level=item.get("bloom_level"),
            domain=item.get("domain"),
        )

        result = gate.check_novelty(signature)

        if result.is_novel:
            gate.register(signature)
            unique.append(item)
        else:
            item["duplicate_of"] = result.similar_tag
            item["similarity_score"] = result.similarity_score
            duplicates.append(item)

    return unique, duplicates
