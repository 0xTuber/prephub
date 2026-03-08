"""Quality metrics for content generation pipeline.

This module provides metrics tracking for quality gates:
- Evidence coverage: Items with valid source evidence
- Quote verification: Quotes found in source chunks
- Single answer rate: Items passing ambiguity gate
- Duplicate rate: Items flagged as duplicates
- Repair rate: Items needing repair loop
- Rejection rate: Items fully rejected

Usage:
    tracker = QualityTracker()

    # Track each item
    tracker.record_item(item, verification_result, novelty_result)

    # Get metrics
    metrics = tracker.get_metrics()
    print(f"Evidence coverage: {metrics.evidence_coverage_ratio:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class QualityMetrics:
    """Quality metrics for a batch of generated items."""

    # Total counts
    total_items: int = 0
    total_capsules: int = 0

    # Evidence metrics
    items_with_evidence: int = 0
    evidence_coverage_ratio: float = 0.0  # items_with_evidence / total_items

    # Quote verification
    total_quotes: int = 0
    verified_quotes: int = 0
    quote_verification_rate: float = 0.0  # verified_quotes / total_quotes

    # Answerability
    items_checked_ambiguity: int = 0
    items_clear_answer: int = 0
    single_answer_rate: float = 0.0  # items_clear_answer / items_checked_ambiguity

    # Novelty
    items_checked_novelty: int = 0
    items_novel: int = 0
    items_duplicate: int = 0
    duplicate_concept_rate: float = 0.0  # items_duplicate / items_checked_novelty

    # Repair loop
    items_passed_first: int = 0
    items_repaired: int = 0
    items_rejected: int = 0
    repair_loop_rate: float = 0.0  # items_repaired / total_items
    rejection_rate: float = 0.0  # items_rejected / total_items

    # Difficulty distribution
    difficulty_distribution: dict[str, int] = field(default_factory=dict)

    # Generation status distribution
    status_distribution: dict[str, int] = field(default_factory=dict)

    # Timestamps
    computed_at: str = ""

    def __post_init__(self):
        if not self.computed_at:
            self.computed_at = datetime.now().isoformat()

    def compute_ratios(self) -> None:
        """Compute all ratio metrics from counts."""
        if self.total_items > 0:
            self.evidence_coverage_ratio = self.items_with_evidence / self.total_items
            self.repair_loop_rate = self.items_repaired / self.total_items
            self.rejection_rate = self.items_rejected / self.total_items

        if self.total_quotes > 0:
            self.quote_verification_rate = self.verified_quotes / self.total_quotes

        if self.items_checked_ambiguity > 0:
            self.single_answer_rate = self.items_clear_answer / self.items_checked_ambiguity

        if self.items_checked_novelty > 0:
            self.duplicate_concept_rate = self.items_duplicate / self.items_checked_novelty

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_items": self.total_items,
            "total_capsules": self.total_capsules,
            "evidence_coverage_ratio": round(self.evidence_coverage_ratio, 4),
            "quote_verification_rate": round(self.quote_verification_rate, 4),
            "single_answer_rate": round(self.single_answer_rate, 4),
            "duplicate_concept_rate": round(self.duplicate_concept_rate, 4),
            "repair_loop_rate": round(self.repair_loop_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
            "difficulty_distribution": self.difficulty_distribution,
            "status_distribution": self.status_distribution,
            "computed_at": self.computed_at,
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Quality Metrics Summary",
            "=" * 40,
            f"Total Items: {self.total_items}",
            f"Total Capsules: {self.total_capsules}",
            "",
            "Coverage & Verification:",
            f"  Evidence Coverage: {self.evidence_coverage_ratio:.1%}",
            f"  Quote Verification: {self.quote_verification_rate:.1%}",
            "",
            "Quality Gates:",
            f"  Single Answer Rate: {self.single_answer_rate:.1%}",
            f"  Duplicate Rate: {self.duplicate_concept_rate:.1%}",
            "",
            "Processing:",
            f"  Passed First Try: {self.items_passed_first}",
            f"  Repaired: {self.items_repaired} ({self.repair_loop_rate:.1%})",
            f"  Rejected: {self.items_rejected} ({self.rejection_rate:.1%})",
            "",
            f"Computed: {self.computed_at}",
        ]
        return "\n".join(lines)


@dataclass
class ItemQualityRecord:
    """Quality record for a single item."""

    item_id: str
    capsule_id: str

    # Evidence
    has_evidence: bool = False
    evidence_chunk_count: int = 0

    # Quote verification
    quote_count: int = 0
    verified_quote_count: int = 0

    # Ambiguity
    checked_ambiguity: bool = False
    is_clear: bool = True

    # Novelty
    checked_novelty: bool = False
    is_novel: bool = True
    similar_to: str | None = None

    # Processing
    passed_first: bool = True
    was_repaired: bool = False
    was_rejected: bool = False

    # Metadata
    difficulty: str = "intermediate"
    status: str = "success"


class QualityTracker:
    """Tracker for accumulating quality metrics across items.

    Usage:
        tracker = QualityTracker()

        for item in items:
            record = tracker.create_record(item, capsule_id)
            # ... perform quality checks ...
            record.has_evidence = True
            record.quote_count = 3
            record.verified_quote_count = 2
            tracker.add_record(record)

        metrics = tracker.get_metrics()
    """

    def __init__(self):
        self._records: list[ItemQualityRecord] = []
        self._capsule_ids: set[str] = set()

    def create_record(self, item_id: str, capsule_id: str) -> ItemQualityRecord:
        """Create a new quality record for an item.

        Args:
            item_id: Item identifier
            capsule_id: Capsule identifier

        Returns:
            New ItemQualityRecord to be filled in
        """
        return ItemQualityRecord(item_id=item_id, capsule_id=capsule_id)

    def add_record(self, record: ItemQualityRecord) -> None:
        """Add a completed quality record.

        Args:
            record: Filled-in quality record
        """
        self._records.append(record)
        self._capsule_ids.add(record.capsule_id)

    def get_metrics(self) -> QualityMetrics:
        """Compute aggregate quality metrics.

        Returns:
            QualityMetrics with computed values
        """
        metrics = QualityMetrics()

        metrics.total_items = len(self._records)
        metrics.total_capsules = len(self._capsule_ids)

        for record in self._records:
            # Evidence
            if record.has_evidence:
                metrics.items_with_evidence += 1

            # Quotes
            metrics.total_quotes += record.quote_count
            metrics.verified_quotes += record.verified_quote_count

            # Ambiguity
            if record.checked_ambiguity:
                metrics.items_checked_ambiguity += 1
                if record.is_clear:
                    metrics.items_clear_answer += 1

            # Novelty
            if record.checked_novelty:
                metrics.items_checked_novelty += 1
                if record.is_novel:
                    metrics.items_novel += 1
                else:
                    metrics.items_duplicate += 1

            # Processing
            if record.passed_first:
                metrics.items_passed_first += 1
            if record.was_repaired:
                metrics.items_repaired += 1
            if record.was_rejected:
                metrics.items_rejected += 1

            # Distributions
            diff = record.difficulty or "unknown"
            metrics.difficulty_distribution[diff] = metrics.difficulty_distribution.get(diff, 0) + 1

            status = record.status or "unknown"
            metrics.status_distribution[status] = metrics.status_distribution.get(status, 0) + 1

        metrics.compute_ratios()
        return metrics

    def clear(self) -> None:
        """Clear all records."""
        self._records = []
        self._capsule_ids = set()

    @property
    def record_count(self) -> int:
        """Return number of records."""
        return len(self._records)


def compute_item_metrics(
    item: Any,
    verification_result: Any | None = None,
    novelty_result: Any | None = None,
) -> ItemQualityRecord:
    """Compute quality metrics for a single item.

    Args:
        item: CapsuleItem or item dict
        verification_result: Optional VerificationResult
        novelty_result: Optional NoveltyCheckResult

    Returns:
        ItemQualityRecord with computed values
    """
    # Handle both CapsuleItem and dict
    if hasattr(item, "item_id"):
        item_id = item.item_id
        difficulty = getattr(item, "difficulty", "intermediate")
        status = getattr(item, "generation_status", "success")
        source_ref = getattr(item, "source_reference", None)
    else:
        item_id = item.get("item_id", "unknown")
        difficulty = item.get("difficulty", "intermediate")
        status = item.get("generation_status", "success")
        source_ref = item.get("source_reference")

    record = ItemQualityRecord(
        item_id=item_id,
        capsule_id="",  # Set by caller
        difficulty=difficulty or "intermediate",
        status=status or "success",
    )

    # Evidence metrics
    if source_ref:
        if hasattr(source_ref, "chunk_ids"):
            record.has_evidence = len(source_ref.chunk_ids) > 0
            record.evidence_chunk_count = len(source_ref.chunk_ids)
        elif isinstance(source_ref, dict):
            record.has_evidence = len(source_ref.get("chunk_ids", [])) > 0
            record.evidence_chunk_count = len(source_ref.get("chunk_ids", []))

        # Quote verification
        if hasattr(source_ref, "quotes_verified"):
            quotes = source_ref.quotes_verified
            record.quote_count = len(quotes)
            record.verified_quote_count = sum(1 for q in quotes if q.found)
        elif isinstance(source_ref, dict):
            quotes = source_ref.get("quotes_verified", [])
            record.quote_count = len(quotes)
            record.verified_quote_count = sum(
                1 for q in quotes if q.get("found", False)
            )

    # Verification result
    if verification_result:
        from course_builder.pipeline.content.verification import VerificationStatus

        record.checked_novelty = False  # This is verification, not novelty
        if hasattr(verification_result, "status"):
            if verification_result.status == VerificationStatus.PASS:
                record.passed_first = True
            elif verification_result.status == VerificationStatus.REPAIR:
                record.was_repaired = True
                record.passed_first = False
            elif verification_result.status == VerificationStatus.REJECT:
                record.was_rejected = True
                record.passed_first = False

    # Novelty result
    if novelty_result:
        record.checked_novelty = True
        if hasattr(novelty_result, "is_novel"):
            record.is_novel = novelty_result.is_novel
            record.similar_to = getattr(novelty_result, "similar_tag", None)
        elif isinstance(novelty_result, dict):
            record.is_novel = novelty_result.get("is_novel", True)
            record.similar_to = novelty_result.get("similar_tag")

    return record


def compute_capsule_metrics(
    items: list[Any],
    capsule_id: str,
    verification_results: list[Any] | None = None,
    novelty_results: list[Any] | None = None,
) -> QualityMetrics:
    """Compute quality metrics for a capsule.

    Args:
        items: List of CapsuleItem or item dicts
        capsule_id: Capsule identifier
        verification_results: Optional list of verification results
        novelty_results: Optional list of novelty results

    Returns:
        QualityMetrics for the capsule
    """
    tracker = QualityTracker()

    for i, item in enumerate(items):
        verification = verification_results[i] if verification_results and i < len(verification_results) else None
        novelty = novelty_results[i] if novelty_results and i < len(novelty_results) else None

        record = compute_item_metrics(item, verification, novelty)
        record.capsule_id = capsule_id
        tracker.add_record(record)

    return tracker.get_metrics()
