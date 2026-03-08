"""Tests for quality metrics module."""

import pytest

from course_builder.domain.content import CapsuleItem, ItemSourceReference, QuoteVerification
from course_builder.metrics.quality import (
    ItemQualityRecord,
    QualityMetrics,
    QualityTracker,
    compute_capsule_metrics,
    compute_item_metrics,
)


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_compute_ratios(self):
        metrics = QualityMetrics(
            total_items=100,
            items_with_evidence=95,
            total_quotes=50,
            verified_quotes=45,
            items_checked_ambiguity=100,
            items_clear_answer=98,
            items_checked_novelty=100,
            items_novel=95,
            items_duplicate=5,
            items_passed_first=80,
            items_repaired=15,
            items_rejected=5,
        )

        metrics.compute_ratios()

        assert metrics.evidence_coverage_ratio == 0.95
        assert metrics.quote_verification_rate == 0.90
        assert metrics.single_answer_rate == 0.98
        assert metrics.duplicate_concept_rate == 0.05
        assert metrics.repair_loop_rate == 0.15
        assert metrics.rejection_rate == 0.05

    def test_compute_ratios_with_zeros(self):
        metrics = QualityMetrics(
            total_items=0,
            total_quotes=0,
            items_checked_ambiguity=0,
            items_checked_novelty=0,
        )

        metrics.compute_ratios()

        # Should not raise division by zero
        assert metrics.evidence_coverage_ratio == 0.0
        assert metrics.quote_verification_rate == 0.0

    def test_to_dict(self):
        metrics = QualityMetrics(
            total_items=10,
            evidence_coverage_ratio=0.95,
        )

        d = metrics.to_dict()

        assert "total_items" in d
        assert "evidence_coverage_ratio" in d
        assert d["total_items"] == 10

    def test_summary(self):
        metrics = QualityMetrics(
            total_items=100,
            total_capsules=10,
            evidence_coverage_ratio=0.95,
        )

        summary = metrics.summary()

        assert "Quality Metrics Summary" in summary
        assert "Total Items: 100" in summary


class TestItemQualityRecord:
    """Tests for ItemQualityRecord dataclass."""

    def test_creation(self):
        record = ItemQualityRecord(
            item_id="item_01",
            capsule_id="capsule_01",
            has_evidence=True,
            evidence_chunk_count=5,
            quote_count=3,
            verified_quote_count=2,
        )

        assert record.item_id == "item_01"
        assert record.has_evidence
        assert record.verified_quote_count == 2

    def test_defaults(self):
        record = ItemQualityRecord(
            item_id="item_01",
            capsule_id="capsule_01",
        )

        assert not record.has_evidence
        assert record.passed_first
        assert not record.was_repaired
        assert record.is_novel


class TestQualityTracker:
    """Tests for QualityTracker class."""

    def test_add_records(self):
        tracker = QualityTracker()

        record1 = tracker.create_record("item_01", "capsule_01")
        record1.has_evidence = True
        tracker.add_record(record1)

        record2 = tracker.create_record("item_02", "capsule_01")
        record2.has_evidence = True
        tracker.add_record(record2)

        assert tracker.record_count == 2

    def test_get_metrics(self):
        tracker = QualityTracker()

        for i in range(10):
            record = tracker.create_record(f"item_{i}", "capsule_01")
            record.has_evidence = i < 9  # 9 out of 10 have evidence
            record.quote_count = 3
            record.verified_quote_count = 2 if i < 8 else 1
            tracker.add_record(record)

        metrics = tracker.get_metrics()

        assert metrics.total_items == 10
        assert metrics.total_capsules == 1
        assert metrics.items_with_evidence == 9
        assert metrics.evidence_coverage_ratio == 0.9

    def test_clear(self):
        tracker = QualityTracker()

        record = tracker.create_record("item_01", "capsule_01")
        tracker.add_record(record)

        assert tracker.record_count == 1

        tracker.clear()

        assert tracker.record_count == 0


class TestComputeItemMetrics:
    """Tests for compute_item_metrics function."""

    def test_with_capsule_item(self):
        source_ref = ItemSourceReference(
            summary="Test summary",
            chunk_ids=["chunk_1", "chunk_2"],
            quotes_verified=[
                QuoteVerification(
                    quote_text="Test quote",
                    found=True,
                    chunk_id="chunk_1",
                    match_type="exact",
                    match_confidence=1.0,
                ),
                QuoteVerification(
                    quote_text="Another quote",
                    found=False,
                    match_type="none",
                    match_confidence=0.0,
                ),
            ],
        )

        item = CapsuleItem(
            item_id="item_01",
            item_type="Multiple Choice",
            title="Test",
            learning_target="Test target",
            difficulty="intermediate",
            generation_status="success",
            source_reference=source_ref,
        )

        record = compute_item_metrics(item)

        assert record.item_id == "item_01"
        assert record.has_evidence
        assert record.evidence_chunk_count == 2
        assert record.quote_count == 2
        assert record.verified_quote_count == 1

    def test_with_dict(self):
        item = {
            "item_id": "item_01",
            "difficulty": "beginner",
            "generation_status": "success",
            "source_reference": {
                "chunk_ids": ["chunk_1"],
                "quotes_verified": [],
            },
        }

        record = compute_item_metrics(item)

        assert record.item_id == "item_01"
        assert record.difficulty == "beginner"

    def test_without_source_reference(self):
        item = CapsuleItem(
            item_id="item_01",
            item_type="Multiple Choice",
            title="Test",
            learning_target="Test target",
        )

        record = compute_item_metrics(item)

        assert not record.has_evidence
        assert record.quote_count == 0


class TestComputeCapsuleMetrics:
    """Tests for compute_capsule_metrics function."""

    def test_computes_capsule_metrics(self):
        items = [
            CapsuleItem(
                item_id=f"item_{i}",
                item_type="Multiple Choice",
                title=f"Test {i}",
                learning_target="Test target",
                difficulty="intermediate" if i % 2 == 0 else "beginner",
                source_reference=ItemSourceReference(
                    summary="Test",
                    chunk_ids=["chunk_1"],
                ),
            )
            for i in range(5)
        ]

        metrics = compute_capsule_metrics(items, "capsule_01")

        assert metrics.total_items == 5
        assert metrics.total_capsules == 1
        assert metrics.items_with_evidence == 5  # All have source_reference

    def test_with_verification_results(self):
        from course_builder.pipeline.content.verification import (
            VerificationResult,
            VerificationStatus,
        )

        items = [
            CapsuleItem(
                item_id=f"item_{i}",
                item_type="Multiple Choice",
                title=f"Test {i}",
                learning_target="Test target",
            )
            for i in range(3)
        ]

        verification_results = [
            VerificationResult(status=VerificationStatus.PASS),
            VerificationResult(status=VerificationStatus.REPAIR),
            VerificationResult(status=VerificationStatus.REJECT),
        ]

        metrics = compute_capsule_metrics(
            items, "capsule_01", verification_results=verification_results
        )

        assert metrics.items_passed_first == 1
        assert metrics.items_repaired == 1
        assert metrics.items_rejected == 1

    def test_with_novelty_results(self):
        from course_builder.pipeline.content.novelty import NoveltyCheckResult

        items = [
            CapsuleItem(
                item_id=f"item_{i}",
                item_type="Multiple Choice",
                title=f"Test {i}",
                learning_target="Test target",
            )
            for i in range(3)
        ]

        novelty_results = [
            NoveltyCheckResult(is_novel=True, similarity_score=0.2),
            NoveltyCheckResult(is_novel=True, similarity_score=0.3),
            NoveltyCheckResult(is_novel=False, similarity_score=0.9, similar_tag="dup"),
        ]

        metrics = compute_capsule_metrics(
            items, "capsule_01", novelty_results=novelty_results
        )

        assert metrics.items_checked_novelty == 3
        assert metrics.items_novel == 2
        assert metrics.items_duplicate == 1
