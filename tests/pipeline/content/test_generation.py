"""Tests for content generation, specifically _select_optimal_images."""

import pytest

from course_builder.domain.content import ItemSourceReference, SourceChunkSnapshot
from course_builder.pipeline.content.generation import _select_optimal_images


class TestSelectOptimalImages:
    """Tests for _select_optimal_images function."""

    def test_selects_images_by_relevance(self):
        """Images from more relevant chunks (lower distance) should be selected first."""
        chunks = [
            {"distance": 0.5, "image_paths": ["img3.png"]},
            {"distance": 0.1, "image_paths": ["img1.png"]},
            {"distance": 0.3, "image_paths": ["img2.png"]},
        ]
        result = _select_optimal_images(chunks, max_images=3)
        assert result == ["img1.png", "img2.png", "img3.png"]

    def test_limits_to_max_images(self):
        """Should return at most max_images."""
        chunks = [
            {"distance": 0.1, "image_paths": ["img1.png", "img2.png"]},
            {"distance": 0.2, "image_paths": ["img3.png", "img4.png"]},
        ]
        result = _select_optimal_images(chunks, max_images=2)
        assert len(result) == 2
        assert result == ["img1.png", "img2.png"]

    def test_deduplicates_images(self):
        """Same image appearing in multiple chunks should only appear once."""
        chunks = [
            {"distance": 0.1, "image_paths": ["img1.png", "img2.png"]},
            {"distance": 0.2, "image_paths": ["img1.png", "img3.png"]},  # img1 is duplicate
        ]
        result = _select_optimal_images(chunks, max_images=3)
        assert result == ["img1.png", "img2.png", "img3.png"]

    def test_handles_chunks_without_images(self):
        """Chunks without image_paths should be skipped gracefully."""
        chunks = [
            {"distance": 0.1, "image_paths": []},
            {"distance": 0.2, "image_paths": ["img1.png"]},
            {"distance": 0.3},  # No image_paths key
        ]
        result = _select_optimal_images(chunks, max_images=3)
        assert result == ["img1.png"]

    def test_returns_empty_list_when_no_images(self):
        """Should return empty list when no chunks have images."""
        chunks = [
            {"distance": 0.1, "image_paths": []},
            {"distance": 0.2},
        ]
        result = _select_optimal_images(chunks)
        assert result == []

    def test_handles_empty_chunks_list(self):
        """Should handle empty input gracefully."""
        result = _select_optimal_images([])
        assert result == []

    def test_default_max_images_is_three(self):
        """Default max_images should be 3."""
        chunks = [
            {"distance": 0.1, "image_paths": ["img1.png", "img2.png", "img3.png", "img4.png"]},
        ]
        result = _select_optimal_images(chunks)
        assert len(result) == 3
        assert result == ["img1.png", "img2.png", "img3.png"]

    def test_handles_missing_distance(self):
        """Chunks without distance should use default 1.0."""
        chunks = [
            {"image_paths": ["img_no_dist.png"]},  # No distance
            {"distance": 0.5, "image_paths": ["img_with_dist.png"]},
        ]
        result = _select_optimal_images(chunks, max_images=2)
        # img_with_dist should come first (0.5 < 1.0)
        assert result[0] == "img_with_dist.png"
        assert result[1] == "img_no_dist.png"

    def test_preserves_order_within_chunk(self):
        """Images within the same chunk should maintain their order."""
        chunks = [
            {"distance": 0.1, "image_paths": ["first.png", "second.png", "third.png"]},
        ]
        result = _select_optimal_images(chunks, max_images=3)
        assert result == ["first.png", "second.png", "third.png"]


class TestSelectOptimalImagesWithAnnotations:
    """Tests for annotation-based image selection."""

    def test_annotation_keyword_match_prioritizes_images(self):
        """Images with descriptions matching learning target should be prioritized."""
        chunks = [
            {"distance": 0.1, "image_paths": ["generic.png"]},
            {"distance": 0.5, "image_paths": ["airway_diagram.png"]},
        ]
        annotations = {
            "generic.png": "A photograph of medical equipment",
            "airway_diagram.png": "Diagram showing airway management techniques including intubation",
        }
        result = _select_optimal_images(
            chunks,
            max_images=2,
            image_annotations=annotations,
            learning_target="airway management intubation",
        )
        # airway_diagram should come first despite higher distance
        assert result[0] == "airway_diagram.png"

    def test_falls_back_to_distance_without_annotations(self):
        """Without annotations, should fall back to distance-based selection."""
        chunks = [
            {"distance": 0.5, "image_paths": ["far.png"]},
            {"distance": 0.1, "image_paths": ["close.png"]},
        ]
        result = _select_optimal_images(
            chunks,
            max_images=2,
            image_annotations=None,
            learning_target="some target",
        )
        assert result == ["close.png", "far.png"]

    def test_falls_back_for_images_without_descriptions(self):
        """Images without descriptions should use chunk distance."""
        chunks = [
            {"distance": 0.1, "image_paths": ["no_desc.png"]},
            {"distance": 0.5, "image_paths": ["has_desc.png"]},
        ]
        annotations = {
            "has_desc.png": "Relevant content about airway management",
        }
        result = _select_optimal_images(
            chunks,
            max_images=2,
            image_annotations=annotations,
            learning_target="airway management",
        )
        # has_desc should come first due to keyword match
        assert result[0] == "has_desc.png"

    def test_case_insensitive_keyword_matching(self):
        """Keyword matching should be case insensitive."""
        chunks = [
            {"distance": 0.3, "image_paths": ["img.png"]},
        ]
        annotations = {
            "img.png": "AIRWAY Management DIAGRAM",
        }
        result = _select_optimal_images(
            chunks,
            max_images=1,
            image_annotations=annotations,
            learning_target="airway diagram",
        )
        assert result == ["img.png"]

    def test_empty_annotations_falls_back_to_distance(self):
        """Empty annotations dict should fall back to distance."""
        chunks = [
            {"distance": 0.5, "image_paths": ["far.png"]},
            {"distance": 0.1, "image_paths": ["close.png"]},
        ]
        result = _select_optimal_images(
            chunks,
            max_images=2,
            image_annotations={},
            learning_target="target",
        )
        assert result == ["close.png", "far.png"]


class TestGetImageDescription:
    """Tests for _get_image_description function."""

    def test_exact_path_match(self):
        """Should find description with exact path match."""
        from course_builder.pipeline.content.generation import _get_image_description

        annotations = {
            "images/fig1.png": "A diagram showing airway anatomy",
        }
        result = _get_image_description("images/fig1.png", annotations)
        assert result == "A diagram showing airway anatomy"

    def test_strips_images_prefix(self):
        """Should find description when images/ prefix needs stripping."""
        from course_builder.pipeline.content.generation import _get_image_description

        annotations = {
            "fig1.png": "A diagram showing airway anatomy",
        }
        result = _get_image_description("images/fig1.png", annotations)
        assert result == "A diagram showing airway anatomy"

    def test_matches_by_filename(self):
        """Should find description by matching filename only."""
        from course_builder.pipeline.content.generation import _get_image_description

        annotations = {
            "data/extracted/EMR/images/fig1.png": "A diagram showing airway anatomy",
        }
        result = _get_image_description("images/fig1.png", annotations)
        assert result == "A diagram showing airway anatomy"

    def test_returns_no_description_when_not_found(self):
        """Should return 'No description' when image not in annotations."""
        from course_builder.pipeline.content.generation import _get_image_description

        annotations = {
            "images/other.png": "Some description",
        }
        result = _get_image_description("images/missing.png", annotations)
        assert result == "No description"

    def test_returns_no_description_for_none_annotations(self):
        """Should return 'No description' when annotations is None."""
        from course_builder.pipeline.content.generation import _get_image_description

        result = _get_image_description("images/fig1.png", None)
        assert result == "No description"

    def test_returns_no_description_for_empty_annotations(self):
        """Should return 'No description' when annotations is empty."""
        from course_builder.pipeline.content.generation import _get_image_description

        result = _get_image_description("images/fig1.png", {})
        assert result == "No description"


class TestSourceChunkSnapshot:
    """Tests for SourceChunkSnapshot model."""

    def test_create_with_required_fields(self):
        """Should create snapshot with required fields."""
        snapshot = SourceChunkSnapshot(
            chunk_id="chunk_1",
            text="Some chunk text",
            book_title="Test Book",
            book_author="Test Author",
        )
        assert snapshot.chunk_id == "chunk_1"
        assert snapshot.text == "Some chunk text"
        assert snapshot.pages == []
        assert snapshot.section_heading is None
        assert snapshot.image_paths == []

    def test_create_with_all_fields(self):
        """Should create snapshot with all fields."""
        snapshot = SourceChunkSnapshot(
            chunk_id="chunk_1",
            text="Some chunk text",
            book_title="Test Book",
            book_author="Test Author",
            pages=[10, 11, 12],
            section_heading="Chapter 3",
            image_paths=["img1.png", "img2.png"],
        )
        assert snapshot.pages == [10, 11, 12]
        assert snapshot.section_heading == "Chapter 3"
        assert snapshot.image_paths == ["img1.png", "img2.png"]


class TestItemSourceReferenceNewFields:
    """Tests for new fields in ItemSourceReference."""

    def test_new_fields_have_defaults(self):
        """New fields should have empty list defaults."""
        ref = ItemSourceReference(summary="Test summary")
        assert ref.optimal_images == []
        assert ref.source_chunks == []

    def test_create_with_new_fields(self):
        """Should create reference with new fields populated."""
        chunks = [
            SourceChunkSnapshot(
                chunk_id="chunk_1",
                text="Text 1",
                book_title="Book 1",
                book_author="Author 1",
                pages=[10],
                image_paths=["fig1.png"],
            ),
            SourceChunkSnapshot(
                chunk_id="chunk_2",
                text="Text 2",
                book_title="Book 1",
                book_author="Author 1",
                pages=[11],
            ),
        ]
        ref = ItemSourceReference(
            summary="Test summary for the item",
            optimal_images=["fig1.png", "fig2.png"],
            source_chunks=chunks,
            chunk_ids=["chunk_1", "chunk_2"],
        )
        assert ref.optimal_images == ["fig1.png", "fig2.png"]
        assert len(ref.source_chunks) == 2
        assert ref.source_chunks[0].chunk_id == "chunk_1"

    def test_serializes_to_json(self):
        """Should serialize correctly to JSON with new fields."""
        chunks = [
            SourceChunkSnapshot(
                chunk_id="chunk_1",
                text="Text 1",
                book_title="Book 1",
                book_author="Author 1",
                pages=[10],
            ),
        ]
        ref = ItemSourceReference(
            summary="Test summary",
            optimal_images=["fig1.png"],
            source_chunks=chunks,
        )
        data = ref.model_dump()
        assert "optimal_images" in data
        assert "source_chunks" in data
        assert data["source_chunks"][0]["chunk_id"] == "chunk_1"
