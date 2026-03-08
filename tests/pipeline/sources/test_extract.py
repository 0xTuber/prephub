"""Tests for content extraction, specifically _chunk_by_headings."""

import pytest

from course_builder.pipeline.sources.extract import _chunk_by_headings


class TestChunkByHeadings:
    """Tests for _chunk_by_headings function."""

    def test_splits_on_headings(self):
        """Basic test: splits markdown on headings."""
        markdown = "# Heading 1\nSome text\n# Heading 2\nMore text"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")

        assert len(chunks) == 2
        assert chunks[0].section_heading == "Heading 1"
        assert chunks[0].text == "Some text"
        assert chunks[1].section_heading == "Heading 2"
        assert chunks[1].text == "More text"

    def test_text_before_first_heading(self):
        """Text before first heading should have section_heading=None."""
        markdown = "Intro text\n# Chapter 1\nContent"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")

        assert len(chunks) == 2
        assert chunks[0].section_heading is None
        assert chunks[0].text == "Intro text"
        assert chunks[1].section_heading == "Chapter 1"

    def test_image_refs_extracted(self):
        """Image references should be extracted into image_paths."""
        markdown = "# Images\nSome text\n![alt](img/photo.png)\nMore ![](img/diagram.jpg)"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")

        assert len(chunks) == 1
        assert chunks[0].image_paths == ["img/photo.png", "img/diagram.jpg"]

    def test_empty_markdown(self):
        """Empty markdown should return empty list."""
        chunks = _chunk_by_headings("", [], "Book", "Author", "file.pdf")
        assert chunks == []

    def test_metadata_carried(self):
        """Book metadata should be carried to chunks."""
        markdown = "# Test\nContent"
        chunks = _chunk_by_headings(
            markdown, [], "My Book", "My Author", "path/to/file.pdf"
        )

        assert chunks[0].book_title == "My Book"
        assert chunks[0].book_author == "My Author"
        assert chunks[0].source_file == "path/to/file.pdf"

    def test_heading_only_no_content(self):
        """Heading without content should be skipped."""
        markdown = "# Empty Heading"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")
        assert chunks == []


class TestChunkByHeadingsPageNumbers:
    """Tests for page number extraction - sequential matching, not aggregation."""

    def test_page_numbers_from_content_list_single_occurrence(self):
        """Single heading occurrence should get its pages."""
        markdown = "# Introduction\nSome text here"
        content_list = [
            {"text": "Introduction", "page_idx": 5},
            {"text": "Some text here", "page_idx": 5},
            {"text": "More content", "page_idx": 6},
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert len(chunks) == 1
        assert chunks[0].section_heading == "Introduction"
        assert chunks[0].page_numbers == [5, 6]

    def test_same_heading_text_not_aggregated_across_book(self):
        """Same heading appearing multiple times should NOT aggregate pages.

        This is the key fix: 'Words of Wisdom' on pages 5, 50, 150 should
        result in each chunk getting only its own section's pages.
        """
        markdown = "# Words of Wisdom\nFirst wisdom\n# Other Section\nOther content\n# Words of Wisdom\nSecond wisdom"
        content_list = [
            {"text": "Words of Wisdom", "page_idx": 5},
            {"text": "First wisdom", "page_idx": 5},
            {"text": "First wisdom continued", "page_idx": 6},
            {"text": "Other Section", "page_idx": 10},
            {"text": "Other content", "page_idx": 10},
            {"text": "Words of Wisdom", "page_idx": 50},
            {"text": "Second wisdom", "page_idx": 50},
            {"text": "Second wisdom continued", "page_idx": 51},
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert len(chunks) == 3

        # First "Words of Wisdom" should only have pages 5-6
        assert chunks[0].section_heading == "Words of Wisdom"
        assert chunks[0].page_numbers == [5, 6]

        # "Other Section" should have page 10
        assert chunks[1].section_heading == "Other Section"
        assert chunks[1].page_numbers == [10]

        # Second "Words of Wisdom" should only have pages 50-51
        assert chunks[2].section_heading == "Words of Wisdom"
        assert chunks[2].page_numbers == [50, 51]

    def test_repeating_heading_throughout_book(self):
        """Heading that repeats many times (like chapter markers) should be tracked per occurrence."""
        markdown = "# Summary\nChapter 1 summary\n# Content\nChapter 1 content\n# Summary\nChapter 2 summary\n# Content\nChapter 2 content\n# Summary\nChapter 3 summary"
        content_list = [
            {"text": "Summary", "page_idx": 10},
            {"text": "Chapter 1 summary", "page_idx": 10},
            {"text": "Content", "page_idx": 11},
            {"text": "Chapter 1 content", "page_idx": 11},
            {"text": "Chapter 1 content continued", "page_idx": 12},
            {"text": "Summary", "page_idx": 50},
            {"text": "Chapter 2 summary", "page_idx": 50},
            {"text": "Content", "page_idx": 51},
            {"text": "Chapter 2 content", "page_idx": 51},
            {"text": "Summary", "page_idx": 100},
            {"text": "Chapter 3 summary", "page_idx": 100},
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert len(chunks) == 5

        # First Summary (pages 10)
        assert chunks[0].section_heading == "Summary"
        assert chunks[0].page_numbers == [10]

        # First Content (pages 11-12)
        assert chunks[1].section_heading == "Content"
        assert chunks[1].page_numbers == [11, 12]

        # Second Summary (page 50)
        assert chunks[2].section_heading == "Summary"
        assert chunks[2].page_numbers == [50]

        # Second Content (page 51)
        assert chunks[3].section_heading == "Content"
        assert chunks[3].page_numbers == [51]

        # Third Summary (page 100)
        assert chunks[4].section_heading == "Summary"
        assert chunks[4].page_numbers == [100]

    def test_empty_content_list(self):
        """Empty content list should result in empty page numbers."""
        markdown = "# Heading\nContent"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")

        assert len(chunks) == 1
        assert chunks[0].page_numbers == []

    def test_content_list_without_matching_headings(self):
        """Content list without matching heading text should result in empty pages."""
        markdown = "# My Heading\nContent"
        content_list = [
            {"text": "Different text", "page_idx": 1},
            {"text": "Other text", "page_idx": 2},
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert len(chunks) == 1
        assert chunks[0].page_numbers == []

    def test_multiple_pages_within_section(self):
        """Section spanning multiple pages should collect all pages."""
        markdown = "# Long Chapter\nLong content here"
        content_list = [
            {"text": "Long Chapter", "page_idx": 10},
            {"text": "Long content here", "page_idx": 10},
            {"text": "More content", "page_idx": 11},
            {"text": "Even more", "page_idx": 12},
            {"text": "Final part", "page_idx": 13},
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert len(chunks) == 1
        assert chunks[0].page_numbers == [10, 11, 12, 13]

    def test_pages_sorted_and_deduplicated(self):
        """Pages should be sorted and deduplicated."""
        markdown = "# Chapter\nContent"
        content_list = [
            {"text": "Chapter", "page_idx": 5},
            {"text": "Content", "page_idx": 5},  # duplicate
            {"text": "More", "page_idx": 7},
            {"text": "Text", "page_idx": 6},  # out of order
            {"text": "Final", "page_idx": 7},  # duplicate
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert chunks[0].page_numbers == [5, 6, 7]

    def test_intro_section_pages_before_first_heading(self):
        """Content before first heading (section_heading=None) should get its pages."""
        markdown = "Intro content\n# First Chapter\nChapter content"
        content_list = [
            {"text": "Intro content", "page_idx": 1},
            {"text": "More intro", "page_idx": 2},
            {"text": "First Chapter", "page_idx": 3},
            {"text": "Chapter content", "page_idx": 3},
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert len(chunks) == 2
        # Intro section
        assert chunks[0].section_heading is None
        assert chunks[0].page_numbers == [1, 2]
        # First chapter
        assert chunks[1].section_heading == "First Chapter"
        assert chunks[1].page_numbers == [3]

    def test_content_list_with_non_dict_entries(self):
        """Non-dict entries in content_list should be skipped gracefully."""
        markdown = "# Heading\nContent"
        content_list = [
            {"text": "Heading", "page_idx": 1},
            None,
            "string entry",
            {"text": "Content", "page_idx": 1},
            123,
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert len(chunks) == 1
        assert chunks[0].page_numbers == [1]

    def test_content_list_entry_missing_page_idx(self):
        """Entries without page_idx should be handled gracefully."""
        markdown = "# Heading\nContent"
        content_list = [
            {"text": "Heading", "page_idx": 5},
            {"text": "Content"},  # missing page_idx
            {"text": "More", "page_idx": 6},
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert chunks[0].page_numbers == [5, 6]
