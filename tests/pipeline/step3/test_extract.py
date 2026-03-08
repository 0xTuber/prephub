from unittest.mock import patch, MagicMock

import pytest

from pipeline.base import PipelineContext
from pipeline.models import DownloadedBook, ExtractedChunk
from pipeline.step3.extract import ContentExtractionStep, _chunk_by_headings


class TestChunkByHeadings:
    def test_splits_on_headings(self):
        markdown = "# Heading 1\nSome text\n# Heading 2\nMore text"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")

        assert len(chunks) == 2
        assert chunks[0].section_heading == "Heading 1"
        assert chunks[0].text == "Some text"
        assert chunks[1].section_heading == "Heading 2"
        assert chunks[1].text == "More text"

    def test_text_before_first_heading(self):
        markdown = "Intro text\n# Chapter 1\nContent"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")

        assert len(chunks) == 2
        assert chunks[0].section_heading is None
        assert chunks[0].text == "Intro text"
        assert chunks[1].section_heading == "Chapter 1"

    def test_image_refs_extracted(self):
        markdown = "# Images\nSome text\n![alt](img/photo.png)\nMore ![](img/diagram.jpg)"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")

        assert len(chunks) == 1
        assert chunks[0].image_paths == ["img/photo.png", "img/diagram.jpg"]

    def test_empty_markdown(self):
        chunks = _chunk_by_headings("", [], "Book", "Author", "file.pdf")
        assert chunks == []

    def test_metadata_carried(self):
        markdown = "# Test\nContent"
        chunks = _chunk_by_headings(
            markdown, [], "My Book", "My Author", "path/to/file.pdf"
        )

        assert chunks[0].book_title == "My Book"
        assert chunks[0].book_author == "My Author"
        assert chunks[0].source_file == "path/to/file.pdf"

    def test_page_numbers_from_content_list(self):
        markdown = "# Introduction\nSome text"
        content_list = [
            {"text": "Introduction", "page_idx": 0},
            {"text": "Introduction", "page_idx": 1},
        ]
        chunks = _chunk_by_headings(
            markdown, content_list, "Book", "Author", "file.pdf"
        )

        assert chunks[0].page_numbers == [0, 1]

    def test_heading_only_no_content(self):
        markdown = "# Empty Heading"
        chunks = _chunk_by_headings(markdown, [], "Book", "Author", "file.pdf")
        assert chunks == []


class TestContentExtractionStep:
    def _make_context(self, books, cert_name="Test Cert"):
        return PipelineContext(
            certification_name=cert_name,
            books_downloaded=books,
        )

    def test_extracts_pdf_books(self, tmp_path):
        book = DownloadedBook(
            title="My PDF",
            author="Author",
            extension="pdf",
            file_path=str(tmp_path / "book.pdf"),
            success=True,
        )
        ctx = self._make_context([book])

        sample_markdown = "# Chapter 1\nHello world\n# Chapter 2\nGoodbye"
        sample_content = []

        step = ContentExtractionStep(extraction_dir=str(tmp_path / "extracted"))
        with patch(
            "pipeline.step3.extract._extract_pdf_with_mineru",
            return_value=(sample_markdown, sample_content),
        ):
            result = step.run(ctx)

        assert len(result["extracted_chunks"]) == 2
        assert len(result["processed_books"]) == 1
        assert result["processed_books"][0].success is True
        assert result["processed_books"][0].title == "My PDF"

    def test_skips_non_pdf(self, tmp_path):
        book = DownloadedBook(
            title="My EPUB",
            author="Author",
            extension="epub",
            file_path=str(tmp_path / "book.epub"),
            success=True,
        )
        ctx = self._make_context([book])

        step = ContentExtractionStep(extraction_dir=str(tmp_path / "extracted"))
        result = step.run(ctx)

        assert len(result["extracted_chunks"]) == 0
        assert len(result["processed_books"]) == 1
        assert result["processed_books"][0].success is False
        assert result["processed_books"][0].error == "Unsupported format"

    def test_handles_extraction_failure(self, tmp_path):
        book = DownloadedBook(
            title="Bad PDF",
            author="Author",
            extension="pdf",
            file_path=str(tmp_path / "bad.pdf"),
            success=True,
        )
        ctx = self._make_context([book])

        step = ContentExtractionStep(extraction_dir=str(tmp_path / "extracted"))
        with patch(
            "pipeline.step3.extract._extract_pdf_with_mineru",
            side_effect=RuntimeError("MinerU crashed"),
        ):
            result = step.run(ctx)

        assert len(result["extracted_chunks"]) == 0
        assert len(result["processed_books"]) == 1
        assert result["processed_books"][0].success is False
        assert "MinerU crashed" in result["processed_books"][0].error

    def test_empty_books_list(self, tmp_path):
        ctx = self._make_context([])

        step = ContentExtractionStep(extraction_dir=str(tmp_path / "extracted"))
        result = step.run(ctx)

        assert result["extracted_chunks"] == []
        assert result["processed_books"] == []

    def test_multiple_pdf_books(self, tmp_path):
        books = [
            DownloadedBook(
                title=f"Book {i}",
                author="Author",
                extension="pdf",
                file_path=str(tmp_path / f"book{i}.pdf"),
                success=True,
            )
            for i in range(3)
        ]
        ctx = self._make_context(books)

        step = ContentExtractionStep(extraction_dir=str(tmp_path / "extracted"))
        with patch(
            "pipeline.step3.extract._extract_pdf_with_mineru",
            return_value=("# Heading\nContent", []),
        ):
            result = step.run(ctx)

        assert len(result["extracted_chunks"]) == 3
        assert len(result["processed_books"]) == 3
        assert all(b.success for b in result["processed_books"])
