"""Integration tests: run the full Step3 pipeline with mocks."""

from unittest.mock import patch, MagicMock

import chromadb
import numpy as np

from pipeline.base import PipelineContext
from pipeline.models import DownloadedBook
from pipeline.step3.pipeline import build_step3_pipeline


def _make_mock_model(n):
    model = MagicMock()
    model.encode.return_value = np.random.rand(n, 384)
    return model


class TestStep3Integration:
    def test_full_pipeline(self, tmp_path):
        books = [
            DownloadedBook(
                title="AWS Guide",
                author="Ben Piper",
                extension="pdf",
                file_path=str(tmp_path / "aws.pdf"),
                success=True,
            ),
        ]

        ctx = PipelineContext(
            certification_name="AWS SAA-C03",
            books_downloaded=books,
        )

        sample_markdown = "# Chapter 1\nIntro text\n# Chapter 2\nAdvanced topics"
        client = chromadb.Client()

        pipeline = build_step3_pipeline(
            extraction_dir=str(tmp_path / "extracted"),
            chroma_client=client,
        )

        mock_model = _make_mock_model(2)

        with (
            patch(
                "pipeline.step3.extract._extract_pdf_with_mineru",
                return_value=(sample_markdown, []),
            ),
            patch.object(
                pipeline.steps[1], "_get_embedding_model", return_value=mock_model
            ),
        ):
            result = pipeline.run(ctx)

        assert result["chunks_stored"] == 2
        assert result["collection_name"] == "AWS_SAA-C03"
        assert len(result["extracted_chunks"]) == 2
        assert len(result["processed_books"]) == 1
        assert result["processed_books"][0].success is True

    def test_pipeline_no_pdfs(self, tmp_path):
        books = [
            DownloadedBook(
                title="EPUB Book",
                author="Author",
                extension="epub",
                file_path=str(tmp_path / "book.epub"),
                success=True,
            ),
        ]

        ctx = PipelineContext(
            certification_name="Test Cert",
            books_downloaded=books,
        )

        client = chromadb.Client()
        pipeline = build_step3_pipeline(
            extraction_dir=str(tmp_path / "extracted"),
            chroma_client=client,
        )

        result = pipeline.run(ctx)

        assert result["chunks_stored"] == 0
        assert result["extracted_chunks"] == []
        assert len(result["processed_books"]) == 1
        assert result["processed_books"][0].success is False

    def test_context_keys(self, tmp_path):
        ctx = PipelineContext(
            certification_name="Key Check",
            books_downloaded=[],
        )

        client = chromadb.Client()
        pipeline = build_step3_pipeline(
            extraction_dir=str(tmp_path / "extracted"),
            chroma_client=client,
        )

        result = pipeline.run(ctx)

        assert "extracted_chunks" in result
        assert "processed_books" in result
        assert "collection_name" in result
        assert "vectorstore_path" in result
        assert "chunks_stored" in result
        assert "certification_name" in result

    def test_mixed_pdf_and_non_pdf(self, tmp_path):
        books = [
            DownloadedBook(
                title="PDF Book",
                author="A",
                extension="pdf",
                file_path=str(tmp_path / "book.pdf"),
                success=True,
            ),
            DownloadedBook(
                title="EPUB Book",
                author="B",
                extension="epub",
                file_path=str(tmp_path / "book.epub"),
                success=True,
            ),
        ]

        ctx = PipelineContext(
            certification_name="Mixed",
            books_downloaded=books,
        )

        client = chromadb.Client()
        pipeline = build_step3_pipeline(
            extraction_dir=str(tmp_path / "extracted"),
            chroma_client=client,
        )

        mock_model = _make_mock_model(1)

        with (
            patch(
                "pipeline.step3.extract._extract_pdf_with_mineru",
                return_value=("# Intro\nContent", []),
            ),
            patch.object(
                pipeline.steps[1], "_get_embedding_model", return_value=mock_model
            ),
        ):
            result = pipeline.run(ctx)

        assert result["chunks_stored"] == 1
        assert len(result["processed_books"]) == 2
        successful = [b for b in result["processed_books"] if b.success]
        failed = [b for b in result["processed_books"] if not b.success]
        assert len(successful) == 1
        assert len(failed) == 1
