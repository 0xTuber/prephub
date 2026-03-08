from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.base import PipelineContext
from pipeline.models import LibgenResult, DownloadedBook
from pipeline.step2.download import BookDownloadStep, _sanitize_filename


class TestSanitizeFilename:
    def test_removes_unsafe_chars(self):
        assert _sanitize_filename('My Book: A "Guide"') == "My Book_ A _Guide_"

    def test_removes_question_marks_and_pipes(self):
        assert _sanitize_filename("What? Why|How") == "What_ Why_How"

    def test_strips_dots_and_spaces(self):
        assert _sanitize_filename("  hello...") == "hello"

    def test_normal_name_unchanged(self):
        assert _sanitize_filename("Clean Title") == "Clean Title"


class TestBookDownloadStep:
    def _make_context(self, books_found, cert_name="Test Cert"):
        return PipelineContext(
            certification_name=cert_name,
            books_found=books_found,
        )

    def test_downloads_book_with_direct_link(self, tmp_path):
        book = LibgenResult(
            title="My Book",
            author="Author",
            found=True,
            extension="pdf",
            direct_download_link="https://example.com/direct.pdf",
            mirror_links=["https://example.com/mirror1"],
        )
        ctx = self._make_context([book])

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"fake pdf content"]
        mock_resp.raise_for_status = MagicMock()

        step = BookDownloadStep(output_dir=str(tmp_path))
        with patch("pipeline.step2.download.requests.get", return_value=mock_resp) as mock_get:
            result = step.run(ctx)

        mock_get.assert_called_once_with(
            "https://example.com/direct.pdf", timeout=60, stream=True
        )
        assert len(result["books_downloaded"]) == 1
        assert len(result["books_failed"]) == 0
        assert result["books_downloaded"][0].success is True

        saved_file = tmp_path / "Test Cert" / "My Book.pdf"
        assert saved_file.exists()
        assert saved_file.read_bytes() == b"fake pdf content"

    def test_falls_back_to_mirror_on_direct_link_failure(self, tmp_path):
        book = LibgenResult(
            title="Fallback Book",
            author="Author",
            found=True,
            extension="epub",
            direct_download_link="https://example.com/broken",
            mirror_links=["https://example.com/mirror1"],
        )
        ctx = self._make_context([book])

        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = Exception("404")

        ok_resp = MagicMock()
        ok_resp.iter_content.return_value = [b"mirror content"]
        ok_resp.raise_for_status = MagicMock()

        step = BookDownloadStep(output_dir=str(tmp_path))
        with patch(
            "pipeline.step2.download.requests.get",
            side_effect=[fail_resp, ok_resp],
        ):
            result = step.run(ctx)

        assert len(result["books_downloaded"]) == 1
        assert result["books_downloaded"][0].success is True

    def test_all_links_fail(self, tmp_path):
        book = LibgenResult(
            title="Bad Book",
            author="Author",
            found=True,
            extension="pdf",
            direct_download_link="https://example.com/broken",
            mirror_links=["https://example.com/also-broken"],
        )
        ctx = self._make_context([book])

        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = Exception("server error")

        step = BookDownloadStep(output_dir=str(tmp_path))
        with patch("pipeline.step2.download.requests.get", return_value=fail_resp):
            result = step.run(ctx)

        assert len(result["books_downloaded"]) == 0
        assert len(result["books_failed"]) == 1
        assert result["books_failed"][0].success is False
        assert "server error" in result["books_failed"][0].error

    def test_no_links_available(self, tmp_path):
        book = LibgenResult(
            title="No Links",
            author="Author",
            found=True,
            extension="pdf",
            direct_download_link=None,
            mirror_links=[],
        )
        ctx = self._make_context([book])

        step = BookDownloadStep(output_dir=str(tmp_path))
        result = step.run(ctx)

        assert len(result["books_downloaded"]) == 0
        assert len(result["books_failed"]) == 1
        assert result["books_failed"][0].error == "No download links available"

    def test_empty_books_list(self, tmp_path):
        ctx = self._make_context([])

        step = BookDownloadStep(output_dir=str(tmp_path))
        result = step.run(ctx)

        assert result["books_downloaded"] == []
        assert result["books_failed"] == []

    def test_creates_cert_subdirectory(self, tmp_path):
        book = LibgenResult(
            title="Dir Test",
            author="A",
            found=True,
            extension="pdf",
            direct_download_link="https://example.com/dl",
        )
        ctx = self._make_context([book], cert_name="AWS SAA-C03")

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.raise_for_status = MagicMock()

        step = BookDownloadStep(output_dir=str(tmp_path))
        with patch("pipeline.step2.download.requests.get", return_value=mock_resp):
            step.run(ctx)

        assert (tmp_path / "AWS SAA-C03").is_dir()

    def test_default_extension_is_pdf(self, tmp_path):
        book = LibgenResult(
            title="No Ext",
            author="A",
            found=True,
            extension=None,
            direct_download_link="https://example.com/dl",
        )
        ctx = self._make_context([book])

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.raise_for_status = MagicMock()

        step = BookDownloadStep(output_dir=str(tmp_path))
        with patch("pipeline.step2.download.requests.get", return_value=mock_resp):
            result = step.run(ctx)

        assert result["books_downloaded"][0].extension == "pdf"
        assert (tmp_path / "Test Cert" / "No Ext.pdf").exists()

    def test_multiple_books(self, tmp_path):
        books = [
            LibgenResult(
                title=f"Book {i}",
                author="A",
                found=True,
                extension="pdf",
                direct_download_link=f"https://example.com/dl{i}",
            )
            for i in range(3)
        ]
        ctx = self._make_context(books)

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.raise_for_status = MagicMock()

        step = BookDownloadStep(output_dir=str(tmp_path))
        with patch("pipeline.step2.download.requests.get", return_value=mock_resp):
            result = step.run(ctx)

        assert len(result["books_downloaded"]) == 3
