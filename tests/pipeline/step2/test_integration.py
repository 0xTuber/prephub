"""Integration tests: run the full Step2 pipeline with mocked HTTP responses."""

from unittest.mock import patch, MagicMock

from pipeline.base import PipelineContext
from pipeline.models import LibgenResult
from pipeline.step2.pipeline import build_step2_pipeline


def _make_mock_response(content=b"file data"):
    resp = MagicMock()
    resp.iter_content.return_value = [content]
    resp.raise_for_status = MagicMock()
    return resp


class TestStep2Integration:
    def test_full_pipeline(self, tmp_path):
        books = [
            LibgenResult(
                title="AWS Guide",
                author="Ben Piper",
                found=True,
                extension="pdf",
                direct_download_link="https://example.com/aws.pdf",
                mirror_links=["https://mirror.com/aws"],
            ),
            LibgenResult(
                title="Python Handbook",
                author="Mark Lutz",
                found=True,
                extension="epub",
                direct_download_link="https://example.com/python.epub",
                mirror_links=[],
            ),
        ]

        ctx = PipelineContext(
            certification_name="AWS SAA-C03",
            books_found=books,
        )

        pipeline = build_step2_pipeline()
        pipeline.steps[0].output_dir = str(tmp_path)

        with patch(
            "pipeline.step2.download.requests.get",
            return_value=_make_mock_response(),
        ):
            result = pipeline.run(ctx)

        assert len(result["books_downloaded"]) == 2
        assert len(result["books_failed"]) == 0
        assert (tmp_path / "AWS SAA-C03" / "AWS Guide.pdf").exists()
        assert (tmp_path / "AWS SAA-C03" / "Python Handbook.epub").exists()

    def test_pipeline_with_mixed_results(self, tmp_path):
        books = [
            LibgenResult(
                title="Good Book",
                author="A",
                found=True,
                extension="pdf",
                direct_download_link="https://example.com/good.pdf",
            ),
            LibgenResult(
                title="Bad Book",
                author="B",
                found=True,
                extension="pdf",
                direct_download_link=None,
                mirror_links=[],
            ),
        ]

        ctx = PipelineContext(
            certification_name="Test Cert",
            books_found=books,
        )

        pipeline = build_step2_pipeline()
        pipeline.steps[0].output_dir = str(tmp_path)

        with patch(
            "pipeline.step2.download.requests.get",
            return_value=_make_mock_response(),
        ):
            result = pipeline.run(ctx)

        assert len(result["books_downloaded"]) == 1
        assert len(result["books_failed"]) == 1
        assert result["books_downloaded"][0].title == "Good Book"
        assert result["books_failed"][0].title == "Bad Book"

    def test_pipeline_with_empty_books(self, tmp_path):
        ctx = PipelineContext(
            certification_name="Empty Cert",
            books_found=[],
        )

        pipeline = build_step2_pipeline()
        pipeline.steps[0].output_dir = str(tmp_path)

        result = pipeline.run(ctx)

        assert result["books_downloaded"] == []
        assert result["books_failed"] == []

    def test_context_keys(self, tmp_path):
        ctx = PipelineContext(
            certification_name="Key Check",
            books_found=[],
        )

        pipeline = build_step2_pipeline()
        pipeline.steps[0].output_dir = str(tmp_path)

        result = pipeline.run(ctx)

        assert "books_downloaded" in result
        assert "books_failed" in result
        assert "certification_name" in result
