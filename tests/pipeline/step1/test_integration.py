"""Integration tests: run the full Step1 pipeline with mocked external services."""

import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from pipeline.base import PipelineContext
from pipeline.step1.pipeline import build_step1_pipeline


GEMINI_RESPONSE_JSON = json.dumps([
    {"title": "AWS Certified Solutions Architect Study Guide", "author": "Ben Piper", "edition": "3rd", "year": "2023", "isbn": "978-1119982623"},
    {"title": "Amazon Web Services in Action", "author": "Michael Wittig", "edition": "3rd", "year": "2023", "isbn": None},
    {"title": "Totally Obscure Book", "author": "Nobody", "edition": None, "year": None, "isbn": None},
])


def _make_libgen_result(title, author):
    obj = SimpleNamespace(
        title=title,
        author=author,
        year="2023",
        language="English",
        pages="500",
        size="10 MB",
        extension="pdf",
        publisher="Publisher",
        mirrors=["https://example.com/download"],
        resolved_download_link=None,
    )
    obj.resolve_direct_download_link = MagicMock(
        side_effect=lambda: setattr(obj, "resolved_download_link", "https://cdn.example.com/get.php?md5=abc&key=123")
    )
    return obj


LIBGEN_DB = {
    "AWS Certified Solutions Architect Study Guide": _make_libgen_result(
        "AWS Certified Solutions Architect Study Guide", "Ben Piper"
    ),
    "Amazon Web Services in Action": _make_libgen_result(
        "Amazon Web Services in Action", "Michael Wittig"
    ),
}


def fake_libgen_search(query):
    for key, val in LIBGEN_DB.items():
        if key.lower() in query.lower() or query.lower() in key.lower():
            return [val]
    return []


class TestStep1Integration:
    def test_full_pipeline(self):
        pipeline = build_step1_pipeline()

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content.return_value = SimpleNamespace(text=GEMINI_RESPONSE_JSON)

        mock_libgen_searcher = MagicMock()
        mock_libgen_searcher.search_title.side_effect = fake_libgen_search

        with (
            patch("builtins.input", return_value="AWS Solutions Architect SAA-C03"),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("pipeline.step1.book_lister.genai.Client", return_value=mock_gemini_client),
            patch("pipeline.step1.libgen_lookup.LibgenSearch", return_value=mock_libgen_searcher),
            patch("pipeline.step1.libgen_lookup.time.sleep"),
        ):
            ctx = pipeline.run()

        # Verify input step
        assert ctx["certification_name"] == "AWS Solutions Architect SAA-C03"

        # Verify book lister step
        assert len(ctx["books_requested"]) == 3

        # Verify libgen lookup step
        assert len(ctx["books_found"]) == 2
        assert len(ctx["books_not_found"]) == 1
        assert ctx["books_not_found"][0].title == "Totally Obscure Book"

        # Verify found books have metadata
        for book in ctx["books_found"]:
            assert book.found is True
            assert book.extension == "pdf"
            assert book.size == "10 MB"
            assert len(book.mirror_links) > 0
            assert book.direct_download_link is not None

    def test_pipeline_with_no_libgen_results(self):
        pipeline = build_step1_pipeline()

        mock_gemini_client = MagicMock()
        gemini_response = json.dumps([
            {"title": "Nonexistent Book", "author": "Ghost", "edition": None, "year": None, "isbn": None},
        ])
        mock_gemini_client.models.generate_content.return_value = SimpleNamespace(text=gemini_response)

        mock_libgen_searcher = MagicMock()
        mock_libgen_searcher.search_title.return_value = []

        with (
            patch("builtins.input", return_value="Fake Cert 101"),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("pipeline.step1.book_lister.genai.Client", return_value=mock_gemini_client),
            patch("pipeline.step1.libgen_lookup.LibgenSearch", return_value=mock_libgen_searcher),
            patch("pipeline.step1.libgen_lookup.time.sleep"),
        ):
            ctx = pipeline.run()

        assert len(ctx["books_found"]) == 0
        assert len(ctx["books_not_found"]) == 1

    def test_context_flows_through_all_steps(self):
        """Verify the pipeline context accumulates all expected keys."""
        pipeline = build_step1_pipeline()

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content.return_value = SimpleNamespace(text="[]")

        mock_libgen_searcher = MagicMock()

        with (
            patch("builtins.input", return_value="Test Cert"),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("pipeline.step1.book_lister.genai.Client", return_value=mock_gemini_client),
            patch("pipeline.step1.libgen_lookup.LibgenSearch", return_value=mock_libgen_searcher),
            patch("pipeline.step1.libgen_lookup.time.sleep"),
        ):
            ctx = pipeline.run()

        expected_keys = {"certification_name", "books_requested", "books_found", "books_not_found"}
        assert expected_keys == set(ctx.keys())
