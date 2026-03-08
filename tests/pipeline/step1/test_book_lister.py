import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from pipeline.base import PipelineContext
from pipeline.step1.book_lister import BookListerStep


SAMPLE_BOOKS_JSON = json.dumps([
    {"title": "AWS Certified Solutions Architect Study Guide", "author": "Ben Piper", "edition": "3rd", "year": "2023", "isbn": "978-1119982623"},
    {"title": "AWS in Action", "author": "Michael Wittig", "edition": None, "year": "2023", "isbn": None},
])


def _make_mock_response(text: str):
    """Create a mock Gemini response."""
    return SimpleNamespace(text=text)


class TestBookListerStep:
    def _run_with_mock(self, response_text: str) -> PipelineContext:
        step = BookListerStep()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_mock_response(response_text)

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("pipeline.step1.book_lister.genai.Client", return_value=mock_client),
        ):
            ctx = PipelineContext(certification_name="AWS SAA-C03")
            return step.run(ctx)

    def test_parses_plain_json(self):
        ctx = self._run_with_mock(SAMPLE_BOOKS_JSON)
        books = ctx["books_requested"]
        assert len(books) == 2
        assert books[0].title == "AWS Certified Solutions Architect Study Guide"
        assert books[0].author == "Ben Piper"
        assert books[0].edition == "3rd"

    def test_parses_json_with_code_fences(self):
        fenced = f"```json\n{SAMPLE_BOOKS_JSON}\n```"
        ctx = self._run_with_mock(fenced)
        assert len(ctx["books_requested"]) == 2

    def test_parses_json_with_bare_code_fences(self):
        fenced = f"```\n{SAMPLE_BOOKS_JSON}\n```"
        ctx = self._run_with_mock(fenced)
        assert len(ctx["books_requested"]) == 2

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            self._run_with_mock("not json at all")

    def test_passes_certification_in_prompt(self):
        step = BookListerStep()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_mock_response(SAMPLE_BOOKS_JSON)

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("pipeline.step1.book_lister.genai.Client", return_value=mock_client),
        ):
            ctx = PipelineContext(certification_name="CKA Kubernetes")
            step.run(ctx)

        call_kwargs = mock_client.models.generate_content.call_args
        prompt_content = call_kwargs.kwargs["contents"]
        assert "CKA Kubernetes" in prompt_content
