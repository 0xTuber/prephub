from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from pipeline.base import PipelineContext
from pipeline.models import Book
from pipeline.step1.libgen_lookup import _search_book, LibgenLookupStep


def _make_libgen_result(title="T", author="A", year="2023", language="English",
                        pages="400", size="5 MB", extension="pdf",
                        publisher="Pub", mirrors=None,
                        resolved_download_link=None):
    """Create a fake libgen_api_enhanced Book-like object."""
    obj = SimpleNamespace(
        title=title,
        author=author,
        year=year,
        language=language,
        pages=pages,
        size=size,
        extension=extension,
        publisher=publisher,
        mirrors=mirrors or ["https://example.com/dl", ""],
        resolved_download_link=resolved_download_link,
    )
    obj.resolve_direct_download_link = MagicMock(
        side_effect=lambda: setattr(obj, "resolved_download_link",
                                     resolved_download_link or "https://cdn.example.com/get.php?md5=abc&key=123")
    )
    return obj


class TestSearchBook:
    def test_found_result(self):
        book = Book(title="AWS Guide", author="John Doe")
        searcher = MagicMock()
        searcher.search_title.return_value = [
            _make_libgen_result(title="AWS Guide", author="John Doe")
        ]

        result = _search_book(searcher, book)
        assert result is not None
        assert result.found is True
        assert result.title == "AWS Guide"
        assert result.author == "John Doe"
        assert result.extension == "pdf"

    def test_no_results(self):
        book = Book(title="Nonexistent Book", author="Nobody")
        searcher = MagicMock()
        searcher.search_title.return_value = []

        result = _search_book(searcher, book)
        assert result is None

    def test_none_results(self):
        book = Book(title="Nonexistent Book", author="Nobody")
        searcher = MagicMock()
        searcher.search_title.return_value = None

        result = _search_book(searcher, book)
        assert result is None

    def test_exception_returns_none(self):
        book = Book(title="Error Book", author="X")
        searcher = MagicMock()
        searcher.search_title.side_effect = Exception("network error")

        result = _search_book(searcher, book)
        assert result is None

    def test_author_matching_prefers_correct_author(self):
        book = Book(title="Python", author="Mark Lutz")
        searcher = MagicMock()
        searcher.search_title.return_value = [
            _make_libgen_result(title="Python Basics", author="Someone Else"),
            _make_libgen_result(title="Learning Python", author="Mark Lutz"),
        ]

        result = _search_book(searcher, book)
        assert result.author == "Mark Lutz"
        assert result.title == "Learning Python"

    def test_falls_back_to_first_result_if_no_author_match(self):
        book = Book(title="Python", author="Unknown Person")
        searcher = MagicMock()
        searcher.search_title.return_value = [
            _make_libgen_result(title="First Result", author="A"),
            _make_libgen_result(title="Second Result", author="B"),
        ]

        result = _search_book(searcher, book)
        assert result.title == "First Result"

    def test_empty_mirrors_filtered(self):
        book = Book(title="T", author="A")
        searcher = MagicMock()
        searcher.search_title.return_value = [
            _make_libgen_result(mirrors=["https://real.link", "", "", ""])
        ]

        result = _search_book(searcher, book)
        assert result.mirror_links == ["https://real.link"]

    def test_no_author_skips_author_matching(self):
        book = Book(title="T", author="")
        searcher = MagicMock()
        searcher.search_title.return_value = [
            _make_libgen_result(title="First", author="X"),
            _make_libgen_result(title="Second", author="Y"),
        ]

        result = _search_book(searcher, book)
        # Should pick the first result since no author to match
        assert result.title == "First"


class TestLibgenLookupStep:
    def test_separates_found_and_not_found(self):
        books = [
            Book(title="Found Book", author="A"),
            Book(title="Missing Book", author="B"),
            Book(title="Also Found", author="C"),
        ]

        def fake_search(query):
            if "Missing" in query:
                return []
            return [_make_libgen_result(title=query, author="X")]

        mock_searcher = MagicMock()
        mock_searcher.search_title.side_effect = fake_search

        step = LibgenLookupStep(delay=0)
        with patch("pipeline.step1.libgen_lookup.LibgenSearch", return_value=mock_searcher):
            ctx = PipelineContext(books_requested=books)
            result = step.run(ctx)

        assert len(result["books_found"]) == 2
        assert len(result["books_not_found"]) == 1
        assert result["books_not_found"][0].title == "Missing Book"

    def test_empty_book_list(self):
        step = LibgenLookupStep(delay=0)
        mock_searcher = MagicMock()
        with patch("pipeline.step1.libgen_lookup.LibgenSearch", return_value=mock_searcher):
            ctx = PipelineContext(books_requested=[])
            result = step.run(ctx)

        assert result["books_found"] == []
        assert result["books_not_found"] == []
        mock_searcher.search_title.assert_not_called()


class TestDirectDownloadLinkResolution:
    def test_resolved_link_stored(self):
        book = Book(title="AWS Guide", author="John Doe")
        searcher = MagicMock()
        searcher.search_title.return_value = [
            _make_libgen_result(title="AWS Guide", author="John Doe")
        ]

        result = _search_book(searcher, book)
        assert result is not None
        assert result.direct_download_link == "https://cdn.example.com/get.php?md5=abc&key=123"

    def test_resolution_failure_does_not_break_search(self):
        book = Book(title="Fragile Book", author="Author")
        fake = _make_libgen_result(title="Fragile Book", author="Author")
        fake.resolve_direct_download_link = MagicMock(side_effect=Exception("network error"))

        searcher = MagicMock()
        searcher.search_title.return_value = [fake]

        result = _search_book(searcher, book)
        assert result is not None
        assert result.found is True
        assert result.direct_download_link is None

    def test_resolved_link_is_none_when_not_available(self):
        book = Book(title="No Link Book", author="Author")
        fake = _make_libgen_result(title="No Link Book", author="Author",
                                    resolved_download_link=None)
        fake.resolve_direct_download_link = MagicMock(
            side_effect=lambda: setattr(fake, "resolved_download_link", None)
        )

        searcher = MagicMock()
        searcher.search_title.return_value = [fake]

        result = _search_book(searcher, book)
        assert result is not None
        assert result.direct_download_link is None
