import pytest
from pydantic import ValidationError

from course_builder.domain.books import Book, LibgenResult, Step1Output


class TestBook:
    def test_minimal_fields(self):
        book = Book(title="AWS Guide", author="John Doe")
        assert book.title == "AWS Guide"
        assert book.author == "John Doe"
        assert book.edition is None
        assert book.year is None
        assert book.isbn is None

    def test_all_fields(self):
        book = Book(
            title="AWS Guide",
            author="John Doe",
            edition="3rd",
            year="2024",
            isbn="978-1234567890",
        )
        assert book.edition == "3rd"
        assert book.year == "2024"
        assert book.isbn == "978-1234567890"

    def test_missing_required_title(self):
        with pytest.raises(ValidationError):
            Book(author="John Doe")

    def test_missing_required_author(self):
        with pytest.raises(ValidationError):
            Book(title="AWS Guide")


class TestLibgenResult:
    def test_defaults(self):
        r = LibgenResult(title="T", author="A")
        assert r.found is False
        assert r.mirror_links == []
        assert r.language is None
        assert r.pages is None
        assert r.size is None
        assert r.extension is None
        assert r.publisher is None

    def test_full(self):
        r = LibgenResult(
            title="T",
            author="A",
            found=True,
            language="English",
            pages="400",
            size="5 MB",
            extension="pdf",
            mirror_links=["https://example.com/dl"],
            publisher="O'Reilly",
        )
        assert r.found is True
        assert r.extension == "pdf"
        assert len(r.mirror_links) == 1


class TestStep1Output:
    def test_construction(self):
        book = Book(title="T", author="A")
        found = LibgenResult(title="T", author="A", found=True)
        not_found = Book(title="T2", author="B")

        output = Step1Output(
            certification_name="AWS SAA-C03",
            books_requested=[book, not_found],
            books_found=[found],
            books_not_found=[not_found],
        )
        assert output.certification_name == "AWS SAA-C03"
        assert len(output.books_requested) == 2
        assert len(output.books_found) == 1
        assert len(output.books_not_found) == 1

    def test_empty_lists(self):
        output = Step1Output(
            certification_name="Test",
            books_requested=[],
            books_found=[],
            books_not_found=[],
        )
        assert output.books_requested == []
