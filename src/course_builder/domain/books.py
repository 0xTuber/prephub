"""Book-related domain models.

This module contains models for book discovery, download, and extraction:
- Book: A book recommendation
- LibgenResult: A book found on LibGen
- DownloadedBook: A downloaded book file
- ProcessedBook: A book that has been processed for content extraction
- ExtractedChunk: A chunk of text extracted from a book
- Step1Output, Step2Output, Step3Output: Pipeline step outputs
"""

from pydantic import BaseModel, field_validator


class Book(BaseModel):
    """A recommended study book."""

    title: str
    author: str
    edition: str | None = None
    year: str | None = None
    isbn: str | None = None

    @field_validator("title", "author", "edition", "year", "isbn", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        if v is None:
            return v
        return str(v)


class LibgenResult(BaseModel):
    """A book found on LibGen."""

    title: str
    author: str
    edition: str | None = None
    year: str | None = None
    isbn: str | None = None
    found: bool = False
    language: str | None = None
    pages: str | None = None
    size: str | None = None
    extension: str | None = None
    mirror_links: list[str] = []
    publisher: str | None = None
    direct_download_link: str | None = None


class DownloadedBook(BaseModel):
    """A downloaded book file."""

    title: str
    author: str
    extension: str | None = None
    file_path: str
    success: bool
    error: str | None = None


class Step1Output(BaseModel):
    """Output of Step 1: Book Discovery."""

    certification_name: str
    books_requested: list[Book]
    books_found: list[LibgenResult]
    books_not_found: list[Book]


class Step2Output(BaseModel):
    """Output of Step 2: Book Download."""

    certification_name: str
    books_downloaded: list[DownloadedBook]
    books_failed: list[DownloadedBook]


class ExtractedChunk(BaseModel):
    """A chunk of text extracted from a book."""

    text: str
    section_heading: str | None = None
    image_paths: list[str] = []
    page_numbers: list[int] = []
    book_title: str
    book_author: str
    source_file: str


class ProcessedBook(BaseModel):
    """A book that has been processed for content extraction."""

    title: str
    author: str
    file_path: str
    success: bool
    chunks: list[ExtractedChunk] = []
    error: str | None = None


class Step3Output(BaseModel):
    """Output of Step 3: Content Extraction and Embedding."""

    certification_name: str
    chunks_extracted: int
    books_processed: int
    books_failed: int
    vectorstore_path: str
    collection_name: str
