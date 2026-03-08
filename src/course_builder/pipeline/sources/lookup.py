"""LibGen book lookup step.

This module provides the LibgenLookupStep which searches LibGen for books
based on title and author.
"""

import time

from libgen_api_enhanced import LibgenSearch

from course_builder.domain.books import Book, LibgenResult
from course_builder.pipeline.base import PipelineContext, PipelineStep


def _search_book(searcher: LibgenSearch, book: Book) -> LibgenResult | None:
    """Search LibGen for a book. Try title first, then title + author filter."""
    try:
        results = searcher.search_title(book.title)
    except Exception:
        return None

    if not results:
        return None

    # Try to find a result that matches the author (loose substring match)
    best = results[0]
    if book.author:
        author_lower = book.author.lower()
        for r in results:
            if author_lower in (r.author or "").lower():
                best = r
                break

    mirrors = [m for m in (best.mirrors or []) if m]

    direct_link = None
    try:
        best.resolve_direct_download_link()
        direct_link = best.resolved_download_link
    except Exception:
        pass

    return LibgenResult(
        title=best.title or book.title,
        author=best.author or book.author,
        edition=book.edition,
        year=best.year or book.year,
        isbn=book.isbn,
        found=True,
        language=best.language,
        pages=best.pages,
        size=best.size,
        extension=best.extension,
        mirror_links=mirrors,
        publisher=best.publisher,
        direct_download_link=direct_link,
    )


class LibgenLookupStep(PipelineStep):
    """Pipeline step that searches LibGen for requested books."""

    def __init__(self, delay: float = 1.5):
        """Initialize the lookup step.

        Args:
            delay: Delay in seconds between LibGen API requests.
        """
        self.delay = delay

    def run(self, context: PipelineContext) -> PipelineContext:
        """Search LibGen for all requested books.

        Expects context["books_requested"] to contain list of Book objects.
        Populates context["books_found"] and context["books_not_found"].
        """
        books: list[Book] = context["books_requested"]
        searcher = LibgenSearch()

        found: list[LibgenResult] = []
        not_found: list[Book] = []

        for i, book in enumerate(books, 1):
            print(f"[{i}/{len(books)}] Searching LibGen for: {book.title}...")
            result = _search_book(searcher, book)

            if result:
                found.append(result)
                print(f"  -> Found ({result.extension}, {result.size})")
            else:
                not_found.append(book)
                print("  -> Not found")

            if i < len(books):
                time.sleep(self.delay)

        print(f"\nLibGen results: {len(found)} found, {len(not_found)} not found.\n")
        context["books_found"] = found
        context["books_not_found"] = not_found
        return context
