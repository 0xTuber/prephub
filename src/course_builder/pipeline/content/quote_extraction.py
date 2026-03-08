"""Quote extraction module for quote-first evidence retrieval.

This module implements the quote-first extraction approach:
1. Use planned queries to retrieve candidate chunks
2. Extract exact quote spans BEFORE generation
3. Iterative retry if first retrieval misses

By extracting quotes before generation, we ensure the LLM is constrained
to use actual source material rather than generating and then failing verification.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from course_builder.pipeline.content.query_planning import QueryPlan, QueryIntent


class ExtractionStatus(str, Enum):
    """Status of quote extraction."""

    SUFFICIENT = "sufficient"  # Found enough quotes (>= min_quotes)
    MARGINAL = "marginal"  # Found some quotes but below ideal
    INSUFFICIENT = "insufficient"  # Could not find minimum required quotes


class ExtractedQuote(BaseModel):
    """A quote extracted from source material before generation."""

    quote_id: str  # Unique ID like Q1, Q2, Q3
    text: str  # Exact quote (30-200 chars)
    chunk_id: str
    chunk_index: int
    start_char: int  # Start position in chunk text
    end_char: int  # End position in chunk text
    page_numbers: list[int] = []
    section_heading: str | None = None
    relevance_score: float = 0.0

    def verify_in_chunk(self, chunk_text: str) -> bool:
        """Verify this quote exists exactly in the chunk text."""
        if self.start_char >= 0 and self.end_char > self.start_char:
            extracted = chunk_text[self.start_char:self.end_char]
            return extracted == self.text
        # Fallback: substring search
        return self.text in chunk_text


class QuoteExtractionResult(BaseModel):
    """Result of quote extraction with iterative retrieval."""

    item_id: str
    quotes: list[ExtractedQuote]
    retrieval_rounds: int
    status: ExtractionStatus
    chunks_searched: int = 0
    keywords_found: list[str] = []
    keywords_missing: list[str] = []


# Patterns that indicate quotable sentences
QUOTABLE_ACTION_PATTERNS = [
    r"\bshould\b",
    r"\bmust\b",
    r"\bensure\b",
    r"\balways\b",
    r"\bnever\b",
    r"\brequires?\b",
    r"\bimportant\b",
    r"\bcritical\b",
    r"\bessential\b",
    r"\bpriority\b",
    r"\bfirst\b",
    r"\bbefore\b",
    r"\bafter\b",
]

QUOTABLE_FACTUAL_PATTERNS = [
    r"\bis defined as\b",
    r"\brefers to\b",
    r"\bmeans\b",
    r"\bis a\b",
    r"\bis an\b",
    r"\bis the\b",
    r"\bare\s+(?:a|an|the)\b",
    r"\bincludes?\b",
    r"\bconsists? of\b",
]

QUOTABLE_PROCEDURAL_PATTERNS = [
    r"\bstep\s+\d+\b",
    r"\bfirst,?\s+\w+",
    r"\bthen,?\s+\w+",
    r"\bnext,?\s+\w+",
    r"\bfinally,?\s+\w+",
    r"\bprocedure\b",
    r"\bprotocol\b",
    r"\bsequence\b",
]


def identify_quotable_sentences(
    chunk_text: str,
    learning_target: str | None = None,
    min_length: int = 30,
    max_length: int = 200,
) -> list[tuple[str, float, int, int]]:
    """Detect quotable sentences in chunk text.

    Quotable sentences contain:
    - Action verbs: "should", "must", "ensure", "always", "never"
    - Factual claims: "is defined as", "refers to"
    - Procedural markers: "before", "after", "first", "then"

    Args:
        chunk_text: Full text of the chunk
        learning_target: Optional learning target for relevance scoring
        min_length: Minimum sentence length
        max_length: Maximum sentence length

    Returns:
        List of (sentence, score, start_char, end_char) tuples, sorted by score descending
    """
    # Split into sentences while tracking positions
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences_with_pos = []
    last_end = 0

    for match in sentence_pattern.finditer(chunk_text):
        sentence = chunk_text[last_end:match.start() + 1].strip()
        if sentence:
            sentences_with_pos.append((sentence, last_end, match.start() + 1))
        last_end = match.end()

    # Don't forget the last sentence
    if last_end < len(chunk_text):
        sentence = chunk_text[last_end:].strip()
        if sentence:
            sentences_with_pos.append((sentence, last_end, len(chunk_text)))

    candidates = []
    target_terms = set()
    if learning_target:
        # Extract meaningful words from learning target
        target_terms = set(
            w.lower()
            for w in re.findall(r"\b\w{4,}\b", learning_target)
            if w.lower() not in {"that", "this", "with", "from", "about"}
        )

    for sentence, start_pos, end_pos in sentences_with_pos:
        sentence = sentence.strip()
        length = len(sentence)

        # Skip sentences that are too short or too long
        if length < min_length or length > max_length:
            continue

        # Skip questions
        if sentence.endswith("?"):
            continue

        # Skip list items and bullet points
        if sentence.startswith(("-", "*", "1.", "2.", "3.")):
            continue

        # Score the sentence
        score = 0.0
        sentence_lower = sentence.lower()

        # Check for action patterns
        for pattern in QUOTABLE_ACTION_PATTERNS:
            if re.search(pattern, sentence_lower):
                score += 0.3
                break

        # Check for factual patterns
        for pattern in QUOTABLE_FACTUAL_PATTERNS:
            if re.search(pattern, sentence_lower):
                score += 0.2
                break

        # Check for procedural patterns
        for pattern in QUOTABLE_PROCEDURAL_PATTERNS:
            if re.search(pattern, sentence_lower):
                score += 0.2
                break

        # Relevance bonus: overlap with learning target terms
        if target_terms:
            sentence_words = set(sentence_lower.split())
            overlap = len(target_terms & sentence_words)
            relevance_bonus = min(overlap * 0.1, 0.3)
            score += relevance_bonus

        # Only include sentences with some quotability score
        if score > 0:
            # Find exact position in original text for verification
            actual_start = chunk_text.find(sentence)
            if actual_start >= 0:
                actual_end = actual_start + len(sentence)
                candidates.append((sentence, score, actual_start, actual_end))
            else:
                # Fallback to approximate position
                candidates.append((sentence, score, start_pos, end_pos))

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _query_vectorstore(
    collection: Any,
    embedding_model: Any,
    query: str,
    n_results: int = 10,
) -> list[dict]:
    """Query the vectorstore and return matching chunks with metadata.

    This is a local implementation to avoid circular imports.
    """
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            chunk_id = results["ids"][0][i] if results["ids"] else f"chunk_{i}"
            distance = results["distances"][0][i] if results["distances"] else None
            chunks.append({
                "chunk_id": chunk_id,
                "text": doc,
                "book_title": metadata.get("book_title", "Unknown"),
                "book_author": metadata.get("book_author", "Unknown"),
                "pages": [int(p) for p in metadata.get("page_numbers", "").split(",") if p],
                "section_heading": metadata.get("section_heading", ""),
                "image_paths": [p.strip() for p in metadata.get("image_paths", "").split("|||") if p.strip()],
                "distance": distance,
            })
    return chunks


def extract_quotes_for_item(
    collection: Any,
    embedding_model: Any,
    query_plan: "QueryPlan",
    min_quotes: int = 2,
    max_rounds: int = 2,
    k_round1: int = 10,
    k_round2: int = 25,
) -> QuoteExtractionResult:
    """Extract quotes using iterative retrieval.

    Round 1 (Precision):
    - Run HIGH_PRECISION + QUOTE_HUNT queries
    - K=10 per query
    - Extract quotable sentences
    - Check must_include keywords

    If quotes < min_quotes OR keywords missing:
    Round 2 (Broaden):
    - Add SYNONYM_VARIANT + SCENARIO_CONTEXT queries
    - K=25 per query
    - Re-extract quotes

    Args:
        collection: ChromaDB collection
        embedding_model: Sentence transformer model
        query_plan: QueryPlan with targeted queries
        min_quotes: Minimum quotes required
        max_rounds: Maximum retrieval rounds
        k_round1: Results per query in round 1
        k_round2: Results per query in round 2

    Returns:
        QuoteExtractionResult with status
    """
    from course_builder.pipeline.content.query_planning import QueryIntent

    all_chunks: dict[str, dict] = {}  # chunk_id -> chunk
    all_quotes: list[ExtractedQuote] = []
    seen_quote_texts: set[str] = set()
    keywords_found: set[str] = set()
    quote_counter = 0  # For generating unique quote_ids

    must_include = set(kw.lower() for kw in query_plan.must_include_keywords)

    def merge_chunks(new_chunks: list[dict]) -> None:
        """Merge new chunks, keeping best distance."""
        for chunk in new_chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = chunk
            elif chunk.get("distance", 1.0) < all_chunks[chunk_id].get("distance", 1.0):
                all_chunks[chunk_id] = chunk

    def extract_from_chunks() -> None:
        """Extract quotable sentences from all collected chunks."""
        nonlocal keywords_found, quote_counter

        for chunk_id, chunk in all_chunks.items():
            text = chunk.get("text", "")
            text_lower = text.lower()

            # Check for must_include keywords
            for kw in must_include:
                if kw in text_lower:
                    keywords_found.add(kw)

            # Extract quotable sentences (now returns positions)
            quotable = identify_quotable_sentences(
                text,
                learning_target=query_plan.learning_target,
            )

            for sentence, score, start_char, end_char in quotable[:3]:  # Top 3 per chunk
                # Deduplicate by normalized text
                normalized = sentence.lower().strip()
                if normalized in seen_quote_texts:
                    continue
                seen_quote_texts.add(normalized)

                # Find chunk index (order by distance)
                sorted_chunks = sorted(
                    all_chunks.items(),
                    key=lambda x: x[1].get("distance", 1.0),
                )
                chunk_index = next(
                    (i for i, (cid, _) in enumerate(sorted_chunks) if cid == chunk_id),
                    0,
                )

                # Generate unique quote_id
                quote_counter += 1
                quote_id = f"Q{quote_counter}"

                all_quotes.append(ExtractedQuote(
                    quote_id=quote_id,
                    text=sentence,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    page_numbers=chunk.get("pages", []),
                    section_heading=chunk.get("section_heading"),
                    relevance_score=score,
                ))

    # Round 1: Precision queries
    round1_queries = [
        q for q in query_plan.queries
        if q.intent in (QueryIntent.HIGH_PRECISION, QueryIntent.QUOTE_HUNT)
    ]

    for query in round1_queries:
        chunks = _query_vectorstore(
            collection,
            embedding_model,
            query.query_text,
            n_results=k_round1,
        )
        merge_chunks(chunks)

    extract_from_chunks()

    # Check if we have enough
    missing_keywords = must_include - keywords_found

    if len(all_quotes) >= min_quotes and not missing_keywords:
        # Sort by relevance score
        all_quotes.sort(key=lambda q: q.relevance_score, reverse=True)
        return QuoteExtractionResult(
            item_id=query_plan.item_id,
            quotes=all_quotes[:min_quotes * 2],  # Return up to 2x min
            retrieval_rounds=1,
            status=ExtractionStatus.SUFFICIENT,
            chunks_searched=len(all_chunks),
            keywords_found=list(keywords_found),
            keywords_missing=[],
        )

    if max_rounds < 2:
        # Can't do round 2, return what we have
        status = (
            ExtractionStatus.MARGINAL
            if all_quotes
            else ExtractionStatus.INSUFFICIENT
        )
        return QuoteExtractionResult(
            item_id=query_plan.item_id,
            quotes=all_quotes,
            retrieval_rounds=1,
            status=status,
            chunks_searched=len(all_chunks),
            keywords_found=list(keywords_found),
            keywords_missing=list(missing_keywords),
        )

    # Round 2: Broaden with synonym and context queries
    round2_queries = [
        q for q in query_plan.queries
        if q.intent in (QueryIntent.SYNONYM_VARIANT, QueryIntent.SCENARIO_CONTEXT)
    ]

    # Also re-run precision queries with larger K
    for query in query_plan.queries:
        chunks = _query_vectorstore(
            collection,
            embedding_model,
            query.query_text,
            n_results=k_round2,
        )
        merge_chunks(chunks)

    extract_from_chunks()

    # Check final status
    missing_keywords = must_include - keywords_found

    if len(all_quotes) >= min_quotes:
        all_quotes.sort(key=lambda q: q.relevance_score, reverse=True)
        return QuoteExtractionResult(
            item_id=query_plan.item_id,
            quotes=all_quotes[:min_quotes * 2],
            retrieval_rounds=2,
            status=ExtractionStatus.SUFFICIENT,
            chunks_searched=len(all_chunks),
            keywords_found=list(keywords_found),
            keywords_missing=list(missing_keywords),
        )
    elif all_quotes:
        all_quotes.sort(key=lambda q: q.relevance_score, reverse=True)
        return QuoteExtractionResult(
            item_id=query_plan.item_id,
            quotes=all_quotes,
            retrieval_rounds=2,
            status=ExtractionStatus.MARGINAL,
            chunks_searched=len(all_chunks),
            keywords_found=list(keywords_found),
            keywords_missing=list(missing_keywords),
        )
    else:
        return QuoteExtractionResult(
            item_id=query_plan.item_id,
            quotes=[],
            retrieval_rounds=2,
            status=ExtractionStatus.INSUFFICIENT,
            chunks_searched=len(all_chunks),
            keywords_found=list(keywords_found),
            keywords_missing=list(missing_keywords),
        )


def format_quotes_for_prompt(quotes: list[ExtractedQuote], max_quotes: int = 4) -> str:
    """Format extracted quotes for inclusion in generation prompt.

    Args:
        quotes: List of ExtractedQuote objects
        max_quotes: Maximum quotes to include

    Returns:
        Formatted string for prompt with quote_ids for reference
    """
    if not quotes:
        return "(No anchor quotes extracted)"

    lines = []
    for quote in quotes[:max_quotes]:
        pages = ", ".join(str(p) for p in quote.page_numbers) if quote.page_numbers else "N/A"
        section = f" [{quote.section_heading}]" if quote.section_heading else ""
        # Include quote_id so generation can reference it: [Q1], [Q2], etc.
        lines.append(f"[{quote.quote_id}] (p.{pages}){section}:\n\"{quote.text}\"")

    return "\n\n".join(lines)


def get_chunks_from_quotes(
    quotes: list[ExtractedQuote],
    all_chunks: dict[str, dict],
) -> list[dict]:
    """Get unique chunks that contain the extracted quotes.

    Args:
        quotes: List of ExtractedQuote objects
        all_chunks: Mapping of chunk_id -> chunk dict

    Returns:
        List of unique chunk dicts
    """
    seen = set()
    chunks = []

    for quote in quotes:
        if quote.chunk_id not in seen and quote.chunk_id in all_chunks:
            chunks.append(all_chunks[quote.chunk_id])
            seen.add(quote.chunk_id)

    return chunks


# ============================================================================
# Mechanical Quote Verification
# ============================================================================


class QuoteVerificationResult(BaseModel):
    """Result of mechanical quote verification."""

    quote_id: str
    found: bool  # Was the quote found in the generated text?
    exact_match: bool  # Was it an exact match?
    match_text: str | None = None  # The matching text if found
    similarity: float = 0.0  # Similarity score if not exact


class GenerationQuoteVerification(BaseModel):
    """Complete verification of quote usage in generated content."""

    item_id: str
    required_quotes: list[str]  # quote_ids that were required
    verified_quotes: list[QuoteVerificationResult]
    all_required_found: bool
    exact_match_count: int
    fuzzy_match_count: int
    missing_quote_ids: list[str]


def verify_quotes_in_text(
    generated_text: str,
    quotes: list[ExtractedQuote],
    required_quote_ids: list[str] | None = None,
    fuzzy_threshold: float = 0.85,
) -> GenerationQuoteVerification:
    """Mechanically verify that required quotes appear in generated text.

    This is a DETERMINISTIC verification - no LLM involved.

    Args:
        generated_text: The generated explanation/content
        quotes: List of ExtractedQuote objects to verify
        required_quote_ids: Specific quote_ids that must appear (default: all)
        fuzzy_threshold: Similarity threshold for fuzzy matching

    Returns:
        GenerationQuoteVerification with detailed results
    """
    if required_quote_ids is None:
        required_quote_ids = [q.quote_id for q in quotes]

    quote_map = {q.quote_id: q for q in quotes}
    verified = []
    missing = []

    for quote_id in required_quote_ids:
        if quote_id not in quote_map:
            missing.append(quote_id)
            verified.append(QuoteVerificationResult(
                quote_id=quote_id,
                found=False,
                exact_match=False,
            ))
            continue

        quote = quote_map[quote_id]
        result = _verify_single_quote(generated_text, quote, fuzzy_threshold)
        verified.append(result)

        if not result.found:
            missing.append(quote_id)

    exact_count = sum(1 for v in verified if v.exact_match)
    fuzzy_count = sum(1 for v in verified if v.found and not v.exact_match)

    return GenerationQuoteVerification(
        item_id=quotes[0].chunk_id if quotes else "",
        required_quotes=required_quote_ids,
        verified_quotes=verified,
        all_required_found=len(missing) == 0,
        exact_match_count=exact_count,
        fuzzy_match_count=fuzzy_count,
        missing_quote_ids=missing,
    )


def _verify_single_quote(
    generated_text: str,
    quote: ExtractedQuote,
    fuzzy_threshold: float,
) -> QuoteVerificationResult:
    """Verify a single quote appears in generated text.

    Checks in order:
    1. Exact substring match
    2. Quoted substring match (with surrounding quotes)
    3. Normalized substring match (whitespace/punctuation)
    4. Fuzzy substring match
    """
    quote_text = quote.text.strip()
    gen_lower = generated_text.lower()
    quote_lower = quote_text.lower()

    # Check 1: Exact substring
    if quote_text in generated_text:
        return QuoteVerificationResult(
            quote_id=quote.quote_id,
            found=True,
            exact_match=True,
            match_text=quote_text,
            similarity=1.0,
        )

    # Check 2: Quoted version (LLM might add quotes around it)
    quoted_patterns = [
        f'"{quote_text}"',
        f"'{quote_text}'",
        f'"{quote_text}"',  # Smart quotes
        f"'{quote_text}'",
    ]
    for pattern in quoted_patterns:
        if pattern in generated_text:
            return QuoteVerificationResult(
                quote_id=quote.quote_id,
                found=True,
                exact_match=True,
                match_text=pattern,
                similarity=1.0,
            )

    # Check 3: Normalized match (collapse whitespace, ignore case)
    normalized_quote = _normalize_for_matching(quote_text)
    normalized_gen = _normalize_for_matching(generated_text)

    if normalized_quote in normalized_gen:
        return QuoteVerificationResult(
            quote_id=quote.quote_id,
            found=True,
            exact_match=False,
            match_text=quote_text,
            similarity=0.95,
        )

    # Check 4: Fuzzy substring match
    similarity = _fuzzy_substring_similarity(quote_lower, gen_lower)
    if similarity >= fuzzy_threshold:
        return QuoteVerificationResult(
            quote_id=quote.quote_id,
            found=True,
            exact_match=False,
            match_text=None,
            similarity=similarity,
        )

    # Not found
    return QuoteVerificationResult(
        quote_id=quote.quote_id,
        found=False,
        exact_match=False,
        similarity=similarity,
    )


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching."""
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", "", text)
    return text.lower().strip()


def _fuzzy_substring_similarity(needle: str, haystack: str) -> float:
    """Calculate best fuzzy match similarity of needle in haystack.

    Uses a sliding window approach to find the best matching substring.
    """
    if not needle or not haystack:
        return 0.0

    needle_len = len(needle)
    best_similarity = 0.0

    # Slide window across haystack
    for i in range(max(1, len(haystack) - needle_len + 1)):
        window = haystack[i:i + needle_len]
        similarity = _char_similarity(needle, window)
        best_similarity = max(best_similarity, similarity)

        # Early exit if we find a very good match
        if best_similarity >= 0.95:
            break

    return best_similarity


def _char_similarity(s1: str, s2: str) -> float:
    """Calculate character-level similarity between two strings."""
    if not s1 or not s2:
        return 0.0

    # Simple character overlap ratio
    matches = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
    return matches / max(len(s1), len(s2))
