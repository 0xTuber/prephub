"""Batch embedding pipeline for already-extracted MinerU content.

Embeds pre-extracted content into vectorstores. Supports shared folder mappings
where one folder can be embedded into multiple collections.

Usage:
    course-builder batch-embed data/sources/extracted/ \
        --shared "paramedic and emt:NREMT_EMT,NREMT_Paramedic" \
        --delete-existing
"""

import json
from pathlib import Path

from course_builder.domain.books import ExtractedChunk
from course_builder.pipeline.sources.embed import EmbeddingStep, _sanitize_collection_name
from course_builder.pipeline.sources.extract import _chunk_by_headings


def find_mineru_outputs(extracted_dir: Path) -> list[tuple[Path, Path | None, str]]:
    """Find all MinerU outputs (markdown + content_list.json) in a directory.

    Handles both structures:
    - Direct: extracted_dir/book_name/book_name.md
    - Nested: extracted_dir/certification/book_name/book_name.md

    Returns list of (md_file, content_list_file, book_title) tuples.
    """
    results = []

    # Find all .md files
    md_files = list(extracted_dir.rglob("*.md"))

    for md_file in md_files:
        # Skip if it's not a MinerU output (should have matching content_list.json)
        content_list_file = md_file.parent / f"{md_file.stem}_content_list.json"

        if not content_list_file.exists():
            # Try to find it with glob
            json_files = list(md_file.parent.glob("*_content_list.json"))
            content_list_file = json_files[0] if json_files else None

        # Use md filename as book title
        book_title = md_file.stem

        results.append((md_file, content_list_file, book_title))

    return results


def rechunk_from_mineru_output(
    md_file: Path,
    content_list_file: Path | None,
    book_title: str = "Unknown Book",
    book_author: str = "Unknown Author",
) -> list[ExtractedChunk]:
    """Re-chunk from existing MinerU output using the chunking logic."""
    markdown = md_file.read_text(encoding="utf-8")

    content_list = []
    if content_list_file and content_list_file.exists():
        content_list = json.loads(content_list_file.read_text(encoding="utf-8"))

    chunks = _chunk_by_headings(
        markdown=markdown,
        content_list=content_list,
        book_title=book_title,
        book_author=book_author,
        source_file=str(md_file),
    )

    return chunks


def build_embedding_plan(
    extracted_dir: Path,
    shared_mappings: dict[str, list[str]] | None = None,
) -> dict[str, list[tuple[Path, Path | None, str]]]:
    """Build a plan mapping collection names to MinerU outputs.

    Args:
        extracted_dir: Root directory containing extracted folders.
        shared_mappings: Dict mapping folder names to list of collection names.

    Returns:
        Dict mapping collection_name -> list of (md_file, content_list_file, book_title).
    """
    shared_mappings = shared_mappings or {}
    plan: dict[str, list[tuple[Path, Path | None, str]]] = {}

    # Iterate through top-level folders
    for folder in sorted(extracted_dir.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name

        # Find MinerU outputs in this folder
        outputs = find_mineru_outputs(folder)
        if not outputs:
            continue

        # Determine target collections
        if folder_name in shared_mappings:
            target_collections = shared_mappings[folder_name]
        else:
            target_collections = [_sanitize_collection_name(folder_name)]

        # Add outputs to each target collection
        for collection in target_collections:
            if collection not in plan:
                plan[collection] = []
            plan[collection].extend(outputs)

    return plan


def run_batch_embed(
    *,
    extracted_dir: str,
    output_dir: str = "data/vectorstore",
    shared_mappings: dict[str, list[str]] | None = None,
    dry_run: bool = False,
    delete_existing: bool = False,
):
    """Embed pre-extracted MinerU content into vectorstores.

    Args:
        extracted_dir: Directory containing extracted folders.
        output_dir: Output directory for vectorstores.
        shared_mappings: Dict mapping folder names to list of collection names.
        dry_run: If True, show what would be done without running.
        delete_existing: If True, delete existing collections before embedding.
    """
    extracted_path = Path(extracted_dir)
    if not extracted_path.exists():
        print(f"ERROR: Directory not found: {extracted_dir}")
        return

    shared_mappings = shared_mappings or {}

    # Header
    print("=" * 60)
    print("Batch Embed (Pre-Extracted Content)")
    print("=" * 60)
    print(f"\nSource: {extracted_path.absolute()}")
    print(f"Vectorstore output: {output_dir}")

    if shared_mappings:
        print("\nShared folder mappings:")
        for folder, collections in shared_mappings.items():
            print(f"  {folder} -> {', '.join(collections)}")

    # Build embedding plan
    plan = build_embedding_plan(extracted_path, shared_mappings)

    if not plan:
        print("\nNo MinerU outputs found to embed.")
        return

    # Show plan summary
    print("\n" + "=" * 60)
    print("Embedding Plan")
    print("=" * 60)

    total_books = sum(len(outputs) for outputs in plan.values())
    print(f"\nCollections to create: {len(plan)}")
    print(f"Total books to embed: {total_books}")

    for collection, outputs in sorted(plan.items()):
        print(f"\n  {collection}: {len(outputs)} book(s)")
        for md_file, _, book_title in outputs:
            print(f"    - {book_title}")

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - No embedding performed")
        print("=" * 60)
        return

    # Delete existing collections if requested
    if delete_existing:
        print("\n" + "=" * 60)
        print("Deleting Existing Collections")
        print("=" * 60)

        import chromadb

        client = chromadb.PersistentClient(path=output_dir)

        for collection_name in plan.keys():
            try:
                client.delete_collection(collection_name)
                print(f"  Deleted: {collection_name}")
            except Exception:
                print(f"  Not found: {collection_name}")

    # Run embedding
    print("\n" + "=" * 60)
    print("Embedding into Collections")
    print("=" * 60)

    for collection_name, outputs in sorted(plan.items()):
        print(f"\n[Collection: {collection_name}]")

        # Gather all chunks for this collection
        all_chunks: list[ExtractedChunk] = []

        for md_file, content_list_file, book_title in outputs:
            print(f"  Processing: {book_title}")
            try:
                chunks = rechunk_from_mineru_output(
                    md_file=md_file,
                    content_list_file=content_list_file,
                    book_title=book_title,
                    book_author="Unknown",
                )
                all_chunks.extend(chunks)
                print(f"    {len(chunks)} chunks")
            except Exception as e:
                print(f"    ERROR: {e}")

        if not all_chunks:
            print("  No chunks to embed, skipping")
            continue

        print(f"  Total: {len(all_chunks)} chunks")

        # Embed
        embedder = EmbeddingStep(vectorstore_dir=output_dir)
        context = {
            "extracted_chunks": all_chunks,
            "certification_name": collection_name,
        }

        try:
            result = embedder.run(context)
            stored = result.get("chunks_stored", 0)
            print(f"  Stored {stored} chunks in '{collection_name}'")
        except Exception as e:
            print(f"  ERROR embedding: {e}")

    print("\n" + "=" * 60)
    print("Batch Embedding Complete")
    print("=" * 60)
