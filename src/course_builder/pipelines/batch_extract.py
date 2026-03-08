"""Batch extraction pipeline.

Extracts and embeds PDFs from organized folder structures into vectorstores.
Supports shared folders that should be embedded into multiple collections.
"""

import re
from pathlib import Path

from course_builder.domain.books import DownloadedBook, ExtractedChunk
from course_builder.pipeline.sources.extract import ContentExtractionStep
from course_builder.pipeline.sources.embed import EmbeddingStep, _sanitize_collection_name


def run_batch_extract(
    *,
    source_dir: str,
    output_dir: str = "data/vectorstore",
    extracted_dir: str = "data/sources/extracted",
    shared_mappings: dict[str, list[str]] | None = None,
    dry_run: bool = False,
    backend: str = "hybrid-auto-engine",
):
    """Extract and embed PDFs from organized folders.

    Args:
        source_dir: Source directory containing certification folders with PDFs.
        output_dir: Output directory for vectorstores.
        extracted_dir: Directory for extracted content.
        shared_mappings: Dict mapping folder names to list of collection names
                        they should be embedded into.
        dry_run: If True, show what would be done without running.
        backend: MinerU backend to use (pipeline, hybrid-auto-engine, vlm-auto-engine).
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return

    shared_mappings = shared_mappings or {}

    # Discover all folders and their PDFs
    print("=" * 60)
    print("Batch PDF Extraction")
    print("=" * 60)
    print(f"\nSource: {source_path.absolute()}")
    print(f"Vectorstore output: {output_dir}")
    print(f"Extraction output: {extracted_dir}")
    print(f"Backend: {backend}")

    # Build extraction plan
    extraction_plan: dict[str, list[Path]] = {}  # collection_name -> list of PDFs
    folder_to_collection: dict[str, str] = {}  # folder_name -> default collection_name

    for folder in sorted(source_path.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name
        pdfs = list(folder.glob("*.pdf"))

        if not pdfs:
            print(f"\n  {folder_name}/")
            print(f"    (no PDFs found, skipping)")
            continue

        # Determine target collections
        if folder_name in shared_mappings:
            # Shared folder - embed into multiple collections
            target_collections = shared_mappings[folder_name]
            print(f"\n  {folder_name}/ (SHARED)")
            print(f"    -> {', '.join(target_collections)}")
        else:
            # Regular folder - use folder name as collection
            collection_name = _sanitize_collection_name(folder_name)
            target_collections = [collection_name]
            folder_to_collection[folder_name] = collection_name
            print(f"\n  {folder_name}/")
            print(f"    -> {collection_name}")

        for pdf in pdfs:
            print(f"      - {pdf.name}")

        # Add PDFs to extraction plan
        for collection in target_collections:
            if collection not in extraction_plan:
                extraction_plan[collection] = []
            extraction_plan[collection].extend(pdfs)

    # Summary
    print("\n" + "=" * 60)
    print("Extraction Plan Summary")
    print("=" * 60)
    total_pdfs = sum(len(pdfs) for pdfs in extraction_plan.values())
    print(f"\nCollections to create: {len(extraction_plan)}")
    print(f"Total PDFs to process: {total_pdfs}")

    for collection, pdfs in sorted(extraction_plan.items()):
        print(f"\n  {collection}: {len(pdfs)} PDF(s)")
        for pdf in pdfs:
            print(f"    - {pdf.name}")

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - No extraction performed")
        print("=" * 60)
        return

    # Run extraction
    print("\n" + "=" * 60)
    print("Starting Extraction")
    print("=" * 60)

    # Extract all unique PDFs first (avoid re-extracting same PDF multiple times)
    all_pdfs = set()
    for pdfs in extraction_plan.values():
        all_pdfs.update(pdfs)

    print(f"\nExtracting {len(all_pdfs)} unique PDF(s)...")

    # Create extraction step with specified backend
    extractor = ContentExtractionStep(extraction_dir=extracted_dir, backend=backend)

    # Cache of extracted chunks per PDF
    pdf_chunks: dict[Path, list[ExtractedChunk]] = {}

    for i, pdf_path in enumerate(sorted(all_pdfs), 1):
        print(f"\n[{i}/{len(all_pdfs)}] {pdf_path.name}")

        # Create a mock context for the extraction step
        book = DownloadedBook(
            title=pdf_path.stem,
            author="Unknown",
            extension="pdf",
            file_path=str(pdf_path.absolute()),
            success=True,
        )

        context = {
            "books_downloaded": [book],
            "certification_name": pdf_path.parent.name,
        }

        try:
            result = extractor.run(context)
            chunks = result.get("extracted_chunks", [])
            pdf_chunks[pdf_path] = chunks
            print(f"  Extracted {len(chunks)} chunks")
        except Exception as e:
            print(f"  ERROR: {e}")
            pdf_chunks[pdf_path] = []

    # Now embed into collections
    print("\n" + "=" * 60)
    print("Embedding into Collections")
    print("=" * 60)

    for collection_name, pdfs in sorted(extraction_plan.items()):
        print(f"\n[Collection: {collection_name}]")

        # Gather all chunks for this collection
        all_chunks = []
        for pdf_path in pdfs:
            chunks = pdf_chunks.get(pdf_path, [])
            all_chunks.extend(chunks)

        if not all_chunks:
            print(f"  No chunks to embed, skipping")
            continue

        print(f"  Total chunks: {len(all_chunks)}")

        # Create embedding step for this collection
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
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Batch Extraction Complete")
    print("=" * 60)
