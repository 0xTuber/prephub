"""Content extraction step.

This module provides the ContentExtractionStep which extracts text content
from PDF files using MinerU.

Supports two backends:
- pipeline: Traditional CPU-friendly approach (~82% accuracy)
- hybrid: VLM-based with vLLM acceleration (~90% accuracy, requires GPU)
"""

import json
import re
import subprocess
from pathlib import Path

from course_builder.domain.books import DownloadedBook, ExtractedChunk, ProcessedBook
from course_builder.pipeline.base import PipelineContext, PipelineStep


def _extract_pdf_with_mineru(
    pdf_path: str,
    output_dir: str,
    backend: str = "hybrid-auto-engine",
) -> tuple[str, list[dict]]:
    """Extract text and images from a PDF using MinerU CLI.

    Uses the MinerU CLI for better backend support (hybrid with vLLM).

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to store extracted content.
        backend: MinerU backend to use:
            - "pipeline": CPU-friendly, ~82% accuracy
            - "hybrid-auto-engine": VLM with auto engine selection (vLLM), ~90% accuracy
            - "vlm-auto-engine": Pure VLM backend

    Returns (markdown_text, content_list).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run MinerU CLI (don't capture output to show real-time progress)
    cmd = [
        "mineru",
        "-p", str(pdf_path),
        "-o", str(output_path),
        "-b", backend,
    ]

    # Set environment for better GPU memory utilization with vLLM
    env = {
        **dict(__import__("os").environ),
        "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
    }

    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0 and result.returncode != -15:  # -15 is SIGTERM from vLLM cleanup
        raise RuntimeError(f"MinerU failed with exit code {result.returncode}")

    # Find the output markdown file
    # MinerU creates: output_dir/<pdf_name>/<pdf_name>.md
    pdf_name = Path(pdf_path).stem
    md_path = output_path / pdf_name / f"{pdf_name}.md"
    content_list_path = output_path / pdf_name / f"{pdf_name}_content_list.json"

    if not md_path.exists():
        # Try alternative path structure
        md_files = list(output_path.rglob("*.md"))
        if md_files:
            md_path = md_files[0]
        else:
            raise FileNotFoundError(f"No markdown output found in {output_path}")

    markdown_text = md_path.read_text(encoding="utf-8")

    # Load content list if available (for page numbers)
    content_list = []
    if content_list_path.exists():
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
    else:
        # Try to find it
        json_files = list(output_path.rglob("*_content_list.json"))
        if json_files:
            content_list = json.loads(json_files[0].read_text(encoding="utf-8"))

    return markdown_text, content_list


def _extract_pdf_with_mineru_legacy(pdf_path: str, output_dir: str) -> tuple[str, list[dict]]:
    """Extract text and images from a PDF using MinerU Python API (legacy pipeline backend).

    Uses the v1.x Dataset API: PymuDocDataset -> classify -> doc_analyze ->
    pipe_txt_mode / pipe_ocr_mode -> get_markdown / get_content_list.

    Returns (markdown_text, content_list).
    """
    import magic_pdf.model as model_config
    from magic_pdf.config.enums import SupportedPdfParseMethod
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

    pdf_bytes = Path(pdf_path).read_bytes()

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_writer = FileBasedDataWriter(str(images_dir))

    ds = PymuDocDataset(pdf_bytes)

    if ds.classify() == SupportedPdfParseMethod.TXT:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

    markdown_text = pipe_result.get_markdown(str(images_dir))
    content_list = pipe_result.get_content_list(str(images_dir))

    return markdown_text, content_list


def _chunk_by_headings(
    markdown: str,
    content_list: list[dict],
    book_title: str,
    book_author: str,
    source_file: str,
) -> list[ExtractedChunk]:
    """Split markdown into chunks based on heading lines.

    Text before the first heading becomes a chunk with section_heading=None.
    Image references like ![...](path) are captured into image_paths.
    Page numbers are extracted from content_list blocks, tracked per section
    rather than aggregated by heading text across the entire book.
    """
    image_pattern = re.compile(r"!\[.*?\]\((.+?)\)")

    # First, identify all heading texts from markdown
    heading_pattern = re.compile(r"^# (.+)$", re.MULTILINE)
    markdown_headings_set = {m.group(1).strip() for m in heading_pattern.finditer(markdown)}

    # Build sequential sections from content_list
    # A section starts when we see a block whose text matches a markdown heading
    # Each section: (heading_text or None, list of page numbers)
    sections: list[tuple[str | None, list[int]]] = []
    current_section_heading: str | None = None
    current_section_pages: set[int] = set()

    for block in content_list:
        if not isinstance(block, dict):
            continue

        page_idx = block.get("page_idx")
        text = block.get("text", "").strip()

        # Check if this block is a heading (its text matches a markdown heading)
        if text in markdown_headings_set:
            # Save previous section
            if current_section_pages or current_section_heading is not None:
                sections.append((current_section_heading, sorted(current_section_pages)))
            # Start new section
            current_section_heading = text
            current_section_pages = set()
            if page_idx is not None:
                current_section_pages.add(page_idx)
        elif page_idx is not None:
            current_section_pages.add(page_idx)

    # Save the last section
    if current_section_pages or current_section_heading is not None:
        sections.append((current_section_heading, sorted(current_section_pages)))

    # Track how many times we've seen each heading text in markdown (for matching)
    heading_usage_count: dict[str, int] = {}

    lines = markdown.split("\n")
    chunks: list[ExtractedChunk] = []
    md_current_heading: str | None = None
    current_lines: list[str] = []

    def _flush():
        nonlocal heading_usage_count
        text = "\n".join(current_lines).strip()
        if not text:
            return
        images = image_pattern.findall(text)

        # Get pages for current heading at current occurrence index
        pages: list[int] = []
        if md_current_heading is None:
            # Get pages from first None section
            for h, p in sections:
                if h is None:
                    pages = p
                    break
        else:
            # Find the (occurrence)th section with this heading
            occurrence = heading_usage_count.get(md_current_heading, 0)
            count = 0
            for h, p in sections:
                if h == md_current_heading:
                    if count == occurrence:
                        pages = p
                        break
                    count += 1
            # Increment usage count after using it
            heading_usage_count[md_current_heading] = occurrence + 1

        chunks.append(
            ExtractedChunk(
                text=text,
                section_heading=md_current_heading,
                image_paths=images,
                page_numbers=pages,
                book_title=book_title,
                book_author=book_author,
                source_file=source_file,
            )
        )

    for line in lines:
        if line.startswith("# "):
            _flush()
            md_current_heading = line[2:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    _flush()
    return chunks


class ContentExtractionStep(PipelineStep):
    """Pipeline step that extracts content from downloaded PDFs."""

    def __init__(
        self,
        extraction_dir: str = "extracted",
        backend: str = "hybrid-auto-engine",
    ):
        """Initialize the extraction step.

        Args:
            extraction_dir: Directory to store extracted content.
            backend: MinerU backend to use:
                - "pipeline": CPU-friendly, ~82% accuracy
                - "hybrid-auto-engine": VLM with vLLM acceleration, ~90% accuracy (default)
                - "vlm-auto-engine": Pure VLM backend
        """
        self.extraction_dir = extraction_dir
        self.backend = backend

    def run(self, context: PipelineContext) -> PipelineContext:
        """Extract content from all downloaded books.

        Expects context["books_downloaded"] and context["certification_name"].
        Populates context["extracted_chunks"] and context["processed_books"].
        """
        books_downloaded: list[DownloadedBook] = context["books_downloaded"]
        certification_name: str = context["certification_name"]

        print(f"\nExtracting content from {len(books_downloaded)} book(s)...")
        print(f"  Backend: {self.backend}")

        all_chunks: list[ExtractedChunk] = []
        processed_books: list[ProcessedBook] = []

        for i, book in enumerate(books_downloaded, 1):
            print(f"\n[{i}/{len(books_downloaded)}] {book.title[:60]}...")
            ext = (book.extension or "").lower()
            if ext != "pdf":
                print(f"  Skipping: unsupported format '{ext}'")
                processed_books.append(
                    ProcessedBook(
                        title=book.title,
                        author=book.author,
                        file_path=book.file_path,
                        success=False,
                        error="Unsupported format",
                    )
                )
                continue

            try:
                output_dir = str(Path(self.extraction_dir) / certification_name / book.title)
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                print(f"  Extracting with MinerU ({self.backend})...")
                markdown, content_list = _extract_pdf_with_mineru(
                    book.file_path, output_dir, backend=self.backend
                )
                print(f"  Chunking content...")
                chunks = _chunk_by_headings(
                    markdown,
                    content_list,
                    book_title=book.title,
                    book_author=book.author,
                    source_file=book.file_path,
                )
                all_chunks.extend(chunks)
                print(f"  Extracted {len(chunks)} chunks")
                processed_books.append(
                    ProcessedBook(
                        title=book.title,
                        author=book.author,
                        file_path=book.file_path,
                        success=True,
                        chunks=chunks,
                    )
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                processed_books.append(
                    ProcessedBook(
                        title=book.title,
                        author=book.author,
                        file_path=book.file_path,
                        success=False,
                        error=str(e),
                    )
                )

        print(f"\nExtraction complete: {len(all_chunks)} total chunks from {len(processed_books)} book(s)")
        context["extracted_chunks"] = all_chunks
        context["processed_books"] = processed_books
        return context
