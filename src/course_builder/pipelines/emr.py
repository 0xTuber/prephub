"""
Certification Course Generation Pipeline.

A generic pipeline for generating AI-powered certification exam preparation courses.
Works with any certification exam by discovering exam format from web search.

Supports checkpoint/resume for iterative development:
- Use --stop-after to save checkpoint after a specific stage
- Use --resume-from to load a checkpoint and continue

Engine configuration:
- Skeleton (exam, course, labs): Always Gemini (needs Google Search)
- Capsule/Item skeleton: Configurable (Gemini or vLLM)
- Content generation: Configurable (Gemini or vLLM)
- Validation: Typically Gemini (needs reasoning)

All step parameters use keyword-only convention (param=value).
"""

import os
from pathlib import Path

from course_builder.config import DataPaths, configure_paths
from course_builder.domain.books import Book, DownloadedBook
from course_builder.engine import EngineProvider, create_engine, create_hybrid_provider
from course_builder.pipeline.base import Pipeline, PipelineContext
from course_builder.pipeline.checkpoint import (
    CHECKPOINT_STAGES,
    save_checkpoint,
    load_checkpoint,
    get_stage_index,
)
from course_builder.pipeline.content import ItemContentGenerationStep
from course_builder.pipeline.skeleton import (
    CapsuleItemSkeletonStep,
    CapsuleSkeletonStep,
    CourseSkeletonStep,
    ExamFormatStep,
    LabSkeletonStep,
)
from course_builder.pipeline.sources import (
    BookDownloadStep,
    ContentExtractionStep,
    EmbeddingStep,
    LibgenLookupStep,
)
from course_builder.pipeline.validation import (
    CorrectionApplicationStep,
    CorrectionQueueStep,
    HierarchicalValidationStep,
)

# Map stage names to step classes for checkpoint tracking
STAGE_TO_STEP = {
    "exam": ExamFormatStep,
    "course": CourseSkeletonStep,
    "labs": LabSkeletonStep,
    "capsules": CapsuleSkeletonStep,
    "items": CapsuleItemSkeletonStep,
    "content": ItemContentGenerationStep,
    "validated": CorrectionApplicationStep,
}


def create_pipeline(
    *,
    paths: DataPaths | None = None,
    # Skeleton generation requiring Google Search (exam, course, labs - always Gemini)
    skeleton_model: str = "gemini-2.0-flash",
    # Capsule skeleton generation (can use vLLM - no Google Search needed)
    capsule_engine: str = "gemini",
    capsule_model: str = "gemini-2.0-flash",
    capsule_base_url: str | None = None,
    # Item skeleton generation (can use vLLM)
    item_engine: str = "gemini",
    item_model: str = "gemini-2.0-flash",
    item_base_url: str | None = None,
    # Content generation (can use vLLM or other engines)
    content_engine: str = "gemini",
    content_model: str = "gemini-2.0-flash",
    content_base_url: str | None = None,
    # Validation (typically smarter model)
    validation_engine: str = "gemini",
    validation_model: str = "gemini-2.0-flash-thinking-exp",
    validation_base_url: str | None = None,
    # Concurrency settings
    max_workers_skeleton: int = 30,
    max_workers_content: int = 30,
    max_workers_validation: int = 10,
) -> tuple[Pipeline, EngineProvider, DataPaths]:
    """Create a certification course generation pipeline.

    Supports different engines for different stages:
    - Skeleton (exam, course, labs): Always Gemini (needs Google Search)
    - Capsule skeleton: Configurable (can use vLLM - no Google Search)
    - Item skeleton: Configurable (can use vLLM)
    - Content: Configurable (can use vLLM - RAG-based, simpler task)
    - Validation: Typically Gemini (needs reasoning)

    Args:
        paths: Data paths configuration. If None, uses default paths.
        skeleton_model: Gemini model for skeleton generation (exam, course, labs).
        capsule_engine: Engine for capsule skeleton generation.
        capsule_model: Model for capsule skeleton generation.
        capsule_base_url: Base URL for vLLM server capsule generation.
        item_engine: Engine for item skeleton generation.
        item_model: Model for item skeleton generation.
        item_base_url: Base URL for vLLM server item generation.
        content_engine: Engine for content generation.
        content_model: Model for content generation.
        content_base_url: Base URL for vLLM server content generation.
        validation_engine: Engine for validation.
        validation_model: Model for validation.
        validation_base_url: Base URL for vLLM server validation.
        max_workers_skeleton: Max concurrent workers for skeleton generation.
        max_workers_content: Max concurrent workers for content generation.
        max_workers_validation: Max concurrent workers for validation.

    Returns:
        Tuple of (pipeline, provider, paths) for running the pipeline.
    """
    # Use provided paths or create default
    if paths is None:
        paths = configure_paths(ensure_dirs=True)

    # Create engine for capsule skeleton generation (no Google Search needed)
    capsule_engine_kwargs = {"model": capsule_model}
    if capsule_base_url:
        capsule_engine_kwargs["base_url"] = capsule_base_url
    capsule_skeleton_engine = create_engine(capsule_engine, **capsule_engine_kwargs)

    # Create engine for item skeleton generation
    item_engine_kwargs = {"model": item_model}
    if item_base_url:
        item_engine_kwargs["base_url"] = item_base_url
    item_skeleton_engine = create_engine(item_engine, **item_engine_kwargs)

    # Create engine provider for content and validation
    if content_engine == validation_engine:
        # Same engine type for both content and validation
        provider_kwargs = {
            "engine_type": content_engine,
            "generation_model": content_model,
            "validation_model": validation_model,
        }
        if content_base_url:
            provider_kwargs["base_url"] = content_base_url
        provider = EngineProvider(**provider_kwargs)
    else:
        # Hybrid: different engines for content vs validation
        provider = create_hybrid_provider(
            generation_engine_type=content_engine,
            generation_model=content_model,
            validation_engine_type=validation_engine,
            validation_model=validation_model,
            generation_base_url=content_base_url,
            validation_base_url=validation_base_url,
        )

    # Build the pipeline with configured paths
    pipeline = Pipeline(
        steps=[
            # Step 1b: Find books on LibGen
            LibgenLookupStep(delay=2.0),
            # Step 2: Download books
            BookDownloadStep(output_dir=str(paths.downloads_path)),
            # Step 3: Extract and embed
            ContentExtractionStep(extraction_dir=str(paths.extracted_path)),
            EmbeddingStep(vectorstore_dir=str(paths.vectorstore_path)),
            # Step 4: Generate course skeleton (always Gemini - needs reasoning)
            ExamFormatStep(model=skeleton_model),
            CourseSkeletonStep(model=skeleton_model, max_workers=8),
            LabSkeletonStep(model=skeleton_model, max_workers=10, target_lab_count=3),
            CapsuleSkeletonStep(engine=capsule_skeleton_engine, max_workers=max_workers_skeleton, target_capsule_count=4),
            CapsuleItemSkeletonStep(engine=item_skeleton_engine, max_workers=max_workers_skeleton, target_item_count=5),
            # Step 5: Generate item content using RAG (can use vLLM)
            ItemContentGenerationStep(
                engine=provider.generation_engine,
                max_workers=max_workers_content,
                vectorstore_dir=str(paths.vectorstore_path),
            ),
            # Step 6: Validation & Correction
            HierarchicalValidationStep(
                engine=provider.validation_engine,
                max_workers=max_workers_validation,
                vectorstore_dir=str(paths.vectorstore_path),
                output_dir=str(paths.corrections_path),
            ),
            CorrectionQueueStep(corrections_dir=str(paths.corrections_path)),
            CorrectionApplicationStep(
                engine=provider.generation_engine,
                max_workers=max_workers_validation,
                corrections_dir=str(paths.corrections_path),
                output_dir=str(paths.skeletons_path),
                vectorstore_dir=str(paths.vectorstore_path),
            ),
        ]
    )

    return pipeline, provider, paths


def run_pipeline(
    *,
    data_root: str = "data",
    engine_type: str | None = None,
    skeleton_model: str = "gemini-2.0-flash",
    capsule_engine: str | None = None,
    capsule_model: str | None = None,
    item_engine: str | None = None,
    item_model: str = "gemini-2.0-flash",
    content_engine: str | None = None,
    content_model: str = "gemini-2.0-flash",
    validation_engine: str | None = None,
    validation_model: str = "gemini-2.0-flash-thinking-exp",
    base_url: str | None = None,
    validation_base_url: str | None = None,
    # Checkpoint options
    stop_after: str | None = None,
    resume_from: str | None = None,
    skip_sources: bool = False,
    book_path: str | None = None,
    certification: str | None = None,
    use_downloads: bool = False,
    collection_name: str | None = None,
):
    """Run the NREMT EMR pipeline with default configuration.

    Args:
        data_root: Root directory for all pipeline data.
        engine_type: Default engine (overridden by capsule/item/content/validation_engine).
        skeleton_model: Gemini model for skeleton generation (exam, course, labs - always Gemini).
        capsule_engine: Engine for capsule skeleton generation (can be vLLM).
        capsule_model: Model for capsule skeleton generation.
        item_engine: Engine for item skeleton generation (can be vLLM).
        item_model: Model for item skeleton generation.
        content_engine: Engine for content generation (can be vLLM).
        content_model: Model for content generation.
        validation_engine: Engine for validation.
        validation_model: Model for validation.
        base_url: Base URL for vLLM server (used for capsule/item/content if not specified).
        validation_base_url: Base URL for vLLM server validation.
        stop_after: Stop and save checkpoint after this stage (exam, course, labs, capsules, items, content).
        resume_from: Path to checkpoint file to resume from.
        skip_sources: Skip LibGen, download, extraction, and embedding steps.
        book_path: Path to local PDF file (skips LibGen/download, runs extraction/embedding).
        certification: Certification name (default: NREMT EMR).
        use_downloads: Use existing PDFs in downloads folder (skips LibGen/download).
        collection_name: Vectorstore collection name for RAG (default: derived from certification).
    """
    # Validate checkpoint options
    if stop_after and stop_after not in CHECKPOINT_STAGES:
        print(f"ERROR: Invalid --stop-after stage '{stop_after}'")
        print(f"  Valid stages: {', '.join(CHECKPOINT_STAGES)}")
        return

    if resume_from and not Path(resume_from).exists():
        print(f"ERROR: Checkpoint file not found: {resume_from}")
        return

    # Resolve engine types
    cap_engine = capsule_engine or engine_type or "gemini"
    itm_engine = item_engine or engine_type or "gemini"
    cont_engine = content_engine or engine_type or "gemini"
    val_engine = validation_engine or engine_type or "gemini"

    # Resolve models (capsule defaults to item_model for convenience)
    cap_model = capsule_model or item_model

    # If resuming, we may not need Gemini (depending on resume stage)
    resume_stage_idx = -1
    if resume_from:
        loaded_skeleton = load_checkpoint(resume_from)
        resume_stage_idx = get_stage_index(loaded_skeleton.checkpoint_stage)
        print(f"Resuming from checkpoint: {resume_from}")
        print(f"  Stage: {loaded_skeleton.checkpoint_stage}")
        print(f"  Version: {loaded_skeleton.version}")

    # Skeleton (exam, course, labs) always uses Gemini - check API key only if needed
    needs_gemini_skeleton = resume_stage_idx < get_stage_index("labs")
    if needs_gemini_skeleton and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set (needed for skeleton generation)")
        return

    # Check capsule engine requirements
    if cap_engine == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set (needed for capsule generation)")
        return
    if cap_engine == "vllm-server" and not base_url:
        print("ERROR: --base-url required for vllm-server capsule engine")
        return

    # Check item engine requirements
    if itm_engine == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set (needed for item generation)")
        return
    if itm_engine == "vllm-server" and not base_url:
        print("ERROR: --base-url required for vllm-server item engine")
        return

    # Check content engine requirements
    if cont_engine == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set (needed for content generation)")
        return
    if cont_engine == "vllm-server" and not base_url:
        print("ERROR: --base-url required for vllm-server content engine")
        return

    # Check validation engine requirements
    if val_engine == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set (needed for validation)")
        return
    if val_engine == "vllm-server" and not (validation_base_url or base_url):
        print("ERROR: --validation-base-url or --base-url required for vllm-server validation engine")
        return

    # Configure paths
    paths = configure_paths(root=data_root, ensure_dirs=True)

    # Define the certification (required for skeleton generation)
    if not certification:
        print("ERROR: --certification is required (e.g., 'NREMT EMT', 'AWS Solutions Architect')")
        return
    certification_name = certification

    # Books are optional - use --skip-sources if vectorstore already exists
    books_requested = []

    print("=" * 60)
    print(f"Running pipeline for: {certification_name}")
    if not skip_sources:
        print(f"Books to find: {len(books_requested)} (use --skip-sources if vectorstore exists)")
    print("=" * 60)
    print(f"\nData directories:")
    print(f"  Downloads:   {paths.downloads_path}")
    print(f"  Extracted:   {paths.extracted_path}")
    print(f"  Vectorstore: {paths.vectorstore_path}")
    print(f"  Skeletons:   {paths.skeletons_path}")
    print(f"  Corrections: {paths.corrections_path}")
    print("=" * 60)

    # Create pipeline and provider
    pipeline, provider, _ = create_pipeline(
        paths=paths,
        skeleton_model=skeleton_model,
        capsule_engine=cap_engine,
        capsule_model=cap_model,
        capsule_base_url=base_url,
        item_engine=itm_engine,
        item_model=item_model,
        item_base_url=base_url,
        content_engine=cont_engine,
        content_model=content_model,
        content_base_url=base_url,
        validation_engine=val_engine,
        validation_model=validation_model,
        validation_base_url=validation_base_url,
    )

    print(f"\nEngine Configuration:")
    print(f"  Skeleton (exam, course, labs - always Gemini):")
    print(f"    - Model: {skeleton_model}")
    print(f"  Capsule skeleton generation:")
    print(f"    - Engine: {cap_engine}")
    print(f"    - Model: {cap_model}")
    if base_url and cap_engine == "vllm-server":
        print(f"    - Base URL: {base_url}")
    print(f"  Item skeleton generation:")
    print(f"    - Engine: {itm_engine}")
    print(f"    - Model: {item_model}")
    if base_url and itm_engine == "vllm-server":
        print(f"    - Base URL: {base_url}")
    print(f"  Content generation:")
    print(f"    - Engine: {cont_engine}")
    print(f"    - Model: {content_model}")
    if base_url and cont_engine == "vllm-server":
        print(f"    - Base URL: {base_url}")
    print(f"  Validation:")
    print(f"    - Engine: {val_engine}")
    print(f"    - Model: {validation_model}")
    if validation_base_url:
        print(f"    - Base URL: {validation_base_url}")
    print("=" * 60)

    # Create initial context with paths info
    context = PipelineContext(
        certification_name=certification_name,
        books_requested=books_requested,
        # Store paths in context for steps that need them
        vectorstore_path=str(paths.vectorstore_path),
    )

    # If collection_name provided, use it for RAG
    if collection_name:
        context["collection_name"] = collection_name

    # If book_path provided, inject it as a downloaded book
    if book_path:
        book_file = Path(book_path)
        if not book_file.exists():
            print(f"ERROR: Book file not found: {book_path}")
            return
        print(f"\nUsing local book: {book_file.name}")
        context["books_downloaded"] = [
            DownloadedBook(
                title=book_file.stem,
                author="Unknown",
                extension=book_file.suffix.lstrip("."),
                file_path=str(book_file.absolute()),
                success=True,
            )
        ]

    # If use_downloads, scan downloads folder for all PDFs
    if use_downloads:
        pdf_files = list(paths.downloads_path.glob("**/*.pdf"))
        if not pdf_files:
            print(f"ERROR: No PDF files found in {paths.downloads_path}")
            return
        print(f"\nUsing {len(pdf_files)} PDF(s) from downloads folder:")
        books_downloaded = []
        for pdf_file in pdf_files:
            print(f"  - {pdf_file.name}")
            books_downloaded.append(
                DownloadedBook(
                    title=pdf_file.stem,
                    author="Unknown",
                    extension="pdf",
                    file_path=str(pdf_file.absolute()),
                    success=True,
                )
            )
        context["books_downloaded"] = books_downloaded

    # If resuming, inject the loaded skeleton
    if resume_from:
        context["course_skeleton"] = loaded_skeleton
        print(f"\n  Loaded skeleton with {len(loaded_skeleton.domain_modules)} domains")

    # Define stage boundaries for checkpoint saves
    # Maps stage name to (step_class, engine_name, model_name)
    stage_info = {
        "exam": (ExamFormatStep, "gemini", skeleton_model),
        "course": (CourseSkeletonStep, "gemini", skeleton_model),
        "labs": (LabSkeletonStep, "gemini", skeleton_model),
        "capsules": (CapsuleSkeletonStep, cap_engine, cap_model),
        "items": (CapsuleItemSkeletonStep, itm_engine, item_model),
        "content": (ItemContentGenerationStep, cont_engine, content_model),
        "validated": (CorrectionApplicationStep, val_engine, validation_model),
    }

    # Source step classes (for --skip-sources)
    all_source_steps = (LibgenLookupStep, BookDownloadStep, ContentExtractionStep, EmbeddingStep)
    download_steps = (LibgenLookupStep, BookDownloadStep)

    # Run the pipeline with checkpoint support
    try:
        stop_stage_idx = get_stage_index(stop_after) if stop_after else len(CHECKPOINT_STAGES)

        for step in pipeline.steps:
            step_class = type(step)

            # Skip all source steps if --skip-sources
            if skip_sources and isinstance(step, all_source_steps):
                print(f"  Skipping {step_class.__name__} (--skip-sources)")
                continue

            # Skip only download steps if --book-path or --use-downloads provided
            if book_path and isinstance(step, download_steps):
                print(f"  Skipping {step_class.__name__} (using --book-path)")
                continue

            if use_downloads and isinstance(step, download_steps):
                print(f"  Skipping {step_class.__name__} (using --use-downloads)")
                continue

            # Find which stage this step belongs to
            current_stage = None
            for stage_name, (stage_class, _, _) in stage_info.items():
                if step_class == stage_class:
                    current_stage = stage_name
                    break

            # Skip steps before resume stage
            if resume_from and current_stage:
                current_idx = get_stage_index(current_stage)
                if current_idx <= resume_stage_idx:
                    print(f"  Skipping {step_class.__name__} (already completed in checkpoint)")
                    continue

            # Run the step
            context = step.run(context)

            # Save checkpoint after stage completion
            if current_stage and "course_skeleton" in context:
                current_idx = get_stage_index(current_stage)
                checkpoint_path = save_checkpoint(
                    context["course_skeleton"],
                    current_stage,
                    paths.skeletons_path,
                    engine=stage_info[current_stage][1],
                    model=stage_info[current_stage][2],
                )
                print(f"  Checkpoint saved: {checkpoint_path.name}")

                # Stop if we've reached the stop_after stage
                if current_idx >= stop_stage_idx:
                    print("\n" + "=" * 60)
                    print(f"PIPELINE STOPPED AFTER '{stop_after}' STAGE")
                    print("=" * 60)
                    print(f"\nCheckpoint saved to: {checkpoint_path}")
                    print(f"\nTo resume, run:")
                    print(f"  course-builder emr --resume-from {checkpoint_path}")
                    return

        result = context

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        # Print summary
        if "books_found" in result:
            print(f"\nBooks found on LibGen: {len(result['books_found'])}")
        if "books_downloaded" in result:
            print(f"Books downloaded: {len(result['books_downloaded'])}")
        if "extracted_chunks" in result:
            print(f"Chunks extracted: {len(result['extracted_chunks'])}")
        if "course_skeleton" in result:
            skeleton = result["course_skeleton"]
            print(f"\nCourse Skeleton:")
            print(f"  - Domains: {len(skeleton.domain_modules)}")
            total_topics = sum(len(m.topics) for m in skeleton.domain_modules)
            print(f"  - Topics: {total_topics}")
            total_subtopics = sum(
                len(t.subtopics)
                for m in skeleton.domain_modules
                for t in m.topics
            )
            print(f"  - Subtopics: {total_subtopics}")
            total_labs = sum(
                len(st.labs)
                for m in skeleton.domain_modules
                for t in m.topics
                for st in t.subtopics
            )
            print(f"  - Labs: {total_labs}")
            total_capsules = sum(
                len(lab.capsules)
                for m in skeleton.domain_modules
                for t in m.topics
                for st in t.subtopics
                for lab in st.labs
            )
            print(f"  - Capsules: {total_capsules}")
            total_items = sum(
                len(cap.items)
                for m in skeleton.domain_modules
                for t in m.topics
                for st in t.subtopics
                for lab in st.labs
                for cap in lab.capsules
            )
            print(f"  - Capsule Items: {total_items}")

            # Count items with content (Step 5 completed)
            items_with_content = sum(
                1
                for m in skeleton.domain_modules
                for t in m.topics
                for st in t.subtopics
                for lab in st.labs
                for cap in lab.capsules
                for item in cap.items
                if item.content is not None
            )
            print(f"  - Items with Content: {items_with_content}")
            print(f"  - Version: {skeleton.version}")
            print(f"  - Validation Status: {skeleton.validation_status}")

        # Print Step 6 summary if available
        if "step6_output" in result:
            step6 = result["step6_output"]
            print(f"\nValidation & Correction (Step 6):")
            print(f"  - Entities validated: {step6.total_entities_validated}")
            print(f"  - Passed: {step6.passed_count}")
            print(f"  - Minor issues: {step6.minor_count}")
            print(f"  - Major issues: {step6.major_count}")
            print(f"  - Critical issues: {step6.critical_count}")
            print(f"  - Corrections applied: {step6.corrections_applied}")
            print(f"  - Corrections failed: {step6.corrections_failed}")
            print(f"  - Final status: {step6.validation_status}")

        # Show output location
        if "course_skeleton" in result:
            skeleton = result["course_skeleton"]
            cert_slug = certification_name.replace(" ", "_").replace("/", "_")
            print(f"\nOutput saved to: {paths.skeletons_path / f'{cert_slug}_skeleton_v{skeleton.version}.json'}")

    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
