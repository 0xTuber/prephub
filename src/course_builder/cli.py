"""Command-line interface for course_builder.

Usage:
    course-builder generate      Generate a certification exam prep course
    course-builder paths         Show configured data paths
    course-builder --help        Show help
"""

import argparse
import sys

from dotenv import load_dotenv


def main():
    """Main CLI entry point."""
    # Load .env file if present
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Course Builder - Generate certification exam preparation courses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  course-builder generate --certification "NREMT EMT"
      Generate an EMT certification prep course using Gemini

  course-builder generate --certification "AWS Solutions Architect" --skip-sources
      Generate course skeleton (assumes vectorstore already exists)

  course-builder generate --certification "NREMT EMR" --stop-after labs
      Run exam/course/labs with Gemini, save checkpoint, stop

  course-builder generate --certification "NREMT EMR" \\
                     --resume-from data/skeletons/NREMT_EMR_v1_labs.json \\
                     --capsule-engine vllm-server --item-engine vllm-server \\
                     --item-model meta-llama/Llama-3-8B-Instruct \\
                     --base-url http://localhost:8000/v1
      Resume from checkpoint, use vLLM for capsules/items/content

  course-builder generate --certification "CompTIA A+" \\
                     --engine-type vllm-server --base-url http://localhost:8000/v1
      Use vLLM server for generation; Gemini for exam format discovery

  course-builder paths
      Show default data paths
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command (main pipeline)
    generate_parser = subparsers.add_parser(
        "generate",
        aliases=["emr"],  # Backwards compatibility
        help="Generate a certification exam preparation course",
    )
    generate_parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for all pipeline data (default: data)",
    )
    generate_parser.add_argument(
        "--skeleton-model",
        default="gemini-2.0-flash",
        help="Gemini model for skeleton (exam format, course, labs). Always uses Gemini.",
    )
    generate_parser.add_argument(
        "--capsule-engine",
        default=None,
        choices=["gemini", "vllm", "vllm-server"],
        help="Engine for capsule skeleton generation (default: same as --engine-type or gemini)",
    )
    generate_parser.add_argument(
        "--capsule-model",
        default=None,
        help="Model for capsule skeleton generation (default: same as --item-model)",
    )
    generate_parser.add_argument(
        "--item-engine",
        default=None,
        choices=["gemini", "vllm", "vllm-server"],
        help="Engine for item skeleton generation (default: same as --engine-type or gemini)",
    )
    generate_parser.add_argument(
        "--item-model",
        default=None,
        help="Model for item skeleton generation (default: same as skeleton-model)",
    )
    generate_parser.add_argument(
        "--content-model",
        default=None,
        help="Model for content generation (default: same as item-model)",
    )
    generate_parser.add_argument(
        "--validation-model",
        default="gemini-2.0-flash-thinking-exp",
        help="Model for validation (default: gemini-2.0-flash-thinking-exp)",
    )
    generate_parser.add_argument(
        "--max-workers",
        type=int,
        default=30,
        help="Max concurrent workers (default: 30)",
    )
    generate_parser.add_argument(
        "--engine-type",
        default=None,
        choices=["gemini", "vllm", "vllm-server"],
        help="LLM engine for both generation and validation (use --generation-engine/--validation-engine for hybrid)",
    )
    generate_parser.add_argument(
        "--content-engine",
        default=None,
        choices=["gemini", "vllm", "vllm-server"],
        help="Engine for content generation (default: gemini). Skeleton always uses Gemini.",
    )
    generate_parser.add_argument(
        "--validation-engine",
        default=None,
        choices=["gemini", "vllm", "vllm-server"],
        help="Engine for validation (overrides --engine-type)",
    )
    generate_parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for vLLM server generation (e.g., http://localhost:8000/v1)",
    )
    generate_parser.add_argument(
        "--validation-base-url",
        default=None,
        help="Base URL for vLLM server validation (if different from --base-url)",
    )

    # Checkpoint options
    generate_parser.add_argument(
        "--stop-after",
        default=None,
        choices=["exam", "course", "labs", "capsules", "items", "content", "validated"],
        help="Stop and save checkpoint after this stage",
    )
    generate_parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to checkpoint file to resume from",
    )
    generate_parser.add_argument(
        "--skip-sources",
        action="store_true",
        help="Skip LibGen lookup, download, extraction, and embedding steps (use if book already processed)",
    )
    generate_parser.add_argument(
        "--book-path",
        default=None,
        help="Path to local PDF file (skips LibGen lookup and download, runs extraction and embedding)",
    )
    generate_parser.add_argument(
        "--certification",
        default=None,
        help="Certification name (default: NREMT EMR). Used for exam format search.",
    )
    generate_parser.add_argument(
        "--use-downloads",
        action="store_true",
        help="Use existing PDFs in data/sources/downloads folder (skips LibGen lookup and download)",
    )
    generate_parser.add_argument(
        "--collection-name",
        default=None,
        help="Vectorstore collection name for RAG (default: derived from certification name)",
    )

    # Checkpoints command - list available checkpoints
    checkpoints_parser = subparsers.add_parser(
        "checkpoints",
        help="List available checkpoints",
    )
    checkpoints_parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for all pipeline data (default: data)",
    )

    # Paths command - show configured paths
    paths_parser = subparsers.add_parser(
        "paths",
        help="Show configured data paths",
    )
    paths_parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for all pipeline data (default: data)",
    )

    # Batch extract command - extract and embed PDFs from folders
    batch_parser = subparsers.add_parser(
        "batch-extract",
        help="Extract and embed PDFs from organized folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  course-builder batch-extract data/
      Extract each subfolder into its own vectorstore

  course-builder batch-extract data/ --shared "paramedic and emt:NREMT_EMT,NREMT_Paramedic"
      Extract 'paramedic and emt' folder into both NREMT_EMT and NREMT_Paramedic vectorstores

  course-builder batch-extract data/ --dry-run
      Show what would be extracted without running
        """,
    )

    # Batch embed command - embed already-extracted MinerU content
    batch_embed_parser = subparsers.add_parser(
        "batch-embed",
        help="Embed already-extracted MinerU content into vectorstores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  course-builder batch-embed data/sources/extracted/
      Embed each subfolder into its own vectorstore

  course-builder batch-embed data/sources/extracted/ --shared "paramedic and emt:NREMT_EMT,NREMT_Paramedic"
      Embed 'paramedic and emt' folder into both NREMT_EMT and NREMT_Paramedic vectorstores

  course-builder batch-embed data/sources/extracted/ --dry-run
      Show what would be embedded without running

  course-builder batch-embed data/sources/extracted/ --delete-existing
      Delete existing collections before embedding
        """,
    )
    batch_embed_parser.add_argument(
        "extracted_dir",
        help="Directory containing extracted MinerU output (markdown + content_list.json)",
    )
    batch_embed_parser.add_argument(
        "--output-dir",
        default="data/vectorstore",
        help="Output directory for vectorstores (default: data/vectorstore)",
    )
    batch_embed_parser.add_argument(
        "--shared",
        action="append",
        default=[],
        help="Shared folder mapping: 'folder_name:collection1,collection2' (can be used multiple times)",
    )
    batch_embed_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be embedded without running",
    )
    batch_embed_parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing collections before embedding",
    )
    batch_parser.add_argument(
        "source_dir",
        help="Source directory containing certification folders with PDFs",
    )
    batch_parser.add_argument(
        "--output-dir",
        default="data/vectorstore",
        help="Output directory for vectorstores (default: data/vectorstore)",
    )
    batch_parser.add_argument(
        "--extracted-dir",
        default="data/sources/extracted",
        help="Directory for extracted content (default: data/sources/extracted)",
    )
    batch_parser.add_argument(
        "--shared",
        action="append",
        default=[],
        help="Shared folder mapping: 'folder_name:collection1,collection2' (can be used multiple times)",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without running",
    )
    batch_parser.add_argument(
        "--backend",
        default="hybrid-auto-engine",
        choices=["pipeline", "hybrid-auto-engine", "vlm-auto-engine"],
        help="MinerU backend: pipeline (CPU, ~82%% acc), hybrid-auto-engine (GPU+vLLM, ~90%% acc, default)",
    )

    args = parser.parse_args()

    if args.command in ("generate", "emr"):
        from course_builder.pipelines.emr import run_pipeline

        # Resolve defaults: item defaults to skeleton, content defaults to item
        item_model = args.item_model or args.skeleton_model
        content_model = args.content_model or item_model

        run_pipeline(
            data_root=args.data_root,
            engine_type=args.engine_type,
            skeleton_model=args.skeleton_model,
            capsule_engine=args.capsule_engine,
            capsule_model=args.capsule_model,
            item_engine=args.item_engine,
            item_model=item_model,
            content_engine=args.content_engine,
            content_model=content_model,
            validation_engine=args.validation_engine,
            validation_model=args.validation_model,
            base_url=args.base_url,
            validation_base_url=args.validation_base_url,
            stop_after=args.stop_after,
            resume_from=args.resume_from,
            skip_sources=args.skip_sources,
            book_path=args.book_path,
            certification=args.certification,
            use_downloads=args.use_downloads,
            collection_name=args.collection_name,
        )

    elif args.command == "checkpoints":
        from course_builder.config import configure_paths
        from course_builder.pipeline.checkpoint import list_checkpoints

        paths = configure_paths(root=args.data_root)
        checkpoints = list_checkpoints(paths.skeletons_path)

        if not checkpoints:
            print("No checkpoints found.")
            print(f"  (Looking in: {paths.skeletons_path.absolute()})")
        else:
            print("Available Checkpoints")
            print("=" * 70)
            for cp in checkpoints:
                print(f"\n  {cp['filepath'].name}")
                print(f"    Stage:    {cp['stage']}")
                print(f"    Version:  {cp['version']}")
                print(f"    Modified: {cp['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print("\n" + "=" * 70)
            print(f"To resume from a checkpoint:")
            print(f"  course-builder emr --resume-from <checkpoint-file>")

    elif args.command == "paths":
        from course_builder.config import configure_paths

        paths = configure_paths(root=args.data_root)
        print("Course Builder Data Paths")
        print("=" * 50)
        print(f"Root:        {paths.root_path.absolute()}")
        print(f"Downloads:   {paths.downloads_path.absolute()}")
        print(f"Extracted:   {paths.extracted_path.absolute()}")
        print(f"Vectorstore: {paths.vectorstore_path.absolute()}")
        print(f"Skeletons:   {paths.skeletons_path.absolute()}")
        print(f"Corrections: {paths.corrections_path.absolute()}")

    elif args.command == "batch-extract":
        from course_builder.pipelines.batch_extract import run_batch_extract

        # Parse shared folder mappings
        shared_mappings = {}
        for mapping in args.shared:
            if ":" not in mapping:
                print(f"ERROR: Invalid --shared format: '{mapping}'")
                print("  Expected format: 'folder_name:collection1,collection2'")
                sys.exit(1)
            folder, collections = mapping.split(":", 1)
            shared_mappings[folder.strip()] = [c.strip() for c in collections.split(",")]

        run_batch_extract(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            extracted_dir=args.extracted_dir,
            shared_mappings=shared_mappings,
            dry_run=args.dry_run,
            backend=args.backend,
        )

    elif args.command == "batch-embed":
        from course_builder.pipelines.batch_embed import run_batch_embed

        # Parse shared folder mappings
        shared_mappings = {}
        for mapping in args.shared:
            if ":" not in mapping:
                print(f"ERROR: Invalid --shared format: '{mapping}'")
                print("  Expected format: 'folder_name:collection1,collection2'")
                sys.exit(1)
            folder, collections = mapping.split(":", 1)
            shared_mappings[folder.strip()] = [c.strip() for c in collections.split(",")]

        run_batch_embed(
            extracted_dir=args.extracted_dir,
            output_dir=args.output_dir,
            shared_mappings=shared_mappings,
            dry_run=args.dry_run,
            delete_existing=args.delete_existing,
        )

    elif args.command is None:
        parser.print_help()
        sys.exit(1)

    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
