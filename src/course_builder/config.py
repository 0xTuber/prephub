"""Configuration management for course_builder.

Centralizes all path configurations for pipeline artifacts and quality settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# =============================================================================
# Quality Mode Configuration
# =============================================================================


class QualityMode(str, Enum):
    """Quality mode for content generation pipeline.

    FAST: Skip expensive checks, generate more items quickly.
    BALANCED: Standard checks, good quality/speed tradeoff.
    STRICT: All checks enabled, fewer but higher quality items.
    """

    FAST = "fast"
    BALANCED = "balanced"
    STRICT = "strict"


@dataclass
class QualityConfig:
    """Configuration for quality checks in content generation.

    Controls which quality gates are enabled and their parameters.
    """

    mode: QualityMode = QualityMode.BALANCED

    # Phase 1: Foundation
    use_capsule_pooling: bool = True  # Pool retrieval at capsule level
    use_source_grounded_skeletons: bool = True  # Generate skeletons from source map
    use_anchor_driven_drafting: bool = True  # Select anchors before generating
    use_verification_loop: bool = True  # Verify claims against evidence

    # Phase 2: Polish
    use_ambiguity_gate: bool = False  # Two-pass answerability check (STRICT only)
    use_novelty_gate: bool = True  # Check for duplicate concepts
    use_controlled_distractors: bool = True  # Generate difficulty-appropriate distractors

    # Repair settings
    max_repair_attempts: int = 2  # Max attempts to fix a failed item
    reject_on_weak_grounding: bool = True  # Reject items with weak evidence

    # Retrieval settings
    capsule_pool_size: int = 80  # Chunks to retrieve per capsule
    chunks_per_item: int = 6  # Chunks allocated to each item
    min_chunks_per_item: int = 4  # Minimum required chunks

    # Thresholds
    evidence_coverage_threshold: float = 0.95  # Target for evidence coverage
    quote_verification_threshold: float = 0.90  # Target for quote verification
    novelty_similarity_threshold: float = 0.85  # Max similarity for novelty gate

    @classmethod
    def fast(cls) -> "QualityConfig":
        """Create a fast mode configuration."""
        return cls(
            mode=QualityMode.FAST,
            use_capsule_pooling=True,
            use_source_grounded_skeletons=False,
            use_anchor_driven_drafting=False,
            use_verification_loop=False,
            use_ambiguity_gate=False,
            use_novelty_gate=False,
            use_controlled_distractors=False,
            max_repair_attempts=0,
            reject_on_weak_grounding=False,
        )

    @classmethod
    def balanced(cls) -> "QualityConfig":
        """Create a balanced mode configuration."""
        return cls(
            mode=QualityMode.BALANCED,
            use_capsule_pooling=True,
            use_source_grounded_skeletons=True,
            use_anchor_driven_drafting=True,
            use_verification_loop=True,
            use_ambiguity_gate=False,
            use_novelty_gate=True,
            use_controlled_distractors=True,
            max_repair_attempts=2,
            reject_on_weak_grounding=True,
        )

    @classmethod
    def strict(cls) -> "QualityConfig":
        """Create a strict mode configuration."""
        return cls(
            mode=QualityMode.STRICT,
            use_capsule_pooling=True,
            use_source_grounded_skeletons=True,
            use_anchor_driven_drafting=True,
            use_verification_loop=True,
            use_ambiguity_gate=True,
            use_novelty_gate=True,
            use_controlled_distractors=True,
            max_repair_attempts=3,
            reject_on_weak_grounding=True,
            evidence_coverage_threshold=0.98,
            quote_verification_threshold=0.95,
        )

    @classmethod
    def from_mode(cls, mode: QualityMode | str) -> "QualityConfig":
        """Create configuration from mode name."""
        if isinstance(mode, str):
            mode = QualityMode(mode)
        if mode == QualityMode.FAST:
            return cls.fast()
        elif mode == QualityMode.STRICT:
            return cls.strict()
        else:
            return cls.balanced()


@dataclass
class DataPaths:
    """Centralized configuration for all data directories.

    Default structure:
        data/
        ├── sources/
        │   ├── downloads/     # Raw downloaded books (PDFs, EPUBs)
        │   └── extracted/     # Extracted content from books
        ├── vectorstore/       # ChromaDB vector embeddings
        └── output/
            ├── skeletons/     # Generated course skeletons (versioned JSON)
            └── corrections/   # Validation reports and correction queues

    Usage:
        # Use defaults (creates data/ in current directory)
        paths = DataPaths()

        # Custom root directory
        paths = DataPaths(root="./my_project_data")

        # Fully custom paths
        paths = DataPaths(
            root="./data",
            downloads="./custom/downloads",
        )

        # Access paths
        paths.downloads      # Path to downloads directory
        paths.extracted      # Path to extracted content
        paths.vectorstore    # Path to vectorstore
        paths.skeletons      # Path to skeleton output
        paths.corrections    # Path to corrections/validation
    """

    root: str | Path = "data"

    # Source materials (relative to root unless absolute)
    downloads: str | Path | None = None
    extracted: str | Path | None = None

    # Vector store
    vectorstore: str | Path | None = None

    # Output
    skeletons: str | Path | None = None
    corrections: str | Path | None = None

    def __post_init__(self):
        """Resolve all paths after initialization."""
        self._root = Path(self.root)

        # Set defaults relative to root
        self._downloads = self._resolve_path(self.downloads, "sources/downloads")
        self._extracted = self._resolve_path(self.extracted, "sources/extracted")
        self._vectorstore = self._resolve_path(self.vectorstore, "vectorstore")
        self._skeletons = self._resolve_path(self.skeletons, "output/skeletons")
        self._corrections = self._resolve_path(self.corrections, "output/corrections")

    def _resolve_path(self, custom: str | Path | None, default: str) -> Path:
        """Resolve a path, using default if not specified."""
        if custom is not None:
            path = Path(custom)
            if path.is_absolute():
                return path
            return self._root / path
        return self._root / default

    @property
    def root_path(self) -> Path:
        """Root data directory."""
        return self._root

    @property
    def downloads_path(self) -> Path:
        """Directory for downloaded books."""
        return self._downloads

    @property
    def extracted_path(self) -> Path:
        """Directory for extracted content."""
        return self._extracted

    @property
    def vectorstore_path(self) -> Path:
        """Directory for vector store."""
        return self._vectorstore

    @property
    def skeletons_path(self) -> Path:
        """Directory for generated skeletons."""
        return self._skeletons

    @property
    def corrections_path(self) -> Path:
        """Directory for validation and corrections."""
        return self._corrections

    def ensure_dirs(self) -> "DataPaths":
        """Create all directories if they don't exist.

        Returns self for method chaining.
        """
        for path in [
            self._downloads,
            self._extracted,
            self._vectorstore,
            self._skeletons,
            self._corrections,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self

    def __str__(self) -> str:
        return (
            f"DataPaths(\n"
            f"  root={self._root}\n"
            f"  downloads={self._downloads}\n"
            f"  extracted={self._extracted}\n"
            f"  vectorstore={self._vectorstore}\n"
            f"  skeletons={self._skeletons}\n"
            f"  corrections={self._corrections}\n"
            f")"
        )


# Global default paths instance
_default_paths: DataPaths | None = None


def get_default_paths() -> DataPaths:
    """Get the global default paths configuration."""
    global _default_paths
    if _default_paths is None:
        _default_paths = DataPaths()
    return _default_paths


def set_default_paths(paths: DataPaths) -> None:
    """Set the global default paths configuration."""
    global _default_paths
    _default_paths = paths


def configure_paths(
    root: str | Path = "data",
    downloads: str | Path | None = None,
    extracted: str | Path | None = None,
    vectorstore: str | Path | None = None,
    skeletons: str | Path | None = None,
    corrections: str | Path | None = None,
    ensure_dirs: bool = False,
) -> DataPaths:
    """Configure and return data paths.

    This is a convenience function that creates a DataPaths instance
    and optionally sets it as the global default.

    Args:
        root: Root directory for all data.
        downloads: Custom path for downloads.
        extracted: Custom path for extracted content.
        vectorstore: Custom path for vector store.
        skeletons: Custom path for skeleton output.
        corrections: Custom path for corrections.
        ensure_dirs: If True, create all directories.

    Returns:
        Configured DataPaths instance.
    """
    paths = DataPaths(
        root=root,
        downloads=downloads,
        extracted=extracted,
        vectorstore=vectorstore,
        skeletons=skeletons,
        corrections=corrections,
    )
    if ensure_dirs:
        paths.ensure_dirs()
    return paths
