"""Checkpoint utilities for pipeline resume.

This module provides save/load functionality for course skeletons,
enabling pipeline resume from any checkpoint stage.

Stages (in order):
- exam: After ExamFormatStep
- course: After CourseSkeletonStep
- labs: After LabSkeletonStep
- capsules: After CapsuleSkeletonStep
- items: After CapsuleItemSkeletonStep
- content: After ItemContentGenerationStep
- validated: After validation steps
"""

import json
from datetime import datetime
from pathlib import Path

from course_builder.domain.course import CourseSkeleton

# Valid checkpoint stages in pipeline order
CHECKPOINT_STAGES = [
    "exam",
    "course",
    "labs",
    "capsules",
    "items",
    "content",
    "validated",
]


def get_checkpoint_filename(
    certification_name: str,
    stage: str,
    version: int = 1,
) -> str:
    """Generate a checkpoint filename.

    Args:
        certification_name: Name of the certification.
        stage: Checkpoint stage name.
        version: Skeleton version number.

    Returns:
        Filename like "NREMT_EMR_v1_labs.json"
    """
    # Sanitize certification name for filename
    safe_name = certification_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    return f"{safe_name}_v{version}_{stage}.json"


def save_checkpoint(
    skeleton: CourseSkeleton,
    stage: str,
    output_dir: Path | str,
    engine: str | None = None,
    model: str | None = None,
) -> Path:
    """Save a skeleton checkpoint to disk.

    Args:
        skeleton: The course skeleton to save.
        stage: The checkpoint stage name.
        output_dir: Directory to save the checkpoint.
        engine: Engine used for this stage (for tracking).
        model: Model used for this stage (for tracking).

    Returns:
        Path to the saved checkpoint file.
    """
    if stage not in CHECKPOINT_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of: {CHECKPOINT_STAGES}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update skeleton with checkpoint info
    skeleton.checkpoint_stage = stage
    skeleton.checkpoint_engine = engine
    skeleton.checkpoint_model = model

    filename = get_checkpoint_filename(
        skeleton.certification_name,
        stage,
        skeleton.version,
    )
    filepath = output_dir / filename

    # Save as JSON
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(skeleton.model_dump_json(indent=2))

    return filepath


def load_checkpoint(filepath: Path | str) -> CourseSkeleton:
    """Load a skeleton checkpoint from disk.

    Args:
        filepath: Path to the checkpoint file.

    Returns:
        The loaded CourseSkeleton.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return CourseSkeleton.model_validate(data)


def list_checkpoints(checkpoint_dir: Path | str) -> list[dict]:
    """List all available checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        List of checkpoint info dicts with keys:
        - filepath: Path to the file
        - certification: Certification name
        - stage: Checkpoint stage
        - version: Skeleton version
        - modified: Last modified timestamp
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for filepath in checkpoint_dir.glob("*.json"):
        try:
            skeleton = load_checkpoint(filepath)
            checkpoints.append({
                "filepath": filepath,
                "certification": skeleton.certification_name,
                "stage": skeleton.checkpoint_stage,
                "version": skeleton.version,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime),
            })
        except Exception:
            # Skip invalid files
            continue

    # Sort by modification time, newest first
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)
    return checkpoints


def get_stage_index(stage: str) -> int:
    """Get the index of a stage in the pipeline order.

    Args:
        stage: Stage name.

    Returns:
        Index of the stage (0-based).
    """
    if stage not in CHECKPOINT_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of: {CHECKPOINT_STAGES}")
    return CHECKPOINT_STAGES.index(stage)


def get_stages_after(stage: str) -> list[str]:
    """Get all stages that come after a given stage.

    Args:
        stage: Stage name.

    Returns:
        List of stage names that come after the given stage.
    """
    idx = get_stage_index(stage)
    return CHECKPOINT_STAGES[idx + 1:]


def get_stages_up_to(stage: str) -> list[str]:
    """Get all stages up to and including a given stage.

    Args:
        stage: Stage name.

    Returns:
        List of stage names up to and including the given stage.
    """
    idx = get_stage_index(stage)
    return CHECKPOINT_STAGES[:idx + 1]
