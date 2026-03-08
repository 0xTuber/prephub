"""Pipeline framework for course building.

This module provides the pipeline orchestration framework:
- base: Core pipeline classes (Pipeline, PipelineStep, PipelineContext)
- sources: Book discovery, download, extraction, and embedding steps
- skeleton: Course structure generation steps
- content: Content generation steps
- validation: Validation and correction steps
"""

from course_builder.pipeline.base import (
    EngineAwareStep,
    Pipeline,
    PipelineContext,
    PipelineStep,
)

__all__ = [
    "Pipeline",
    "PipelineStep",
    "PipelineContext",
    "EngineAwareStep",
]
