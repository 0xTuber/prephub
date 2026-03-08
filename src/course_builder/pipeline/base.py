"""Pipeline framework base classes.

This module provides the core pipeline infrastructure:
- Pipeline: Orchestrates step execution
- PipelineStep: Base class for pipeline steps
- PipelineContext: Data container passed between steps
- EngineAwareStep: Base class for steps that use generation engines
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from course_builder.engine import GenerationEngine


class PipelineContext(dict):
    """Dict-like object that accumulates data as it flows through steps."""

    pass


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute this pipeline step.

        Args:
            context: The pipeline context containing data from previous steps.

        Returns:
            Updated pipeline context with this step's output.
        """
        ...


class EngineAwareStep(PipelineStep):
    """Base class for pipeline steps that use generation engines.

    This establishes the standard pattern for engine integration:
    - Engine is passed via keyword-only parameter
    - Engine can be None for backward compatibility (falls back to legacy behavior)
    - Steps should prefer using the provided engine over creating their own

    Example:
        class MyStep(EngineAwareStep):
            def __init__(
                self,
                *,  # Force keyword-only
                engine: GenerationEngine | None = None,
                max_workers: int = 8,
            ):
                super().__init__(engine=engine)
                self.max_workers = max_workers

            def run(self, context: PipelineContext) -> PipelineContext:
                engine = self.get_engine()
                if engine:
                    result = engine.generate("prompt")
                else:
                    # Legacy fallback
                    ...
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        engine: "GenerationEngine | None" = None,
    ):
        """Initialize with an optional generation engine.

        Args:
            engine: Generation engine to use. If None, step may fall back to
                   legacy behavior or raise an error.
        """
        self._engine = engine

    def get_engine(self) -> "GenerationEngine | None":
        """Get the configured generation engine."""
        return self._engine

    def require_engine(self) -> "GenerationEngine":
        """Get the engine, raising an error if not configured.

        Returns:
            The configured generation engine.

        Raises:
            RuntimeError: If no engine is configured.
        """
        if self._engine is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a generation engine. "
                f"Pass engine=... to the constructor."
            )
        return self._engine


class Pipeline:
    """Orchestrates execution of pipeline steps in sequence."""

    def __init__(self, steps: list[PipelineStep]):
        """Initialize the pipeline with a list of steps.

        Args:
            steps: List of pipeline steps to execute in order.
        """
        self.steps = steps

    def run(self, context: PipelineContext | None = None) -> PipelineContext:
        """Execute all pipeline steps in sequence.

        Args:
            context: Initial pipeline context. If None, creates empty context.

        Returns:
            Final pipeline context after all steps have executed.
        """
        if context is None:
            context = PipelineContext()
        for step in self.steps:
            context = step.run(context)
        return context
