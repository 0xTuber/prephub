"""Course Builder - AI-powered exam prep course generation.

This package provides tools for generating comprehensive exam preparation courses
using AI-driven content generation, RAG-based question creation, and automated
validation.

Main modules:
- config: Path configuration and settings
- engine: LLM generation engine abstraction (Gemini, vLLM)
- domain: Core data models (books, courses, content, validation)
- pipeline: Processing pipeline steps and orchestration
- pipelines: Pre-built pipeline configurations
"""

from course_builder.config import DataPaths, configure_paths

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DataPaths",
    "configure_paths",
]
