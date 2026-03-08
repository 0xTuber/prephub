"""Source material acquisition and processing steps.

This module handles the first three pipeline phases:
- lookup: Book discovery via LibGen
- download: Book downloading
- extract: Content extraction from PDFs
- embed: Vector embedding generation
"""

from course_builder.pipeline.sources.download import BookDownloadStep
from course_builder.pipeline.sources.embed import EmbeddingStep
from course_builder.pipeline.sources.extract import ContentExtractionStep
from course_builder.pipeline.sources.lookup import LibgenLookupStep

__all__ = [
    "LibgenLookupStep",
    "BookDownloadStep",
    "ContentExtractionStep",
    "EmbeddingStep",
]
