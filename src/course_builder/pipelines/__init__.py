"""Pre-built pipeline configurations.

This module provides ready-to-use pipeline configurations for generating
AI-powered certification exam preparation courses.
"""

from course_builder.pipelines.emr import create_pipeline, run_pipeline

# Backwards compatibility aliases
create_emr_pipeline = create_pipeline
run_emr_pipeline = run_pipeline

__all__ = [
    "create_pipeline",
    "run_pipeline",
    # Backwards compatibility
    "create_emr_pipeline",
    "run_emr_pipeline",
]
