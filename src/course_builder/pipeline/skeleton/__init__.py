"""Course skeleton generation steps.

This module handles course structure generation:
- exam_format: Exam format discovery
- modules: Course module/topic generation
- labs: Lab structure generation
- capsules: Capsule generation
- items: Capsule item skeleton generation
"""

from course_builder.pipeline.skeleton.capsules import CapsuleSkeletonStep
from course_builder.pipeline.skeleton.exam_format import ExamFormatStep
from course_builder.pipeline.skeleton.items import CapsuleItemSkeletonStep
from course_builder.pipeline.skeleton.labs import LabSkeletonStep
from course_builder.pipeline.skeleton.modules import CourseSkeletonStep

__all__ = [
    "ExamFormatStep",
    "CourseSkeletonStep",
    "LabSkeletonStep",
    "CapsuleSkeletonStep",
    "CapsuleItemSkeletonStep",
]
