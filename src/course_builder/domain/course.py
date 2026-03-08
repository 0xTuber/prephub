"""Course structure domain models.

This module contains models for course structure generation:
- ExamFormat: Exam format and details (legacy)
- ExamFormatV2: Improved exam format with full structure
- CourseSkeleton: Complete course structure
- CourseModule, CourseTopic, SubTopic: Course hierarchy
- Lab, Capsule: Learning units
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from course_builder.domain.content import CapsuleItem


# =============================================================================
# Exam Format V2 - Improved schema with provenance and full structure
# =============================================================================


class SourceFact(BaseModel):
    """A fact with provenance tracking."""

    field: str  # Which field this fact relates to
    value: Any  # The value
    source_type: str = "inferred"  # "official", "third_party", "inferred"
    source_title: str | None = None  # e.g., "NREMT Candidate Handbook 2024"
    source_url: str | None = None
    confidence: str = "medium"  # "high", "medium", "low"
    notes: str | None = None  # Additional context


class ItemClass(BaseModel):
    """A question/item type with full classification.

    Captures both the interaction model (how the user responds) and
    the context model (scenario vs direct question).
    """

    name: str  # "multiple_choice", "multiple_response", "build_list", "drag_and_drop", "options_table"
    display_name: str | None = None  # Human-readable name
    is_tei: bool = False  # Technology Enhanced Item (non-traditional format)
    interaction_model: str = "single_select"  # "single_select", "multi_select", "order_list", "categorize", "table_select"
    allowed_context_types: list[str] = ["scenario", "direct"]  # What stem types are allowed
    patient_presence: str = "optional"  # "required", "optional", "not_required"
    age_representation: list[str] = ["exact", "broad", "none"]  # How patient age is represented
    description: str | None = None
    grading: str = "unknown"  # "all_or_nothing", "partial_credit", "unknown"
    estimated_percentage: float | None = None  # Approximate % of exam
    source_type: str = "inferred"  # "official", "third_party", "inferred"


class ExamDomainV2(BaseModel):
    """An exam domain with weight ranges."""

    name: str
    weight_min_pct: float | None = None
    weight_max_pct: float | None = None
    weight_pct: float | None = None  # Single value if range not available
    description: str | None = None
    num_items: int | None = None  # If known
    source_type: str = "inferred"


class ExamComponent(BaseModel):
    """A component of the exam (cognitive, psychomotor, etc.).

    Many certifications have multiple components - e.g., NREMT has both
    a cognitive (written) exam and psychomotor (skills) exam.
    """

    name: str  # "cognitive", "psychomotor", "practical", "oral"
    display_name: str | None = None
    description: str | None = None

    # Adaptive testing
    adaptive: bool = False
    adaptive_algorithm: str | None = None  # "CAT", "linear", etc.

    # Question counts (supports ranges for CAT exams)
    num_questions: int | None = None  # Single value if fixed
    num_questions_min: int | None = None  # Minimum for CAT
    num_questions_max: int | None = None  # Maximum for CAT
    pilot_unscored_items: int | None = None  # Unscored pilot questions

    # Time
    time_limit_minutes: int | None = None

    # Delivery
    delivery_methods: list[str] = []  # ["Pearson VUE", "OnVUE", "PSI"]

    # Structure
    domains: list[ExamDomainV2] = []
    item_classes: list[ItemClass] = []

    # Scoring
    passing_model: str | None = None  # "criterion-referenced", "norm-referenced", "competency-based"
    passing_score: str | None = None
    passing_score_source: str = "unknown"  # "official", "inferred", "unpublished"


class ExamFormatV2(BaseModel):
    """Improved exam format with full structure and provenance.

    Key improvements over ExamFormat:
    - Separates exam components (cognitive vs psychomotor)
    - Supports question count ranges (for CAT exams)
    - Tracks provenance for each fact
    - Distinguishes context types (scenario vs direct)
    - Captures TEI vs traditional item types
    - Handles patient presence and age representation rules
    """

    certification_name: str
    certification_code: str | None = None  # e.g., "EMR", "EMT", "AEMT", "Paramedic"
    certifying_body: str | None = None  # e.g., "NREMT", "AWS", "CompTIA"
    exam_code: str | None = None

    # Components (cognitive, psychomotor, etc.)
    exam_components: list[ExamComponent] = []

    # General info
    prerequisites: list[str] = []
    cost_usd: str | None = None
    validity_years: int | None = None
    languages: list[str] = []
    recertification_policy: str | None = None

    # Provenance tracking
    source_facts: list[SourceFact] = []
    discovery_queries: list[str] = []  # What searches were used

    # Raw data
    raw_sources: dict[str, str] = {}  # source_name -> raw_text

    # Metadata
    discovered_at: datetime | None = None
    schema_version: str = "2.0"


# =============================================================================
# Legacy Exam Format (V1) - Kept for backwards compatibility
# =============================================================================


class ExamDomain(BaseModel):
    """An exam domain or content area."""

    name: str
    weight_pct: float | None = None
    description: str | None = None


class QuestionType(BaseModel):
    """A type of question used in the exam."""

    name: str
    description: str | None = None
    purpose: str | None = None
    skeleton: str | None = None
    example: str | None = None
    grading_notes: str | None = None


class ExamFormat(BaseModel):
    """Exam format and details (legacy v1 schema)."""

    certification_name: str
    exam_code: str | None = None
    num_questions: int | None = None
    time_limit_minutes: int | None = None
    question_types: list[QuestionType] = []
    passing_score: str | None = None
    domains: list[ExamDomain] = []
    prerequisites: list[str] = []
    cost_usd: str | None = None
    validity_years: int | None = None
    delivery_methods: list[str] = []
    languages: list[str] = []
    recertification_policy: str | None = None
    additional_notes: str | None = None
    raw_description: str


class Step4Output(BaseModel):
    """Output of Step 4: Exam Format Discovery."""

    certification_name: str
    exam_code: str | None = None
    num_questions: int | None = None
    time_limit_minutes: int | None = None
    passing_score: str | None = None
    num_domains: int = 0
    num_question_types: int = 0
    delivery_methods: list[str] = []
    raw_description_length: int = 0


class ReasoningTemplate(BaseModel):
    """Template for reasoning through a question."""

    approach_steps: list[str]
    time_allocation_advice: str | None = None
    common_traps: list[str] = []


class ExplanationTemplate(BaseModel):
    """Template for explaining answers."""

    correct_answer_template: str
    wrong_answer_template: str
    partial_credit_template: str | None = None


class QuestionTypeGuide(BaseModel):
    """Guide for a specific question type."""

    question_type_name: str
    detailed_structure: str | None = None
    reasoning_template: ReasoningTemplate | None = None
    explanation_template: ExplanationTemplate | None = None
    difficulty_scaling_notes: str | None = None
    answer_choice_design_notes: str | None = None


class Capsule(BaseModel):
    """A micro-learning unit within a lab."""

    capsule_id: str  # e.g., "cap_01"
    title: str
    description: str | None = None
    learning_goal: str
    capsule_type: str  # "conceptual", "procedural", "case_study", "practice", "review"
    estimated_duration_minutes: float | None = None
    items: list[Any] = []  # CapsuleItem, using Any to avoid circular import
    prerequisites_within_lab: list[str] = []
    assessment_criteria: list[str] = []
    common_errors: list[str] = []

    @model_validator(mode="after")
    def deserialize_items(self) -> "Capsule":
        """Convert dict items to CapsuleItem objects."""
        from course_builder.domain.content import CapsuleItem

        if self.items:
            self.items = [
                CapsuleItem.model_validate(item) if isinstance(item, dict) else item
                for item in self.items
            ]
        return self


class Lab(BaseModel):
    """A hands-on learning activity within a subtopic."""

    lab_id: str  # e.g., "lab_01"
    title: str
    description: str | None = None
    objective: str
    lab_type: str  # "guided", "exploratory", "challenge", "simulation"
    estimated_duration_minutes: float | None = None
    tools_required: list[str] = []
    capsules: list[Capsule] = []
    prerequisites_within_subtopic: list[str] = []
    success_criteria: list[str] = []
    real_world_application: str | None = None


class SubTopic(BaseModel):
    """A subtopic within a course topic."""

    name: str
    description: str | None = None
    key_concepts: list[str] = []
    practical_skills: list[str] = []
    common_misconceptions: list[str] = []
    labs: list[Lab] = []


class LearningObjective(BaseModel):
    """A learning objective for a topic."""

    objective: str
    bloom_level: str | None = None
    relevant_question_types: list[str] = []


class CourseTopic(BaseModel):
    """A topic within a course module."""

    name: str
    description: str | None = None
    learning_objectives: list[LearningObjective] = []
    subtopics: list[SubTopic] = []
    estimated_study_hours: float | None = None


class CourseModule(BaseModel):
    """A module within a course, corresponding to an exam domain."""

    domain_name: str
    domain_weight_pct: float | None = None
    overview: str | None = None
    topics: list[CourseTopic] = []
    prerequisites_for_domain: list[str] = []
    recommended_study_order: list[str] = []
    official_references: list[str] = []


class StudyStrategy(BaseModel):
    """A recommended study strategy."""

    name: str
    description: str | None = None
    when_to_use: str | None = None


class CourseOverview(BaseModel):
    """Overview information for a course."""

    target_audience: str | None = None
    course_description: str | None = None
    total_estimated_study_hours: float | None = None
    study_strategies: list[StudyStrategy] = []
    exam_day_tips: list[str] = []
    prerequisites_detail: list[str] = []


class DomainModuleResult(BaseModel):
    """Result of generating a domain module."""

    domain_name: str
    module: CourseModule | None = None
    success: bool = True
    error: str | None = None


class LabGenerationResult(BaseModel):
    """Result of generating labs for a subtopic."""

    domain_name: str
    topic_name: str
    subtopic_name: str
    labs: list[Lab] = []
    success: bool = True
    error: str | None = None


class CapsuleGenerationResult(BaseModel):
    """Result of generating capsules for a lab."""

    domain_name: str
    topic_name: str
    subtopic_name: str
    lab_id: str
    capsules: list[Capsule] = []
    success: bool = True
    error: str | None = None


class CourseSkeleton(BaseModel):
    """Complete course skeleton structure."""

    certification_name: str
    exam_code: str | None = None
    overview: CourseOverview
    question_type_guides: list[QuestionTypeGuide] = []
    domain_modules: list[CourseModule] = []
    failed_domains: list[DomainModuleResult] = []
    version: int = 1
    validated_at: datetime | None = None
    validation_status: str | None = None  # "passed", "corrected", "needs_review"
    # Checkpoint tracking for pipeline resume
    checkpoint_stage: str | None = None  # "exam", "course", "labs", "capsules", "items", "content", "validated"
    checkpoint_engine: str | None = None  # Engine used for this stage (e.g., "gemini", "vllm")
    checkpoint_model: str | None = None  # Model used for this stage
