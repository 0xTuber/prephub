"""Schema learning domain models.

This module contains models for question type schema learning:
- CertificationSchema: Complete schema for a certification's question types
- QuestionTypeSchema: Schema for a specific question type
- ExhibitType, ScenarioTemplate: Supporting types
"""

from pydantic import BaseModel


class ExhibitField(BaseModel):
    """A field within an exhibit template."""

    name: str
    description: str | None = None
    field_type: str = "text"  # "text", "number", "date", "list", "table"
    required: bool = True
    example: str | None = None


class ExhibitType(BaseModel):
    """A type of document/artifact that can be embedded in questions."""

    name: str  # e.g., "medication_administration_record", "network_diagram"
    display_name: str  # e.g., "Medication Administration Record (MAR)"
    description: str | None = None
    category: str | None = None  # e.g., "clinical_documentation", "diagnostic"
    fields: list[ExhibitField] = []
    example_content: str | None = None
    typical_usage: str | None = None  # When this exhibit type is typically used


class ScenarioTemplate(BaseModel):
    """A template for situation/scenario structure."""

    name: str  # e.g., "emergency_admission", "routine_checkup"
    display_name: str
    description: str | None = None
    context_type: str | None = None  # e.g., "inpatient", "outpatient", "community"
    typical_elements: list[str] = []  # What info is usually included
    compatible_exhibit_types: list[str] = []  # Which exhibits commonly appear
    example_opening: str | None = None


class QuestionTypeSchema(BaseModel):
    """Certification-specific schema for a question type."""

    type_name: str  # Normalized name like "clinical_scenario_mc"
    display_name: str  # Human-readable like "Clinical Scenario Multiple Choice"
    base_type: str  # Which base type it extends: "multiple_choice", "case_study", etc.
    description: str | None = None

    # Structure customizations
    has_situation: bool = False
    situation_template: str | None = None  # Reference to ScenarioTemplate name
    supports_exhibits: bool = False
    allowed_exhibit_types: list[str] = []  # References to ExhibitType names
    max_exhibits: int | None = None

    # Answer format
    num_choices: int = 4
    num_correct: int = 1  # 1 for MC, 2+ for multiple response

    # Domain-specific fields
    domain_specific_fields: dict[str, str] = {}  # field_name -> description

    # Prompt guidance for LLM
    generation_guidance: str | None = None  # Tips for generating this question type
    common_pitfalls: list[str] = []  # What to avoid


class CertificationSchema(BaseModel):
    """Complete schema for a certification's question types."""

    certification_name: str
    certification_slug: str  # Folder name, e.g., "oiiq", "aws_saa_c03"
    version: str = "1.0"
    created_at: str | None = None
    updated_at: str | None = None

    # Domain context
    professional_domain: str | None = None  # "nursing", "cloud_computing", "law"
    domain_description: str | None = None

    # The schemas themselves
    question_types: list[QuestionTypeSchema] = []
    exhibit_types: list[ExhibitType] = []
    scenario_templates: list[ScenarioTemplate] = []

    # Mappings from exam format question types to our schemas
    type_mappings: dict[str, str] = {}  # "Multiple Choice" -> "clinical_scenario_mc"


class SchemaLearningOutput(BaseModel):
    """Output of the schema learning step."""

    certification_name: str
    certification_slug: str
    schema_dir: str
    question_types_count: int
    exhibit_types_count: int
    scenario_templates_count: int
    user_refinements_made: bool = False
