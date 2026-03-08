"""Validation rules for the Validation & Correction Pipeline.

Rules are organized by level in the hierarchy and type:
- Structural rules: Fast checks without LLM (content presence, format, etc.)
- Grounding rules: RAG verification against source material
- Quality rules: LLM-based review of content quality
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Severity(str, Enum):
    """Validation severity levels."""

    PASSED = "passed"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

    def _severity_order(self) -> int:
        order = {
            Severity.PASSED: 0,
            Severity.MINOR: 1,
            Severity.MAJOR: 2,
            Severity.CRITICAL: 3,
        }
        return order[self]

    def __lt__(self, other: "Severity") -> bool:
        return self._severity_order() < other._severity_order()

    def __le__(self, other: "Severity") -> bool:
        return self._severity_order() <= other._severity_order()

    def __gt__(self, other: "Severity") -> bool:
        return self._severity_order() > other._severity_order()

    def __ge__(self, other: "Severity") -> bool:
        return self._severity_order() >= other._severity_order()


@dataclass
class ValidationRule:
    """Definition of a validation rule."""

    name: str
    description: str
    entity_type: str  # "skeleton", "module", "topic", "subtopic", "lab", "capsule", "item"
    rule_type: str  # "structural", "grounding", "quality"
    severity_if_failed: Severity
    check_fn: Callable[[Any], tuple[bool, str | None, str | None]]  # (passed, description, suggested_fix)


# ============================================================================
# STRUCTURAL RULES (Fast, No LLM)
# ============================================================================


def check_content_not_empty(item) -> tuple[bool, str | None, str | None]:
    """Check that item content is not empty."""
    if item.content is None or len(str(item.content).strip()) < 10:
        return False, "Item content is missing or too short (< 10 chars)", None
    return True, None, None


def check_options_present(item) -> tuple[bool, str | None, str | None]:
    """Check that item has answer options."""
    if item.options is None or len(item.options) < 2:
        return False, "Item must have at least 2 answer options", None
    return True, None, None


def check_options_count_valid(item) -> tuple[bool, str | None, str | None]:
    """Check option count matches item type (MCQ=4, MR=5-6)."""
    if item.options is None:
        return True, None, None  # Handled by options_present

    item_type_lower = (item.item_type or "").lower()
    num_options = len(item.options)

    if "multiple choice" in item_type_lower or "mcq" in item_type_lower:
        if num_options != 4:
            return (
                False,
                f"MCQ should have exactly 4 options, found {num_options}",
                "Regenerate with correct option count",
            )
    elif "multiple response" in item_type_lower or "multi-select" in item_type_lower:
        if num_options < 5 or num_options > 6:
            return (
                False,
                f"Multiple response should have 5-6 options, found {num_options}",
                "Regenerate with correct option count",
            )

    return True, None, None


def check_correct_index_valid(item) -> tuple[bool, str | None, str | None]:
    """Check that correct answer index is within bounds."""
    if item.options is None or item.correct_answer_index is None:
        return True, None, None  # Handled by other rules

    if not (0 <= item.correct_answer_index < len(item.options)):
        return (
            False,
            f"Correct answer index {item.correct_answer_index} out of bounds for {len(item.options)} options",
            None,
        )
    return True, None, None


def check_explanation_present(item) -> tuple[bool, str | None, str | None]:
    """Check that explanation is present and meaningful."""
    if item.explanation is None or len(str(item.explanation).strip()) < 20:
        return False, "Explanation is missing or too short (< 20 chars)", None
    return True, None, None


def check_source_reference_present(item) -> tuple[bool, str | None, str | None]:
    """Check that source reference is present."""
    if item.source_reference is None:
        return False, "Source reference is missing", None
    return True, None, None


def check_chunk_ids_present(item) -> tuple[bool, str | None, str | None]:
    """Check that chunk IDs are present in source reference."""
    if item.source_reference is None:
        return True, None, None  # Handled by source_reference_present

    if not item.source_reference.chunk_ids or len(item.source_reference.chunk_ids) == 0:
        return False, "No chunk IDs in source reference", None
    return True, None, None


def check_no_duplicate_options(item) -> tuple[bool, str | None, str | None]:
    """Check that all options are unique."""
    if item.options is None:
        return True, None, None

    unique = set(o.strip().lower() for o in item.options)
    if len(unique) != len(item.options):
        return False, "Duplicate answer options detected", "Regenerate with unique options"
    return True, None, None


def check_module_not_empty(module) -> tuple[bool, str | None, str | None]:
    """Check that module has topics."""
    if not module.topics or len(module.topics) == 0:
        return False, "Module has no topics", None
    return True, None, None


def check_module_has_overview(module) -> tuple[bool, str | None, str | None]:
    """Check that module has an overview."""
    if not module.overview or len(str(module.overview).strip()) < 10:
        return False, "Module overview is missing or too short", None
    return True, None, None


def check_topic_has_subtopics(topic) -> tuple[bool, str | None, str | None]:
    """Check that topic has subtopics."""
    if not topic.subtopics or len(topic.subtopics) == 0:
        return False, "Topic has no subtopics", None
    return True, None, None


def check_topic_has_learning_objectives(topic) -> tuple[bool, str | None, str | None]:
    """Check that topic has learning objectives."""
    if not topic.learning_objectives or len(topic.learning_objectives) == 0:
        return False, "Topic has no learning objectives", None
    return True, None, None


def check_subtopic_has_labs(subtopic) -> tuple[bool, str | None, str | None]:
    """Check that subtopic has at least 1 lab."""
    if not subtopic.labs or len(subtopic.labs) == 0:
        return False, "Subtopic has no labs", None
    return True, None, None


def check_subtopic_has_description(subtopic) -> tuple[bool, str | None, str | None]:
    """Check that subtopic has a description."""
    if not subtopic.description or len(str(subtopic.description).strip()) < 10:
        return False, "Subtopic description is missing or too short", None
    return True, None, None


def check_lab_has_capsules(lab) -> tuple[bool, str | None, str | None]:
    """Check that lab has at least 1 capsule."""
    if not lab.capsules or len(lab.capsules) == 0:
        return False, "Lab has no capsules", None
    return True, None, None


def check_lab_has_objective(lab) -> tuple[bool, str | None, str | None]:
    """Check that lab has a clear objective."""
    if not lab.objective or len(str(lab.objective).strip()) < 10:
        return False, "Lab objective is missing or too short", None
    return True, None, None


def check_lab_type_valid(lab) -> tuple[bool, str | None, str | None]:
    """Check that lab type is valid."""
    valid_types = {"guided", "exploratory", "challenge", "simulation"}
    if not lab.lab_type or lab.lab_type.lower() not in valid_types:
        return (
            False,
            f"Invalid lab type: {lab.lab_type}. Valid: {valid_types}",
            f"Use one of: {', '.join(valid_types)}",
        )
    return True, None, None


def check_capsule_has_items(capsule) -> tuple[bool, str | None, str | None]:
    """Check that capsule has at least 1 item."""
    if not capsule.items or len(capsule.items) == 0:
        return False, "Capsule has no items", None
    return True, None, None


def check_capsule_has_learning_goal(capsule) -> tuple[bool, str | None, str | None]:
    """Check that capsule has a learning goal."""
    if not capsule.learning_goal or len(str(capsule.learning_goal).strip()) < 10:
        return False, "Capsule learning goal is missing or too short", None
    return True, None, None


def check_capsule_type_valid(capsule) -> tuple[bool, str | None, str | None]:
    """Check that capsule type is valid."""
    valid_types = {"conceptual", "procedural", "case_study", "practice", "review"}
    if not capsule.capsule_type or capsule.capsule_type.lower() not in valid_types:
        return (
            False,
            f"Invalid capsule type: {capsule.capsule_type}. Valid: {valid_types}",
            f"Use one of: {', '.join(valid_types)}",
        )
    return True, None, None


def check_skeleton_has_domains(skeleton) -> tuple[bool, str | None, str | None]:
    """Check that skeleton has domain modules."""
    if not skeleton.domain_modules or len(skeleton.domain_modules) == 0:
        return False, "Skeleton has no domain modules", None
    return True, None, None


def check_skeleton_has_overview(skeleton) -> tuple[bool, str | None, str | None]:
    """Check that skeleton has an overview."""
    if not skeleton.overview:
        return False, "Skeleton overview is missing", None
    return True, None, None


def check_domain_weights_sum(skeleton) -> tuple[bool, str | None, str | None]:
    """Check that domain weights sum to approximately 100%."""
    if not skeleton.domain_modules:
        return True, None, None

    weights = [m.domain_weight_pct for m in skeleton.domain_modules if m.domain_weight_pct is not None]
    if not weights:
        return True, None, None  # No weights specified, skip

    total = sum(weights)
    if not (95 <= total <= 105):
        return (
            False,
            f"Domain weights sum to {total}%, should be ~100%",
            "Adjust domain weights to sum to 100%",
        )
    return True, None, None


# ============================================================================
# RULE REGISTRY
# ============================================================================

# Structural rules for each entity type
STRUCTURAL_RULES: dict[str, list[ValidationRule]] = {
    "skeleton": [
        ValidationRule(
            name="skeleton_has_domains",
            description="Skeleton must have domain modules",
            entity_type="skeleton",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_skeleton_has_domains,
        ),
        ValidationRule(
            name="skeleton_has_overview",
            description="Skeleton must have an overview",
            entity_type="skeleton",
            rule_type="structural",
            severity_if_failed=Severity.MAJOR,
            check_fn=check_skeleton_has_overview,
        ),
        ValidationRule(
            name="domain_weights_sum",
            description="Domain weights should sum to ~100%",
            entity_type="skeleton",
            rule_type="structural",
            severity_if_failed=Severity.MINOR,
            check_fn=check_domain_weights_sum,
        ),
    ],
    "module": [
        ValidationRule(
            name="module_not_empty",
            description="Module must have topics",
            entity_type="module",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_module_not_empty,
        ),
        ValidationRule(
            name="module_has_overview",
            description="Module should have an overview",
            entity_type="module",
            rule_type="structural",
            severity_if_failed=Severity.MINOR,
            check_fn=check_module_has_overview,
        ),
    ],
    "topic": [
        ValidationRule(
            name="topic_has_subtopics",
            description="Topic must have subtopics",
            entity_type="topic",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_topic_has_subtopics,
        ),
        ValidationRule(
            name="topic_has_learning_objectives",
            description="Topic should have learning objectives",
            entity_type="topic",
            rule_type="structural",
            severity_if_failed=Severity.MAJOR,
            check_fn=check_topic_has_learning_objectives,
        ),
    ],
    "subtopic": [
        ValidationRule(
            name="subtopic_has_labs",
            description="Subtopic must have at least 1 lab",
            entity_type="subtopic",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_subtopic_has_labs,
        ),
        ValidationRule(
            name="subtopic_has_description",
            description="Subtopic should have a description",
            entity_type="subtopic",
            rule_type="structural",
            severity_if_failed=Severity.MINOR,
            check_fn=check_subtopic_has_description,
        ),
    ],
    "lab": [
        ValidationRule(
            name="lab_has_capsules",
            description="Lab must have at least 1 capsule",
            entity_type="lab",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_lab_has_capsules,
        ),
        ValidationRule(
            name="lab_has_objective",
            description="Lab must have a clear objective",
            entity_type="lab",
            rule_type="structural",
            severity_if_failed=Severity.MAJOR,
            check_fn=check_lab_has_objective,
        ),
        ValidationRule(
            name="lab_type_valid",
            description="Lab type must be valid",
            entity_type="lab",
            rule_type="structural",
            severity_if_failed=Severity.MINOR,
            check_fn=check_lab_type_valid,
        ),
    ],
    "capsule": [
        ValidationRule(
            name="capsule_has_items",
            description="Capsule must have at least 1 item",
            entity_type="capsule",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_capsule_has_items,
        ),
        ValidationRule(
            name="capsule_has_learning_goal",
            description="Capsule must have a learning goal",
            entity_type="capsule",
            rule_type="structural",
            severity_if_failed=Severity.MAJOR,
            check_fn=check_capsule_has_learning_goal,
        ),
        ValidationRule(
            name="capsule_type_valid",
            description="Capsule type must be valid",
            entity_type="capsule",
            rule_type="structural",
            severity_if_failed=Severity.MINOR,
            check_fn=check_capsule_type_valid,
        ),
    ],
    "item": [
        ValidationRule(
            name="content_not_empty",
            description="Item content must not be empty",
            entity_type="item",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_content_not_empty,
        ),
        ValidationRule(
            name="options_present",
            description="Item must have answer options",
            entity_type="item",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_options_present,
        ),
        ValidationRule(
            name="options_count_valid",
            description="Option count must match question type",
            entity_type="item",
            rule_type="structural",
            severity_if_failed=Severity.MAJOR,
            check_fn=check_options_count_valid,
        ),
        ValidationRule(
            name="correct_index_valid",
            description="Correct answer index must be valid",
            entity_type="item",
            rule_type="structural",
            severity_if_failed=Severity.CRITICAL,
            check_fn=check_correct_index_valid,
        ),
        ValidationRule(
            name="explanation_present",
            description="Explanation must be present",
            entity_type="item",
            rule_type="structural",
            severity_if_failed=Severity.MAJOR,
            check_fn=check_explanation_present,
        ),
        ValidationRule(
            name="source_reference_present",
            description="Source reference must be present",
            entity_type="item",
            rule_type="structural",
            severity_if_failed=Severity.MAJOR,
            check_fn=check_source_reference_present,
        ),
        ValidationRule(
            name="chunk_ids_present",
            description="Chunk IDs must be present",
            entity_type="item",
            rule_type="structural",
            severity_if_failed=Severity.MINOR,
            check_fn=check_chunk_ids_present,
        ),
        ValidationRule(
            name="no_duplicate_options",
            description="All options must be unique",
            entity_type="item",
            rule_type="structural",
            severity_if_failed=Severity.MAJOR,
            check_fn=check_no_duplicate_options,
        ),
    ],
}


def get_structural_rules(entity_type: str) -> list[ValidationRule]:
    """Get structural validation rules for an entity type."""
    return STRUCTURAL_RULES.get(entity_type, [])


def get_worst_severity(severities: list[Severity]) -> Severity:
    """Get the worst (highest) severity from a list."""
    if not severities:
        return Severity.PASSED
    return max(severities)


# ============================================================================
# QUALITY REVIEW PROMPT (for LLM-based validation)
# ============================================================================

QUALITY_REVIEW_PROMPT = """You are a quality reviewer for certification exam practice questions.

Review this practice question and identify any issues:

CERTIFICATION: {certification}
DOMAIN: {domain_name}
TOPIC: {topic_name}
LEARNING TARGET: {learning_target}

QUESTION:
{content}

OPTIONS:
{options_formatted}

CORRECT ANSWER: {correct_answer}
EXPLANATION: {explanation}

SOURCE MATERIAL USED:
{source_chunks}

Review for these issues:
1. CLARITY: Is the question unambiguous? Could multiple answers seem correct?
2. CORRECTNESS: Is the marked answer definitively correct based on the source material?
3. GROUNDING: Is the content factually accurate per the source material?
4. ALIGNMENT: Does this question test the stated learning target?
5. DISTRACTORS: Are the wrong answers plausible but clearly wrong?

Return a JSON object:
{{
  "overall_quality": "passed" | "minor" | "major" | "critical",
  "issues": [
    {{
      "rule_name": "...",
      "severity": "...",
      "description": "...",
      "suggested_fix": "..." (optional)
    }}
  ],
  "recommendation": "keep" | "fix" | "regenerate"
}}
"""


GROUNDING_CHECK_PROMPT = """Verify if the following answer is supported by the source material.

QUESTION: {content}
CORRECT ANSWER: {correct_answer}
EXPLANATION: {explanation}

SOURCE MATERIAL:
{source_chunks}

Is the correct answer directly supported by the source material?
Return a JSON object:
{{
  "is_grounded": true | false,
  "confidence": 0.0-1.0,
  "evidence": "Quote from source that supports/refutes the answer",
  "issues": "Description of any grounding problems" (if not grounded)
}}
"""
