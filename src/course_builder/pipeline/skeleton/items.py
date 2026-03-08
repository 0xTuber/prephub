"""Capsule items skeleton generation step.

This module provides the CapsuleItemSkeletonStep which generates practice
question skeletons for each capsule in the course skeleton.

Supports any engine (Gemini, vLLM, etc.) via the engine abstraction.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from course_builder.domain.content import CapsuleItem, CapsuleItemGenerationResult
from course_builder.domain.course import CourseSkeleton
from course_builder.engine.base import GenerationEngine, GenerationConfig
from course_builder.pipeline.base import EngineAwareStep, PipelineContext

ITEMS_SYSTEM_PROMPT = (
    "You are an expert exam preparation content architect. "
    "Design practice question skeletons that will later be filled with content from source materials. "
    "Each item must be a practice question matching the exam's actual question types."
)

ITEMS_USER_PROMPT_TEMPLATE = """Design practice question skeletons for the following capsule in a "{certification}" ({exam_code}) certification preparation course.

CONTEXT:
- Domain: {domain_name} ({domain_weight}% of exam)
- Topic: {topic_name}
- Subtopic: {subtopic_name}
- Lab: {lab_title}

CAPSULE DETAILS:
- Title: {capsule_title}
- Learning Goal: {learning_goal}
- Capsule Type: {capsule_type}
- Assessment Criteria: {assessment_criteria}
- Common Errors to Address: {common_errors}

EXAM QUESTION TYPES (use ONLY these types):
{question_types}

Learning Objectives for This Topic:
{learning_objectives}

Generate exactly {target_item_count} practice question skeletons. Each item:
1. MUST use an item_type from the EXAM QUESTION TYPES above (e.g., "Multiple Choice", "Multiple Response", "Drag-and-Drop")
2. Should target a specific concept/skill the learner needs to master
3. DIFFICULTY CALIBRATION (5-Dimension Framework):

   BEGINNER (1-2 items):
   - Bloom's Level: Remember/Understand - direct recall
   - Reasoning: Single-step (if X, then Y)
   - Scenario: ONE finding, no hazards, controlled environment
   - Stem patterns: "What is...", "Which describes...", "Identify the..."
   - FORBIDDEN: "first", "priority", "most important", "initially"

   INTERMEDIATE (2-3 items):
   - Bloom's Level: Apply/Analyze - use concept in context
   - Reasoning: Two-step (Given A+B, therefore C)
   - Scenario: 2-3 findings, may have minor complication
   - Stem patterns: "Based on this presentation...", "Which action addresses..."
   - ALLOWED: "most appropriate" but NOT "first priority"

   ADVANCED (1-2 items):
   - Bloom's Level: Evaluate/Synthesize - prioritization judgment
   - Reasoning: Multi-step with competing priorities
   - Scenario: Multiple problems OR hazards OR time pressure
   - Stem patterns: "What is your FIRST priority?", "Before X, you should FIRST..."
   - REQUIRED: Must use "first"/"priority"/"initially" in stem

NOTE: You are only generating the skeleton/outline. The actual question content will be generated later using source materials.

Return a JSON object with an "items" key containing a list, each with:
- "item_id" (str, e.g., "item_01", "item_02")
- "item_type" (str — MUST be one of the exam question types listed above)
- "title" (str — brief title describing what this question tests)
- "learning_target" (str — the specific concept, skill, or knowledge this question assesses)
- "difficulty" (str — one of: "beginner", "intermediate", "advanced")

Return ONLY the JSON object, no other text."""


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _call_with_retry(fn, max_retries=3, base_delay=2.0):
    """Call function with exponential backoff retry."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_error


def _generate_items_for_capsule(
    engine: GenerationEngine,
    cert_name: str,
    exam_code: str | None,
    domain_name: str,
    domain_weight: float | None,
    topic,
    subtopic_name: str,
    lab_title: str,
    capsule,
    question_types: list[dict],  # Full question type info from exam format
    target_item_count: int,
) -> list[CapsuleItem]:
    """Generate item skeletons for a single capsule."""
    assessment_criteria = (
        "\n".join(f"- {c}" for c in capsule.assessment_criteria)
        if capsule.assessment_criteria
        else "None specified"
    )
    common_errors = (
        "\n".join(f"- {e}" for e in capsule.common_errors)
        if capsule.common_errors
        else "None specified"
    )

    # Extract question types from topic's learning objectives
    topic_question_types = set()
    for lo in topic.learning_objectives:
        topic_question_types.update(lo.relevant_question_types)

    # Filter to relevant types for this topic, or use all if none specified
    relevant_types = []
    for qt in question_types:
        if not topic_question_types or qt["name"] in topic_question_types:
            relevant_types.append(qt)

    if not relevant_types:
        relevant_types = question_types

    # Format question types with descriptions
    question_types_str = (
        "\n".join(
            f"- {qt['name']}: {qt.get('description', 'No description')}"
            for qt in relevant_types
        )
        if relevant_types
        else "- Multiple Choice: Standard 4-option question"
    )

    # Format learning objectives
    learning_objectives_str = (
        "\n".join(
            f"- {lo.objective} (Bloom level: {lo.bloom_level or 'N/A'})"
            for lo in topic.learning_objectives
        )
        if topic.learning_objectives
        else "None specified"
    )

    # Build prompt with system instruction included
    user_prompt = ITEMS_USER_PROMPT_TEMPLATE.format(
        certification=cert_name,
        exam_code=exam_code or 'N/A',
        domain_name=domain_name,
        domain_weight=domain_weight or 'unknown',
        topic_name=topic.name,
        subtopic_name=subtopic_name,
        lab_title=lab_title,
        capsule_title=capsule.title,
        learning_goal=capsule.learning_goal,
        capsule_type=capsule.capsule_type,
        assessment_criteria=assessment_criteria,
        common_errors=common_errors,
        question_types=question_types_str,
        learning_objectives=learning_objectives_str,
        target_item_count=target_item_count,
    )
    full_prompt = f"{ITEMS_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _call():
        return engine.generate(full_prompt, config=GenerationConfig(temperature=0.7))

    response = _call_with_retry(_call)
    raw_json = _strip_code_fences(response.text)
    data = json.loads(raw_json)

    items = []
    for item_data in data.get("items", []):
        items.append(
            CapsuleItem(
                item_id=item_data["item_id"],
                item_type=item_data["item_type"],
                title=item_data["title"],
                learning_target=item_data["learning_target"],
                difficulty=item_data.get("difficulty"),
                # Content fields are None - filled in Step 5
                content=None,
                options=None,
                correct_answer_index=None,
                explanation=None,
                source_reference=None,
            )
        )
    return items


def _fan_out_items_generation(
    engine: GenerationEngine,
    skeleton: CourseSkeleton,
    max_workers: int,
    target_item_count: int,
) -> tuple[CourseSkeleton, list[CapsuleItemGenerationResult]]:
    """Generate item skeletons for all capsules in parallel."""
    cert_name = skeleton.certification_name
    exam_code = skeleton.exam_code

    # Extract full question type info from skeleton
    question_types = [
        {"name": g.question_type_name, "description": g.detailed_structure}
        for g in skeleton.question_type_guides
    ]

    # Collect all capsules with their parent info
    capsule_tasks = []
    for module in skeleton.domain_modules:
        for topic in module.topics:
            for subtopic in topic.subtopics:
                for lab in subtopic.labs:
                    for capsule in lab.capsules:
                        capsule_tasks.append(
                            {
                                "domain_name": module.domain_name,
                                "domain_weight": module.domain_weight_pct,
                                "topic": topic,
                                "subtopic_name": subtopic.name,
                                "lab_id": lab.lab_id,
                                "lab_title": lab.title,
                                "capsule": capsule,
                            }
                        )

    if not capsule_tasks:
        return skeleton, []

    results: dict[int, CapsuleItemGenerationResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, task in enumerate(capsule_tasks):
            future = executor.submit(
                _generate_items_for_capsule,
                engine,
                cert_name,
                exam_code,
                task["domain_name"],
                task["domain_weight"],
                task["topic"],
                task["subtopic_name"],
                task["lab_title"],
                task["capsule"],
                question_types,
                target_item_count,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            task = capsule_tasks[idx]
            try:
                items = future.result()
                results[idx] = CapsuleItemGenerationResult(
                    domain_name=task["domain_name"],
                    topic_name=task["topic"].name,
                    subtopic_name=task["subtopic_name"],
                    lab_id=task["lab_id"],
                    capsule_id=task["capsule"].capsule_id,
                    items=items,
                    success=True,
                )
            except Exception as e:
                results[idx] = CapsuleItemGenerationResult(
                    domain_name=task["domain_name"],
                    topic_name=task["topic"].name,
                    subtopic_name=task["subtopic_name"],
                    lab_id=task["lab_id"],
                    capsule_id=task["capsule"].capsule_id,
                    items=[],
                    success=False,
                    error=str(e),
                )

    # Update skeleton with generated items
    failed = []
    for idx, task in enumerate(capsule_tasks):
        result = results[idx]
        if result.success:
            # Find and update the capsule
            for module in skeleton.domain_modules:
                if module.domain_name == task["domain_name"]:
                    for topic in module.topics:
                        if topic.name == task["topic"].name:
                            for subtopic in topic.subtopics:
                                if subtopic.name == task["subtopic_name"]:
                                    for lab in subtopic.labs:
                                        if lab.lab_id == task["lab_id"]:
                                            for capsule in lab.capsules:
                                                if (
                                                    capsule.capsule_id
                                                    == task["capsule"].capsule_id
                                                ):
                                                    capsule.items = result.items
        else:
            failed.append(result)

    return skeleton, failed


class CapsuleItemSkeletonStep(EngineAwareStep):
    """Generate practice question skeletons for each capsule.

    This step creates the outline for practice questions (item_type, title,
    learning_target, difficulty) but NOT the actual content. Content generation
    happens in Step 5 using RAG from source materials.

    Supports any engine (Gemini, vLLM, etc.) via the engine abstraction.
    """

    def __init__(
        self,
        *,
        engine: GenerationEngine | None = None,
        max_workers: int = 4,
        target_item_count: int = 5,
    ):
        """Initialize the capsule items skeleton step.

        Args:
            engine: The generation engine to use.
            max_workers: Maximum parallel workers.
            target_item_count: Number of items to generate per capsule.
        """
        super().__init__(engine=engine)
        self.max_workers = max_workers
        self.target_item_count = target_item_count

    def run(self, context: PipelineContext) -> PipelineContext:
        """Generate item skeletons for all capsules.

        Expects context["course_skeleton"].
        Updates context["course_skeleton"] with items.
        """
        engine = self.require_engine()
        skeleton: CourseSkeleton = context["course_skeleton"]
        cert_name = skeleton.certification_name
        print(f"Generating practice question skeletons for '{cert_name}'...")
        print(f"  Using engine: {engine.engine_type} / {engine.model_name}")

        # Count capsules
        capsule_count = sum(
            len(lab.capsules)
            for module in skeleton.domain_modules
            for topic in module.topics
            for subtopic in topic.subtopics
            for lab in subtopic.labs
        )
        print(
            f"  Generating skeletons for {capsule_count} capsules ({self.target_item_count} items each)..."
        )

        skeleton, failed = _fan_out_items_generation(
            engine, skeleton, self.max_workers, self.target_item_count
        )

        if failed:
            for f in failed:
                print(
                    f"  WARNING: Items for capsule '{f.capsule_id}' in lab '{f.lab_id}' failed: {f.error}"
                )

        # Store failed results for tracking
        if "failed_capsule_items" not in context:
            context["failed_capsule_items"] = []
        context["failed_capsule_items"].extend(failed)

        context["course_skeleton"] = skeleton
        print(f"Practice question skeletons complete for '{cert_name}'.\n")
        return context
