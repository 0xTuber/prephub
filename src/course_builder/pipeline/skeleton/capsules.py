"""Capsules skeleton generation step.

This module provides the CapsuleSkeletonStep which generates micro-learning
capsule structures for each lab in the course skeleton.

Supports any engine (Gemini, vLLM, etc.) via the engine abstraction.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from course_builder.domain.course import Capsule, CapsuleGenerationResult, CourseSkeleton
from course_builder.engine.base import GenerationConfig, GenerationEngine
from course_builder.pipeline.base import EngineAwareStep, PipelineContext

CAPSULES_SYSTEM_PROMPT = (
    "You are an expert instructional designer specializing in micro-learning content. "
    "Create detailed, complete capsules that leave no gaps in the learner's understanding "
    "and provide comprehensive coverage of all learning objectives."
)

CAPSULES_USER_PROMPT_TEMPLATE = """Design comprehensive micro-learning capsules for the following lab in a "{certification}" ({exam_code}) certification preparation course.

CONTEXT:
- Domain: {domain_name}
- Topic: {topic_name}
- Subtopic: {subtopic_name}

LAB DETAILS:
- Title: {lab_title}
- Objective: {lab_objective}
- Lab Type: {lab_type}
- Success Criteria: {success_criteria}
- Real-World Application: {real_world_application}

EXAM QUESTION TYPES TO PREPARE FOR:
{question_types}

Learning Objectives for This Topic:
{learning_objectives}

Generate exactly {target_capsule_count} capsules that together ensure the learner can achieve the lab objective.
IMPORTANT: Design capsules that specifically prepare learners to answer the exam question types listed above.

Capsule Types (use a variety as appropriate):
- "conceptual": Theory and background knowledge
- "procedural": Step-by-step how-to
- "case_study": Real-world examples and analysis
- "practice": Hands-on exercises
- "review": Summary and self-assessment

Return a JSON object with a "capsules" key containing a list of capsule objects, each with:
- "capsule_id" (str, e.g., "cap_01", "cap_02")
- "title" (str)
- "description" (str or null)
- "learning_goal" (str — specific, measurable goal for this capsule)
- "capsule_type" (str — one of: "conceptual", "procedural", "case_study", "practice", "review")
- "estimated_duration_minutes" (float or null)
- "prerequisites_within_lab" (list of str — capsule_ids that should be completed first)
- "assessment_criteria" (list of str — how to evaluate mastery)
- "common_errors" (list of str — typical mistakes learners make)

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


def _generate_capsules_for_lab(
    engine: GenerationEngine,
    cert_name: str,
    exam_code: str | None,
    domain_name: str,
    topic,
    subtopic_name: str,
    lab,
    question_type_names: list[str],
    target_capsule_count: int,
) -> list[Capsule]:
    """Generate capsules for a single lab."""
    success_criteria = (
        "\n".join(f"- {c}" for c in lab.success_criteria)
        if lab.success_criteria
        else "None specified"
    )

    # Extract question types from topic's learning objectives
    topic_question_types = set()
    for lo in topic.learning_objectives:
        topic_question_types.update(lo.relevant_question_types)

    # Use topic-specific types if available, otherwise fall back to all exam types
    relevant_types = list(topic_question_types) if topic_question_types else question_type_names
    question_types_str = (
        "\n".join(f"- {qt}" for qt in relevant_types)
        if relevant_types
        else "None specified"
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

    # Build full prompt with system instruction
    user_prompt = CAPSULES_USER_PROMPT_TEMPLATE.format(
        certification=cert_name,
        exam_code=exam_code or 'N/A',
        domain_name=domain_name,
        topic_name=topic.name,
        subtopic_name=subtopic_name,
        lab_title=lab.title,
        lab_objective=lab.objective,
        lab_type=lab.lab_type,
        success_criteria=success_criteria,
        real_world_application=lab.real_world_application or 'Not specified',
        question_types=question_types_str,
        learning_objectives=learning_objectives_str,
        target_capsule_count=target_capsule_count,
    )
    full_prompt = f"{CAPSULES_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _call():
        return engine.generate(full_prompt, config=GenerationConfig(temperature=0.7))

    response = _call_with_retry(_call)
    raw_json = _strip_code_fences(response.text)
    data = json.loads(raw_json)

    capsules = []
    for cap_data in data.get("capsules", []):
        capsules.append(
            Capsule(
                capsule_id=cap_data["capsule_id"],
                title=cap_data["title"],
                description=cap_data.get("description"),
                learning_goal=cap_data["learning_goal"],
                capsule_type=cap_data["capsule_type"],
                estimated_duration_minutes=cap_data.get("estimated_duration_minutes"),
                prerequisites_within_lab=cap_data.get("prerequisites_within_lab", []),
                assessment_criteria=cap_data.get("assessment_criteria", []),
                common_errors=cap_data.get("common_errors", []),
            )
        )
    return capsules


def _fan_out_capsules_generation(
    engine: GenerationEngine,
    skeleton: CourseSkeleton,
    max_workers: int,
    target_capsule_count: int,
) -> tuple[CourseSkeleton, list[CapsuleGenerationResult]]:
    """Generate capsules for all labs in parallel."""
    cert_name = skeleton.certification_name
    exam_code = skeleton.exam_code

    # Extract question type names from skeleton
    question_type_names = [g.question_type_name for g in skeleton.question_type_guides]

    # Collect all labs with their parent info
    lab_tasks = []
    for module in skeleton.domain_modules:
        for topic in module.topics:
            for subtopic in topic.subtopics:
                for lab in subtopic.labs:
                    lab_tasks.append(
                        {
                            "domain_name": module.domain_name,
                            "topic": topic,
                            "subtopic_name": subtopic.name,
                            "lab": lab,
                        }
                    )

    if not lab_tasks:
        return skeleton, []

    results: dict[int, CapsuleGenerationResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, task in enumerate(lab_tasks):
            future = executor.submit(
                _generate_capsules_for_lab,
                engine,
                cert_name,
                exam_code,
                task["domain_name"],
                task["topic"],
                task["subtopic_name"],
                task["lab"],
                question_type_names,
                target_capsule_count,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            task = lab_tasks[idx]
            try:
                capsules = future.result()
                results[idx] = CapsuleGenerationResult(
                    domain_name=task["domain_name"],
                    topic_name=task["topic"].name,
                    subtopic_name=task["subtopic_name"],
                    lab_id=task["lab"].lab_id,
                    capsules=capsules,
                    success=True,
                )
            except Exception as e:
                results[idx] = CapsuleGenerationResult(
                    domain_name=task["domain_name"],
                    topic_name=task["topic"].name,
                    subtopic_name=task["subtopic_name"],
                    lab_id=task["lab"].lab_id,
                    capsules=[],
                    success=False,
                    error=str(e),
                )

    # Update skeleton with generated capsules
    failed = []
    for idx, task in enumerate(lab_tasks):
        result = results[idx]
        if result.success:
            # Find and update the lab
            for module in skeleton.domain_modules:
                if module.domain_name == task["domain_name"]:
                    for topic in module.topics:
                        if topic.name == task["topic"].name:
                            for subtopic in topic.subtopics:
                                if subtopic.name == task["subtopic_name"]:
                                    for lab in subtopic.labs:
                                        if lab.lab_id == task["lab"].lab_id:
                                            lab.capsules = result.capsules
        else:
            failed.append(result)

    return skeleton, failed


class CapsuleSkeletonStep(EngineAwareStep):
    """Pipeline step that generates capsule structures for labs.

    Supports any engine (Gemini, vLLM, etc.) via the engine abstraction.
    """

    def __init__(
        self,
        *,
        engine: GenerationEngine | None = None,
        max_workers: int = 4,
        target_capsule_count: int = 4,
    ):
        """Initialize the capsules skeleton step.

        Args:
            engine: The generation engine to use.
            max_workers: Maximum parallel workers.
            target_capsule_count: Number of capsules to generate per lab.
        """
        super().__init__(engine=engine)
        self.max_workers = max_workers
        self.target_capsule_count = target_capsule_count

    def run(self, context: PipelineContext) -> PipelineContext:
        """Generate capsules for all labs.

        Expects context["course_skeleton"].
        Updates context["course_skeleton"] with capsules.
        """
        engine = self.require_engine()
        skeleton: CourseSkeleton = context["course_skeleton"]
        cert_name = skeleton.certification_name
        print(f"Generating capsules for '{cert_name}'...")
        print(f"  Using engine: {engine.engine_type} / {engine.model_name}")

        # Count labs
        lab_count = sum(
            len(subtopic.labs)
            for module in skeleton.domain_modules
            for topic in module.topics
            for subtopic in topic.subtopics
        )
        print(f"  Generating capsules for {lab_count} labs...")

        skeleton, failed = _fan_out_capsules_generation(
            engine, skeleton, self.max_workers, self.target_capsule_count
        )

        if failed:
            for f in failed:
                print(
                    f"  WARNING: Capsules for lab '{f.lab_id}' in '{f.domain_name}/{f.topic_name}/{f.subtopic_name}' failed: {f.error}"
                )

        # Store failed results for tracking
        if "failed_capsules" not in context:
            context["failed_capsules"] = []
        context["failed_capsules"].extend(failed)

        context["course_skeleton"] = skeleton
        print(f"Capsules generation complete for '{cert_name}'.\n")
        return context
