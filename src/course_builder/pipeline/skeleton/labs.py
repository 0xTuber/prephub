"""Labs skeleton generation step.

This module provides the LabSkeletonStep which generates hands-on lab
structures for each subtopic in the course skeleton.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from course_builder.domain.course import CourseSkeleton, Lab, LabGenerationResult
from course_builder.pipeline.base import PipelineContext, PipelineStep

LABS_SYSTEM_PROMPT = (
    "You are an expert instructional designer specializing in hands-on learning experiences. "
    "Design comprehensive, thorough labs that cover ALL aspects of the topic and leave no gaps "
    "in the learner's understanding."
)

LABS_USER_PROMPT_TEMPLATE = """Design comprehensive hands-on labs for the following subtopic in a "{certification}" ({exam_code}) certification preparation course.

DOMAIN: {domain_name}
TOPIC: {topic_name}
SUBTOPIC: {subtopic_name}
SUBTOPIC DESCRIPTION: {subtopic_description}

Key Concepts to Cover:
{key_concepts}

Practical Skills to Develop:
{practical_skills}

Common Misconceptions to Address:
{common_misconceptions}

EXAM QUESTION TYPES TO PREPARE FOR:
{question_types}

Learning Objectives for This Topic:
{learning_objectives}

Generate exactly {target_lab_count} labs that together provide exhaustive coverage of this subtopic.
IMPORTANT: Design labs that specifically prepare learners to answer the exam question types listed above.

Lab Types (use a variety):
- "guided": Step-by-step walkthrough with explicit instructions
- "exploratory": Open-ended investigation with guiding questions
- "challenge": Problem-solving exercise with minimal guidance
- "simulation": Realistic scenario-based practice

Return a JSON object with a "labs" key containing a list of lab objects, each with:
- "lab_id" (str, e.g., "lab_01", "lab_02")
- "title" (str)
- "description" (str or null)
- "objective" (str — what the learner will achieve)
- "lab_type" (str — one of: "guided", "exploratory", "challenge", "simulation")
- "estimated_duration_minutes" (float or null)
- "tools_required" (list of str)
- "prerequisites_within_subtopic" (list of str — lab_ids that should be completed first)
- "success_criteria" (list of str — how to know the lab is complete)
- "real_world_application" (str or null — how this applies in practice)

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


def _generate_labs_for_subtopic(
    client: genai.Client,
    cert_name: str,
    exam_code: str | None,
    domain_name: str,
    topic,
    subtopic,
    question_type_names: list[str],
    target_lab_count: int,
    model: str,
) -> list[Lab]:
    """Generate labs for a single subtopic."""
    key_concepts = (
        "\n".join(f"- {c}" for c in subtopic.key_concepts)
        if subtopic.key_concepts
        else "None specified"
    )
    practical_skills = (
        "\n".join(f"- {s}" for s in subtopic.practical_skills)
        if subtopic.practical_skills
        else "None specified"
    )
    common_misconceptions = (
        "\n".join(f"- {m}" for m in subtopic.common_misconceptions)
        if subtopic.common_misconceptions
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

    prompt = LABS_USER_PROMPT_TEMPLATE.format(
        certification=cert_name,
        exam_code=exam_code or "N/A",
        domain_name=domain_name,
        topic_name=topic.name,
        subtopic_name=subtopic.name,
        subtopic_description=subtopic.description or "No description",
        key_concepts=key_concepts,
        practical_skills=practical_skills,
        common_misconceptions=common_misconceptions,
        question_types=question_types_str,
        learning_objectives=learning_objectives_str,
        target_lab_count=target_lab_count,
    )

    config = types.GenerateContentConfig(
        system_instruction=LABS_SYSTEM_PROMPT,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )

    def _call():
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

    response = _call_with_retry(_call)
    raw_json = _strip_code_fences(response.text)
    data = json.loads(raw_json)

    labs = []
    for lab_data in data.get("labs", []):
        labs.append(
            Lab(
                lab_id=lab_data["lab_id"],
                title=lab_data["title"],
                description=lab_data.get("description"),
                objective=lab_data["objective"],
                lab_type=lab_data["lab_type"],
                estimated_duration_minutes=lab_data.get("estimated_duration_minutes"),
                tools_required=lab_data.get("tools_required", []),
                prerequisites_within_subtopic=lab_data.get(
                    "prerequisites_within_subtopic", []
                ),
                success_criteria=lab_data.get("success_criteria", []),
                real_world_application=lab_data.get("real_world_application"),
            )
        )
    return labs


def _fan_out_labs_generation(
    client: genai.Client,
    skeleton: CourseSkeleton,
    model: str,
    max_workers: int,
    target_lab_count: int,
) -> tuple[CourseSkeleton, list[LabGenerationResult]]:
    """Generate labs for all subtopics in parallel."""
    cert_name = skeleton.certification_name
    exam_code = skeleton.exam_code

    # Extract question type names from skeleton
    question_type_names = [g.question_type_name for g in skeleton.question_type_guides]

    # Collect all subtopics with their parent info
    subtopic_tasks = []
    for module in skeleton.domain_modules:
        for topic in module.topics:
            for subtopic in topic.subtopics:
                subtopic_tasks.append(
                    {
                        "domain_name": module.domain_name,
                        "topic": topic,
                        "subtopic": subtopic,
                    }
                )

    if not subtopic_tasks:
        return skeleton, []

    results: dict[int, LabGenerationResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, task in enumerate(subtopic_tasks):
            future = executor.submit(
                _generate_labs_for_subtopic,
                client,
                cert_name,
                exam_code,
                task["domain_name"],
                task["topic"],
                task["subtopic"],
                question_type_names,
                target_lab_count,
                model,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            task = subtopic_tasks[idx]
            try:
                labs = future.result()
                results[idx] = LabGenerationResult(
                    domain_name=task["domain_name"],
                    topic_name=task["topic"].name,
                    subtopic_name=task["subtopic"].name,
                    labs=labs,
                    success=True,
                )
            except Exception as e:
                results[idx] = LabGenerationResult(
                    domain_name=task["domain_name"],
                    topic_name=task["topic"].name,
                    subtopic_name=task["subtopic"].name,
                    labs=[],
                    success=False,
                    error=str(e),
                )

    # Update skeleton with generated labs
    failed = []
    for idx, task in enumerate(subtopic_tasks):
        result = results[idx]
        if result.success:
            # Find and update the subtopic
            for module in skeleton.domain_modules:
                if module.domain_name == task["domain_name"]:
                    for topic in module.topics:
                        if topic.name == task["topic"].name:
                            for subtopic in topic.subtopics:
                                if subtopic.name == task["subtopic"].name:
                                    subtopic.labs = result.labs
        else:
            failed.append(result)

    return skeleton, failed


class LabSkeletonStep(PipelineStep):
    """Pipeline step that generates lab structures for subtopics."""

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        max_workers: int = 4,
        target_lab_count: int = 3,
    ):
        """Initialize the labs skeleton step.

        Args:
            model: The Gemini model to use.
            max_workers: Maximum parallel workers.
            target_lab_count: Number of labs to generate per subtopic.
        """
        self.model = model
        self.max_workers = max_workers
        self.target_lab_count = target_lab_count

    def run(self, context: PipelineContext) -> PipelineContext:
        """Generate labs for all subtopics.

        Expects context["course_skeleton"].
        Updates context["course_skeleton"] with labs.
        """
        skeleton: CourseSkeleton = context["course_skeleton"]
        cert_name = skeleton.certification_name
        print(f"Generating labs for '{cert_name}'...")

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        # Count subtopics
        subtopic_count = sum(
            len(topic.subtopics)
            for module in skeleton.domain_modules
            for topic in module.topics
        )
        print(f"  Generating labs for {subtopic_count} subtopics...")

        skeleton, failed = _fan_out_labs_generation(
            client, skeleton, self.model, self.max_workers, self.target_lab_count
        )

        if failed:
            for f in failed:
                print(
                    f"  WARNING: Labs for '{f.domain_name}/{f.topic_name}/{f.subtopic_name}' failed: {f.error}"
                )

        # Store failed results for tracking
        if "failed_labs" not in context:
            context["failed_labs"] = []
        context["failed_labs"].extend(failed)

        context["course_skeleton"] = skeleton
        print(f"Labs generation complete for '{cert_name}'.\n")
        return context
