"""Course skeleton generation step (V2).

This module provides the CourseSkeletonStep which generates the course
structure including overview, question type guides, and domain modules.

V2 Improvements:
- Uses ExamFormatV2 with item classes and exam components
- Deduplicates topics across domains
- Generates component-specific content (cognitive vs psychomotor)
- Provides detailed item class guides with TEI/context info
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from course_builder.domain.course import (
    CourseModule,
    CourseOverview,
    CourseSkeleton,
    CourseTopic,
    DomainModuleResult,
    ExamComponent,
    ExamFormat,
    ExamFormatV2,
    ExplanationTemplate,
    ItemClass,
    LearningObjective,
    QuestionTypeGuide,
    ReasoningTemplate,
    StudyStrategy,
    SubTopic,
)
from course_builder.pipeline.base import PipelineContext, PipelineStep

SCAFFOLD_SYSTEM_PROMPT = (
    "You are an expert instructional designer specializing in certification exam preparation. "
    "Design high-quality course scaffolding that helps candidates study effectively."
)

SCAFFOLD_USER_PROMPT_TEMPLATE = """Design the cross-cutting scaffold for a "{certification}" ({exam_code}) certification preparation course.

EXAM COMPONENTS:
{exam_components_section}

ITEM TYPES (Question Formats):
{item_types_section}

CONTENT DOMAINS:
{domains_section}

Prerequisites: {prerequisites_section}

Return a JSON object with these keys:
- "overview" (object with:
    "target_audience" (str),
    "course_description" (str),
    "total_estimated_study_hours" (float or null),
    "study_strategies" (list of objects with "name", "description", "when_to_use"),
    "exam_day_tips" (list of str),
    "prerequisites_detail" (list of str))
- "question_type_guides" (list of objects, one per ITEM TYPE listed above, each with:
    "question_type_name" (str - must match item type name exactly),
    "is_tei" (bool - Technology Enhanced Item),
    "context_types" (list of str - "scenario" and/or "direct"),
    "detailed_structure" (str - how this item type is structured),
    "reasoning_template" (object with "approach_steps" (list of str), "time_allocation_advice" (str or null), "common_traps" (list of str)) or null,
    "explanation_template" (object with "correct_answer_template" (str), "wrong_answer_template" (str), "partial_credit_template" (str or null)) or null,
    "difficulty_scaling_notes" (str or null),
    "answer_choice_design_notes" (str or null),
    "scenario_guidance" (str or null - how to approach scenario-based stems),
    "direct_question_guidance" (str or null - how to approach direct questions))

IMPORTANT: Create a guide for EACH item type listed. Be specific to this certification's exam format.

Return ONLY the JSON object, no other text."""

# V2 Scaffold prompt with exam components
SCAFFOLD_USER_PROMPT_V2 = """Design the cross-cutting scaffold for a "{certification}" ({exam_code}) certification preparation course.

{exam_structure_section}

Return a JSON object with these keys:
- "overview" (object with:
    "target_audience" (str),
    "course_description" (str),
    "total_estimated_study_hours" (float or null),
    "study_strategies" (list of objects with "name", "description", "when_to_use"),
    "exam_day_tips" (list of str),
    "prerequisites_detail" (list of str))
- "question_type_guides" (list of objects, one per item type, each with:
    "question_type_name" (str),
    "is_tei" (bool),
    "allowed_context_types" (list: "scenario", "direct"),
    "interaction_model" (str: single_select, multi_select, order_list, categorize, table_select),
    "detailed_structure" (str),
    "reasoning_template" (object with "approach_steps" (list), "time_allocation_advice" (str), "common_traps" (list)),
    "explanation_template" (object with "correct_answer_template", "wrong_answer_template", "partial_credit_template"),
    "scenario_guidance" (str - tips for scenario-based questions of this type),
    "direct_question_guidance" (str - tips for direct questions of this type),
    "difficulty_scaling_notes" (str),
    "answer_choice_design_notes" (str))

Return ONLY the JSON object, no other text."""

DOMAIN_SYSTEM_PROMPT = (
    "You are an expert certification exam content developer. "
    "Create detailed, accurate topic breakdowns for exam domains using official exam guides. "
    "AVOID duplicating topics that belong to other domains."
)

DOMAIN_USER_PROMPT_TEMPLATE = """Create a detailed topic breakdown for the "{domain_name}" domain of the "{certification}" ({exam_code}) certification exam.

This domain covers approximately {weight_pct}% of the exam.
The exam tests these item types: {question_type_names}

{deduplication_section}

Search for the official exam guide and topic breakdown for this domain.

IMPORTANT GUIDELINES:
1. Focus ONLY on topics specific to "{domain_name}" - do not include topics that belong to other domains
2. Each topic should be unique and not overlap with topics from other domains
3. Learning objectives should specify which item types they're tested with
4. Distinguish between cognitive (knowledge) and psychomotor (hands-on) skills

Return a JSON object with these keys:
- "domain_name" (str)
- "domain_weight_pct" (float or null)
- "overview" (str)
- "topics" (list of objects, each with:
    "name" (str - must be unique, not duplicating other domains),
    "description" (str or null),
    "content_type" (str - "cognitive" for knowledge, "psychomotor" for hands-on skills, "mixed" for both),
    "learning_objectives" (list of objects with:
        "objective" (str),
        "bloom_level" (str — Remember, Understand, Apply, Analyze, Evaluate, Create),
        "relevant_question_types" (list of str - which item types test this objective),
        "is_psychomotor" (bool - true if this is a hands-on skill)),
    "subtopics" (list of objects with:
        "name" (str),
        "description" (str or null),
        "key_concepts" (list of str),
        "practical_skills" (list of str),
        "common_misconceptions" (list of str)),
    "estimated_study_hours" (float or null))
- "prerequisites_for_domain" (list of str)
- "recommended_study_order" (list of str — topic names in recommended order)
- "official_references" (list of str — URLs or document names)

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


def _format_exam_structure_v2(exam_format_v2: ExamFormatV2) -> str:
    """Format V2 exam structure for the prompt."""
    lines = []

    for comp in exam_format_v2.exam_components:
        lines.append(f"\n=== {comp.name.upper()} COMPONENT ===")
        if comp.adaptive:
            lines.append(f"Format: Adaptive ({comp.adaptive_algorithm or 'CAT'})")
            lines.append(f"Questions: {comp.num_questions_min}-{comp.num_questions_max}")
            if comp.pilot_unscored_items:
                lines.append(f"Pilot Items: {comp.pilot_unscored_items} (unscored)")
        else:
            lines.append(f"Questions: {comp.num_questions}")
        lines.append(f"Time: {comp.time_limit_minutes} minutes")

        if comp.domains:
            lines.append("\nDomains:")
            for d in comp.domains:
                if d.weight_min_pct and d.weight_max_pct:
                    lines.append(f"  - {d.name}: {d.weight_min_pct}-{d.weight_max_pct}%")
                elif d.weight_pct:
                    lines.append(f"  - {d.name}: {d.weight_pct}%")
                else:
                    lines.append(f"  - {d.name}")

        if comp.item_classes:
            lines.append("\nItem Types:")
            for ic in comp.item_classes:
                tei = "[TEI]" if ic.is_tei else ""
                lines.append(f"  - {ic.display_name or ic.name} {tei}")
                lines.append(f"    Interaction: {ic.interaction_model}")
                lines.append(f"    Context: {', '.join(ic.allowed_context_types)}")
                if ic.patient_presence != "optional":
                    lines.append(f"    Patient: {ic.patient_presence}")
                lines.append(f"    Grading: {ic.grading}")

    return "\n".join(lines)


def _generate_scaffold(
    client: genai.Client,
    exam_format: ExamFormat,
    model: str,
    exam_format_v2: ExamFormatV2 | None = None,
) -> tuple[CourseOverview, list[QuestionTypeGuide]]:
    """Generate the course scaffold including overview and question type guides."""

    # Use V2 format if available for richer item class info
    if exam_format_v2 and exam_format_v2.exam_components:
        exam_structure_section = _format_exam_structure_v2(exam_format_v2)

        # Build item types section from V2
        item_lines = []
        for comp in exam_format_v2.exam_components:
            for ic in comp.item_classes:
                tei = " [TEI]" if ic.is_tei else ""
                item_lines.append(f"- {ic.display_name or ic.name}{tei}")
                item_lines.append(f"  Interaction: {ic.interaction_model}")
                item_lines.append(f"  Context types: {', '.join(ic.allowed_context_types)}")
                item_lines.append(f"  Patient presence: {ic.patient_presence}")
                item_lines.append(f"  Grading: {ic.grading}")
        item_types_section = "\n".join(item_lines) if item_lines else "None specified"

        # Build domains section from V2
        domain_lines = []
        for comp in exam_format_v2.exam_components:
            for d in comp.domains:
                if d.weight_min_pct and d.weight_max_pct:
                    domain_lines.append(f"- {d.name}: {d.weight_min_pct}-{d.weight_max_pct}%")
                elif d.weight_pct:
                    domain_lines.append(f"- {d.name}: {d.weight_pct}%")
                else:
                    domain_lines.append(f"- {d.name}")
        domains_section = "\n".join(domain_lines) if domain_lines else "None specified"

        # Build exam components section
        comp_lines = []
        for comp in exam_format_v2.exam_components:
            if comp.adaptive:
                comp_lines.append(f"- {comp.name}: Adaptive ({comp.num_questions_min}-{comp.num_questions_max} questions, {comp.time_limit_minutes} min)")
            else:
                comp_lines.append(f"- {comp.name}: {comp.num_questions} questions, {comp.time_limit_minutes} min")
        exam_components_section = "\n".join(comp_lines) if comp_lines else "Single component exam"

        prerequisites_section = (
            ", ".join(exam_format_v2.prerequisites) if exam_format_v2.prerequisites else "None"
        )

        prompt = SCAFFOLD_USER_PROMPT_TEMPLATE.format(
            certification=exam_format_v2.certification_name,
            exam_code=exam_format_v2.exam_code or "N/A",
            exam_components_section=exam_components_section,
            item_types_section=item_types_section,
            domains_section=domains_section,
            prerequisites_section=prerequisites_section,
        )
    else:
        # Fall back to V1 format
        qt_lines = []
        for qt in exam_format.question_types:
            qt_lines.append(f"- {qt.name}: {qt.description or 'No description'}")
        item_types_section = "\n".join(qt_lines) if qt_lines else "None specified"

        domain_lines = []
        for d in exam_format.domains:
            weight = f" ({d.weight_pct}%)" if d.weight_pct else ""
            domain_lines.append(f"- {d.name}{weight}")
        domains_section = "\n".join(domain_lines) if domain_lines else "None specified"

        prerequisites_section = (
            ", ".join(exam_format.prerequisites) if exam_format.prerequisites else "None"
        )

        prompt = SCAFFOLD_USER_PROMPT_TEMPLATE.format(
            certification=exam_format.certification_name,
            exam_code=exam_format.exam_code or "N/A",
            exam_components_section="Single component exam",
            item_types_section=item_types_section,
            domains_section=domains_section,
            prerequisites_section=prerequisites_section,
        )

    config = types.GenerateContentConfig(
        system_instruction=SCAFFOLD_SYSTEM_PROMPT,
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

    overview_data = data.get("overview", {})
    strategies = [
        StudyStrategy(**s) for s in overview_data.get("study_strategies", [])
    ]
    overview = CourseOverview(
        target_audience=overview_data.get("target_audience"),
        course_description=overview_data.get("course_description"),
        total_estimated_study_hours=overview_data.get("total_estimated_study_hours"),
        study_strategies=strategies,
        exam_day_tips=overview_data.get("exam_day_tips", []),
        prerequisites_detail=overview_data.get("prerequisites_detail", []),
    )

    guides = []
    for g in data.get("question_type_guides", []):
        rt_data = g.get("reasoning_template")
        reasoning = ReasoningTemplate(**rt_data) if rt_data else None

        et_data = g.get("explanation_template")
        explanation = ExplanationTemplate(**et_data) if et_data else None

        guides.append(
            QuestionTypeGuide(
                question_type_name=g["question_type_name"],
                detailed_structure=g.get("detailed_structure"),
                reasoning_template=reasoning,
                explanation_template=explanation,
                difficulty_scaling_notes=g.get("difficulty_scaling_notes"),
                answer_choice_design_notes=g.get("answer_choice_design_notes"),
            )
        )

    return overview, guides


def _generate_domain_module(
    client: genai.Client,
    cert_name: str,
    exam_code: str | None,
    domain,
    question_type_names: list[str],
    model: str,
    other_domain_names: list[str] | None = None,
    already_covered_topics: list[str] | None = None,
) -> CourseModule:
    """Generate a domain module with topics and subtopics.

    Args:
        other_domain_names: Names of other domains (for context)
        already_covered_topics: Topics already generated in other domains (for deduplication)
    """
    # Build deduplication section
    dedup_lines = []
    if other_domain_names:
        dedup_lines.append("Other domains in this exam (do NOT duplicate their content):")
        for name in other_domain_names:
            dedup_lines.append(f"  - {name}")

    if already_covered_topics:
        dedup_lines.append("\nTopics already covered (do NOT include these):")
        for topic in already_covered_topics:
            dedup_lines.append(f"  - {topic}")

    deduplication_section = "\n".join(dedup_lines) if dedup_lines else ""

    # Get weight - handle both V1 and V2 domain formats
    if hasattr(domain, 'weight_min_pct') and domain.weight_min_pct and hasattr(domain, 'weight_max_pct') and domain.weight_max_pct:
        weight_pct = f"{domain.weight_min_pct}-{domain.weight_max_pct}"
    elif hasattr(domain, 'weight_pct') and domain.weight_pct:
        weight_pct = str(domain.weight_pct)
    else:
        weight_pct = "unknown"

    prompt = DOMAIN_USER_PROMPT_TEMPLATE.format(
        domain_name=domain.name,
        certification=cert_name,
        exam_code=exam_code or "N/A",
        weight_pct=weight_pct,
        question_type_names=", ".join(question_type_names) if question_type_names else "N/A",
        deduplication_section=deduplication_section,
    )

    config = types.GenerateContentConfig(
        system_instruction=DOMAIN_SYSTEM_PROMPT,
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

    # Normalize domain_weight_pct if returned as decimal (0.06 -> 6.0)
    raw_weight = data.get("domain_weight_pct")
    if raw_weight is not None and isinstance(raw_weight, (int, float)):
        # If weight is less than 1, it's likely a decimal that should be percentage
        if raw_weight < 1:
            data["domain_weight_pct"] = raw_weight * 100

    topics = []
    for t in data.get("topics", []):
        objectives = [LearningObjective(**lo) for lo in t.get("learning_objectives", [])]
        subtopics = [SubTopic(**st) for st in t.get("subtopics", [])]
        topics.append(
            CourseTopic(
                name=t["name"],
                description=t.get("description"),
                learning_objectives=objectives,
                subtopics=subtopics,
                estimated_study_hours=t.get("estimated_study_hours"),
            )
        )

    # Compute fallback weight from domain (handle V2 ranges)
    fallback_weight = None
    if hasattr(domain, 'weight_pct') and domain.weight_pct is not None:
        fallback_weight = domain.weight_pct
    elif hasattr(domain, 'weight_min_pct') and hasattr(domain, 'weight_max_pct'):
        if domain.weight_min_pct is not None and domain.weight_max_pct is not None:
            # Use midpoint of range
            fallback_weight = (domain.weight_min_pct + domain.weight_max_pct) / 2

    return CourseModule(
        domain_name=data.get("domain_name", domain.name),
        domain_weight_pct=data.get("domain_weight_pct", fallback_weight),
        overview=data.get("overview"),
        topics=topics,
        prerequisites_for_domain=data.get("prerequisites_for_domain", []),
        recommended_study_order=data.get("recommended_study_order", []),
        official_references=data.get("official_references", []),
    )


def _fan_out_domain_modules(
    client: genai.Client,
    exam_format: ExamFormat,
    model: str,
    max_workers: int,
    exam_format_v2: ExamFormatV2 | None = None,
) -> tuple[list[CourseModule], list[DomainModuleResult]]:
    """Generate domain modules with deduplication.

    Note: We generate sequentially (not in parallel) to enable deduplication.
    Each domain is aware of topics already generated in previous domains.
    """
    # Get item type names from V2 or fall back to V1
    if exam_format_v2 and exam_format_v2.exam_components:
        question_type_names = []
        domains = []
        for comp in exam_format_v2.exam_components:
            for ic in comp.item_classes:
                question_type_names.append(ic.display_name or ic.name)
            domains.extend(comp.domains)
        # Deduplicate
        question_type_names = list(dict.fromkeys(question_type_names))
    else:
        question_type_names = [qt.name for qt in exam_format.question_types]
        domains = exam_format.domains

    if not domains:
        return [], []

    # Get all domain names for context
    all_domain_names = [d.name for d in domains]

    # Track generated topics for deduplication
    generated_topics: list[str] = []

    modules: list[CourseModule] = []
    failed: list[DomainModuleResult] = []

    # Generate sequentially for deduplication (can still use some parallelism within)
    for idx, domain in enumerate(domains):
        other_domains = [name for i, name in enumerate(all_domain_names) if i != idx]

        print(f"    Generating domain {idx + 1}/{len(domains)}: {domain.name}")

        try:
            module = _generate_domain_module(
                client,
                exam_format.certification_name,
                exam_format.exam_code,
                domain,
                question_type_names,
                model,
                other_domain_names=other_domains,
                already_covered_topics=generated_topics.copy(),
            )

            # Track topics from this module for next iteration
            for topic in module.topics:
                generated_topics.append(topic.name)
                # Also track subtopic names
                for st in topic.subtopics:
                    generated_topics.append(st.name)

            modules.append(module)

        except Exception as e:
            print(f"      ERROR: {e}")
            failed.append(DomainModuleResult(
                domain_name=domain.name,
                module=None,
                success=False,
                error=str(e),
            ))

    return modules, failed


class CourseSkeletonStep(PipelineStep):
    """Pipeline step that generates the course skeleton structure.

    V2 improvements:
    - Uses ExamFormatV2 for richer item class information
    - Deduplicates topics across domains
    - Generates component-specific content
    """

    def __init__(self, model: str = "gemini-2.0-flash", max_workers: int = 4):
        """Initialize the course skeleton step.

        Args:
            model: The Gemini model to use.
            max_workers: Maximum parallel workers for domain generation.
        """
        self.model = model
        self.max_workers = max_workers

    def run(self, context: PipelineContext) -> PipelineContext:
        """Generate the complete course skeleton.

        Expects context["exam_format"] and optionally context["exam_format_v2"].
        Populates context["course_skeleton"].
        """
        exam_format: ExamFormat = context["exam_format"]
        exam_format_v2: ExamFormatV2 | None = context.get("exam_format_v2")

        cert_name = exam_format.certification_name
        print(f"\nGenerating course skeleton for '{cert_name}'...")

        if exam_format_v2:
            print(f"  Using V2 exam format with {len(exam_format_v2.exam_components)} component(s)")
            for comp in exam_format_v2.exam_components:
                print(f"    - {comp.name}: {len(comp.item_classes)} item types, {len(comp.domains)} domains")
        else:
            print(f"  Using V1 exam format")

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        # Phase 1: Generate scaffold (uses V2 if available)
        print("\n  Phase 1: Generating scaffold...")
        try:
            overview, guides = _generate_scaffold(
                client, exam_format, self.model, exam_format_v2
            )
            print(f"    Generated {len(guides)} question type guides")
        except Exception as e:
            raise RuntimeError(
                f"Scaffold generation failed for '{cert_name}': {e}"
            ) from e

        # Phase 2: Domain deep-dives with deduplication
        num_domains = len(exam_format.domains)
        if exam_format_v2 and exam_format_v2.exam_components:
            num_domains = sum(len(comp.domains) for comp in exam_format_v2.exam_components)

        print(f"\n  Phase 2: Generating {num_domains} domain modules (with deduplication)...")
        modules, failed = _fan_out_domain_modules(
            client, exam_format, self.model, self.max_workers, exam_format_v2
        )

        if failed:
            for f in failed:
                print(f"    WARNING: Domain '{f.domain_name}' failed: {f.error}")

        # Count total topics and subtopics
        total_topics = sum(len(m.topics) for m in modules)
        total_subtopics = sum(
            sum(len(t.subtopics) for t in m.topics) for m in modules
        )

        # Phase 3: Programmatic merge
        skeleton = CourseSkeleton(
            certification_name=cert_name,
            exam_code=exam_format.exam_code,
            overview=overview,
            question_type_guides=guides,
            domain_modules=modules,
            failed_domains=failed,
        )

        context["course_skeleton"] = skeleton

        print(f"\n  {'='*50}")
        print(f"  Course Skeleton Complete")
        print(f"  {'='*50}")
        print(f"  Domains: {len(modules)}")
        print(f"  Topics: {total_topics}")
        print(f"  Subtopics: {total_subtopics}")
        print(f"  Question Type Guides: {len(guides)}")
        if failed:
            print(f"  Failed Domains: {len(failed)}")
        print(f"  {'='*50}\n")

        return context
