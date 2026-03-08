"""Exam format discovery step (V2).

This module provides the ExamFormatStep which discovers and structures
exam format information using web search and LLM extraction.

V2 Improvements:
- Separates factual discovery from example generation
- Multi-query retrieval (handbook, sample items, TEI docs)
- Tracks provenance for each fact
- Supports exam components (cognitive vs psychomotor)
- Handles question count ranges (for CAT exams)
- Distinguishes context types (scenario vs direct)
"""

import json
import os
from datetime import datetime

from google import genai
from google.genai import types

from course_builder.domain.course import (
    ExamComponent,
    ExamDomain,
    ExamDomainV2,
    ExamFormat,
    ExamFormatV2,
    ItemClass,
    QuestionType,
    SourceFact,
)
from course_builder.pipeline.base import PipelineContext, PipelineStep


# =============================================================================
# V2 Prompts - Factual discovery with provenance
# =============================================================================

PHASE1A_SYSTEM_PROMPT = """You are an expert certification exam researcher.
Your job is to find FACTUAL, VERIFIABLE information about certification exams.

IMPORTANT RULES:
1. Only report facts you can attribute to a specific source
2. Clearly distinguish between official sources and third-party sources
3. If information is not publicly available, say "not published" rather than guessing
4. Do NOT generate example questions - only report real published examples if they exist
5. For each fact, note where it came from (handbook, website, sample packet, etc.)"""

PHASE1A_QUERIES = [
    '"{certification}" official candidate handbook exam format',
    '"{certification}" exam number of questions time limit passing score',
    '"{certification}" exam question types item formats TEI',
    '"{certification}" exam content domains objectives weights',
    '"{certification}" sample questions practice test official',
    '"{certification}" exam delivery Pearson VUE PSI online proctored',
]

PHASE1A_USER_PROMPT_TEMPLATE = """Find FACTUAL information about the "{certification}" certification exam.

For each piece of information, note the source (official handbook, exam website, third-party, etc.).

Gather facts about:

1. EXAM STRUCTURE
   - Number of questions (exact number OR range for adaptive exams)
   - Number of unscored/pilot items (if known)
   - Time limit
   - Is it adaptive (CAT) or fixed-form?

2. EXAM COMPONENTS
   - Is there more than one component? (e.g., cognitive + psychomotor/skills)
   - Details for each component separately

3. QUESTION/ITEM TYPES (be precise)
   - List each item type used (multiple choice, multiple response, drag-drop, etc.)
   - For each type, note:
     * Is it a TEI (Technology Enhanced Item) or traditional?
     * Does it use scenario/case-based stems or direct questions?
     * Is patient/client presence required, optional, or not applicable?

4. CONTENT DOMAINS
   - List all domains with their weight percentages (or ranges)
   - Note if weights are approximate or exact

5. SCORING
   - Passing score/cut score (if published)
   - Note if passing score is "not published" or "criterion-referenced"

6. LOGISTICS
   - Prerequisites
   - Cost
   - Validity period
   - Delivery methods (testing center name, online proctoring)
   - Available languages

Be precise. If something is not officially published, say "not published" rather than inferring."""

PHASE1B_SYSTEM_PROMPT = """You are a precise JSON extraction assistant.
Extract structured data from the provided research notes.

IMPORTANT:
- Only extract facts that are explicitly stated in the source text
- Use null for unknown/unpublished fields - do NOT invent values
- Track source_type for each major section: "official", "third_party", or "inferred"
- For ranges, use min/max fields rather than single values"""

PHASE1B_EXTRACTION_SCHEMA = """{
  "certification_name": "string",
  "certification_code": "string or null (e.g., EMR, EMT)",
  "certifying_body": "string or null (e.g., NREMT, AWS)",
  "exam_code": "string or null",

  "exam_components": [
    {
      "name": "string (e.g., cognitive, psychomotor)",
      "adaptive": "boolean",
      "adaptive_algorithm": "string or null (e.g., CAT)",
      "num_questions": "int or null (if fixed)",
      "num_questions_min": "int or null (if adaptive)",
      "num_questions_max": "int or null (if adaptive)",
      "pilot_unscored_items": "int or null",
      "time_limit_minutes": "int or null",
      "delivery_methods": ["string"],
      "passing_score": "string or null",
      "passing_score_source": "official | inferred | unpublished",

      "domains": [
        {
          "name": "string",
          "weight_min_pct": "float or null",
          "weight_max_pct": "float or null",
          "weight_pct": "float or null (if single value)",
          "source_type": "official | third_party | inferred"
        }
      ],

      "item_classes": [
        {
          "name": "string (e.g., multiple_choice, multiple_response, build_list)",
          "display_name": "string",
          "is_tei": "boolean",
          "interaction_model": "single_select | multi_select | order_list | categorize | table_select",
          "allowed_context_types": ["scenario", "direct"],
          "patient_presence": "required | optional | not_required",
          "grading": "all_or_nothing | partial_credit | unknown",
          "estimated_percentage": "float or null",
          "source_type": "official | third_party | inferred"
        }
      ]
    }
  ],

  "prerequisites": ["string"],
  "cost_usd": "string or null",
  "validity_years": "int or null",
  "languages": ["string"],
  "recertification_policy": "string or null",

  "source_facts": [
    {
      "field": "string (which field this fact supports)",
      "value": "the value",
      "source_type": "official | third_party | inferred",
      "source_title": "string (e.g., NREMT Candidate Handbook 2024)",
      "confidence": "high | medium | low"
    }
  ]
}"""

PHASE1B_USER_PROMPT_TEMPLATE = """Extract exam format details from the following research about "{certification}".

Return a JSON object following this schema:
{schema}

IMPORTANT RULES:
1. Use null for any field where information is not available - do NOT guess
2. For adaptive exams, use num_questions_min/max instead of num_questions
3. Track source_type for item_classes and domains
4. Include source_facts for key pieces of information with their provenance

Research notes:
{raw_description}

Return ONLY the JSON object, no other text."""


# =============================================================================
# V1 Prompts - Legacy (kept for backwards compatibility)
# =============================================================================

PHASE1_SYSTEM_PROMPT = (
    "You are an expert certification exam researcher. "
    "Your job is to find exhaustive, accurate details about certification exams."
)

PHASE1_USER_PROMPT_TEMPLATE = """Find a complete and exhaustive description of the "{certification}" certification exam format.

Cover ALL of the following aspects:
1. Number of questions
2. Time limit
3. Question types — for EACH type found (e.g. multiple choice, multiple response, performance-based, drag-and-drop, hotspot, case study, fill-in-the-blank, simulation, etc.):
   a. What the question type is and how it works
   b. WHY this type is used — what skill or competency does it measure that other types cannot?
   c. A clear skeleton / template showing the structure of the question (stem, options layout, interaction model)
   d. A realistic example question with answer choices (use plausible but clearly fictional content)
   e. How it is scored / graded (partial credit, all-or-nothing, etc.)
4. Passing score / cut score
5. Exam domains / objectives with their percentage weights
6. Prerequisites
7. Cost (USD)
8. Validity / renewal period
9. Delivery methods (testing center, online proctored, etc.)
10. Available languages

Be as detailed and specific as possible. Include exact numbers and percentages where available.
For question types, be truly exhaustive — the goal is to fully understand the nature of every question format a candidate will encounter so we can later reproduce realistic practice questions."""

PHASE2_SYSTEM_PROMPT = (
    "You are a precise JSON extraction assistant. "
    "Extract structured data from the provided text and return ONLY valid JSON."
)

PHASE2_USER_PROMPT_TEMPLATE = """Extract the exam format details from the following text about the "{certification}" certification exam.

Return a JSON object with these keys:
- "certification_name" (str)
- "exam_code" (str or null)
- "num_questions" (int or null)
- "time_limit_minutes" (int or null)
- "question_types" (list of objects, each with:
    "name" (str) — the question type name,
    "description" (str or null) — how the question type works,
    "purpose" (str or null) — why this type is used, what competency it measures,
    "skeleton" (str or null) — a template showing the structure (stem, options layout, interaction),
    "example" (str or null) — a realistic example question with answer choices,
    "grading_notes" (str or null) — how it is scored: partial credit, all-or-nothing, etc.)
- "passing_score" (str or null)
- "domains" (list of objects with "name", "weight_pct" (float or null), "description" (str or null))
- "prerequisites" (list of str)
- "cost_usd" (str or null)
- "validity_years" (int or null)
- "delivery_methods" (list of str)
- "languages" (list of str)
- "recertification_policy" (str or null)
- "additional_notes" (str or null)

Return ONLY the JSON object, no other text.

Text:
{raw_description}"""


# =============================================================================
# V2 Discovery Functions
# =============================================================================


def _search_exam_format_v2(client: genai.Client, cert_name: str, model: str) -> dict[str, str]:
    """Phase 1A: Multi-query search for exam format information.

    Returns a dict mapping query type to raw search results.
    """
    config = types.GenerateContentConfig(
        system_instruction=PHASE1A_SYSTEM_PROMPT,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )

    raw_sources = {}

    # Main comprehensive search
    print(f"  Searching: comprehensive exam format...")
    response = client.models.generate_content(
        model=model,
        contents=PHASE1A_USER_PROMPT_TEMPLATE.format(certification=cert_name),
        config=config,
    )
    raw_sources["comprehensive"] = response.text

    # Additional targeted searches for specific info
    targeted_queries = [
        ("item_types", f'"{cert_name}" exam item types question formats TEI technology enhanced'),
        ("domains", f'"{cert_name}" exam content domains objectives percentage weights'),
    ]

    for query_name, query in targeted_queries:
        try:
            print(f"  Searching: {query_name}...")
            response = client.models.generate_content(
                model=model,
                contents=f"Find specific information about: {query}",
                config=config,
            )
            raw_sources[query_name] = response.text
        except Exception as e:
            print(f"    Warning: {query_name} search failed: {e}")

    return raw_sources


def _structure_exam_format_v2(
    client: genai.Client, raw_sources: dict[str, str], cert_name: str, model: str
) -> ExamFormatV2:
    """Phase 1B: Structure raw search results into ExamFormatV2 model."""

    # Combine all sources
    combined_raw = "\n\n---\n\n".join(
        f"=== {name.upper()} ===\n{text}" for name, text in raw_sources.items()
    )

    response = client.models.generate_content(
        model=model,
        contents=PHASE1B_USER_PROMPT_TEMPLATE.format(
            certification=cert_name,
            schema=PHASE1B_EXTRACTION_SCHEMA,
            raw_description=combined_raw,
        ),
        config={"system_instruction": PHASE1B_SYSTEM_PROMPT},
    )

    raw_json = response.text.strip()

    # Strip markdown code fences if present
    if raw_json.startswith("```"):
        lines = raw_json.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_json = "\n".join(lines)

    data = json.loads(raw_json)

    # Build ExamFormatV2
    exam_components = []
    for comp_data in data.get("exam_components", []):
        domains = [
            ExamDomainV2(
                name=d.get("name", "Unknown"),
                weight_min_pct=d.get("weight_min_pct"),
                weight_max_pct=d.get("weight_max_pct"),
                weight_pct=d.get("weight_pct"),
                description=d.get("description"),
                source_type=d.get("source_type", "inferred"),
            )
            for d in comp_data.get("domains", [])
        ]

        item_classes = [
            ItemClass(
                name=ic.get("name", "unknown"),
                display_name=ic.get("display_name"),
                is_tei=ic.get("is_tei", False),
                interaction_model=ic.get("interaction_model", "single_select"),
                allowed_context_types=ic.get("allowed_context_types", ["scenario", "direct"]),
                patient_presence=ic.get("patient_presence", "optional"),
                age_representation=ic.get("age_representation", ["exact", "broad", "none"]),
                description=ic.get("description"),
                grading=ic.get("grading", "unknown"),
                estimated_percentage=ic.get("estimated_percentage"),
                source_type=ic.get("source_type", "inferred"),
            )
            for ic in comp_data.get("item_classes", [])
        ]

        exam_components.append(
            ExamComponent(
                name=comp_data.get("name", "cognitive"),
                display_name=comp_data.get("display_name"),
                description=comp_data.get("description"),
                adaptive=comp_data.get("adaptive", False),
                adaptive_algorithm=comp_data.get("adaptive_algorithm"),
                num_questions=comp_data.get("num_questions"),
                num_questions_min=comp_data.get("num_questions_min"),
                num_questions_max=comp_data.get("num_questions_max"),
                pilot_unscored_items=comp_data.get("pilot_unscored_items"),
                time_limit_minutes=comp_data.get("time_limit_minutes"),
                delivery_methods=comp_data.get("delivery_methods", []),
                passing_score=comp_data.get("passing_score"),
                passing_score_source=comp_data.get("passing_score_source", "unknown"),
                domains=domains,
                item_classes=item_classes,
            )
        )

    source_facts = [
        SourceFact(
            field=sf.get("field", ""),
            value=sf.get("value"),
            source_type=sf.get("source_type", "inferred"),
            source_title=sf.get("source_title"),
            source_url=sf.get("source_url"),
            confidence=sf.get("confidence", "medium"),
        )
        for sf in data.get("source_facts", [])
    ]

    return ExamFormatV2(
        certification_name=data.get("certification_name", cert_name),
        certification_code=data.get("certification_code"),
        certifying_body=data.get("certifying_body"),
        exam_code=data.get("exam_code"),
        exam_components=exam_components,
        prerequisites=data.get("prerequisites", []),
        cost_usd=data.get("cost_usd"),
        validity_years=data.get("validity_years"),
        languages=data.get("languages", []),
        recertification_policy=data.get("recertification_policy"),
        source_facts=source_facts,
        raw_sources=raw_sources,
        discovered_at=datetime.now(),
    )


# =============================================================================
# V1 Discovery Functions - Legacy
# =============================================================================


def _search_exam_format(client: genai.Client, cert_name: str, model: str) -> str:
    """Phase 1: Search the web for exam format information (legacy)."""
    config = types.GenerateContentConfig(
        system_instruction=PHASE1_SYSTEM_PROMPT,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    response = client.models.generate_content(
        model=model,
        contents=PHASE1_USER_PROMPT_TEMPLATE.format(certification=cert_name),
        config=config,
    )
    return response.text


def _structure_exam_format(
    client: genai.Client, raw_description: str, cert_name: str, model: str
) -> ExamFormat:
    """Phase 2: Structure raw text into an ExamFormat model (legacy)."""
    response = client.models.generate_content(
        model=model,
        contents=PHASE2_USER_PROMPT_TEMPLATE.format(
            certification=cert_name, raw_description=raw_description
        ),
        config={
            "system_instruction": PHASE2_SYSTEM_PROMPT,
        },
    )

    raw_json = response.text.strip()
    # Strip markdown code fences if present
    if raw_json.startswith("```"):
        lines = raw_json.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_json = "\n".join(lines)

    data = json.loads(raw_json)
    domains = [ExamDomain(**d) for d in data.get("domains", [])]
    raw_qtypes = data.get("question_types", [])
    question_types = []
    for qt in raw_qtypes:
        if isinstance(qt, str):
            question_types.append(QuestionType(name=qt))
        elif isinstance(qt, dict):
            question_types.append(QuestionType(**qt))
    return ExamFormat(
        certification_name=data.get("certification_name", cert_name),
        exam_code=data.get("exam_code"),
        num_questions=data.get("num_questions"),
        time_limit_minutes=data.get("time_limit_minutes"),
        question_types=question_types,
        passing_score=data.get("passing_score"),
        domains=domains,
        prerequisites=data.get("prerequisites", []),
        cost_usd=data.get("cost_usd"),
        validity_years=data.get("validity_years"),
        delivery_methods=data.get("delivery_methods", []),
        languages=data.get("languages", []),
        recertification_policy=data.get("recertification_policy"),
        additional_notes=data.get("additional_notes"),
        raw_description=raw_description,
    )


# =============================================================================
# Pipeline Steps
# =============================================================================


class ExamFormatStep(PipelineStep):
    """Pipeline step that discovers and structures exam format information.

    Supports both V1 (legacy) and V2 (improved) discovery modes.
    """

    def __init__(self, model: str = "gemini-2.0-flash", use_v2: bool = True):
        """Initialize the exam format step.

        Args:
            model: The Gemini model to use for generation.
            use_v2: If True, use improved V2 discovery with provenance tracking.
        """
        self.model = model
        self.use_v2 = use_v2

    def run(self, context: PipelineContext) -> PipelineContext:
        """Discover exam format via web search and structure it.

        Expects context["certification_name"].
        Populates context["exam_format"] (and context["exam_format_v2"] if using V2).
        """
        cert_name = context["certification_name"]
        print(f"\nDiscovering exam format for '{cert_name}'...")
        print(f"  Mode: {'V2 (improved)' if self.use_v2 else 'V1 (legacy)'}")

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        if self.use_v2:
            return self._run_v2(context, client, cert_name)
        else:
            return self._run_v1(context, client, cert_name)

    def _run_v2(
        self, context: PipelineContext, client: genai.Client, cert_name: str
    ) -> PipelineContext:
        """Run V2 discovery with improved schema and provenance."""
        # Phase 1A: Multi-query search
        print("\nPhase 1A: Searching for exam format information...")
        raw_sources = _search_exam_format_v2(client, cert_name, self.model)

        if not raw_sources.get("comprehensive"):
            raise RuntimeError(
                f"Phase 1A failed: no exam format information found for '{cert_name}'"
            )

        # Phase 1B: Structure the raw data
        print("\nPhase 1B: Structuring exam format...")
        try:
            exam_format_v2 = _structure_exam_format_v2(
                client, raw_sources, cert_name, self.model
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: V2 parsing failed ({e}), falling back to V1")
            return self._run_v1(context, client, cert_name)

        # Store V2 format
        context["exam_format_v2"] = exam_format_v2

        # Also create a V1 format for backwards compatibility
        v1_domains = []
        v1_qtypes = []
        num_questions = None
        time_limit = None
        delivery_methods = []

        for comp in exam_format_v2.exam_components:
            # Use first component's data for V1 (usually cognitive)
            if num_questions is None:
                num_questions = comp.num_questions or comp.num_questions_max
            if time_limit is None:
                time_limit = comp.time_limit_minutes
            delivery_methods.extend(comp.delivery_methods)

            for domain in comp.domains:
                v1_domains.append(
                    ExamDomain(
                        name=domain.name,
                        weight_pct=domain.weight_pct
                        or (
                            (domain.weight_min_pct + domain.weight_max_pct) / 2
                            if domain.weight_min_pct and domain.weight_max_pct
                            else None
                        ),
                        description=domain.description,
                    )
                )

            for ic in comp.item_classes:
                v1_qtypes.append(
                    QuestionType(
                        name=ic.display_name or ic.name,
                        description=ic.description,
                        purpose=f"TEI: {ic.is_tei}, Context: {ic.allowed_context_types}",
                        grading_notes=ic.grading,
                    )
                )

        v1_format = ExamFormat(
            certification_name=exam_format_v2.certification_name,
            exam_code=exam_format_v2.exam_code,
            num_questions=num_questions,
            time_limit_minutes=time_limit,
            question_types=v1_qtypes,
            passing_score=exam_format_v2.exam_components[0].passing_score
            if exam_format_v2.exam_components
            else None,
            domains=v1_domains,
            prerequisites=exam_format_v2.prerequisites,
            cost_usd=exam_format_v2.cost_usd,
            validity_years=exam_format_v2.validity_years,
            delivery_methods=list(set(delivery_methods)),
            languages=exam_format_v2.languages,
            recertification_policy=exam_format_v2.recertification_policy,
            raw_description=raw_sources.get("comprehensive", ""),
        )

        context["exam_format"] = v1_format
        context["exam_format_raw"] = raw_sources.get("comprehensive", "")

        # Print summary
        print(f"\n{'='*60}")
        print(f"Exam Format Discovery Complete")
        print(f"{'='*60}")
        print(f"Certification: {exam_format_v2.certification_name}")
        print(f"Certifying Body: {exam_format_v2.certifying_body or 'Unknown'}")
        print(f"Components: {len(exam_format_v2.exam_components)}")

        for comp in exam_format_v2.exam_components:
            print(f"\n  [{comp.name.upper()}]")
            if comp.adaptive:
                print(f"    Adaptive: Yes ({comp.adaptive_algorithm or 'CAT'})")
                print(
                    f"    Questions: {comp.num_questions_min}-{comp.num_questions_max}"
                    + (f" (+{comp.pilot_unscored_items} pilot)" if comp.pilot_unscored_items else "")
                )
            else:
                print(f"    Questions: {comp.num_questions}")
            print(f"    Time: {comp.time_limit_minutes} minutes")
            print(f"    Domains: {len(comp.domains)}")
            print(f"    Item Types: {len(comp.item_classes)}")
            for ic in comp.item_classes:
                tei_flag = " [TEI]" if ic.is_tei else ""
                print(f"      - {ic.display_name or ic.name}{tei_flag}")

        print(f"\nSource Facts Tracked: {len(exam_format_v2.source_facts)}")
        print(f"{'='*60}\n")

        return context

    def _run_v1(
        self, context: PipelineContext, client: genai.Client, cert_name: str
    ) -> PipelineContext:
        """Run V1 (legacy) discovery."""
        # Phase 1: Search the web
        raw_description = _search_exam_format(client, cert_name, self.model)
        if not raw_description:
            raise RuntimeError(
                f"Phase 1 failed: no exam format information found for '{cert_name}'"
            )

        context["exam_format_raw"] = raw_description

        # Phase 2: Structure the raw text
        try:
            exam_format = _structure_exam_format(
                client, raw_description, cert_name, self.model
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            print("Warning: could not parse structured exam format, using raw only.")
            exam_format = ExamFormat(
                certification_name=cert_name,
                raw_description=raw_description,
            )

        context["exam_format"] = exam_format
        print(f"Exam format retrieved for '{cert_name}'.\n")
        return context
