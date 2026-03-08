"""Query planning module for targeted evidence retrieval.

This module implements LLM-driven query generation to improve retrieval accuracy.
Instead of naive concatenation of topic + learning_target, we generate 4-6 targeted
queries with different intents to maximize the chance of finding supporting evidence.

Key intents:
- HIGH_PRECISION: Exact terminology, section headings
- SYNONYM_VARIANT: Alternative phrasings (EMT jargon, layman terms)
- QUOTE_HUNT: Targets sentences with action verbs ("should", "must", "ensure")
- SCENARIO_CONTEXT: Contextual keywords for realistic scenarios
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from course_builder.domain.content import CapsuleItem
    from course_builder.engine import GenerationEngine


class QueryIntent(str, Enum):
    """Intent classification for planned queries."""

    HIGH_PRECISION = "high_precision"  # Exact terms, section headings
    SYNONYM_VARIANT = "synonym_variant"  # Paraphrased terminology
    QUOTE_HUNT = "quote_hunt"  # Targets citable sentences
    SCENARIO_CONTEXT = "scenario_context"  # Contextual keywords


class PlannedQuery(BaseModel):
    """A single planned query with its intent and constraints."""

    query_text: str
    intent: QueryIntent
    must_include: list[str] = []  # Keywords that MUST appear in results
    priority: int = 1  # 1 = highest priority


class QueryPlan(BaseModel):
    """Complete query plan for retrieving evidence for an item."""

    item_id: str
    learning_target: str
    queries: list[PlannedQuery]  # 4-6 targeted queries
    must_include_keywords: list[str] = []  # Global must-include strings


QUERY_PLANNING_SYSTEM_PROMPT = """You are an expert at designing search queries to find textbook evidence for exam questions.
Your queries should be diverse and targeted to maximize the chance of finding relevant source material."""


QUERY_PLANNING_USER_PROMPT = """Generate search queries to find textbook evidence for this exam question.

TOPIC: {topic_name} > {subtopic_name}
LEARNING TARGET: {learning_target}

AVAILABLE SECTIONS (from source material):
{available_sections}

Generate 4-6 queries with different intents:

1. HIGH_PRECISION (1-2 queries): Use exact terminology from the learning target, section titles, or medical terms.
   Examples: "scene safety assessment procedures", "electrical hazard isolation perimeter"

2. SYNONYM_VARIANT (1-2 queries): Use alternative phrasings, EMT jargon, or layman terms.
   Examples: If target says "BSI", include "body substance isolation" or "infection control"

3. QUOTE_HUNT (1 query): Target sentences likely to contain citable evidence.
   Use patterns like: "[topic] should", "[topic] must", "ensure [topic]", "always [topic]"

4. SCENARIO_CONTEXT (1 query): Keywords for realistic clinical/field scenarios.
   Examples: "traffic accident scene", "patient assessment priority"

Also specify 2-3 must_include keywords that MUST appear in useful chunks.

Return a JSON object:
{{
  "queries": [
    {{
      "query_text": "exact search string",
      "intent": "high_precision|synonym_variant|quote_hunt|scenario_context",
      "must_include": ["optional", "keywords"],
      "priority": 1
    }}
  ],
  "must_include_keywords": ["key", "terms"]
}}

Return ONLY the JSON object."""


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


def plan_queries_for_item(
    engine: "GenerationEngine",
    item: "CapsuleItem",
    topic_name: str,
    subtopic_name: str,
    available_sections: list[str] | None = None,
) -> QueryPlan:
    """Generate 4-6 targeted queries for an item using LLM.

    This function uses an LLM to generate diverse queries with different intents
    to maximize retrieval coverage for the learning target.

    Args:
        engine: Generation engine for LLM calls
        item: The CapsuleItem to plan queries for
        topic_name: Parent topic name
        subtopic_name: Parent subtopic name
        available_sections: Optional list of section headings from source material

    Returns:
        QueryPlan with 4-6 targeted queries
    """
    from course_builder.engine import GenerationConfig

    # Format available sections
    sections_str = "\n".join(f"- {s}" for s in (available_sections or [])[:10])
    if not sections_str:
        sections_str = "(No section information available)"

    prompt = QUERY_PLANNING_USER_PROMPT.format(
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        learning_target=item.learning_target,
        available_sections=sections_str,
    )

    config = GenerationConfig(system_prompt=QUERY_PLANNING_SYSTEM_PROMPT)
    result = engine.generate(prompt, config=config)

    try:
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)

        queries = []
        for q in data.get("queries", []):
            try:
                intent = QueryIntent(q.get("intent", "high_precision"))
            except ValueError:
                intent = QueryIntent.HIGH_PRECISION

            queries.append(PlannedQuery(
                query_text=q.get("query_text", ""),
                intent=intent,
                must_include=q.get("must_include", []),
                priority=q.get("priority", 1),
            ))

        return QueryPlan(
            item_id=item.item_id,
            learning_target=item.learning_target,
            queries=queries,
            must_include_keywords=data.get("must_include_keywords", []),
        )

    except (json.JSONDecodeError, KeyError) as e:
        # Fallback: generate basic queries without LLM
        return _generate_fallback_queries(item, topic_name, subtopic_name)


def _generate_fallback_queries(
    item: "CapsuleItem",
    topic_name: str,
    subtopic_name: str,
) -> QueryPlan:
    """Generate basic queries without LLM (fallback).

    Used when LLM query planning fails.
    """
    learning_target = item.learning_target

    queries = [
        # HIGH_PRECISION: Full context
        PlannedQuery(
            query_text=f"{topic_name} {subtopic_name} {learning_target}",
            intent=QueryIntent.HIGH_PRECISION,
            priority=1,
        ),
        # HIGH_PRECISION: Just learning target
        PlannedQuery(
            query_text=learning_target,
            intent=QueryIntent.HIGH_PRECISION,
            priority=1,
        ),
        # QUOTE_HUNT: Action verb pattern
        PlannedQuery(
            query_text=f"{learning_target} should must ensure",
            intent=QueryIntent.QUOTE_HUNT,
            priority=2,
        ),
        # SCENARIO_CONTEXT: With procedure keywords
        PlannedQuery(
            query_text=f"{subtopic_name} procedures assessment",
            intent=QueryIntent.SCENARIO_CONTEXT,
            priority=2,
        ),
    ]

    # Extract likely keywords from learning target
    words = re.findall(r"\b\w{4,}\b", learning_target.lower())
    stop_words = {"that", "this", "with", "from", "about", "should", "would"}
    keywords = [w for w in words if w not in stop_words][:3]

    return QueryPlan(
        item_id=item.item_id,
        learning_target=learning_target,
        queries=queries,
        must_include_keywords=keywords,
    )


def plan_queries_batch(
    engine: "GenerationEngine",
    items: list["CapsuleItem"],
    topic_name: str,
    subtopic_name: str,
    available_sections: list[str] | None = None,
) -> list[QueryPlan]:
    """Plan queries for multiple items.

    Args:
        engine: Generation engine for LLM calls
        items: List of CapsuleItems
        topic_name: Parent topic name
        subtopic_name: Parent subtopic name
        available_sections: Optional list of section headings

    Returns:
        List of QueryPlans, one per item
    """
    plans = []
    for item in items:
        plan = plan_queries_for_item(
            engine=engine,
            item=item,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            available_sections=available_sections,
        )
        plans.append(plan)
    return plans


def get_queries_by_intent(
    plan: QueryPlan,
    intent: QueryIntent,
) -> list[PlannedQuery]:
    """Filter queries by intent type.

    Args:
        plan: The QueryPlan to filter
        intent: The intent type to filter for

    Returns:
        List of queries matching the intent
    """
    return [q for q in plan.queries if q.intent == intent]


def get_precision_queries(plan: QueryPlan) -> list[PlannedQuery]:
    """Get high-precision queries for initial retrieval."""
    return get_queries_by_intent(plan, QueryIntent.HIGH_PRECISION)


def get_broadening_queries(plan: QueryPlan) -> list[PlannedQuery]:
    """Get synonym and context queries for expanded retrieval."""
    synonym = get_queries_by_intent(plan, QueryIntent.SYNONYM_VARIANT)
    context = get_queries_by_intent(plan, QueryIntent.SCENARIO_CONTEXT)
    return synonym + context
