"""Tests for query planning module."""

from unittest.mock import MagicMock, patch

import pytest

from course_builder.domain.content import CapsuleItem
from course_builder.pipeline.content.query_planning import (
    QueryIntent,
    QueryPlan,
    PlannedQuery,
    get_broadening_queries,
    get_precision_queries,
    get_queries_by_intent,
    plan_queries_for_item,
    _generate_fallback_queries,
)


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_intent_values(self):
        assert QueryIntent.HIGH_PRECISION == "high_precision"
        assert QueryIntent.SYNONYM_VARIANT == "synonym_variant"
        assert QueryIntent.QUOTE_HUNT == "quote_hunt"
        assert QueryIntent.SCENARIO_CONTEXT == "scenario_context"

    def test_all_intents_defined(self):
        intents = list(QueryIntent)
        assert len(intents) == 4


class TestPlannedQuery:
    """Tests for PlannedQuery model."""

    def test_create_basic_query(self):
        query = PlannedQuery(
            query_text="scene safety procedures",
            intent=QueryIntent.HIGH_PRECISION,
        )

        assert query.query_text == "scene safety procedures"
        assert query.intent == QueryIntent.HIGH_PRECISION
        assert query.must_include == []
        assert query.priority == 1

    def test_create_query_with_must_include(self):
        query = PlannedQuery(
            query_text="BSI infection control",
            intent=QueryIntent.SYNONYM_VARIANT,
            must_include=["gloves", "protection"],
            priority=2,
        )

        assert query.must_include == ["gloves", "protection"]
        assert query.priority == 2

    def test_query_json_serialization(self):
        query = PlannedQuery(
            query_text="test query",
            intent=QueryIntent.QUOTE_HUNT,
        )

        data = query.model_dump()
        assert data["query_text"] == "test query"
        assert data["intent"] == "quote_hunt"


class TestQueryPlan:
    """Tests for QueryPlan model."""

    def test_create_empty_plan(self):
        plan = QueryPlan(
            item_id="item_01",
            learning_target="scene safety assessment",
            queries=[],
        )

        assert plan.item_id == "item_01"
        assert plan.learning_target == "scene safety assessment"
        assert plan.queries == []
        assert plan.must_include_keywords == []

    def test_create_plan_with_queries(self):
        queries = [
            PlannedQuery(
                query_text="query 1",
                intent=QueryIntent.HIGH_PRECISION,
            ),
            PlannedQuery(
                query_text="query 2",
                intent=QueryIntent.SYNONYM_VARIANT,
            ),
        ]

        plan = QueryPlan(
            item_id="item_01",
            learning_target="hazard assessment",
            queries=queries,
            must_include_keywords=["hazard", "safety"],
        )

        assert len(plan.queries) == 2
        assert plan.must_include_keywords == ["hazard", "safety"]

    def test_plan_json_serialization(self):
        plan = QueryPlan(
            item_id="item_01",
            learning_target="test target",
            queries=[
                PlannedQuery(
                    query_text="test",
                    intent=QueryIntent.HIGH_PRECISION,
                ),
            ],
        )

        data = plan.model_dump()
        assert data["item_id"] == "item_01"
        assert len(data["queries"]) == 1


class TestPlanQueriesForItem:
    """Tests for plan_queries_for_item function."""

    @pytest.fixture
    def sample_item(self):
        return CapsuleItem(
            item_id="item_01",
            item_type="Multiple Choice",
            title="Scene Safety Assessment",
            learning_target="electrical hazard perimeter establishment",
        )

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        return engine

    def test_successful_query_planning(self, mock_engine, sample_item):
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.text = '''```json
{
  "queries": [
    {"query_text": "electrical hazard perimeter", "intent": "high_precision", "priority": 1},
    {"query_text": "downed power line safety", "intent": "synonym_variant", "priority": 2},
    {"query_text": "electrical hazard should must ensure", "intent": "quote_hunt", "priority": 2},
    {"query_text": "scene safety electrical incident", "intent": "scenario_context", "priority": 3}
  ],
  "must_include_keywords": ["electrical", "perimeter", "safety"]
}
```'''
        mock_engine.generate.return_value = mock_response

        plan = plan_queries_for_item(
            engine=mock_engine,
            item=sample_item,
            topic_name="Scene Safety",
            subtopic_name="Hazard Assessment",
        )

        assert plan.item_id == "item_01"
        assert len(plan.queries) == 4
        assert plan.must_include_keywords == ["electrical", "perimeter", "safety"]

        # Check intent distribution
        intents = [q.intent for q in plan.queries]
        assert QueryIntent.HIGH_PRECISION in intents
        assert QueryIntent.QUOTE_HUNT in intents

    def test_fallback_on_json_error(self, mock_engine, sample_item):
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        mock_engine.generate.return_value = mock_response

        plan = plan_queries_for_item(
            engine=mock_engine,
            item=sample_item,
            topic_name="Scene Safety",
            subtopic_name="Hazard Assessment",
        )

        # Should fall back to basic queries
        assert plan.item_id == "item_01"
        assert len(plan.queries) >= 3
        assert any(q.intent == QueryIntent.HIGH_PRECISION for q in plan.queries)

    def test_with_available_sections(self, mock_engine, sample_item):
        mock_response = MagicMock()
        mock_response.text = '{"queries": [], "must_include_keywords": []}'
        mock_engine.generate.return_value = mock_response

        sections = [
            "Electrical Hazards",
            "Scene Size-Up Procedures",
            "Personal Safety",
        ]

        plan_queries_for_item(
            engine=mock_engine,
            item=sample_item,
            topic_name="Scene Safety",
            subtopic_name="Hazard Assessment",
            available_sections=sections,
        )

        # Verify sections were passed to prompt
        call_args = mock_engine.generate.call_args
        prompt = call_args[0][0]
        assert "Electrical Hazards" in prompt


class TestGenerateFallbackQueries:
    """Tests for _generate_fallback_queries function."""

    @pytest.fixture
    def sample_item(self):
        return CapsuleItem(
            item_id="item_01",
            item_type="Multiple Choice",
            title="PPE Selection",
            learning_target="select appropriate personal protective equipment",
        )

    def test_generates_multiple_queries(self, sample_item):
        plan = _generate_fallback_queries(
            item=sample_item,
            topic_name="BSI",
            subtopic_name="PPE",
        )

        assert len(plan.queries) >= 3

    def test_includes_high_precision_query(self, sample_item):
        plan = _generate_fallback_queries(
            item=sample_item,
            topic_name="BSI",
            subtopic_name="PPE",
        )

        precision_queries = [q for q in plan.queries if q.intent == QueryIntent.HIGH_PRECISION]
        assert len(precision_queries) >= 1

    def test_includes_quote_hunt_query(self, sample_item):
        plan = _generate_fallback_queries(
            item=sample_item,
            topic_name="BSI",
            subtopic_name="PPE",
        )

        quote_queries = [q for q in plan.queries if q.intent == QueryIntent.QUOTE_HUNT]
        assert len(quote_queries) >= 1

    def test_extracts_keywords_from_target(self, sample_item):
        plan = _generate_fallback_queries(
            item=sample_item,
            topic_name="BSI",
            subtopic_name="PPE",
        )

        # Should extract meaningful keywords
        assert len(plan.must_include_keywords) > 0
        # Should include substantive words
        keywords_lower = [kw.lower() for kw in plan.must_include_keywords]
        assert any(kw in keywords_lower for kw in ["appropriate", "personal", "protective", "equipment", "select"])


class TestQueryFiltering:
    """Tests for query filtering functions."""

    @pytest.fixture
    def sample_plan(self):
        return QueryPlan(
            item_id="item_01",
            learning_target="test target",
            queries=[
                PlannedQuery(query_text="q1", intent=QueryIntent.HIGH_PRECISION),
                PlannedQuery(query_text="q2", intent=QueryIntent.HIGH_PRECISION),
                PlannedQuery(query_text="q3", intent=QueryIntent.SYNONYM_VARIANT),
                PlannedQuery(query_text="q4", intent=QueryIntent.QUOTE_HUNT),
                PlannedQuery(query_text="q5", intent=QueryIntent.SCENARIO_CONTEXT),
            ],
        )

    def test_get_queries_by_intent(self, sample_plan):
        precision = get_queries_by_intent(sample_plan, QueryIntent.HIGH_PRECISION)
        assert len(precision) == 2

        synonym = get_queries_by_intent(sample_plan, QueryIntent.SYNONYM_VARIANT)
        assert len(synonym) == 1

    def test_get_precision_queries(self, sample_plan):
        precision = get_precision_queries(sample_plan)
        assert len(precision) == 2
        assert all(q.intent == QueryIntent.HIGH_PRECISION for q in precision)

    def test_get_broadening_queries(self, sample_plan):
        broadening = get_broadening_queries(sample_plan)
        assert len(broadening) == 2  # SYNONYM_VARIANT + SCENARIO_CONTEXT
        intents = {q.intent for q in broadening}
        assert QueryIntent.SYNONYM_VARIANT in intents
        assert QueryIntent.SCENARIO_CONTEXT in intents
