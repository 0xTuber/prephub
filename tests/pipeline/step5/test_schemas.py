import json
import tempfile
from pathlib import Path

import pytest

from pipeline.models import QuestionType
from pipeline.step5.schemas import (
    PREDEFINED_SCHEMAS,
    ensure_schemas_exist,
    get_schema_for_prompt,
    load_schemas,
    normalize_type_name,
)


class TestNormalizeTypeName:
    def test_lowercase_with_spaces(self):
        assert normalize_type_name("Multiple Choice") == "multiple_choice"

    def test_mixed_case(self):
        assert normalize_type_name("MULTIPLE CHOICE") == "multiple_choice"

    def test_hyphenated(self):
        assert normalize_type_name("Multiple-Response") == "multiple_response"

    def test_already_normalized(self):
        assert normalize_type_name("drag_drop") == "drag_drop"

    def test_with_and(self):
        assert normalize_type_name("Drag and Drop") == "drag_and_drop"

    def test_extra_spaces(self):
        assert normalize_type_name("  fill  blank  ") == "fill_blank"

    def test_special_characters(self):
        assert normalize_type_name("Case-Study (Complex)") == "case_study_complex"

    def test_empty_string(self):
        assert normalize_type_name("") == ""

    def test_only_spaces(self):
        assert normalize_type_name("   ") == ""


class TestLoadSchemas:
    def test_creates_file_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_schemas(tmpdir)

            assert "version" in result
            assert "schemas" in result
            assert "multiple_choice" in result["schemas"]

            # Verify file was created
            schema_path = Path(tmpdir) / "question_types.json"
            assert schema_path.exists()

    def test_loads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "question_types.json"
            existing_data = {
                "version": "2.0",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "schemas": {
                    "custom_type": {"fields": ["custom"], "scoring": "custom"}
                },
            }
            schema_path.write_text(json.dumps(existing_data))

            result = load_schemas(tmpdir)

            assert result["version"] == "2.0"
            assert "custom_type" in result["schemas"]

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "schemas"
            result = load_schemas(str(nested_dir))

            assert "schemas" in result
            assert (nested_dir / "question_types.json").exists()


class TestEnsureSchemasExist:
    def test_preserves_existing_schemas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "question_types.json"
            existing_data = {
                "version": "1.0",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "schemas": {
                    "multiple_choice": {"fields": ["custom"], "scoring": "custom"}
                },
            }
            schema_path.write_text(json.dumps(existing_data))

            question_types = [QuestionType(name="multiple choice")]
            result = ensure_schemas_exist(tmpdir, question_types)

            # Should preserve existing schema, not overwrite
            assert result["multiple_choice"]["fields"] == ["custom"]

    def test_adds_new_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "question_types.json"
            existing_data = {
                "version": "1.0",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "schemas": {
                    "multiple_choice": {"fields": ["stem"], "scoring": "all_or_nothing"}
                },
            }
            schema_path.write_text(json.dumps(existing_data))

            # Add a new type
            question_types = [
                QuestionType(name="multiple choice"),
                QuestionType(name="drag drop"),
            ]
            result = ensure_schemas_exist(tmpdir, question_types)

            assert "multiple_choice" in result
            assert "drag_drop" in result

            # Verify file was updated
            updated_data = json.loads(schema_path.read_text())
            assert "drag_drop" in updated_data["schemas"]

    def test_unknown_type_gets_default_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            question_types = [QuestionType(name="brand new type")]
            result = ensure_schemas_exist(tmpdir, question_types)

            assert "brand_new_type" in result
            assert result["brand_new_type"]["scoring"] == "manual"

    def test_empty_question_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_schemas_exist(tmpdir, [])

            # Should return default schemas
            assert "multiple_choice" in result
            assert "drag_drop" in result


class TestGetSchemaForPrompt:
    def test_known_schema(self):
        schemas = {"multiple_choice": {"fields": ["stem", "choices"], "choice_count": 4}}
        result = get_schema_for_prompt("multiple_choice", schemas)

        parsed = json.loads(result)
        assert parsed["fields"] == ["stem", "choices"]
        assert parsed["choice_count"] == 4

    def test_unknown_schema_uses_default(self):
        schemas = {}
        result = get_schema_for_prompt("unknown_type", schemas)

        parsed = json.loads(result)
        assert parsed["scoring"] == "manual"
        assert "Custom question type" in parsed["description"]

    def test_predefined_fallback(self):
        schemas = {}
        result = get_schema_for_prompt("ordering", schemas)

        parsed = json.loads(result)
        assert parsed == PREDEFINED_SCHEMAS["ordering"]


class TestPredefinedSchemas:
    def test_all_common_types_defined(self):
        expected_types = [
            "multiple_choice",
            "multiple_response",
            "drag_drop",
            "hotspot",
            "fill_blank",
            "ordering",
            "simulation",
            "case_study",
        ]
        for type_name in expected_types:
            assert type_name in PREDEFINED_SCHEMAS

    def test_schemas_have_required_fields(self):
        for type_name, schema in PREDEFINED_SCHEMAS.items():
            assert "fields" in schema, f"{type_name} missing 'fields'"
            assert "scoring" in schema, f"{type_name} missing 'scoring'"
