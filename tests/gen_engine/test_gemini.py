"""Tests for Gemini generation engine."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from course_builder.engine.base import (
    GenerationConfig,
    GenerationError,
    Message,
    StopReason,
)
from course_builder.engine.gemini import GeminiEngine


def _make_mock_response(text: str = "Test response", finish_reason: str = "STOP"):
    """Create a mock Gemini API response."""
    response = SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(
                finish_reason=finish_reason,
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=text)]
                ),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
        ),
    )
    return response


class TestGeminiEngineInit:
    """Tests for GeminiEngine initialization."""

    def test_init_with_api_key(self):
        engine = GeminiEngine(model="gemini-pro", api_key="test-key")
        assert engine.model_name == "gemini-pro"
        assert engine.engine_type == "gemini"
        assert engine._api_key == "test-key"

    def test_init_from_env_var(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}):
            engine = GeminiEngine(model="gemini-pro")
            assert engine._api_key == "env-key"

    def test_init_without_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            import os

            original = os.environ.pop("GOOGLE_API_KEY", None)
            try:
                with pytest.raises(GenerationError) as exc_info:
                    GeminiEngine(model="gemini-pro")
                assert "No API key" in str(exc_info.value)
            finally:
                if original:
                    os.environ["GOOGLE_API_KEY"] = original

    def test_custom_retry_settings(self):
        engine = GeminiEngine(
            model="gemini-pro",
            api_key="test",
            max_retries=5,
            retry_delay=1.0,
        )
        assert engine._max_retries == 5
        assert engine._retry_delay == 1.0


class TestGeminiEngineGenerate:
    """Tests for GeminiEngine.generate()."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Gemini client."""
        client = MagicMock()
        client.models.generate_content.return_value = _make_mock_response("Hello!")
        return client

    @pytest.fixture
    def engine(self, mock_client):
        """Create engine with mocked client."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            engine = GeminiEngine(model="gemini-flash-latest")
            engine._client = mock_client
            return engine

    def test_generate_basic(self, engine, mock_client):
        result = engine.generate("Hello")
        assert result.text == "Hello!"
        assert result.stop_reason == StopReason.END_OF_SEQUENCE
        assert result.model == "gemini-flash-latest"

        mock_client.models.generate_content.assert_called_once()

    def test_generate_with_config(self, engine, mock_client):
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            system_prompt="Be helpful.",
        )
        result = engine.generate("Hello", config=config)

        call_args = mock_client.models.generate_content.call_args
        assert call_args is not None

    def test_generate_parses_usage(self, engine):
        result = engine.generate("Test")
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    def test_generate_max_tokens_stop_reason(self, engine, mock_client):
        mock_client.models.generate_content.return_value = _make_mock_response(
            "Partial", finish_reason="MAX_TOKENS"
        )
        result = engine.generate("Test")
        assert result.stop_reason == StopReason.MAX_TOKENS

    def test_generate_retries_on_failure(self, engine, mock_client):
        mock_client.models.generate_content.side_effect = [
            Exception("Temporary error"),
            _make_mock_response("Success"),
        ]

        with patch("course_builder.engine.gemini.time.sleep"):
            result = engine.generate("Test")

        assert result.text == "Success"
        assert mock_client.models.generate_content.call_count == 2

    def test_generate_raises_after_max_retries(self, engine, mock_client):
        mock_client.models.generate_content.side_effect = Exception("Persistent error")

        with patch("course_builder.engine.gemini.time.sleep"):
            with pytest.raises(GenerationError) as exc_info:
                engine.generate("Test")

        assert "retries" in str(exc_info.value)
        assert mock_client.models.generate_content.call_count == 3  # default max_retries


class TestGeminiEngineChat:
    """Tests for GeminiEngine.chat()."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.models.generate_content.return_value = _make_mock_response("Chat response")
        return client

    @pytest.fixture
    def engine(self, mock_client):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            engine = GeminiEngine(model="gemini-pro")
            engine._client = mock_client
            return engine

    def test_chat_basic(self, engine, mock_client):
        messages = [
            Message(role="user", content="Hello"),
        ]
        result = engine.chat(messages)
        assert result.text == "Chat response"

    def test_chat_with_history(self, engine, mock_client):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        result = engine.chat(messages)
        assert result.text == "Chat response"

    def test_chat_with_system_message(self, engine, mock_client):
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]
        result = engine.chat(messages)

        # System message should be extracted to config
        call_args = mock_client.models.generate_content.call_args
        assert call_args is not None


class TestGeminiEngineBuildConfig:
    """Tests for config building."""

    @pytest.fixture
    def engine(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            return GeminiEngine(model="gemini-pro")

    def test_build_config_none(self, engine):
        result = engine._build_config(None)
        assert result is None

    def test_build_config_with_values(self, engine):
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
            system_prompt="Be helpful.",
        )
        result = engine._build_config(config)
        assert result is not None

    def test_build_config_json_mode(self, engine):
        config = GenerationConfig(json_mode=True)
        result = engine._build_config(config)
        # Should set response_mime_type
        assert result is not None


class TestGeminiEngineAvailability:
    """Tests for engine availability checking."""

    def test_is_available_returns_bool(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            engine = GeminiEngine(model="gemini-pro")
            engine._client = MagicMock()
            engine._client.models.generate_content.return_value = _make_mock_response("OK")

            assert engine.is_available() is True

    def test_is_available_false_on_error(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            engine = GeminiEngine(model="gemini-pro")
            engine._client = MagicMock()
            engine._client.models.generate_content.side_effect = Exception("Error")

            assert engine.is_available() is False
