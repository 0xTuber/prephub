"""Tests for gen_engine base types and protocols."""

import pytest

from course_builder.engine.base import (
    GenerationConfig,
    GenerationError,
    GenerationResult,
    Message,
    StopReason,
    StreamChunk,
    TokenUsage,
)


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_defaults(self):
        config = GenerationConfig()
        assert config.max_tokens is None
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.top_k is None
        assert config.stop_sequences == []
        assert config.system_prompt is None
        assert config.json_mode is False
        assert config.seed is None

    def test_custom_values(self):
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            stop_sequences=["END", "STOP"],
            system_prompt="You are helpful.",
            json_mode=True,
            seed=42,
        )
        assert config.max_tokens == 100
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.stop_sequences == ["END", "STOP"]
        assert config.system_prompt == "You are helpful."
        assert config.json_mode is True
        assert config.seed == 42

    def test_to_dict_excludes_none(self):
        config = GenerationConfig(temperature=0.5, max_tokens=100)
        d = config.to_dict()
        assert "temperature" in d
        assert "max_tokens" in d
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 100


class TestMessage:
    """Tests for Message."""

    def test_user_message(self):
        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_assistant_message(self):
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_system_message(self):
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."


class TestTokenUsage:
    """Tests for TokenUsage."""

    def test_defaults(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestStopReason:
    """Tests for StopReason enum."""

    def test_values(self):
        assert StopReason.END_OF_SEQUENCE.value == "end_of_sequence"
        assert StopReason.MAX_TOKENS.value == "max_tokens"
        assert StopReason.STOP_SEQUENCE.value == "stop_sequence"
        assert StopReason.ERROR.value == "error"
        assert StopReason.UNKNOWN.value == "unknown"


class TestGenerationResult:
    """Tests for GenerationResult."""

    def test_minimal(self):
        result = GenerationResult(text="Hello!")
        assert result.text == "Hello!"
        assert result.stop_reason == StopReason.UNKNOWN
        assert result.model is None

    def test_full(self):
        usage = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        result = GenerationResult(
            text="Response",
            stop_reason=StopReason.END_OF_SEQUENCE,
            usage=usage,
            model="test-model",
            raw_response={"raw": "data"},
        )
        assert result.text == "Response"
        assert result.stop_reason == StopReason.END_OF_SEQUENCE
        assert result.usage.total_tokens == 15
        assert result.model == "test-model"
        assert result.raw_response == {"raw": "data"}

    def test_is_complete_true(self):
        result = GenerationResult(text="", stop_reason=StopReason.END_OF_SEQUENCE)
        assert result.is_complete is True

        result = GenerationResult(text="", stop_reason=StopReason.STOP_SEQUENCE)
        assert result.is_complete is True

    def test_is_complete_false(self):
        result = GenerationResult(text="", stop_reason=StopReason.MAX_TOKENS)
        assert result.is_complete is False

        result = GenerationResult(text="", stop_reason=StopReason.ERROR)
        assert result.is_complete is False


class TestStreamChunk:
    """Tests for StreamChunk."""

    def test_defaults(self):
        chunk = StreamChunk(text="Hello")
        assert chunk.text == "Hello"
        assert chunk.is_final is False
        assert chunk.stop_reason is None

    def test_final_chunk(self):
        chunk = StreamChunk(
            text="!",
            is_final=True,
            stop_reason=StopReason.END_OF_SEQUENCE,
        )
        assert chunk.text == "!"
        assert chunk.is_final is True
        assert chunk.stop_reason == StopReason.END_OF_SEQUENCE


class TestGenerationError:
    """Tests for GenerationError."""

    def test_basic_error(self):
        error = GenerationError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_error_with_context(self):
        cause = ValueError("invalid value")
        error = GenerationError(
            "Generation failed",
            engine="test",
            model="test-model",
            cause=cause,
        )
        assert "Generation failed" in str(error)
        assert "engine=test" in str(error)
        assert "model=test-model" in str(error)
        assert error.engine == "test"
        assert error.model == "test-model"
        assert error.cause == cause

    def test_error_is_exception(self):
        error = GenerationError("Test")
        assert isinstance(error, Exception)

        with pytest.raises(GenerationError):
            raise error
