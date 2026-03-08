"""Tests for vLLM generation engines."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from course_builder.engine.base import (
    GenerationConfig,
    GenerationError,
    Message,
    StopReason,
)

# Mock vllm module before importing VLLMEngine
mock_vllm = MagicMock()
mock_vllm.SamplingParams = MagicMock(return_value=MagicMock())
mock_vllm.LLM = MagicMock()
sys.modules["vllm"] = mock_vllm

from course_builder.engine.vllm import VLLMEngine, VLLMServerEngine


class TestVLLMEngineInit:
    """Tests for VLLMEngine initialization."""

    def test_init_basic(self):
        engine = VLLMEngine(model="meta-llama/Llama-3.1-8B-Instruct")
        assert engine.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert engine.engine_type == "vllm"

    def test_init_with_options(self):
        engine = VLLMEngine(
            model="test-model",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            dtype="float16",
            quantization="awq",
        )
        assert engine._tensor_parallel_size == 2
        assert engine._gpu_memory_utilization == 0.8
        assert engine._max_model_len == 4096
        assert engine._dtype == "float16"
        assert engine._quantization == "awq"

    def test_lazy_initialization(self):
        engine = VLLMEngine(model="test-model")
        # LLM should not be loaded yet
        assert engine._llm is None


class TestVLLMEngineGenerate:
    """Tests for VLLMEngine.generate()."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock vLLM LLM object."""
        llm = MagicMock()

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.encode.return_value = [1, 2, 3, 4]
        llm.get_tokenizer.return_value = tokenizer

        # Mock generation output
        output = SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[
                SimpleNamespace(
                    text="Generated response",
                    token_ids=[4, 5, 6, 7],
                    finish_reason="stop",
                )
            ],
        )
        llm.generate.return_value = [output]

        return llm

    @pytest.fixture
    def engine(self, mock_llm):
        """Create engine with mocked LLM."""
        with patch.object(VLLMEngine, "_build_sampling_params", return_value=MagicMock()):
            engine = VLLMEngine(model="test-model")
            engine._llm = mock_llm
            engine._tokenizer = mock_llm.get_tokenizer()
            return engine

    def test_generate_basic(self, engine, mock_llm):
        with patch.object(engine, "_build_sampling_params", return_value=MagicMock()):
            result = engine.generate("Hello")
            assert result.text == "Generated response"
            assert result.stop_reason == StopReason.END_OF_SEQUENCE
            assert result.model == "test-model"

    def test_generate_with_config(self, engine, mock_llm):
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
        )
        with patch.object(engine, "_build_sampling_params", return_value=MagicMock()):
            result = engine.generate("Hello", config=config)

            # Check that sampling params were passed
            call_args = mock_llm.generate.call_args
            assert call_args is not None

    def test_generate_parses_usage(self, engine):
        with patch.object(engine, "_build_sampling_params", return_value=MagicMock()):
            result = engine.generate("Test")
            # prompt_token_ids has 3 tokens, output token_ids has 4
            assert result.usage.prompt_tokens == 3
            assert result.usage.completion_tokens == 4
            assert result.usage.total_tokens == 7

    def test_generate_length_stop_reason(self, engine, mock_llm):
        output = SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[
                SimpleNamespace(
                    text="Truncated",
                    token_ids=[4, 5],
                    finish_reason="length",
                )
            ],
        )
        mock_llm.generate.return_value = [output]

        with patch.object(engine, "_build_sampling_params", return_value=MagicMock()):
            result = engine.generate("Test")
            assert result.stop_reason == StopReason.MAX_TOKENS

    def test_count_tokens(self, engine):
        count = engine.count_tokens("Hello world")
        # Mock tokenizer returns [1, 2, 3, 4] for any input
        assert count == 4


class TestVLLMEngineChat:
    """Tests for VLLMEngine.chat()."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted chat"
        llm.get_tokenizer.return_value = tokenizer

        output = SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[
                SimpleNamespace(
                    text="Chat response",
                    token_ids=[4, 5, 6],
                    finish_reason="stop",
                )
            ],
        )
        llm.generate.return_value = [output]

        return llm

    @pytest.fixture
    def engine(self, mock_llm):
        engine = VLLMEngine(model="test-model")
        engine._llm = mock_llm
        engine._tokenizer = mock_llm.get_tokenizer()
        return engine

    def test_chat_basic(self, engine, mock_llm):
        messages = [Message(role="user", content="Hello")]
        with patch.object(engine, "_build_sampling_params", return_value=MagicMock()):
            result = engine.chat(messages)
            assert result.text == "Chat response"

    def test_chat_uses_template(self, engine, mock_llm):
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
        ]
        with patch.object(engine, "_build_sampling_params", return_value=MagicMock()):
            engine.chat(messages)

            # Should have called apply_chat_template
            mock_llm.get_tokenizer().apply_chat_template.assert_called()


class TestVLLMEngineBatch:
    """Tests for VLLMEngine.generate_batch()."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()

        tokenizer = MagicMock()
        llm.get_tokenizer.return_value = tokenizer

        # Return multiple outputs for batch
        outputs = [
            SimpleNamespace(
                prompt_token_ids=[1, 2],
                outputs=[SimpleNamespace(text="Response 1", token_ids=[3, 4], finish_reason="stop")],
            ),
            SimpleNamespace(
                prompt_token_ids=[5, 6],
                outputs=[SimpleNamespace(text="Response 2", token_ids=[7, 8], finish_reason="stop")],
            ),
        ]
        llm.generate.return_value = outputs

        return llm

    @pytest.fixture
    def engine(self, mock_llm):
        engine = VLLMEngine(model="test-model")
        engine._llm = mock_llm
        engine._tokenizer = mock_llm.get_tokenizer()
        return engine

    def test_batch_generate(self, engine, mock_llm):
        prompts = ["Prompt 1", "Prompt 2"]
        with patch.object(engine, "_build_sampling_params", return_value=MagicMock()):
            results = engine.generate_batch(prompts)

            assert len(results) == 2
            assert results[0].text == "Response 1"
            assert results[1].text == "Response 2"

            # Should be called once with all prompts
            mock_llm.generate.assert_called_once()


class TestVLLMServerEngineInit:
    """Tests for VLLMServerEngine initialization."""

    def test_init_basic(self):
        engine = VLLMServerEngine(base_url="http://localhost:8000/v1")
        assert engine._base_url == "http://localhost:8000/v1"
        assert engine.engine_type == "vllm-server"

    def test_init_with_model(self):
        engine = VLLMServerEngine(
            base_url="http://localhost:8000/v1",
            model="llama-3.1-8b",
        )
        assert engine._model == "llama-3.1-8b"

    def test_init_strips_trailing_slash(self):
        engine = VLLMServerEngine(base_url="http://localhost:8000/v1/")
        assert engine._base_url == "http://localhost:8000/v1"


class TestVLLMServerEngineGenerate:
    """Tests for VLLMServerEngine.generate()."""

    @pytest.fixture
    def mock_response(self):
        return {
            "id": "gen-123",
            "object": "text_completion",
            "choices": [
                {
                    "text": "Server response",
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        }

    @pytest.fixture
    def engine(self, mock_response):
        engine = VLLMServerEngine(
            base_url="http://localhost:8000/v1",
            model="test-model",
        )

        # Mock the HTTP request
        def mock_request(endpoint, method="GET", json_data=None, stream=False):
            return mock_response

        engine._make_request = MagicMock(side_effect=mock_request)
        return engine

    def test_generate_basic(self, engine):
        result = engine.generate("Hello")
        assert result.text == "Server response"
        assert result.stop_reason == StopReason.END_OF_SEQUENCE

    def test_generate_parses_usage(self, engine):
        result = engine.generate("Test")
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 15

    def test_generate_with_config(self, engine):
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            stop_sequences=["END"],
        )
        engine.generate("Test", config=config)

        call_args = engine._make_request.call_args
        json_data = call_args[1]["json_data"]

        assert json_data["max_tokens"] == 100
        assert json_data["temperature"] == 0.5
        assert json_data["stop"] == ["END"]


class TestVLLMServerEngineChat:
    """Tests for VLLMServerEngine.chat()."""

    @pytest.fixture
    def mock_chat_response(self):
        return {
            "id": "chat-123",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Chat server response",
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

    @pytest.fixture
    def engine(self, mock_chat_response):
        engine = VLLMServerEngine(
            base_url="http://localhost:8000/v1",
            model="test-model",
        )

        def mock_request(endpoint, method="GET", json_data=None, stream=False):
            return mock_chat_response

        engine._make_request = MagicMock(side_effect=mock_request)
        return engine

    def test_chat_basic(self, engine):
        messages = [Message(role="user", content="Hello")]
        result = engine.chat(messages)
        assert result.text == "Chat server response"

    def test_chat_sends_messages(self, engine):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="How are you?"),
        ]
        engine.chat(messages)

        call_args = engine._make_request.call_args
        json_data = call_args[1]["json_data"]

        assert len(json_data["messages"]) == 3
        assert json_data["messages"][0]["role"] == "user"
        assert json_data["messages"][0]["content"] == "Hello"


class TestVLLMServerEngineAvailability:
    """Tests for VLLMServerEngine availability checking."""

    def test_is_available_true(self):
        engine = VLLMServerEngine(base_url="http://localhost:8000/v1")

        def mock_list_models():
            return ["model-1", "model-2"]

        engine._list_models = MagicMock(side_effect=mock_list_models)

        assert engine.is_available() is True

    def test_is_available_false_on_error(self):
        engine = VLLMServerEngine(base_url="http://localhost:8000/v1")
        engine._list_models = MagicMock(side_effect=Exception("Connection refused"))

        assert engine.is_available() is False

    def test_is_available_false_no_models(self):
        engine = VLLMServerEngine(base_url="http://localhost:8000/v1")
        engine._list_models = MagicMock(return_value=[])

        assert engine.is_available() is False
